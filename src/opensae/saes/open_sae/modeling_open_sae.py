import os
import sys

import torch
from torch import Tensor
import einops

import transformers

from ...sae_utils import (
    PreTrainedSae, 
    SaeEncoderOutput, 
    SaeDecoderOutput, 
    SaeForwardOutput,
    torch_decode,
    triton_decode
)
from ...sparse_activations import (
    TopK,
    GlobalTopK,
    JumpReLU
)
from .configuration_open_sae import OpenSaeConfig


class PreTrainedOpenSae(PreTrainedSae):
    """
    Interface class to set config class and weight initialization
    """
    is_parallelizable = False
    config_class = OpenSaeConfig
    
    def _init_weights(self, module: torch.nn.Module):
        """Initialize the weights."""
        return

import torch
import torch.distributed as dist  # 必须引入这个
from torch import Tensor
import einops
class OpenSae(PreTrainedOpenSae):
    def __init__(
        self, 
        config: OpenSaeConfig,
        device: str | torch.device = None,
        decoder: bool = True,
        model_parallel_group: dist.ProcessGroup = None,  # <--- 【修改1】增加这个参数
        **kwargs
    ):
        super().__init__(config, **kwargs)
        
        device = torch.device("cpu") if device is None else torch.device(device)
        self.decoder = decoder
        self.group = model_parallel_group  # 保存进程组
        
        # --- 【修改2】计算当前卡的特征数 (Local Feature Size) ---
        self.mp_world_size = dist.get_world_size(self.group) if self.group else 1
        self.mp_rank = dist.get_rank(self.group) if self.group else 0
        
        if self.config.feature_size % self.mp_world_size != 0:
            raise ValueError(
                f"Feature size ({self.config.feature_size}) must be divisible by "
                f"MP size ({self.mp_world_size})"
            )
            
        self.local_feature_size = self.config.feature_size // self.mp_world_size
        # -----------------------------------------------------

        # --- 【修改3】使用 local_feature_size 初始化 Encoder ---
        self.encoder = torch.nn.Linear(
            in_features = self.config.hidden_size, 
            out_features = self.local_feature_size,  # <--- 这里改成了 local
            device = device, 
            dtype = self.config.get_torch_dtype()
        )
        self.encoder.bias.data.zero_()

        # --- 【修改4】使用 local_feature_size 初始化 Decoder ---
        # 你的 Decoder 权重是从 Encoder 克隆的，这没问题，因为它会自动继承 local shape
        self.W_dec = torch.nn.Parameter(self.encoder.weight.data.clone()) if self.decoder else None
        
        if self.decoder and self.config.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()
            
        # Decoder Bias 保持完整大小 (Hidden Size)，不需要切分
        self.b_dec = torch.nn.Parameter(
            torch.zeros(
                self.config.hidden_size,
                dtype = self.config.get_torch_dtype(), 
                device = device
            )
        )

        self.sparse_activation = None
        if self.config.activation == "topk":
            # --- 修改开始 ---
            # 判断是否启用了 MP，如果启用了，就用 GlobalTopK
            if self.group is not None and dist.get_world_size(self.group) > 1:
                # 传入 k 和 通信组
                self.sparse_activation = GlobalTopK(k=self.config.k, process_group=self.group)
                if self.config.multi_topk:
                # multi_topk 通常用于辅助 Loss，建议保持 Local 即可，或者同理修改
                    self.multi_topk = GlobalTopK(k = self.config.k * self.config.multi_topk,process_group=self.group)
                # 打印一下确认你用到它了
                if dist.get_rank(self.group) == 0:
                    print(f"Algorithm: Using Global Top-K (k={self.config.k}) with Communication.")
            else:
                # 单卡模式保持原样
                self.sparse_activation = TopK(k=self.config.k)
                if self.config.multi_topk:
                    self.multi_topk=TopK(k=self.config.k*self.config.multi_topk)
            # --- 修改结束 ---
            

        if self.config.decoder_impl == "triton":
            self.decode_fn = triton_decode
        elif self.config.decoder_impl == "torch":
            self.decode_fn = torch_decode
            
        # 打印调试信息
        print(f"[Rank {self.mp_rank}] Encoder weight shape: {self.encoder.weight.shape}")

    # ... 下面的方法 (forward, encode, decode 等) 保持原样 ...
    # ... 但是在 forward 里，你的 TritonDecoder 会负责处理 all-reduce ...

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
        
    def normalization(self, x: Tensor, eps: float = 1e-5) -> Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + eps)
        return x, mu, std

    def pre_process(self, hidden: Tensor) -> Tensor:
        if self.config.input_normalize:
            hidden, mu, std = self.normalization(hidden, self.config.input_normalize_eps)
        
        if not self.config.normalize_shift_back:
            mu, std = None, None
        
        # Remove decoder bias as per Anthropic
        return hidden.to(self.b_dec.dtype) - self.b_dec, mu, std


    def encode(self, hidden: Tensor, return_all_features: bool = False) -> SaeEncoderOutput:
        sae_input, input_mean, input_std = self.pre_process(hidden)
        all_features = self.encoder(sae_input)
        # Remove negative features
        all_features = torch.nn.functional.relu(all_features)
        
        feature_activation, feature_indices = self.sparse_activation(all_features)
        
        return SaeEncoderOutput(
            sparse_feature_activations = feature_activation,
            sparse_feature_indices = feature_indices,
            all_features = all_features if return_all_features else None,
            input_mean = input_mean if self.config.input_normalize else None,
            input_std = input_std if self.config.input_normalize else None
        )


    def decode(
        self, 
        feature_indices: Tensor, 
        feature_activation: Tensor,
        input_mean: Tensor | None = None,
        input_std: Tensor | None = None
    ) -> SaeDecoderOutput:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        if self.config.normalize_shift_back:
            assert input_mean is not None and input_std is not None, "Input mean and std must be provided for shift back normalization."            

        with torch.cuda.device(self.W_dec.device.index):
            # 【修改5】如果是 Triton 实现，确保传入 process_group
            if self.config.decoder_impl == "triton":
                # 注意：这里假设你的 triton_decode 已经被修改为接受 process_group 参数
                # 或者你直接调用了 kernels.py 里的 TritonDecoder.apply
                reconstruction = self.decode_fn(
                    feature_indices,
                    feature_activation.to(torch.float32),
                    self.W_dec.mT.to(torch.float32),
                    process_group = self.group  # <--- 必须传入这个！
                )
            else:
                # 如果是 Torch 实现，你也需要手动做 all-reduce
                reconstruction = self.decode_fn(...)
                if self.group is not None and dist.get_world_size(self.group) > 1:
                    dist.all_reduce(reconstruction, op=dist.ReduceOp.SUM, group=self.group)
        reconstruction = reconstruction + self.b_dec
        
        if self.config.normalize_shift_back:
            reconstruction = reconstruction * (input_std + self.config.input_normalize_eps) + input_mean

        return SaeDecoderOutput(sae_output = reconstruction)


    def reconstruction_loss(
        self,
        hidden: Tensor,
        hidden_variance: Tensor,
        sae_output: Tensor
    ) -> Tensor:
        reconstruction_error = sae_output - hidden
        dimensional_l2_loss = reconstruction_error.pow(2).sum(0)    # size = (hidden_size,), per-dimensional L2 loss
        normalized_l2_loss = dimensional_l2_loss / hidden_variance  # putting everything on a reasonable scale
        reconstruction_loss = torch.mean(normalized_l2_loss)
        l2_loss = dimensional_l2_loss.mean()
        
        return (
            reconstruction_error,
            l2_loss,
            reconstruction_loss,
        )
        
    
    def auxk_loss(
        self, 
        hidden: Tensor, 
        sae_output: Tensor,
        reconstruction_error: Tensor,
        hidden_variance: Tensor, 
        dead_mask: Tensor,
        all_features: Tensor,
        input_mean: Tensor | None = None,
        input_std: Tensor | None = None
    ) -> Tensor:
        assert dead_mask is not None, "Dead mask is not provided."
        num_dead = int(dead_mask.sum())
        if num_dead == 0:
            return sae_output.new_tensor(0.0)


        # Heuristic from Appendix B.1 in the paper
        k_aux = hidden.shape[-1] // 2

        # Reduce the scale of the loss if there are a small number of dead latents
        scale = min(num_dead / k_aux, 1.0)
        k_aux = min(k_aux, num_dead)

        # Don't include living latents in this loss
        auxk_all_features = torch.where(dead_mask[None], all_features, -torch.inf)

        # Top-k dead latents
        auxk_feature_activations, auxk_feature_indices = auxk_all_features.topk(k_aux, sorted=False)

        # Encourage the top ~50% of dead latents to predict the residual of the
        # top k living latents
        auxk_sae_decoder_output = self.decode(
            auxk_feature_indices, 
            auxk_feature_activations,
            input_mean, input_std
        ).sae_output
        auxk_loss = (auxk_sae_decoder_output - reconstruction_error).pow(2).sum(0)
        auxk_loss = scale * torch.mean(auxk_loss / hidden_variance)
        
        return auxk_loss


    def forward(
        self, 
        hidden: Tensor, 
        dead_mask: Tensor | None = None
    ) -> SaeForwardOutput:
        # 1. SAE computation: hidden --> [encode] --> features --> [decode] --> reconstruction
        sae_encoder_output = self.encode(hidden, return_all_features = self.config.multi_topk)
        
        # [Decode] Inside here, you already did All-Reduce. 
        # So sae_decoder_output is the FULL reconstructed vector (e.g. shape [Batch, 4096])
        sae_decoder_output = self.decode(
            sae_encoder_output.sparse_feature_indices, 
            sae_encoder_output.sparse_feature_activations,
            sae_encoder_output.input_mean,
            sae_encoder_output.input_std
        ).sae_output
        
        assert sae_decoder_output.shape == hidden.shape, f"Output shape {sae_decoder_output.shape} does not match input shape {hidden.shape}"
        
        # 2. Prepare per-dimensional variance to make the training-loss stable
        per_dimension_variance = (hidden - hidden.mean(0)).pow(2).sum(0)       # size = (hidden_size,)
        per_dimension_variance = torch.clamp(per_dimension_variance, min=1.0)  # clip to ensure total_variance < 5.0
        
        # 3. Compute losses
        
        # 3.1. Reconstruction loss
        # This loss is calculated on the full vector. It is the CORRECT global loss.
        # DO NOT divide by mp_world_size.
        reconstruction_error, l2_loss, reconstruction_loss = self.reconstruction_loss(
            hidden = hidden, 
            hidden_variance =  per_dimension_variance, 
            sae_output = sae_decoder_output
        )

        # 3.2. AuxK loss
        if self.config.auxk_alpha > 1e-6 and dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            auxk_loss = self.auxk_loss(
                hidden = hidden,
                sae_output = sae_decoder_output,
                reconstruction_error = reconstruction_error,
                hidden_variance = per_dimension_variance,
                dead_mask = dead_mask,
                all_features = sae_encoder_output.all_features,
                input_mean = sae_encoder_output.input_mean,
                input_std = sae_encoder_output.input_std
            )
            # DO NOT divide by mp_world_size.
        else:
            auxk_loss = sae_decoder_output.new_tensor(0.0)

        # 3.3. Multi-TopK loss
        if self.config.multi_topk:
            multi_topk_feature_activations, multi_topk_feature_indices = self.multi_topk(sae_encoder_output.all_features)
            
            # Note: You need to make sure multi_topk also uses GlobalTopK logic if you want strict consistency,
            # but usually multi_topk is just an auxiliary loss so Local TopK is fine.
            
            multi_topk_sae_decoder_output = self.decode(
                multi_topk_feature_indices, multi_topk_feature_activations,
                sae_encoder_output.input_mean, sae_encoder_output.input_std
            ).sae_output

            _, _, multi_topk_loss = self.reconstruction_loss(
                hidden = hidden, 
                hidden_variance = per_dimension_variance, 
                sae_output = multi_topk_sae_decoder_output
            )
            # DO NOT divide by mp_world_size.
        else:
            multi_topk_loss = sae_decoder_output.new_tensor(0.0)


        # 3.4. L1 loss
        l1_loss = torch.tensor(0.0, device=hidden.device)
        if self.config.l1_coef is not None and self.config.l1_coef > 1e-8:
            # Note: L1 loss is calculated on LOCAL features only.
            # But since L1 is "average of absolute values", average of local parts == average of global parts
            # (assuming uniform distribution). So this is fine.
            l1_loss = torch.norm(sae_encoder_output.all_features, p=1, dim=-1).mean() * self.config.l1_coef

        # Final Sum
        final_loss = reconstruction_loss + multi_topk_loss / 8 + auxk_loss * self.config.auxk_alpha
        if l1_loss > 1e-8:
            final_loss += l1_loss

        return SaeForwardOutput(
            # Encoder Outputs
            sparse_feature_activations = sae_encoder_output.sparse_feature_activations,
            sparse_feature_indices = sae_encoder_output.sparse_feature_indices,
            all_features = sae_encoder_output.all_features,
            # Decoder Outputs
            sae_output = sae_decoder_output,
            # Loss Outputs
            reconstruction_loss = reconstruction_loss,
            multi_topk_loss = multi_topk_loss,
            auxk_loss = auxk_loss,
            l1_loss = l1_loss,
            loss = final_loss,
        )
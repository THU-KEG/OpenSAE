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
import torch.distributed as dist
from torch import Tensor
import einops

class OpenSae(PreTrainedOpenSae):
    def __init__(
        self, 
        config: OpenSaeConfig,
        device: str | torch.device = None,
        decoder: bool = True,
        model_parallel_group: dist.ProcessGroup = None,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        
        device = torch.device("cpu") if device is None else torch.device(device)
        self.decoder = decoder
        self.group = model_parallel_group
        
        # --- 计算当前卡的特征数 (Local Feature Size) ---
        self.mp_world_size = dist.get_world_size(self.group) if self.group else 1
        self.mp_rank = dist.get_rank(self.group) if self.group else 0
        
        if self.config.feature_size % self.mp_world_size != 0:
            raise ValueError(
                f"Feature size ({self.config.feature_size}) must be divisible by "
                f"MP size ({self.mp_world_size})"
            )
            
        self.local_feature_size = self.config.feature_size // self.mp_world_size

        # --- 使用 local_feature_size 初始化 Encoder (保证随机性一致) ---
        full_encoder = torch.nn.Linear(
            in_features=self.config.hidden_size, 
            out_features=self.config.feature_size, 
            bias=True)
        print("全块权重",full_encoder.weight.sum().item())

        # 手动切分权重
        start_idx = self.mp_rank * self.local_feature_size
        end_idx = (self.mp_rank + 1) * self.local_feature_size
        
        with torch.no_grad():
            local_weight = full_encoder.weight.data[start_idx:end_idx, :].clone()
            local_bias = full_encoder.bias.data[start_idx:end_idx].clone()

        # 赋值给本地 Encoder
        self.encoder = torch.nn.Linear(
            in_features=self.config.hidden_size, 
            out_features=self.local_feature_size,
            device=device,
            dtype=self.config.get_torch_dtype()
        )
        print("过一会的decoder,",self.encoder.weight.sum().item())
        self.encoder.weight.data.copy_(local_weight)
        print("copy之后的,",self.encoder.weight.sum().item())
        self.encoder.bias.data.copy_(local_bias)

        del full_encoder
        self.encoder.bias.data.zero_()

        # --- 初始化 Decoder ---
        self.W_dec = torch.nn.Parameter(self.encoder.weight.data.clone()) if self.decoder else None
        
        if self.decoder and self.config.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()
            
        self.b_dec = torch.nn.Parameter(
            torch.zeros(
                self.config.hidden_size,
                dtype = self.config.get_torch_dtype(), 
                device = device
            )
        )

        self.sparse_activation = None
        if self.config.activation == "topk":
            # 判断是否启用了 MP，如果启用了，就用 GlobalTopK
            if self.group is not None and dist.get_world_size(self.group) > 1:
                self.sparse_activation = GlobalTopK(k=self.config.k, process_group=self.group)
                if self.config.multi_topk:
                    self.multi_topk = GlobalTopK(k = self.config.k * self.config.multi_topk, process_group=self.group)
                if dist.get_rank(self.group) == 0:
                    print(f"Algorithm: Using Global Top-K (k={self.config.k}) with Communication.")
            else:
                self.sparse_activation = TopK(k=self.config.k)
                if self.config.multi_topk:
                    self.multi_topk=TopK(k=self.config.k*self.config.multi_topk)

        if self.config.decoder_impl == "triton":
            self.decode_fn = triton_decode
        elif self.config.decoder_impl == "torch":
            self.decode_fn = torch_decode
            
        print(f"[Rank {self.mp_rank}] Encoder weight shape: {self.encoder.weight.shape}")
        print("平均权重为",self.encoder.weight.sum().item())

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        assert self.W_dec.grad is not None 

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
        return hidden.to(self.b_dec.dtype) - self.b_dec, mu, std

    def encode(self, hidden: Tensor, return_all_features: bool = False) -> SaeEncoderOutput:
        sae_input, input_mean, input_std = self.pre_process(hidden)
        all_features = self.encoder(sae_input)
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
            assert input_mean is not None and input_std is not None, "Input mean/std missing."            

        with torch.cuda.device(self.W_dec.device.index):
            if self.config.decoder_impl == "triton":
                reconstruction = self.decode_fn(
                    feature_indices,
                    feature_activation.to(torch.float32),
                    self.W_dec.mT.to(torch.float32),
                    process_group = self.group 
                )
            else:
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
        dimensional_l2_loss = reconstruction_error.pow(2).sum(0)
        normalized_l2_loss = dimensional_l2_loss / hidden_variance
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

        # 设定全局目标 k_aux (例如 2048)
        # 即使在 TP 模式下，我们也希望总共只复活 2048 个，所以不除以 mp_world_size
        k_aux = hidden.shape[-1] // 2 

        scale = min(num_dead / k_aux, 1.0)
        
        # 如果当前卡的死特征连全局 k_aux 都不到，那就以死特征数量为上限
        k_aux_limit = min(k_aux, num_dead)

        # 排除活特征，设为 -inf
        auxk_all_features = torch.where(dead_mask[None], all_features, -torch.inf)

        # 1. 本地海选：先在本地选出前 k_aux 个候选者
        # (即使全局只需要 2048 个，我们本地也提供 2048 个最好的供全局挑选)
        local_vals, local_inds = auxk_all_features.topk(k_aux_limit, sorted=False)

        # ================== 通信筛选逻辑 (Communication Selection) ==================
        if self.mp_world_size > 1:
            # 2. 如果本地不足 k_aux 个，我们需要填充 padding 以便 all_gather
            # 这一步是为了防止某些卡死特征很少，导致张量形状不一致
            if k_aux_limit < k_aux:
                pad_size = k_aux - k_aux_limit
                local_vals = torch.cat([local_vals, torch.full((local_vals.shape[0], pad_size), -float('inf'), device=local_vals.device)], dim=1)
                # indices 填充无所谓，反正值是 -inf
                local_inds = torch.cat([local_inds, torch.zeros((local_inds.shape[0], pad_size), dtype=local_inds.dtype, device=local_inds.device)], dim=1)

            # 3. 收集所有卡的候选分值
            local_vals = local_vals.contiguous()
            gathered_vals = [torch.zeros_like(local_vals) for _ in range(self.mp_world_size)]
            dist.all_gather(gathered_vals, local_vals, group=self.group)
            
            # 4. 全局排序与划线
            all_candidates = torch.cat(gathered_vals, dim=1) # [Batch, k_aux * mp_size]
            # 找出全局第 k_aux 大的值
            global_topk_vals, _ = torch.topk(all_candidates, k_aux, dim=-1)
            threshold = global_topk_vals[:, -1].unsqueeze(1) # [Batch, 1]
            
            # 5. 本地过滤：只有大于等于全局阈值的才保留
            # (注意：我们要切回原来的 k_aux_limit 长度，去掉 padding)
            real_local_vals = local_vals[:, :k_aux_limit]
            mask = real_local_vals >= threshold
            
            auxk_feature_activations = real_local_vals * mask.to(real_local_vals.dtype)
            auxk_feature_indices = local_inds[:, :k_aux_limit]
        else:
            # 单卡模式
            auxk_feature_activations = local_vals
            auxk_feature_indices = local_inds
        # =========================================================================

        # Decode & Compute Loss
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
        dead_mask: Tensor | None = None,
        external_variance: Tensor | None = None
    ) -> SaeForwardOutput:
        # 1. SAE computation
        sae_encoder_output = self.encode(hidden, return_all_features = self.config.multi_topk)
        
        sae_decoder_output = self.decode(
            sae_encoder_output.sparse_feature_indices, 
            sae_encoder_output.sparse_feature_activations,
            sae_encoder_output.input_mean,
            sae_encoder_output.input_std
        ).sae_output
        
        assert sae_decoder_output.shape == hidden.shape, f"Output shape mismatch"
        
        # 2. Variance
        if external_variance is not None:
            # 如果传了全局方差，直接用它（这就是我们想要的！）
            per_dimension_variance = external_variance
        else:
            # 否则回退到计算当前切片的局部方差
            per_dimension_variance = (hidden - hidden.mean(0)).pow(2).sum(0)
            per_dimension_variance = torch.clamp(per_dimension_variance, min=1.0)
        
        # 3. Compute losses
        
        # 3.1. Reconstruction loss (DO NOT DIVIDE BY MP_SIZE)
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
            # DO NOT DIVIDE BY MP_SIZE
        else:
            auxk_loss = sae_decoder_output.new_tensor(0.0)

        # 3.3. Multi-TopK loss
        if self.config.multi_topk:
            multi_topk_feature_activations, multi_topk_feature_indices = self.multi_topk(sae_encoder_output.all_features)
            
            multi_topk_sae_decoder_output = self.decode(
                multi_topk_feature_indices, multi_topk_feature_activations,
                sae_encoder_output.input_mean, sae_encoder_output.input_std
            ).sae_output

            _, _, multi_topk_loss = self.reconstruction_loss(
                hidden = hidden, 
                hidden_variance = per_dimension_variance, 
                sae_output = multi_topk_sae_decoder_output
            )
            # DO NOT DIVIDE BY MP_SIZE
        else:
            multi_topk_loss = sae_decoder_output.new_tensor(0.0)

        # 3.4. L1 loss
        l1_loss = torch.tensor(0.0, device=hidden.device)
        if self.config.l1_coef is not None and self.config.l1_coef > 1e-8:
            l1_loss = torch.norm(sae_encoder_output.all_features, p=1, dim=-1).mean() * self.config.l1_coef

        final_loss = reconstruction_loss + multi_topk_loss / 8 + auxk_loss * self.config.auxk_alpha
        if l1_loss > 1e-8:
            final_loss += l1_loss

        return SaeForwardOutput(
            sparse_feature_activations = sae_encoder_output.sparse_feature_activations,
            sparse_feature_indices = sae_encoder_output.sparse_feature_indices,
            all_features = sae_encoder_output.all_features,
            sae_output = sae_decoder_output,
            reconstruction_loss = reconstruction_loss,
            multi_topk_loss = multi_topk_loss,
            auxk_loss = auxk_loss,
            l1_loss = l1_loss,
            loss = final_loss,
        )
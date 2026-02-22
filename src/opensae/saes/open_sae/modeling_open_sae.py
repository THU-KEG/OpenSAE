import os
import sys
import torch
import torch.distributed as dist
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
        
        # --- 1. 计算并行参数 ---
        self.mp_world_size = dist.get_world_size(self.group) if self.group else 1
        self.mp_rank = dist.get_rank(self.group) if self.group else 0
        
        if self.config.feature_size % self.mp_world_size != 0:
            raise ValueError(
                f"Feature size ({self.config.feature_size}) must be divisible by "
                f"MP size ({self.mp_world_size})"
            )
            
        self.local_feature_size = self.config.feature_size // self.mp_world_size

        # --- 2. 关键步骤：先生成全量，再进行切分 ---
        # 目的：确保所有Rank消耗相同数量的随机数，从而保证初始化的一致性
        # 注意：在CPU上生成以避免大模型初始化时的显存峰值
        full_encoder = torch.nn.Linear(
            in_features=self.config.hidden_size, 
            out_features=self.config.feature_size, 
            bias=True
        )
# ================= NEW: 打印前几个数值 =================
        with torch.no_grad():
            # 1. 取出权重数据，展平，取前 5 个数
            # .flatten() 把矩阵变成一维数组
            # .tolist() 转成 Python 列表，打印出来好看
            head_values = full_encoder.weight.data.flatten()[:5].tolist()
            
            # 2. 计算所有权重的总和 (这是验证两张卡初始化是否完全一样的最强证据)
            total_sum = full_encoder.weight.data.sum().item()
            
            print(f"\n[Global Init Check] Full Encoder Head (前5个): {[round(x, 6) for x in head_values]}")
            print(f"[Global Init Check] Full Encoder Sum  (总和): {total_sum:.6f}\n")
        # =======================================================

        # --- 3. 手动切分权重 ---
        start_idx = self.mp_rank * self.local_feature_size
        end_idx = (self.mp_rank + 1) * self.local_feature_size
        
        with torch.no_grad():
            # 从全量权重中“抠”出属于当前Rank的部分
            local_weight = full_encoder.weight.data[start_idx:end_idx, :].clone()
            local_bias = full_encoder.bias.data[start_idx:end_idx].clone()

        # --- 4. 赋值给本地 Encoder ---
        self.encoder = torch.nn.Linear(
            in_features=self.config.hidden_size, 
            out_features=self.local_feature_size,
            device=device,
            dtype=self.config.get_torch_dtype()
        )
        
        self.encoder.weight.data.copy_(local_weight)
        self.encoder.bias.data.copy_(local_bias)

        # 立即删除全量模型，释放内存
        del full_encoder
        self.encoder.bias.data.zero_()

        # --- 5. 初始化 Decoder (Tied Weights) ---
        # 直接使用切分好的 encoder 权重初始化 decoder，保证 decoder 也是正确切分的
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

        # --- 6. 稀疏激活函数 ---
        self.sparse_activation = None
        if self.config.activation == "topk":
            if self.group is not None and dist.get_world_size(self.group) > 1:
                self.sparse_activation = GlobalTopK(k=self.config.k, process_group=self.group)
                if self.config.multi_topk:
                    self.multi_topk = GlobalTopK(k = self.config.k * self.config.multi_topk, process_group=self.group)
                if dist.get_rank(self.group) == 0:
                    print(f"Algorithm: Using Global Top-K (k={self.config.k}) with Communication.")
            else:
                self.sparse_activation = TopK(k=self.config.k)
                if self.config.multi_topk:
                    self.multi_topk = TopK(k=self.config.k*self.config.multi_topk)

        if self.config.decoder_impl == "triton":
            self.decode_fn = triton_decode
        elif self.config.decoder_impl == "torch":
            self.decode_fn = torch_decode
            
        # --- 7. MoE Router 初始化 ---
        self.use_moe = config.num_experts > 1
        if self.use_moe:
            self.router = torch.nn.Linear(config.hidden_size, config.num_experts, bias=False, device=device)
            # 这里的 normal_ 初始化是安全的，因为前面的 full_encoder 已经消耗了固定的随机数
            torch.nn.init.normal_(self.router.weight, std=0.02)
            
            self.global_features_per_expert = config.feature_size // config.num_experts
            
            if config.num_experts % self.mp_world_size != 0:
                raise ValueError("MoE Error: num_experts must be divisible by MP size for simple alignment.")
            
            self.experts_per_rank = config.num_experts // self.mp_world_size
            self.my_expert_start_idx = self.mp_rank * self.experts_per_rank
            self.my_expert_end_idx = (self.mp_rank + 1) * self.experts_per_rank
            
            print(f"[Rank {self.mp_rank}] MoE Active: Managing Experts {self.my_expert_start_idx} to {self.my_expert_end_idx-1}")

        # ==========================================
        # [Debug Print] 验证初始化一致性 (保留这个以供检查)
        # ==========================================
        with torch.no_grad():
            global_rank = dist.get_rank() if dist.is_initialized() else 0
            
            enc_flat = self.encoder.weight.data.flatten()
            enc_head = enc_flat[:3].cpu().numpy().tolist()
            enc_sum = enc_flat.sum().item()
            
            router_msg = "N/A"
            if self.use_moe:
                r_flat = self.router.weight.data.flatten()
                r_head = r_flat[:3].cpu().numpy().tolist()
                r_sum = r_flat.sum().item()
                router_msg = f"Sum={r_sum:.6f} | Head={[round(x, 6) for x in r_head]}"
            
            print(
                f"\n[🔍 Init Check] GlobalRank {global_rank} | MP_Rank {self.mp_rank}/{self.mp_world_size}\n"
                f"   >>> Encoder Slice: Sum={enc_sum:.6f} | Head={[round(x, 6) for x in enc_head]}\n"
                f"   >>> Router Weights: {router_msg} (All ranks MUST match this!)"
            )
            
            if dist.is_initialized():
                dist.barrier()

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
    def _compute_moe_losses(self, router_logits: Tensor):
        """实现 DeepSeek-V3 风格的负载均衡 Loss"""
        # 1. Router Z-Loss: 提升数值稳定性，防止 Logits 过大
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean() * getattr(self.config, "router_z_loss_coef", 1e-4)

        # 2. 计算软概率
        probs = torch.softmax(router_logits, dim=-1) # [Batch, Num_Experts]
        local_prob_sum = probs.sum(0)
        local_count = torch.tensor([probs.shape[0]], device=probs.device, dtype=probs.dtype)
        
        # 3. 全局同步统计数据
        if dist.is_initialized():
            stats = torch.cat([local_prob_sum, local_count])
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            global_prob_sum = stats[:self.config.num_experts]
            global_total_count = stats[-1]
        else:
            global_prob_sum = local_prob_sum
            global_total_count = local_count

        # 4. CV Loss (变异系数平方)
        # f_i: 每个专家的平均激活概率
        f_i = global_prob_sum / (global_total_count + 1e-6)
        f_bar = f_i.mean()
        # DeepSeek 公式: CV^2 = mean((f_i/f_bar - 1)^2)
        moe_cv_loss = torch.mean((f_i / (f_bar + 1e-6) - 1).pow(2)) * self.config.moe_loss_coef

        return moe_cv_loss, z_loss
    def encode(self, hidden: Tensor, expert_scale: Tensor | None = None, return_all_features: bool = False) -> SaeEncoderOutput:
        sae_input, input_mean, input_std = self.pre_process(hidden)
        all_features = self.encoder(sae_input)
        
        # [Fix] 使用乘法应用 Expert 权重（梯度可导）
        if expert_scale is not None:
            all_features = all_features * expert_scale
            
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
                reconstruction = self.decode_fn(
                    feature_indices,
                    feature_activation,
                    self.W_dec,
                    process_group = self.group
                )
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

        k_aux = hidden.shape[-1] // 2 
        scale = min(num_dead / k_aux, 1.0)
        k_aux_limit = min(k_aux, num_dead)

        # 排除活特征
        auxk_all_features = torch.where(dead_mask[None], all_features, -torch.inf)

        # 1. 本地海选
        local_vals, local_inds = auxk_all_features.topk(k_aux_limit, sorted=False)

        # ================== 通信筛选逻辑 ==================
        if self.mp_world_size > 1:
            if k_aux_limit < k_aux:
                pad_size = k_aux - k_aux_limit
                local_vals = torch.cat([local_vals, torch.full((local_vals.shape[0], pad_size), -float('inf'), device=local_vals.device)], dim=1)
                local_inds = torch.cat([local_inds, torch.zeros((local_inds.shape[0], pad_size), dtype=local_inds.dtype, device=local_inds.device)], dim=1)

            local_vals = local_vals.contiguous()
            gathered_vals = [torch.zeros_like(local_vals) for _ in range(self.mp_world_size)]
            dist.all_gather(gathered_vals, local_vals, group=self.group)
            
            all_candidates = torch.cat(gathered_vals, dim=1) 
            global_topk_vals, _ = torch.topk(all_candidates, k_aux, dim=-1)
            threshold = global_topk_vals[:, -1].unsqueeze(1) 
            
            real_local_vals = local_vals[:, :k_aux_limit]
            mask = real_local_vals >= threshold
            
            auxk_feature_activations = real_local_vals * mask.to(real_local_vals.dtype)
            auxk_feature_indices = local_inds[:, :k_aux_limit]
        else:
            auxk_feature_activations = local_vals
            auxk_feature_indices = local_inds

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
        moe_cv_loss = torch.tensor(0.0, device=hidden.device)
        router_z_loss = torch.tensor(0.0, device=hidden.device)
        expert_scale = None 
        expert_mask_for_logging = None 
        
        if self.use_moe:
            router_logits = self.router(hidden) 
            # 1. 计算现代化 MoE Loss
            moe_cv_loss, router_z_loss = self._compute_moe_losses(router_logits)

            topk_vals, selected_experts = torch.topk(router_logits, self.config.k_experts, dim=-1)
            routing_weights = torch.softmax(topk_vals, dim=-1) 
            with torch.no_grad():
                # 1. 看看当前 Batch 里的路由权重分布
                # routing_weights shape: [Batch, k_experts]
                mean_weight = routing_weights.mean(dim=0) # 每个被选中的专家的平均权重
                max_w = routing_weights.max()
                min_w = routing_weights.min()
                
                # 2. 看看 Logits 的数值范围 (检查 Z-Loss 是否有效)
                logits_std, logits_mean = router_logits.std(), router_logits.mean()

                print(f"\n--- [Router Debug")
                print(f"Logits: Mean={logits_mean:.4f}, Std={logits_std:.4f}")
                print(f"Weights: Max={max_w:.4f}, Min={min_w:.4f}, Mean_of_TopK={mean_weight.mean():.4f}")
                # 打印前 8 个被选中专家的索引，看看是不是每次都选一样的
                #print(f"Selected Experts (First 3 tokens): \n{selected_experts[:3]}")
                print("-------------------------------------------\n")
                # 2. 路由选择与权重生成
            global_expert_weights = torch.zeros_like(router_logits)
            global_expert_weights.scatter_(1, selected_experts, routing_weights)
            
            # 3. TP 权重切分与扩展 (将 Expert 权重映射到对应的 Feature 上)
            local_expert_weights = global_expert_weights[:, self.my_expert_start_idx : self.my_expert_end_idx]
            expert_scale = local_expert_weights.repeat_interleave(self.global_features_per_expert, dim=1)
            expert_mask_for_logging = (expert_scale > 0)

        # SAE 主体计算
        sae_enc_out = self.encode(hidden, expert_scale=expert_scale, return_all_features=self.config.multi_topk)
        sae_dec_out = self.decode(sae_enc_out.sparse_feature_indices, sae_enc_out.sparse_feature_activations,
                                 sae_enc_out.input_mean, sae_enc_out.input_std).sae_output
        
        # Loss 计算
        var = external_variance if external_variance is not None else torch.clamp((hidden - hidden.mean(0)).pow(2).sum(0), min=1.0)
        recon_err, l2_loss, recon_loss = self.reconstruction_loss(hidden, var, sae_dec_out)

        # AuxK 与 Multi-TopK (省略细节实现，保持逻辑结构)
        auxk_loss = self.auxk_loss(hidden, sae_dec_out, recon_err, var, dead_mask, sae_enc_out.all_features, 
                                  sae_enc_out.input_mean, sae_enc_out.input_std) if dead_mask is not None else torch.tensor(0.0, device=hidden.device)
        
        l1_loss = torch.norm(sae_enc_out.all_features, p=1, dim=-1).mean() * self.config.l1_coef if self.config.l1_coef else torch.tensor(0.0, device=hidden.device)

        # 最终 Loss 合成
        final_loss = recon_loss + auxk_loss * self.config.auxk_alpha + moe_cv_loss + router_z_loss + l1_loss
            
        return SaeForwardOutput(
            sparse_feature_activations = sae_enc_out.sparse_feature_activations,
            sparse_feature_indices = sae_enc_out.sparse_feature_indices,
            sae_output = sae_dec_out,
            reconstruction_loss = recon_loss,
            loss = final_loss,
            aux_moe_loss = moe_cv_loss,
            expert_mask = expert_mask_for_logging
        )
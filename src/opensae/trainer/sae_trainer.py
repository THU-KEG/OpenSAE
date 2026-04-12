import os
import math
import time
import functools
from collections import defaultdict
from dataclasses import asdict
from typing import Sized
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor, nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, get_linear_schedule_with_warmup, get_wsd_schedule, get_cosine_schedule_with_warmup
import numpy as np

# 假设这些模块在你原本的路径下
from ..saes import OpenSae, OpenSaeConfig
from .train_arguments import TrainConfig, SaeConfig, ModelConfig
from .train_utils import geometric_median
from ..data.collator import packing_collate_fn


class SaeTrainer:
    def __init__(
        self, 
        train_cfg: TrainConfig, 
        sae_cfg: SaeConfig,
        model_cfg: ModelConfig,
        dataset: Dataset, 
        model: PreTrainedModel,
        data_parallel_group: dist.ProcessGroup | None = None,
        model_parallel_group: dist.ProcessGroup | None = None,
        pipeline_parallel_group: dist.ProcessGroup | None = None,
        device_mesh = None
    ):
        self.train_cfg = train_cfg
        self.data_parallel_group = data_parallel_group
        self.model_parallel_group = model_parallel_group
        self.pipeline_parallel_group = pipeline_parallel_group
        self.device_mesh = device_mesh

        # PP 参数
        self.pp_group = pipeline_parallel_group
        self.pp_rank = dist.get_rank(self.pp_group) if self.pp_group else 0
        self.pp_size = dist.get_world_size(self.pp_group) if self.pp_group else 1
        
        # 层切分逻辑
        full_model_layers = model.config.num_hidden_layers
        target_layers = train_cfg.early_exit_inference_layer_num + 1
        effective_num_layers = min(target_layers, full_model_layers)
        
        layers_per_stage = math.ceil(effective_num_layers / self.pp_size)
        
        self.my_start_layer = self.pp_rank * layers_per_stage
        self.my_end_layer = min((self.pp_rank + 1) * layers_per_stage, effective_num_layers)

        # Hookpoint 解析
        target_layer_name = train_cfg.hookpoint
        try:
            self.target_layer_idx = int(target_layer_name.split('.')[1])
        except:
            print(f"Warning: Could not parse layer index from {train_cfg.hookpoint}")
            self.target_layer_idx = -1
             
        self.hook_owner_pp_rank = min(self.target_layer_idx // layers_per_stage, self.pp_size - 1)

        assert self.train_cfg.hookpoint is not None
        assert isinstance(dataset, Sized)
        
        device = torch.device("cuda", torch.cuda.current_device())
        input_width = model.config.hidden_size
        
        # === 关键：完全确定性的初始化 ===
        torch.manual_seed(train_cfg.seed)
        np.random.seed(train_cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(train_cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        open_sae_config = OpenSaeConfig(
            hidden_size = input_width,
            feature_size = sae_cfg.num_latents or input_width * sae_cfg.expansion_factor,
            input_normalize = sae_cfg.input_normalize,
            normalize_shift_back = sae_cfg.shift_back,
            input_hookpoint = self.train_cfg.hookpoint,
            output_hookpoint = self.train_cfg.hookpoint,
            model_name = model_cfg.model,
            activation = "topk",
            k = sae_cfg.k,
            normalize_decoder = sae_cfg.normalize_decoder,
            auxk_alpha = train_cfg.auxk_alpha,
            l1_coef = sae_cfg.l1_coef,
            decoder_impl = sae_cfg.decoder_impl if hasattr(sae_cfg, 'decoder_impl') else "triton",
            num_experts = getattr(sae_cfg, "num_experts", 1),
            k_experts = getattr(sae_cfg, "k_experts", 1),
            moe_loss_coef = getattr(sae_cfg, "moe_loss_coef", 0.01)
        )
        
        self.sae = OpenSae(open_sae_config, device, model_parallel_group=self.model_parallel_group)
        
        # === 初始化后立即同步所有并行组 ===
        if dist.is_initialized():
            # 同步MP组
            if self.model_parallel_group and dist.get_world_size(self.model_parallel_group) > 1:
                mp_src = dist.get_global_rank(self.model_parallel_group, 0)
                for buffer in self.sae.buffers():
                    dist.broadcast(buffer.data, src=mp_src, group=self.model_parallel_group)
            
            # 同步PP组
            if self.pp_group and self.pp_size > 1:
                pp_src = dist.get_global_rank(self.pp_group, 0)
                for param in self.sae.parameters():
                    dist.broadcast(param.data, src=pp_src, group=self.pp_group)
                for buffer in self.sae.buffers():
                    dist.broadcast(buffer.data, src=pp_src, group=self.pp_group)
            
            # 同步DP组
            if self.data_parallel_group and dist.get_world_size(self.data_parallel_group) > 1:
                dp_src = dist.get_global_rank(self.data_parallel_group, 0)
                for param in self.sae.parameters():
                    dist.broadcast(param.data, src=dp_src, group=self.data_parallel_group)
                for buffer in self.sae.buffers():
                    dist.broadcast(buffer.data, src=dp_src, group=self.data_parallel_group)
            
            dist.barrier()
        
        self.model = model
        self.model.eval()

        num_sae_params = sum(p.numel() for p in self.sae.parameters())
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Number of SAE parameters: {num_sae_params:_}")
            print(f"PP Rank {self.pp_rank} manages layers {self.my_start_layer}-{self.my_end_layer}")

        sae_lr = train_cfg.lr or 2e-4 / (self.sae.config.feature_size / (2**14)) ** 0.5
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Learning rate: {sae_lr}")

        if train_cfg.adam_in_8bit:
            try:
                from bitsandbytes.optim import Adam8bit as Adam
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print("Using 8-bit Adam")
            except ImportError:
                from torch.optim import Adam
                print("Using standard Adam")
        else:
            from torch.optim import Adam

        self.optimizer = Adam(params=self.sae.parameters(), lr=sae_lr)
        
        feature_dim = self.sae.local_feature_size if hasattr(self.sae, "local_feature_size") else self.sae.config.feature_size
        self.did_fire = torch.zeros(feature_dim, device=device, dtype=torch.bool)
        self.num_tokens_since_fired = torch.zeros(feature_dim, device=device, dtype=torch.long)
        self.manual_global_norm = sae_cfg.input_normalize
        
        self.dataset = dataset
        if self.data_parallel_group is not None:
            dp_rank = dist.get_rank(self.data_parallel_group)
            dp_size = dist.get_world_size(self.data_parallel_group)
        else:
            dp_rank = 0
            dp_size = 1
        
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=True,
            seed=self.train_cfg.seed,
            drop_last=True
        )

        self.dl = StatefulDataLoader(
            self.dataset,
            batch_size=self.train_cfg.local_batch_size,
            sampler=sampler, 
            collate_fn=functools.partial(packing_collate_fn, max_length=self.train_cfg.ctx_len),
            drop_last=True
        )
        
        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        self.dl_pbar = tqdm(desc="Training", disable=not rank_zero, 
                           total=math.ceil(len(self.dl) / self.train_cfg.grad_acc_steps))
        self.i_start = 0
        
        if dist.is_initialized():
            for _, buffer in self.sae.named_buffers():
                dist.broadcast(buffer.data, src=0)
            dist.barrier()
            
        self.num_training_steps = int(len(self.dl) / self.train_cfg.grad_acc_steps)
        if train_cfg.lr_warmup_ratio:
            self.train_cfg.lr_warmup_steps = int(self.num_training_steps * train_cfg.lr_warmup_ratio)
        if train_cfg.lr_decay_ratio:
            self.train_cfg.lr_decay_steps = int(self.num_training_steps * train_cfg.lr_decay_ratio)
        if train_cfg.lr_decay_steps:
            self.train_cfg.lr_stable_steps = max(
                int(self.num_training_steps - self.train_cfg.lr_warmup_steps - self.train_cfg.lr_decay_steps), 0
            )
        
        if rank_zero:
            print(f"Total Training Steps: {self.num_training_steps}")
            print(f"Warmup/Stable/Decay: {self.train_cfg.lr_warmup_steps}/{self.train_cfg.lr_stable_steps}/{self.train_cfg.lr_decay_steps}")
        
        if self.train_cfg.lr_scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.train_cfg.lr_warmup_steps, 
                num_training_steps=self.num_training_steps
            )
        elif self.train_cfg.lr_scheduler_type == "wsd":
            self.lr_scheduler = get_wsd_schedule(
                self.optimizer, 
                num_warmup_steps=self.train_cfg.lr_warmup_steps, 
                num_stable_steps=self.train_cfg.lr_stable_steps, 
                num_decay_steps=self.train_cfg.lr_decay_steps, 
                min_lr_ratio=self.train_cfg.min_lr_ratio
            )
        else:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.train_cfg.lr_warmup_steps, 
                num_training_steps=self.num_training_steps
            )
        
        self.loss_history = list()
        
        if train_cfg.load_dir is not None:
            self.resume_training()
            
        print(f"[Rank {dist.get_rank()}] Hookpoint: {train_cfg.hookpoint} -> Layer {self.target_layer_idx} (Owner: PP Rank {self.hook_owner_pp_rank})")

    @torch.no_grad()
    def pipeline_forward(self, batch):
        """
        Pipeline forward pass (Robust Handshake Version)
        """
        device = torch.device("cuda", torch.cuda.current_device())
        
        # 1. 准备 Input IDs
        input_ids = batch["input_ids"]
        # 强制展平再升维，确保是 [1, Total_Seq] (Batch=1 模式)
        input_ids = input_ids.view(-1).unsqueeze(0)
        
        # 默认形状
        batch_size, seq_len = input_ids.shape
        hidden_dim = self.model.config.hidden_size

        attention_mask = None 
        hidden_state = None
        cu_seqlens = batch.get("cu_seqlens", None)
        max_seqlens = batch.get("max_seqlens", None)
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(device)
        if max_seqlens is not None and isinstance(max_seqlens, torch.Tensor):
            max_seqlens = max_seqlens.to(device)

        # 2. Pipeline 通信逻辑 (Shape Handshake)
        if self.pp_rank == 0:
            input_ids = input_ids.to(device)
            hidden_state = self.model.embed_tokens(input_ids)
        else:
            src_rank = dist.get_global_rank(self.pp_group, self.pp_rank - 1)
            
            # [Handshake Step 1] 接收形状
            shape_tensor = torch.zeros(3, dtype=torch.long, device=device)
            dist.recv(shape_tensor, src=src_rank, group=self.pp_group)
            
            # [Handshake Step 2] 解析形状
            recv_shape = tuple(shape_tensor.tolist())
            bs_recv, seq_recv, dim_recv = recv_shape
            batch_size = bs_recv
            seq_len = seq_recv 
            
            # [Handshake Step 3] 接收数据
            hidden_state = torch.zeros(recv_shape, device=device, dtype=self.model.dtype)
            dist.recv(hidden_state, src=src_rank, group=self.pp_group)
            hidden_state = hidden_state.view(batch_size, seq_len, hidden_dim)

        # 3. Position IDs 处理
        if "position_ids" in batch:
            position_ids = batch["position_ids"].to(device)
            if position_ids.numel() != batch_size * seq_len:
                position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
            elif position_ids.dim() != 2:
                position_ids = position_ids.view(batch_size, seq_len)
        else:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
            if position_ids.shape[0] != batch_size:
                position_ids = position_ids.repeat(batch_size, 1)

        # 4. RoPE 计算
        local_rotary_emb = self.model.rotary_emb
        full_cos, full_sin = local_rotary_emb(hidden_state, position_ids)

        target_activation = None
        
        # 5. Layer 循环
        for i in range(self.my_start_layer, self.my_end_layer):
            if i >= len(self.model.layers): break
            layer = self.model.layers[i]
            
            # RoPE 维度适配
            layer_head_dim = layer.self_attn.head_dim
            if full_cos.shape[-1] > layer_head_dim:
                current_cos = full_cos[..., :layer_head_dim]
                current_sin = full_sin[..., :layer_head_dim]
            else:
                current_cos = full_cos
                current_sin = full_sin
            position_embeddings = (current_cos, current_sin)
            
            layer_out = layer(
                hidden_state, 
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens, 
                max_seqlens=max_seqlens 
            )[0]
            
            hidden_state = layer_out

            if i == self.target_layer_idx:
                target_activation = hidden_state.unsqueeze(0)
            hidden_state=hidden_state.unsqueeze(0)
        
        # 6. 发送给下一个 Rank
        if self.pp_rank < self.pp_size - 1:
            dst_rank = dist.get_global_rank(self.pp_group, self.pp_rank + 1)
            curr_shape = torch.tensor(hidden_state.shape, dtype=torch.long, device=device)
            dist.send(curr_shape, dst=dst_rank, group=self.pp_group)
            dist.send(hidden_state.contiguous(), dst=dst_rank, group=self.pp_group)

        torch.cuda.synchronize()
        return target_activation

    def maybe_all_reduce(self, x: Tensor, op: str = "mean", skip_pp: bool = False) -> Tensor:
        """Helper for cross-replica reduction"""
        if not dist.is_initialized(): 
            return x
        
        if self.data_parallel_group and dist.get_world_size(self.data_parallel_group) > 1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.data_parallel_group)
            
        if not skip_pp and self.pipeline_parallel_group and dist.get_world_size(self.pipeline_parallel_group) > 1:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.pipeline_parallel_group)

        if op == "mean":
            dp_size = dist.get_world_size(self.data_parallel_group) if self.data_parallel_group else 1
            pp_size = 1 if skip_pp else (dist.get_world_size(self.pipeline_parallel_group) if self.pipeline_parallel_group else 1)
            x /= (dp_size * pp_size)
            
        return x

    def fit(self):
        """Main training loop"""
        torch.set_float32_matmul_precision("high")

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        device = torch.device("cuda", torch.cuda.current_device())
        hook_name = self.train_cfg.hookpoint

        # WandB
        if self.train_cfg.log_to_wandb and rank_zero:
            try:
                import wandb
                wandb.init(
                    project=self.train_cfg.wandb_project,
                    name=self.train_cfg.run_name,
                    id=self.train_cfg.wandb_id,
                    config=asdict(self.train_cfg),
                    save_code=True,
                    resume="allow",
                )
            except ImportError:
                print("Weights & Biases not installed, skipping.")
                self.train_cfg.log_to_wandb = False

        # DDP wrapping
        if self.data_parallel_group and dist.get_world_size(self.data_parallel_group) > 1:
            DP_wrapped_sae = DDP(self.sae, process_group=self.data_parallel_group)
        else:
            DP_wrapped_sae = self.sae

        if dist.is_initialized():
            dist.barrier()
        
        grad_acc_steps = self.train_cfg.grad_acc_steps
        micro_acc_steps = self.train_cfg.micro_acc_steps
        total_acc_steps = grad_acc_steps * micro_acc_steps 
        log_denom = total_acc_steps * self.train_cfg.wandb_log_frequency

        avg_loss = defaultdict(float)
        avg_reconstruction_loss = defaultdict(float)
        avg_l1_loss = defaultdict(float)
        avg_auxk_loss = defaultdict(float)
        avg_moe_loss = defaultdict(float)

        num_tokens_in_step = 0
        
        tokens_eligible_per_feature = torch.zeros(
            self.sae.local_feature_size, 
            device=device, 
            dtype=torch.long
        )

        # === Training Loop ===
        for i, batch in enumerate(self.dl, start=self.i_start):
            if i == 0 and self.train_cfg.save_at_init and self.i_start == 0:
                self.save(0)
            
            step, substep = divmod(i + 1, grad_acc_steps)
            start_time = time.time()

            num_tokens_in_step += batch["input_ids"].numel()
            
            # === Pipeline Forward ===
            local_activation = self.pipeline_forward(batch)
            
            input_ids = batch["input_ids"]
            if input_ids.dim() == 1:
                bs, seq = 1, input_ids.shape[0]
            else:
                bs, seq = input_ids.shape
            
            hidden_dim = self.model.config.hidden_size
            
            # === Broadcast ===
            if local_activation is None:
                broadcast_buffer = torch.zeros((bs, seq, hidden_dim), device=device, dtype=self.model.dtype)
            else:
                broadcast_buffer = local_activation.contiguous()
            
            src_global_rank = dist.get_global_rank(self.pp_group, self.hook_owner_pp_rank)
            dist.broadcast(broadcast_buffer, src=src_global_rank, group=self.pp_group)
            
            # === 几何中位数初始化 ===
            if i == 0 and self.i_start == 0:
                if dist.get_rank() == 0:
                    with torch.random.fork_rng():
                        torch.manual_seed(self.train_cfg.seed)
                        median = geometric_median(broadcast_buffer.flatten(0, 1))
                        self.sae.b_dec.data = median.to(self.sae.config.get_torch_dtype())
                
                if dist.is_initialized():
                    dist.broadcast(self.sae.b_dec.data, src=0)
                    dist.barrier()
            
            # =======================================================
            # [Fix 1] 完美的“整除”数据切分逻辑 (Perfect Alignment Strategy)
            # =======================================================
            # 1. 强制展平
            flat_hiddens = broadcast_buffer.view(-1, hidden_dim)
            total_tokens = flat_hiddens.shape[0]
            
            # 2. 关键修改：使用整除 (//) 而非 ceil
            # 这样保证每个 Rank 分到的 token 数完全一致 (tokens_per_rank)
            tokens_per_rank = total_tokens // self.pp_size
            
            start_idx = self.pp_rank * tokens_per_rank
            end_idx = (self.pp_rank + 1) * tokens_per_rank
            
            # 3. 切分
            my_hiddens = flat_hiddens[start_idx:end_idx]

            # 立即释放大显存
            del broadcast_buffer
            del flat_hiddens

            # =======================================================
            # [Fix 2] 全程 float64 高精度统计 (保持不变)
            # =======================================================
            if self.sae.config.normalize_decoder:
                self.sae.set_decoder_norm_to_unit_norm()
            
            hiddens_f32 = my_hiddens.detach().to(torch.float32)

            # 1. Count
            local_count = torch.tensor([hiddens_f32.shape[0]], device=device, dtype=torch.float64)
            
            # 2. Sum & SumSq
            local_sum = hiddens_f32.sum(dim=0).contiguous()
            local_sum_sq = hiddens_f32.pow(2).sum(dim=0).contiguous()


            # 3. Sync
            if self.pp_size > 1:
                dist.all_reduce(local_count, op=dist.ReduceOp.SUM, group=self.pp_group)
                dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=self.pp_group)
                dist.all_reduce(local_sum_sq, op=dist.ReduceOp.SUM, group=self.pp_group)
            
            # 4. Variance
            global_mean = local_sum / local_count
            global_mean_sq = local_sum_sq / local_count
            global_per_token_variance = global_mean_sq - global_mean.pow(2)
            
            global_per_token_variance = global_per_token_variance.to(torch.float32)
            global_per_token_variance = torch.clamp(global_per_token_variance, min=1e-6)
            
            del hiddens_f32

            # === Micro-batch训练 ===
            chunks = my_hiddens.chunk(micro_acc_steps)
            
            for chunk_idx, chunk in enumerate(chunks):
                if chunk.numel() == 0: 
                    continue
                current_chunk_variance = global_per_token_variance * chunk.shape[0]
                current_chunk_variance = torch.clamp(current_chunk_variance, min=1.0)
                chunk_f32 = chunk.to(torch.float32)
                out = DP_wrapped_sae(
                    chunk_f32,
                    dead_mask=(
                        self.num_tokens_since_fired > self.train_cfg.dead_feature_threshold
                        if self.train_cfg.auxk_alpha > 0
                        else None
                    ),
                    external_variance=current_chunk_variance
                )
                
                if out.expert_mask is not None:
                    tokens_eligible_per_feature += out.expert_mask.long().sum(dim=0)
                else:
                    tokens_eligible_per_feature += chunk.shape[0]

                # =======================================================
                # [Fix 3] Loss 还原为 Sum 后聚合 (保持不变)
                # =======================================================
                local_loss = out.loss.detach()
                local_recon = out.reconstruction_loss.detach()
                num_samples_in_chunk = chunk.shape[0]
                
                loss_sum = local_loss * num_samples_in_chunk
                recon_sum = local_recon * num_samples_in_chunk
                
                local_moe_loss = out.aux_moe_loss if hasattr(out, 'aux_moe_loss') and out.aux_moe_loss is not None else torch.tensor(0.0, device=device)
                moe_loss_sum = local_moe_loss * num_samples_in_chunk

                # AllReduce SUM
                if self.pp_group and self.pp_size > 1:
                    num_samples_tensor = torch.tensor([num_samples_in_chunk], device=device, dtype=torch.float32)
                    total_samples_tensor = num_samples_tensor.clone()
                    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM, group=self.pp_group)
                    total_samples = int(total_samples_tensor.item())
                    
                    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=self.pp_group)
                    dist.all_reduce(recon_sum, op=dist.ReduceOp.SUM, group=self.pp_group)
                    dist.all_reduce(moe_loss_sum, op=dist.ReduceOp.SUM, group=self.pp_group)
                    
                    global_loss_mean = loss_sum / total_samples
                    global_recon_mean = recon_sum / total_samples
                    global_moe_loss_mean = moe_loss_sum / total_samples
                else:
                    global_loss_mean = local_loss
                    global_recon_mean = local_recon
                    global_moe_loss_mean = local_moe_loss
                
                # DP Reduce
                if self.data_parallel_group and dist.get_world_size(self.data_parallel_group) > 1:
                    dist.all_reduce(global_loss_mean, op=dist.ReduceOp.SUM, group=self.data_parallel_group)
                    dist.all_reduce(global_recon_mean, op=dist.ReduceOp.SUM, group=self.data_parallel_group)
                    dist.all_reduce(global_moe_loss_mean, op=dist.ReduceOp.SUM, group=self.data_parallel_group)
                    
                    dp_size = dist.get_world_size(self.data_parallel_group)
                    global_loss_mean /= dp_size
                    global_recon_mean /= dp_size
                    global_moe_loss_mean /= dp_size
              
                avg_loss[hook_name] += float(global_loss_mean / log_denom)
                avg_reconstruction_loss[hook_name] += float(global_recon_mean / log_denom)
                avg_moe_loss[hook_name] += float(global_moe_loss_mean / log_denom)

                # L1 处理
                if hasattr(out, "l1_loss") and out.l1_loss is not None:
                    l1_sum = out.l1_loss.detach() * num_samples_in_chunk
                    if self.pp_group and self.pp_size > 1:
                        dist.all_reduce(l1_sum, op=dist.ReduceOp.SUM, group=self.pp_group)
                        l1_mean = l1_sum / total_samples
                    else:
                        l1_mean = out.l1_loss.detach()
                    
                    if self.data_parallel_group and dist.get_world_size(self.data_parallel_group) > 1:
                        dist.all_reduce(l1_mean, op=dist.ReduceOp.SUM, group=self.data_parallel_group)
                        l1_mean /= dist.get_world_size(self.data_parallel_group)
                    
                    avg_l1_loss[hook_name] += float(l1_mean / log_denom)
                
                # AuxK 处理
                if hasattr(out, "auxk_loss") and self.train_cfg.auxk_alpha > 0 and out.auxk_loss is not None:
                    auxk_sum = out.auxk_loss.detach() * num_samples_in_chunk
                    if self.pp_group and self.pp_size > 1:
                        dist.all_reduce(auxk_sum, op=dist.ReduceOp.SUM, group=self.pp_group)
                        auxk_mean = auxk_sum / total_samples
                    else:
                        auxk_mean = out.auxk_loss.detach()
                    
                    if self.data_parallel_group and dist.get_world_size(self.data_parallel_group) > 1:
                        dist.all_reduce(auxk_mean, op=dist.ReduceOp.SUM, group=self.data_parallel_group)
                        auxk_mean /= dist.get_world_size(self.data_parallel_group)
                    
                    avg_auxk_loss[hook_name] += float(auxk_mean / log_denom)

                # === Backward ===
                # 因为上面用了整除保证数据量一致，这里的 local loss 就可以直接用于 backward
                # 后续的 all_reduce(AVG) 梯度将是数学上完美的
                loss_for_backward = out.loss.div(total_acc_steps)

                # Spike detection
                spike_threshold = 1000.0
                if len(self.loss_history) > self.train_cfg.spike_detection_window_size:
                    current_mean = np.mean(self.loss_history[-self.train_cfg.spike_detection_window_size:])
                    spike_threshold = current_mean * self.train_cfg.spike_detection_threshold_ratio
                
                if global_loss_mean.item() < spike_threshold:
                    loss_for_backward.backward()
                    
                    # ================= NEW: Backward Graph Debug (绝对防漏版) =================
                    if dist.is_initialized() and (not hasattr(self, "_printed_backward") or not self._printed_backward):
                        self._printed_backward = True  # 只要进来一次就锁死，防止刷屏
                        global_rank = dist.get_rank() if dist.is_initialized() else 0
                        
                        # 检查 Router 梯度
                        router_grad = "N/A"
                        if hasattr(self.sae, "router"):
                            if self.sae.router.weight.grad is not None:
                                router_grad = f"{self.sae.router.weight.grad.norm().item():.6f}"
                            else:
                                router_grad = "NONE (🚨 计算图断裂！)"
                                
                        # 检查 Encoder 梯度
                        enc_grad = "N/A"
                        if self.sae.encoder.weight.grad is not None:
                            enc_grad = f"{self.sae.encoder.weight.grad.norm().item():.6f}"

                        print(
                            f"🧨 [Backward | Rank {global_rank}]\n"
                            f"    ├─ Encoder Grad Norm: {enc_grad}\n"
                            f"    └─ Router Grad Norm : {router_grad}  <-- 请盯紧这个值！\n"
                            f"---------------------------------------------------"
                        )
                    # ==============================================================
                else:
                    if rank_zero:
                        print(f"⚠️ Spike: {global_loss_mean.item():.2f} > {spike_threshold:.2f}")

                # Dead features 标记
                active_mask = out.sparse_feature_activations.flatten() > 0
                real_active_indices = out.sparse_feature_indices.flatten()[active_mask]
                self.did_fire[real_active_indices] = True

            # === 梯度同步 ===
            if self.pp_group and self.pp_size > 1:
                for param in self.sae.parameters():
                    if param.grad is not None:
                        # 完美对齐：由于数据切分完全均等，梯度的 AVG 是正确的
                        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=self.pp_group)
            # [新增] Router 梯度强行同步 (针对 TP 组)
            # 这一步保证 TP 组内所有 Rank 的 Router 梯度完全比特级一致
            if self.model_parallel_group and dist.get_world_size(self.model_parallel_group) > 1:
                if hasattr(self.sae, "router") and self.sae.router.weight.grad is not None:
                    # 使用 AVG 或 SUM 都可以，只要统一。这里用 AVG 消除浮点噪音。
                    dist.all_reduce(self.sae.router.weight.grad, op=dist.ReduceOp.AVG, group=self.model_parallel_group)
            # === 验证梯度 ===
            if i == 0 and substep == 0:
                grad_norm = 0.0
                for param in self.sae.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                print(f"[Rank {dist.get_rank()}] Grad norm: {grad_norm:.8f}")
            
            # === Optimizer Step ===
            if substep == 0:
                self.dl_pbar.update(1)
                
                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
                
                if self.sae.config.normalize_decoder:
                    self.sae.remove_gradient_parallel_to_decoder_directions()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                
                # Dead features 更新
                with torch.no_grad():
                    self.num_tokens_since_fired += tokens_eligible_per_feature
                    
                    if dist.is_initialized():
                        did_fire_float = self.did_fire.float()
                        for g in [self.data_parallel_group, self.pp_group]:
                            if g and dist.get_world_size(g) > 1:
                                dist.all_reduce(did_fire_float, op=dist.ReduceOp.MAX, group=g)
                        self.did_fire = did_fire_float.bool()
                    
                    self.num_tokens_since_fired[self.did_fire] = 0
                    
                    # 清零状态
                    self.did_fire.zero_()
                    num_tokens_in_step = 0 
                    tokens_eligible_per_feature.zero_() 

                # Loss history
                if global_loss_mean.item() < spike_threshold:
                    self.loss_history.append(global_loss_mean.item())

                # Logging
                if self.train_cfg.log_to_wandb and rank_zero and step % self.train_cfg.wandb_log_frequency == 0:
                    import wandb
                    lr = self.optimizer.param_groups[0]["lr"]
                    mask = self.num_tokens_since_fired > self.train_cfg.dead_feature_threshold
                    
                    info = {
                        f"loss/loss/{hook_name}": avg_loss[hook_name],
                        f"loss/fvu/{hook_name}": avg_reconstruction_loss[hook_name],
                        f"loss/l1_reg_loss/{hook_name}": avg_l1_loss[hook_name],
                        f"dead_pct/{hook_name}": mask.float().mean().item(),
                        f"moe_loss/{hook_name}": avg_moe_loss[hook_name], 
                        "train/lr": lr,
                        "train/step_time": time.time() - start_time,
                    }
                    
                    if self.train_cfg.auxk_alpha > 0:
                        info[f"auxk_loss/{hook_name}"] = avg_auxk_loss[hook_name]
                    
                    print(f"Step {step} | Loss: {avg_loss[hook_name]:.4f} | FVU: {avg_reconstruction_loss[hook_name]:.4f} | MoE: {avg_moe_loss[hook_name]:.4f}" )
                    wandb.log(info, step=step)

                    avg_loss.clear()
                    avg_reconstruction_loss.clear()
                    avg_l1_loss.clear()
                    avg_auxk_loss.clear()
                    avg_moe_loss.clear()

                if step > 0 and step % self.train_cfg.save_every == 0:
                    self.save(step)

        self.save(step)
        self.dl_pbar.close()

    def resume_training(self):
        """Resume Training with Tensor Parallelism Support."""
        mp_rank = dist.get_rank(self.model_parallel_group) if self.model_parallel_group else 0
        iter_num = 0
        
        sae_base_path = os.path.join(self.train_cfg.load_dir, self.train_cfg.run_name, "saes", f"{self.train_cfg.hookpoint}")
        
        if os.path.exists(sae_base_path):
            latest_file = os.path.join(sae_base_path, "latest_checkpoint.txt")
            if os.path.exists(latest_file):
                with open(latest_file, "r") as f:
                    try:
                        iter_num = int(f.read().strip())
                    except ValueError:
                        pass
        
        if iter_num > 0:
            print(f"[Rank {dist.get_rank()}] Loading SAEs from disk, iteration: {iter_num} (MP Rank: {mp_rank})")
            load_path = os.path.join(sae_base_path, f"iter_{iter_num:07d}.pt")
            
            if not os.path.exists(load_path):
                print(f"Checkpoint not found at {load_path}")
                return

            state_dict = torch.load(load_path, map_location="cpu", weights_only=False)
            start_idx = mp_rank * self.sae.local_feature_size
            end_idx = (mp_rank + 1) * self.sae.local_feature_size
            new_state_dict = {}
            if "encoder.weight" in state_dict:
                new_state_dict["encoder.weight"] = state_dict["encoder.weight"][start_idx:end_idx, :]
            if "encoder.bias" in state_dict:
                new_state_dict["encoder.bias"] = state_dict["encoder.bias"][start_idx:end_idx]
            if "W_dec" in state_dict:
                new_state_dict["W_dec"] = state_dict["W_dec"][start_idx:end_idx, :]
            if "b_dec" in state_dict:
                new_state_dict["b_dec"] = state_dict["b_dec"]
            # ================= NEW: 加载 MoE Router 权重 =================
            if getattr(self.sae, "use_moe", False) and "router.weight" in state_dict:
                # 不切片，直接全量赋给本地的 router
                new_state_dict["router.weight"] = state_dict["router.weight"]
            # ==========================================================

            self.sae.load_state_dict(new_state_dict, strict=False)
            print(f"[Rank {dist.get_rank()}] Model weights sliced and loaded.")
        else:
            print("No SAEs found in the disk, starting fresh.")
            return
                
        optimizer_save_dir = self.train_cfg.hookpoint
        optimizer_load_base = os.path.join(self.train_cfg.load_dir, self.train_cfg.run_name, "optimizer", optimizer_save_dir)
        optimizer_load_path = os.path.join(optimizer_load_base, f"iter_{iter_num:07d}_mp{mp_rank}.pt")
        
        if not os.path.exists(optimizer_load_path) and self.sae.mp_world_size == 1:
             fallback = os.path.join(optimizer_load_base, f"iter_{iter_num:07d}.pt")
             if os.path.exists(fallback):
                 optimizer_load_path = fallback

        if os.path.exists(optimizer_load_path):
            print(f"[Rank {dist.get_rank()}] Loading optimization states from {optimizer_load_path}...")
            optimization_dict = torch.load(optimizer_load_path, map_location=self.model.device, weights_only=False)
            try:
                self.optimizer.load_state_dict(optimization_dict["optimizer"])
                self.lr_scheduler.load_state_dict(optimization_dict["lr_scheduler"])
                self.did_fire.copy_(optimization_dict["did_fire"].to(self.model.device))
                self.num_tokens_since_fired.copy_(optimization_dict["num_tokens_since_fired"].to(self.model.device))
                self.loss_history = optimization_dict["loss_history"]

                dp_size = dist.get_world_size(self.data_parallel_group) if self.data_parallel_group else 1
                self.i_start = (iter_num * self.train_cfg.global_batch_size) // (dp_size * self.train_cfg.local_batch_size)
                
                if "dataloader_state" in optimization_dict:
                    self.dl.load_state_dict(optimization_dict["dataloader_state"])
                if hasattr(self, 'dl_pbar'):
                    self.dl_pbar.n = iter_num
                    self.dl_pbar.refresh()
            except Exception as e:
                print(f"[Rank {dist.get_rank()}] Failed to load optimizer state: {e}. Starting optimizer fresh.")
        else:
            print(f"[Rank {dist.get_rank()}] Warning: Optimizer checkpoint not found. Starting optimizer fresh.")
        
        if dist.is_initialized():
            dist.barrier()
        
    def save(self, iter):
        """Save the SAEs to disk with Auto-Merge for Tensor Parallelism."""
        
        mp_rank = getattr(self.sae, "mp_rank", 0)
        mp_world_size = getattr(self.sae, "mp_world_size", 1)
        
        # 获取全集群唯一的物理卡号 (0 ~ 7)
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        
        def gather_and_merge(local_tensor, dim=0):
            if mp_world_size == 1:
                return local_tensor.cpu()
            
            gathered_list = [torch.zeros_like(local_tensor) for _ in range(mp_world_size)] if mp_rank == 0 else None
            # 注意：只要是属于 model_parallel_group 的都需要执行 gather
            dist.gather(local_tensor, gathered_list, dst=0, group=self.model_parallel_group)
            
            if mp_rank == 0:
                full_tensor = torch.cat(gathered_list, dim=dim).cpu()
                return full_tensor
            return None

        # 1. 所有人（按MP组）一起执行 gather，但只有 mp_rank=0 会拿到合并后的 Tensor
        full_encoder_weight = gather_and_merge(self.sae.encoder.weight.data, dim=0)
        full_encoder_bias = gather_and_merge(self.sae.encoder.bias.data, dim=0)
        
        if self.sae.decoder:
            full_decoder_weight = gather_and_merge(self.sae.W_dec.data, dim=0)
        else:
            full_decoder_weight = None
            
        full_decoder_bias = self.sae.b_dec.data.cpu() if mp_rank == 0 else None

        # ===============================================================
        # 2. 【终极防并发锁】全集群 8 张卡，只允许 Global Rank 0 写合并后的主模型！
        # ===============================================================
        if global_rank == 0:
            print(f"[Iter {iter}] Gathering SAE weights to Global Rank 0 and saving...")
            save_path = os.path.join(self.train_cfg.save_dir, self.train_cfg.run_name, 'saes', self.train_cfg.hookpoint)
            Path(save_path).mkdir(parents=True, exist_ok=True)

            full_state_dict = {
                "encoder.weight": full_encoder_weight,
                "encoder.bias": full_encoder_bias,
                "b_dec": full_decoder_bias,
            }
            if full_decoder_weight is not None:
                full_state_dict["W_dec"] = full_decoder_weight
                
            # 保存 MoE Router 权重
            if getattr(self.sae, "use_moe", False) and hasattr(self.sae, "router"):
                full_state_dict["router.weight"] = self.sae.router.weight.data.cpu()

            torch.save(full_state_dict, os.path.join(save_path, f"iter_{iter:07d}.pt"))
            with open(os.path.join(save_path, "latest_checkpoint.txt"), "w") as f:
                f.write(str(iter))
            print(f"Saved merged checkpoint to {save_path}")

        # ===============================================================
        # 3. 【终极防并发锁】写优化器时，只允许 Global Rank 等于自己的 MP Rank 才能写！
        # (比如 MP=2 时，只有 0号物理卡写 mp0.pt，1号物理卡写 mp1.pt，其余 6 张卡直接罚站！)
        # ===============================================================
        if global_rank == mp_rank:
            print(f"Model Parallel {mp_rank} Saving Optimization States (Sharded) from Global Rank {global_rank}")
            optimizer_save_dir = self.train_cfg.hookpoint
            save_path_opt = os.path.join(self.train_cfg.save_dir, self.train_cfg.run_name, 'optimizer', optimizer_save_dir)
            Path(save_path_opt).mkdir(parents=True, exist_ok=True)
            
            optimization_dict = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "dataloader_state": self.dl.state_dict(),
                "did_fire": self.did_fire, 
                "num_tokens_since_fired": self.num_tokens_since_fired,
                "loss_history": self.loss_history,
                "is_sharded_optimizer": True 
            }
            
            filename_opt = f"iter_{iter:07d}_mp{mp_rank}.pt"
            torch.save(optimization_dict, os.path.join(save_path_opt, filename_opt))
            
            if mp_rank == 0:
                with open(os.path.join(save_path_opt, "latest_checkpoint.txt"), "w") as f:
                    f.write(str(iter))

        # 所有人等写盘的兄弟写完
        if dist.is_initialized():
            dist.barrier()
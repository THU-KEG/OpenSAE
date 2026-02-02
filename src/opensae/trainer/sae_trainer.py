import os
import sys
import math
import time
import functools
from collections import defaultdict
from dataclasses import asdict
from typing import Sized
from pathlib import Path
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, get_linear_schedule_with_warmup, get_wsd_schedule, get_cosine_schedule_with_warmup
import numpy as np

# 假设你的项目结构如下
from ..saes import OpenSae, OpenSaeConfig
from .train_arguments import TrainConfig, SaeConfig, ModelConfig
from .train_utils import geometric_median, resolve_width
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
        pipeline_parallel_group: dist.ProcessGroup | None = None, # [新增] PP 组
        device_mesh = None # [新增] Device Mesh (可选)
    ):
    
        self.train_cfg = train_cfg

        self.data_parallel_group = data_parallel_group
        self.model_parallel_group = model_parallel_group
        self.pipeline_parallel_group = pipeline_parallel_group # [新增]
        self.device_mesh = device_mesh

        # [新增] PP 相关参数
        self.pp_group = pipeline_parallel_group
        self.pp_rank = dist.get_rank(self.pp_group) if self.pp_group else 0
        self.pp_size = dist.get_world_size(self.pp_group) if self.pp_group else 1
        
        # === [修改开始] ===
        # 必须与 main.py 的切分逻辑完全一致
        full_model_layers = model.config.num_hidden_layers
        # 注意：这里要确保 train_cfg 里有 early_exit_inference_layer_num
        target_layers = train_cfg.early_exit_inference_layer_num + 1
        effective_num_layers = min(target_layers, full_model_layers)
        
        layers_per_stage = math.ceil(effective_num_layers / self.pp_size)
        
        self.my_start_layer = self.pp_rank * layers_per_stage
        self.my_end_layer = min((self.pp_rank + 1) * layers_per_stage, effective_num_layers)
        # === [修改结束] ===

        # [新增] 解析 Hookpoint 位置
        target_layer_name = train_cfg.hookpoint # e.g., "layers.26"
        try:
            self.target_layer_idx = int(target_layer_name.split('.')[1])
        except:
             print(f"Warning: Could not parse layer index from {train_cfg.hookpoint}")
             self.target_layer_idx = -1
             
        # [新增] 判断 Hookpoint 归谁管
        self.hook_owner_pp_rank = self.target_layer_idx // layers_per_stage
        if self.hook_owner_pp_rank >= self.pp_size:
            self.hook_owner_pp_rank = self.pp_size - 1

        assert self.train_cfg.hookpoint is not None
        assert isinstance(dataset, Sized)
        
        device = torch.device("cuda", torch.cuda.current_device())
        input_width = model.config.hidden_size
        
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
            decoder_impl = sae_cfg.decoder_impl if hasattr(sae_cfg, 'decoder_impl') else "triton"
        )
        
        # 初始化 OpenSae，传入 model_parallel_group 以便内部进行切分和 GlobalTopK
        self.sae = OpenSae(open_sae_config, device, model_parallel_group=self.model_parallel_group)
        self.model = model

        # 统计参数量
        num_sae_params = sum(p.numel() for p in self.sae.parameters())
        # LLM 参数量统计可能不准确（因为只加载了部分层），但不影响运行
        
        # 如果是分布式，只在 Rank 0 打印
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Number of SAE parameters: {num_sae_params:_}")
            print(f"PP Rank {self.pp_rank} responsible for layers {self.my_start_layer} - {self.my_end_layer}")

        sae_params = self.sae.parameters()
        # Auto-select LR using 1 / sqrt(d) scaling law
        sae_lr = train_cfg.lr or 2e-4 / (self.sae.config.feature_size / (2**14)) ** 0.5
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Learning rates: {sae_lr}")

        if train_cfg.adam_in_8bit:
            try:
                from bitsandbytes.optim import Adam8bit as Adam
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print("Using 8-bit Adam from bitsandbytes")
            except ImportError:
                print("bitsandbytes 8-bit Adam not available, using torch.optim.Adam")
                raise ImportError
        else:
            from torch.optim import Adam

        # init Optimization state
        self.optimizer = Adam(params=sae_params, lr=sae_lr)
        
        # Init Information 
        # 获取切分后的特征维度，用于初始化死特征统计
        if hasattr(self.sae, "local_feature_size"):
            feature_dim = self.sae.local_feature_size
        else:
            feature_dim = self.sae.config.feature_size

        self.did_fire = torch.zeros(feature_dim, device=device, dtype=torch.bool)
        self.num_tokens_since_fired = torch.zeros(feature_dim, device=device, dtype=torch.long)
        
        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Tracking dead features for {feature_dim} latents (Local Part).")
        
        # Init Dataset & Dataloader
        self.dataset = dataset
        self.dl = StatefulDataLoader(
            self.dataset,
            batch_size=self.train_cfg.local_batch_size,
            shuffle=False, 
            collate_fn = functools.partial(packing_collate_fn, max_length = self.train_cfg.ctx_len),
            drop_last=True
        )
        
        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        self.dl_pbar = tqdm(desc="Training", disable=not rank_zero, total = math.ceil(len(self.dl) / self.train_cfg.grad_acc_steps))
        self.i_start = 0
        
        self.num_training_steps = int(len(self.dl) / self.train_cfg.grad_acc_steps)
        if train_cfg.lr_warmup_ratio:
            self.train_cfg.lr_warmup_steps = int(self.num_training_steps * train_cfg.lr_warmup_ratio)
        if train_cfg.lr_decay_ratio:
            self.train_cfg.lr_decay_steps = int(self.num_training_steps * train_cfg.lr_decay_ratio)
        if train_cfg.lr_decay_steps:
            self.train_cfg.lr_stable_steps = int(self.num_training_steps - self.train_cfg.lr_warmup_steps - self.train_cfg.lr_decay_steps)
            self.train_cfg.lr_stable_steps = max(self.train_cfg.lr_stable_steps, 0)
        
        if rank_zero:
            print(f"Total Training Steps: {self.num_training_steps}")
            print(f"Warmup Steps: {self.train_cfg.lr_warmup_steps}")
            print(f"Stable Steps: {self.train_cfg.lr_stable_steps}")
            print(f"Decay Steps:  {self.train_cfg.lr_decay_steps}")
        
        if self.train_cfg.lr_scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps = self.train_cfg.lr_warmup_steps, 
                num_training_steps = self.num_training_steps
            )
        elif self.train_cfg.lr_scheduler_type == "wsd":
            self.lr_scheduler = get_wsd_schedule(
                self.optimizer, 
                num_warmup_steps = self.train_cfg.lr_warmup_steps, 
                num_stable_steps = self.train_cfg.lr_stable_steps, 
                num_decay_steps = self.train_cfg.lr_decay_steps, 
                min_lr_ratio = self.train_cfg.min_lr_ratio
            )
        elif self.train_cfg.lr_scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps = self.train_cfg.lr_warmup_steps, 
                num_training_steps = self.num_training_steps
            )
        # 兜底 constant
        else:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=0, num_training_steps=self.num_training_steps
            )
        
        self.loss_history = list()
        
        if train_cfg.load_dir is not None:
            self.resume_training()
        print(f"[Rank {dist.get_rank()}] Hookpoint: {train_cfg.hookpoint} -> Parsed Index: {self.target_layer_idx} (Owner PP Rank: {self.hook_owner_pp_rank})")

    # [保留原样] 你的 Resume 逻辑非常完善，不需要修改
    def resume_training(self):
        """
        Resume Training with Tensor Parallelism Support.
        """
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
    def pipeline_forward(self, batch):
        """
        手动执行 Pipeline Forward。
        返回: 如果当前 rank 是 hook rank，返回 activation Tensor；否则返回 None。
        """
        input_ids = batch["input_ids"]
        device = torch.device("cuda", torch.cuda.current_device())
        hidden_state = None
        
        # 获取 Batch 和 Seq Len
        if input_ids.dim() == 2:
            batch_size, seq_len = input_ids.shape
        else:
            batch_size = 1 
            seq_len = input_ids.shape[0]
        attention_mask=batch.get("attention_mask",None)
        if attention_mask is None:
            attention_mask=torch.ones((batch_size,seq_len),device=device,dtype=torch.bool)
        else:
            attention_mask=attention_mask.to(device)
        dummy_embeds=torch.empty((batch_size,seq_len,self.model.config.hidden_size),dtype=self.model.dtype,device=device)
        extended_attention_mask=_prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size,seq_len),
            dummy_embeds,
            past_key_values_length=0
        )
        # === 1. PP Rank 0: Embedding ===
        if self.pp_rank == 0:
            input_ids = input_ids.to(device)
            hidden_state = self.model.embed_tokens(input_ids)
        else:
            # 接收上一个 Rank 的 Tensor
            hidden_dim = self.model.config.hidden_size
            if input_ids.dim() == 2:
                recv_shape = (batch_size, seq_len, hidden_dim)
            else:
                recv_shape = (seq_len, hidden_dim)
                
            hidden_state = torch.zeros(recv_shape, device=device, dtype=self.model.dtype)
            src_rank = dist.get_global_rank(self.pp_group, self.pp_rank - 1)
            dist.recv(hidden_state, src=src_rank, group=self.pp_group)



        # === 2. 准备 Position IDs 和 Rotary Embeddings ===
        
        # 2.1 生成 Position IDs
        if "position_ids" in batch:
            position_ids = batch["position_ids"].to(device)
        else:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
            if input_ids.dim() == 2:
                position_ids = position_ids.repeat(batch_size, 1)

        # 2.2 计算 Rotary Embeddings (cos, sin)
        rotary_emb = self.model.rotary_emb
        
        # [修改] 强制使用 position_ids 调用
        cos, sin = rotary_emb(hidden_state, position_ids)
            
        position_embeddings = (cos, sin)


        # === 3. Run Local Layers ===
        target_activation = None
        
        for i in range(self.my_start_layer, self.my_end_layer):
            if i >= len(self.model.layers): break
            
            layer = self.model.layers[i]
            
            layer_out = layer(
                hidden_state, 
                attention_mask=extended_attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]
            
            hidden_state = layer_out
            
            if i == self.target_layer_idx:
                target_activation = hidden_state.detach().clone()
        
        # === 4. Send to Next Rank ===
        if self.pp_rank < self.pp_size - 1:
             dst_rank = dist.get_global_rank(self.pp_group, self.pp_rank + 1)
             dist.send(hidden_state.contiguous(), dst=dst_rank, group=self.pp_group)

        return target_activation

    def maybe_all_reduce(self, x: Tensor, op: str = "mean") -> Tensor:
        if not dist.is_initialized(): return x
        
        # 聚合所有数据并行维度的值
        if self.data_parallel_group:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.data_parallel_group)
        if self.pipeline_parallel_group:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.pipeline_parallel_group)

        if op == "mean":
            dp_size = dist.get_world_size(self.data_parallel_group) if self.data_parallel_group else 1
            x /= (dp_size * self.pp_size)
        return x
    def fit(self):
        # 使用 Tensor Cores 加速 fp32 矩阵乘法
        torch.set_float32_matmul_precision("high")

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        device = torch.device("cuda", torch.cuda.current_device())
        hook_name = self.train_cfg.hookpoint

        # WandB 初始化
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
                print("Weights & Biases not installed, skipping logging.")
                self.train_cfg.log_to_wandb = False

        # 初始化 SAE 包装逻辑
        is_wrapped = False
        DP_wrapped_sae = None
        if not is_wrapped:
            if self.data_parallel_group and dist.get_world_size(self.data_parallel_group) > 1:
                DP_wrapped_sae = DDP(self.sae, process_group=self.data_parallel_group)
            else:
                DP_wrapped_sae = self.sae
            is_wrapped = True

        dist.barrier()
        
        # #### [修改开始] 核心修复：强制同步 PP 组内的 SAE 初始权重 ####
        # 原因：由于不同 PP Rank 加载模型层数不同，随机数生成器状态已偏移。
        # 必须以 PP Rank 0 为“真理来源”，将其初始权重广播给同组的其他 Rank。
        if self.pp_group and self.pp_size > 1:
            # 1. 获取 PP 组内 Rank 0 的 全局 Rank (Global Rank)
            # 注意：dist.broadcast 的 src 参数必须是全局 rank
            src_global_rank = dist.get_global_rank(self.pp_group, 0)
            
            if rank_zero:
                print(f"DEBUG: Broadcasting SAE initialization from Global Rank {src_global_rank} to ensure PP consistency.")

            # 2. 遍历所有参数进行广播
            # 包括 encoder.weight, encoder.bias, W_dec, b_dec 等
            for name, param in self.sae.named_parameters():
                # src=src_global_rank: 数据来源是 PP Rank 0
                # group=self.pp_group: 只在当前 PP 管道内部广播
                dist.broadcast(param.data, src=src_global_rank, group=self.pp_group)
                
            # 3. 如果使用了 Buffer (如 running_mean 等)，也要同步
            for name, buffer in self.sae.named_buffers():
                dist.broadcast(buffer.data, src=src_global_rank, group=self.pp_group)

            dist.barrier() # 确保所有人同步完成再开始训练
        grad_acc_steps = self.train_cfg.grad_acc_steps
        micro_acc_steps = self.train_cfg.micro_acc_steps
        total_steps_in_accumulation = grad_acc_steps * micro_acc_steps

        # 初始化累加器
        current_accumulation_metrics = defaultdict(float)

        for i, batch in enumerate(self.dl, start=self.i_start):
            if i == 0 and self.train_cfg.save_at_init and self.i_start == 0:
                self.save(0)
            local_input_sum = batch["input_ids"].sum().item()
            
            # 打印格式: [Rank Global_Rank] PP_Rank X | Step X | Input Sum: XXXXX
            #print(f"[Rank {dist.get_rank()}] PP{self.pp_rank} | Step {i} | Input Sum: {local_input_sum}")
            step, substep_idx = divmod(i + 1, grad_acc_steps)
            start_time = time.time()
            
            # 1. Pipeline Forward & Broadcast
            local_activation = self.pipeline_forward(batch)
            bs, seq = batch["input_ids"].shape
            hidden_dim = self.model.config.hidden_size
            
            if local_activation is None:
                broadcast_buffer = torch.zeros((bs, seq, hidden_dim), device=device, dtype=self.model.dtype)
            else:
                broadcast_buffer = local_activation.contiguous()
            
            src_global_rank = dist.get_global_rank(self.pp_group, self.hook_owner_pp_rank)
            dist.broadcast(broadcast_buffer, src=src_global_rank, group=self.pp_group)
            # 在 fit() 中，dist.broadcast(broadcast_buffer, ...) 之后：

            if rank_zero: # 每10步打一次
                act_mean = broadcast_buffer.mean().item()
                act_std = broadcast_buffer.std().item()
                act_max = broadcast_buffer.max().item()
                
                #print(f"🔍 [Step {step}] Activation Stats: Mean={act_mean:.4f}, Std={act_std:.4f}, Max={act_max:.4f}")
                
                if act_max == 0:
                    print("❌ CRITICAL: Activations are ALL ZEROS! Check hookpoint index or pipeline communication.")
                if math.isnan(act_mean):
                    print("❌ CRITICAL: Activations contain NaNs!")
            # #### [Fix 1] 几何中位数初始化 & 强制同步 Bias ####
            # 这一步必须在拿到 broadcast_buffer (全量数据) 之后，切分数据之前做
            if i == 0 and self.i_start == 0:
                # 展平数据 [BS, Seq, Hidden] -> [N, Hidden]
                all_activations = broadcast_buffer.flatten(0, 1)
                
                # 计算几何中位数
                median = geometric_median(all_activations)
                self.sae.b_dec.data = median.to(self.sae.config.get_torch_dtype())

                # [关键] 强制在 PP 组内广播这个 Bias，确保所有 Rank 起点一致
                if self.pp_group is not None and self.pp_size > 1:
                    src_global_rank_0 = dist.get_global_rank(self.pp_group, 0)
                    dist.broadcast(self.sae.b_dec.data, src=src_global_rank_0, group=self.pp_group)
                    if self.pp_rank == 0 and rank_zero:
                        print(f"DEBUG: Initialized and broadcasted geometric median bias.")
            # #################################################
            # 2. 切分 Batch (PP -> DP)
            flat_hiddens = broadcast_buffer.flatten(0, 1)
            total_tokens_batch = flat_hiddens.shape[0]
            tokens_per_pp_rank = total_tokens_batch // self.pp_size
            start_idx = self.pp_rank * tokens_per_pp_rank
            end_idx = (self.pp_rank + 1) * tokens_per_pp_rank if self.pp_rank != self.pp_size - 1 else total_tokens_batch
            my_hiddens = flat_hiddens[start_idx:end_idx]

            if self.sae.config.normalize_decoder:
                self.sae.set_decoder_norm_to_unit_norm()

            # 3. Micro-batch 训练循环
            chunks = my_hiddens.chunk(micro_acc_steps)
            for chunk in chunks:
                if chunk.numel() == 0: continue

                out = DP_wrapped_sae(
                    chunk,
                    dead_mask=(
                        self.num_tokens_since_fired > self.train_cfg.dead_feature_threshold
                        if self.train_cfg.auxk_alpha > 0
                        else None
                    )
                )

                # 梯度反传用的 Loss (除以总步数)
                loss_for_backward = out.loss / total_steps_in_accumulation
                loss_for_backward.backward()
                
                # 记录指标
                with torch.no_grad():
                    current_accumulation_metrics["loss"] += out.loss.detach()
                    current_accumulation_metrics["fvu"] += out.reconstruction_loss.detach()
                    
                    # 记录 L1 (如果有)
                    if hasattr(out, "l1_loss"):
                        current_accumulation_metrics["l1"] += out.l1_loss.detach()
                    
                    # 【核心修复：记录 AuxK Loss】
                    if hasattr(out, "auxk_loss"):
                        current_accumulation_metrics["auxk"] += out.auxk_loss.detach()
                    
                    if hasattr(out, "multi_topk_loss"):
                        current_accumulation_metrics["multi_topk"] += out.multi_topk_loss.detach()

                # 更新本地死特征统计
                active_mask = out.sparse_feature_activations.flatten() > 0
                real_active_indices = out.sparse_feature_indices.flatten()[active_mask]
                self.did_fire[real_active_indices] = True

            # 4. PP 组梯度同步
            for param in self.sae.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=self.pp_group)

            time_elapsed = time.time() - start_time
            
            # 5. 梯度累积周期结束 (进行更新和日志记录)
            if substep_idx == 0:
                self.dl_pbar.update(1)
                
                with torch.no_grad():
                    # a. 本地平均 + 跨进程归约
                    for key in list(current_accumulation_metrics.keys()):
                        current_accumulation_metrics[key] /= total_steps_in_accumulation
                        current_accumulation_metrics[key] = self.maybe_all_reduce(current_accumulation_metrics[key], op="mean")

                final_step_loss = current_accumulation_metrics["loss"].item()
                
                # Spike Detection
                spike_threshold = 1000.0
                if len(self.loss_history) > self.train_cfg.spike_detection_window_size:
                    spike_threshold = np.mean(self.loss_history[-self.train_cfg.spike_detection_window_size:]) * self.train_cfg.spike_detection_threshold_ratio
                
                if final_step_loss < spike_threshold:
                    self.loss_history.append(final_step_loss)
                    if self.sae.config.normalize_decoder:
                        self.sae.remove_gradient_parallel_to_decoder_directions()
                    self.optimizer.step()
                else:
                    if rank_zero: print(f"⚠️ Spike! Loss {final_step_loss:.2f} > {spike_threshold:.2f}")
                
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                
                # 6. 死特征跨并行组同步 (MAX 归约)
                with torch.no_grad():
                    if self.pp_size > 1 or (self.data_parallel_group and dist.get_world_size(self.data_parallel_group) > 1):
                        did_fire_float = self.did_fire.float()
                        target_groups = [self.pp_group, self.data_parallel_group]
                        for g in target_groups:
                            if g is not None and dist.get_world_size(g) > 1:
                                dist.all_reduce(did_fire_float, op=dist.ReduceOp.MAX, group=g)
                        self.did_fire = did_fire_float.bool()

                    global_tokens = self.train_cfg.global_batch_size * self.train_cfg.ctx_len
                    self.num_tokens_since_fired += global_tokens
                    self.num_tokens_since_fired[self.did_fire] = 0
                    self.did_fire.zero_()

                # 7. 日志上传
                if self.train_cfg.log_to_wandb and rank_zero and step % self.train_cfg.wandb_log_frequency == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    info = {
                        f"loss/loss/{hook_name}": final_step_loss,
                        f"loss/fvu/{hook_name}": current_accumulation_metrics["fvu"].item(),
                        f"dead_pct/{hook_name}": (self.num_tokens_since_fired > self.train_cfg.dead_feature_threshold).float().mean().item(),
                        "train/lr": lr,
                        "train/step_time": time_elapsed,
                    }
                    
                    # 显式添加可选 Loss 项到日志
                    if "auxk" in current_accumulation_metrics:
                        info[f"auxk_loss/{hook_name}"] = current_accumulation_metrics["auxk"].item()
                    if "l1" in current_accumulation_metrics:
                        info[f"loss/l1_reg_loss/{hook_name}"] = current_accumulation_metrics["l1"].item()
                    if "multi_topk" in current_accumulation_metrics:
                        info[f"multi_topk_fvu/{hook_name}"] = current_accumulation_metrics["multi_topk"].item()
                    
                    wandb.log(info, step=step)

                # 重置累加器进入下一个梯度周期
                current_accumulation_metrics.clear()

                if step > 0 and step % self.train_cfg.save_every == 0:
                    self.save(step)

        self.save(step)
        self.dl_pbar.close()
    def save(self, iter):
        """
        Save the SAEs to disk with Auto-Merge for Tensor Parallelism.
        1. Model Weights: Gathered from all MP ranks and merged into a single file on Rank 0.
        2. Optimizer States: Sharded (one file per MP rank) to save memory/time.
        """
        
        # 获取 Rank 信息
        mp_rank = dist.get_rank(self.model_parallel_group) if self.model_parallel_group else 0
        mp_world_size = dist.get_world_size(self.model_parallel_group) if self.model_parallel_group else 1
        dp_rank = dist.get_rank(self.data_parallel_group) if self.data_parallel_group else 0
        
        # ---------------- Helper Function ----------------
        def gather_and_merge(local_tensor, dim=0):
            if mp_world_size == 1:
                return local_tensor.cpu()
            
            # Rank 0 准备接收容器
            gathered_list = [torch.zeros_like(local_tensor) for _ in range(mp_world_size)] if mp_rank == 0 else None
            
            # 通信
            dist.gather(local_tensor, gathered_list, dst=0, group=self.model_parallel_group)
            
            # 拼接
            if mp_rank == 0:
                full_tensor = torch.cat(gathered_list, dim=dim).cpu()
                return full_tensor
            return None
        # -------------------------------------------------

        # 只有 DP Rank 0 执行保存逻辑 (避免重复)
        if dp_rank == 0:
            if mp_rank == 0:
                print(f"[Iter {iter}] Gathering SAE weights from all MP ranks to Rank 0...")

            # --- 1. 保存模型 (合并) ---
            # Encoder Weight / Bias: 切分维度是 0 (Feature Dim)
            full_encoder_weight = gather_and_merge(self.sae.encoder.weight.data, dim=0)
            full_encoder_bias = gather_and_merge(self.sae.encoder.bias.data, dim=0)
            
            # Decoder Weight: 切分维度是 0 (Feature Dim)
            if self.sae.decoder:
                full_decoder_weight = gather_and_merge(self.sae.W_dec.data, dim=0)
            else:
                full_decoder_weight = None
            
            # Decoder Bias: 没切分，直接取
            full_decoder_bias = self.sae.b_dec.data.cpu() if mp_rank == 0 else None

            # 写盘 (仅 MP Rank 0)
            if mp_rank == 0:
                save_path = os.path.join(self.train_cfg.save_dir, self.train_cfg.run_name, 'saes', self.train_cfg.hookpoint)
                Path(save_path).mkdir(parents=True, exist_ok=True)

                full_state_dict = {
                    "encoder.weight": full_encoder_weight,
                    "encoder.bias": full_encoder_bias,
                    "b_dec": full_decoder_bias,
                }
                if full_decoder_weight is not None:
                    full_state_dict["W_dec"] = full_decoder_weight

                torch.save(full_state_dict, os.path.join(save_path, f"iter_{iter:07d}.pt"))
                with open(os.path.join(save_path, "latest_checkpoint.txt"), "w") as f:
                    f.write(str(iter))
                
                print(f"Saved merged checkpoint to {save_path}")

            # --- 2. 保存优化器 (分片) ---
            # 优化器必须分片存，否则显存爆炸且难以 Resume
            print(f"Model Parallel {mp_rank} Saving Optimization States (Sharded)")
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

        if dist.is_initialized():
            dist.barrier()
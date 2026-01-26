import os
import sys
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
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, get_linear_schedule_with_warmup, get_wsd_schedule, get_cosine_schedule_with_warmup
import numpy as np

# 假设你的项目结构如下，请根据实际情况调整 import
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
    ):
    
        self.train_cfg = train_cfg
        
        self.data_parallel_group = data_parallel_group
        self.model_parallel_group = model_parallel_group

        assert self.train_cfg.hookpoint is not None

        assert isinstance(dataset, Sized)
        
        device = model.device
        input_width = resolve_width(model, train_cfg.hookpoint)
        
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
        num_model_params = sum(p.numel() for p in self.model.parameters())
        
        # 如果是分布式，只在 Rank 0 打印
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Number of SAE parameters: {num_sae_params:_}")
            print(f"Number of model parameters: {num_model_params:_}")

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
        
        self.loss_history = list()
        
        if train_cfg.load_dir is not None:
            self.resume_training()
        
        
    def resume_training(self):
        """
        Resume Training with Tensor Parallelism Support.
        Strategy:
        1. Model Weights: Load merged checkpoint (iter_xxx.pt) -> Slice to local part.
        2. Optimizer States: Load sharded checkpoint (iter_xxx_mp{rank}.pt).
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

            # -----------------------------------------------------
            # 1. 加载模型 (Merged Checkpoint -> Sliced Loading)
            # -----------------------------------------------------
            load_path = os.path.join(sae_base_path, f"iter_{iter_num:07d}.pt")
            
            if not os.path.exists(load_path):
                print(f"Checkpoint not found at {load_path}")
                return

            # 先读到 CPU，避免爆显存
            state_dict = torch.load(load_path, map_location="cpu", weights_only=False)
            
            # 计算切片范围
            start_idx = mp_rank * self.sae.local_feature_size
            end_idx = (mp_rank + 1) * self.sae.local_feature_size
            
            new_state_dict = {}
            
            # 切分 Encoder Weight (Column Parallel)
            if "encoder.weight" in state_dict:
                # Shape: [Total_Features, Hidden] -> Slice dim 0
                new_state_dict["encoder.weight"] = state_dict["encoder.weight"][start_idx:end_idx, :]
                
            # 切分 Encoder Bias (Column Parallel)
            if "encoder.bias" in state_dict:
                # Shape: [Total_Features] -> Slice dim 0
                new_state_dict["encoder.bias"] = state_dict["encoder.bias"][start_idx:end_idx]
                
            # 切分 Decoder Weight (Row Parallel)
            if "W_dec" in state_dict:
                # Shape: [Total_Features, Hidden] (因为你是 OpenSAE，通常是 Encoder 的转置或者独立存储)
                # 无论如何，TP 下 Decoder 是 Row Parallel，如果 W_dec 是 [Features, Hidden]，切 dim 0
                new_state_dict["W_dec"] = state_dict["W_dec"][start_idx:end_idx, :]
                
            # Decoder Bias (Replicated, 不切分)
            if "b_dec" in state_dict:
                new_state_dict["b_dec"] = state_dict["b_dec"]

            # 加载切好的权重
            self.sae.load_state_dict(new_state_dict, strict=False)
            print(f"[Rank {dist.get_rank()}] Model weights sliced and loaded.")

        else:
            print("No SAEs found in the disk, starting fresh.")
            return
                
        # -----------------------------------------------------
        # 2. 加载优化器 (Sharded Checkpoint)
        # -----------------------------------------------------
        optimizer_save_dir = self.train_cfg.hookpoint
        optimizer_load_base = os.path.join(self.train_cfg.load_dir, self.train_cfg.run_name, "optimizer", optimizer_save_dir)
        
        # 优先加载带 _mp{rank} 的分片文件
        optimizer_load_path = os.path.join(optimizer_load_base, f"iter_{iter_num:07d}_mp{mp_rank}.pt")
        
        # 兼容旧版本单卡训练的文件 (如果 MP=1)
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
                
                # 加载 Dead feature 统计 (这些是 Sharded 的，直接覆盖即可)
                self.did_fire.copy_(optimization_dict["did_fire"].to(self.model.device))
                self.num_tokens_since_fired.copy_(optimization_dict["num_tokens_since_fired"].to(self.model.device))
                self.loss_history = optimization_dict["loss_history"]

                # 恢复 DataLoader 位置
                dp_size = dist.get_world_size(self.data_parallel_group) if self.data_parallel_group else 1
                self.i_start = (iter_num * self.train_cfg.global_batch_size) // (dp_size * self.train_cfg.local_batch_size)
                
                # 恢复 DataLoader 随机状态
                if "dataloader_state" in optimization_dict:
                    self.dl.load_state_dict(optimization_dict["dataloader_state"])
                
                # 更新进度条
                if hasattr(self, 'dl_pbar'):
                    self.dl_pbar.n = iter_num
                    self.dl_pbar.refresh()
            except Exception as e:
                print(f"[Rank {dist.get_rank()}] Failed to load optimizer state: {e}. Starting optimizer fresh.")

        else:
            print(f"[Rank {dist.get_rank()}] Warning: Optimizer checkpoint not found. Starting optimizer fresh.")
        
        if dist.is_initialized():
            dist.barrier()


    def fit(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        is_data_parallel = dist.is_initialized() and self.train_cfg.dp_size > 1
        device = self.model.device

        if self.train_cfg.log_to_wandb and rank_zero:
            try:
                import wandb
                wandb.init(
                    project=self.train_cfg.wandb_project,
                    name=self.train_cfg.run_name,
                    id=self.train_cfg.wandb_id,
                    config=asdict(self.train_cfg),
                    save_code=True,
                    resume = "allow",
                )
            except ImportError:
                print("Weights & Biases not installed, skipping logging.")
                self.train_cfg.log_to_wandb = False

        num_tokens_in_step = 0

        # For logging purposes
        avg_auxk_loss = defaultdict(float)
        average_reconstruction_loss = defaultdict(float)
        avg_l1_loss = defaultdict(float)
        average_multi_topk_loss = defaultdict(float)
        avg_loss = defaultdict(float)

        hidden_dict: dict[str, Tensor] = {}
        module_for_sae = self.model.get_submodule(self.train_cfg.hookpoint)

        def hook(module: nn.Module, _, outputs):
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            hidden_dict[self.train_cfg.hookpoint] = outputs.flatten(0, 1)

        dist.barrier()
        is_wrapped = False
        DP_wrapped_sae = None
        
        for i, batch in enumerate(self.dl, start = self.i_start):
            if i == 0 and self.train_cfg.save_at_init and self.i_start == 0:
                self.save(0)
            
            step, substep = divmod(i + 1, self.train_cfg.grad_acc_steps)
            start_time = time.time()
            
            hidden_dict.clear()
            num_tokens_in_step += batch["input_ids"].numel()

            forward_hook_handle = module_for_sae.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    if self.train_cfg.varlen and "cu_seqlens" in batch:
                        self.model(
                            batch["input_ids"].to(device),
                            cu_seqlens = batch["cu_seqlens"].to(device),
                            max_seqlens = batch["max_seqlens"].to(device),
                            max_layer_num = self.train_cfg.early_exit_inference_layer_num
                        )                    
                    else:
                        self.model(batch["input_ids"].to(device), max_layer_num = self.train_cfg.early_exit_inference_layer_num)
            finally:
                forward_hook_handle.remove()


            for name, hiddens in hidden_dict.items():
                if i == 0 and self.i_start == 0:
                    median = geometric_median(self.maybe_all_cat(hiddens))
                    self.sae.b_dec.data = median.to(self.sae.config.get_torch_dtype())

                if not is_wrapped:
                    if self.train_cfg.fsdp and is_data_parallel:
                        DP_wrapped_sae = FSDP(self.sae, process_group=self.data_parallel_group)
                    elif is_data_parallel:
                        DP_wrapped_sae = DDP(self.sae, process_group=self.data_parallel_group)
                    else:
                        DP_wrapped_sae = self.sae
                    is_wrapped = True

                if self.sae.config.normalize_decoder:
                    self.sae.set_decoder_norm_to_unit_norm()

                acc_steps = self.train_cfg.grad_acc_steps * self.train_cfg.micro_acc_steps
                denom = acc_steps * self.train_cfg.wandb_log_frequency

                for chunk in hiddens.chunk(self.train_cfg.micro_acc_steps):
                    out = DP_wrapped_sae(
                        chunk,
                        dead_mask=(
                            self.num_tokens_since_fired > self.train_cfg.dead_feature_threshold
                            if self.train_cfg.auxk_alpha > 0
                            else None
                        )
                    )

                    average_reconstruction_loss[name] += float(
                        self.maybe_all_reduce(out.reconstruction_loss.detach()) / denom
                    )
                    average_multi_topk_loss[name] += float(
                        self.maybe_all_reduce(out.multi_topk_loss.detach()) / denom
                    )
                    avg_l1_loss[name] += float(
                        self.maybe_all_reduce(out.l1_loss.detach()) / denom
                    )
                    if self.train_cfg.auxk_alpha > 0:
                        avg_auxk_loss[name] += float(
                            self.maybe_all_reduce(out.auxk_loss.detach()) / denom
                        )

                    loss = out.loss
                    loss = loss.div(acc_steps)
                    
                    loss_for_spike_analysis = loss.clone().detach()
                    self.maybe_all_reduce(loss_for_spike_analysis)
                    
                    spike_threshold = 100
                    if step > self.train_cfg.spike_detection_start and len(self.loss_history) > self.train_cfg.spike_detection_window_size:
                         spike_threshold = np.mean(self.loss_history[-self.train_cfg.spike_detection_window_size:]) * self.train_cfg.spike_detection_threshold_ratio
                         spike_threshold /= denom
                    
                    if loss_for_spike_analysis < spike_threshold:
                        loss.backward()
                    else:
                        print(f"Omit Loss {loss}, threshold {spike_threshold}")

                    avg_loss[name] += float((loss_for_spike_analysis / denom) * acc_steps)

                    self.did_fire[out.sparse_feature_indices.flatten()] = True
                    # did_fire 只是本地统计，不需要 all-reduce，因为 AuxK 是本地算的

                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
                
            time_elapsed = time.time() - start_time
            
            if substep == 0:
                self.dl_pbar.update(1)
                if self.sae.config.normalize_decoder:
                        self.sae.remove_gradient_parallel_to_decoder_directions()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                
                lr = self.optimizer.state_dict()["param_groups"][0]["lr"]

                with torch.no_grad():
                    self.num_tokens_since_fired += num_tokens_in_step
    # === [修复开始] 同步 did_fire ===
                    # 如果使用了 DP，需要把所有 DP 组内的 did_fire 取并集 (Logical OR / MAX)
                    if self.data_parallel_group is not None and dist.get_world_size(self.data_parallel_group) > 1:
                        # bool 转 float/byte 才能 reduce，建议用 MAX (相当于 OR)
                        did_fire_float = self.did_fire.float()
                        dist.all_reduce(did_fire_float, op=dist.ReduceOp.MAX, group=self.data_parallel_group)
                        self.did_fire = did_fire_float.bool()
                    # === [修复结束] ===
                    self.num_tokens_since_fired[self.did_fire] = 0
                    num_tokens_in_step = 0
                    self.did_fire.zero_()

                mask = self.num_tokens_since_fired > self.train_cfg.dead_feature_threshold
                
                # Spike Detection Check
                if step > self.train_cfg.spike_detection_start and len(self.loss_history) > self.train_cfg.spike_detection_window_size:
                    avg_loss_spike_threshold = np.mean(self.loss_history[-self.train_cfg.spike_detection_window_size:]) * self.train_cfg.spike_detection_threshold_ratio
                else:
                    avg_loss_spike_threshold = 1000.0 # big enough
                    
                if avg_loss[self.train_cfg.hookpoint] < avg_loss_spike_threshold:
                    self.loss_history.append(avg_loss[self.train_cfg.hookpoint])

                info = {
                    f"loss/fvu/{self.train_cfg.hookpoint}": average_reconstruction_loss[self.train_cfg.hookpoint],
                    f"loss/loss/{self.train_cfg.hookpoint}": avg_loss[self.train_cfg.hookpoint],
                    f"loss/l1_reg_loss/{self.train_cfg.hookpoint}": avg_l1_loss[self.train_cfg.hookpoint],
                    f"dead_pct/{self.train_cfg.hookpoint}": mask.float().mean().item(),
                    "train/lr": lr,
                    "train/topk": 128,
                    "train/step_time": time_elapsed,
                    "train/tokens": i * self.train_cfg.local_batch_size * self.train_cfg.ctx_len * self.train_cfg.dp_size,
                    "train/total_tokens": step * self.train_cfg.global_batch_size * self.train_cfg.ctx_len,
                }
                if self.train_cfg.auxk_alpha > 0:
                    info[f"auxk/{self.train_cfg.hookpoint}"] = avg_auxk_loss[self.train_cfg.hookpoint]
                info[f"multi_topk_fvu/{self.train_cfg.hookpoint}"] = average_multi_topk_loss[self.train_cfg.hookpoint]

                avg_auxk_loss.clear()
                average_reconstruction_loss.clear()
                average_multi_topk_loss.clear()
                avg_loss.clear()
                avg_l1_loss.clear()

                if self.train_cfg.distribute_modules:
                    # Gather logs from MP ranks if needed
                    # (Simplified: assuming Rank 0 is enough for WandB)
                    pass

                if self.train_cfg.log_to_wandb and rank_zero and step % self.train_cfg.wandb_log_frequency == 0:
                    wandb.log(info, step=step)

                if step > 0 and step % self.train_cfg.save_every == 0:
                    self.save(step)
        
        if (i + 1) % 1000 == 0:
            torch.cuda.empty_cache()

        self.save(step)
        self.dl_pbar.close()


    def maybe_all_cat(self, x: Tensor) -> Tensor:
        if not dist.is_initialized():
            return x
        if self.data_parallel_group is None:
             return x
        
        buffer = x.new_empty([dist.get_world_size(self.data_parallel_group) * x.shape[0], *x.shape[1:]])
        dist.all_gather_into_tensor(buffer, x, group = self.data_parallel_group)
        return buffer


    def maybe_all_reduce(self, x: Tensor, op: str = "mean") -> Tensor:
        if not dist.is_initialized():
            return x
        if self.data_parallel_group is None:
            return x

        if op == "sum":
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group = self.data_parallel_group)
        elif op == "mean":
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group = self.data_parallel_group)
            x /= dist.get_world_size(self.data_parallel_group)
        elif op == "max":
            dist.all_reduce(x, op=dist.ReduceOp.MAX, group = self.data_parallel_group)
        
        return x


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
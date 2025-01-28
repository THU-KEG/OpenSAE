import os
import sys
import math

from collections import defaultdict
from dataclasses import asdict
from typing import Sized
from pathlib import Path
import functools

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from safetensors.torch import load_model
from fnmatch import fnmatchcase
from natsort import natsorted
import numpy as np

from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel, get_linear_schedule_with_warmup, get_wsd_schedule, get_cosine_schedule_with_warmup

from ..saes import OpenSae, OpenSaeConfig

from .train_arguments import TrainConfig, SaeConfig, ModelConfig
from .train_utils import geometric_median, get_layer_list, resolve_width
from ..data.collator import packing_collate_fn
from .topk_scheduler import k_scheduler
import time


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
        num_examples = len(dataset)

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
            l1_coef = sae_cfg.l1_coef
        )
        self.sae = OpenSae(open_sae_config, device)
        self.model = model

        num_sae_params = sum(p.numel() for p in self.sae.parameters())
        num_model_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"Number of SAE parameters: {num_sae_params:_}")
        print(f"Number of model parameters: {num_model_params:_}")

        sae_params = self.sae.parameters()
        # Auto-select LR using 1 / sqrt(d) scaling law from Fig 3 of the paper
        sae_lr = train_cfg.lr or 2e-4 / (self.sae.config.feature_size / (2**14)) ** 0.5
        print(f"Learning rates: {sae_lr}")

        if train_cfg.adam_in_8bit:
            try:
                from bitsandbytes.optim import Adam8bit as Adam
                print("Using 8-bit Adam from bitsandbytes")
            except ImportError:
                print("bitsandbytes 8-bit Adam not available, using torch.optim.Adam")
                print("Run `pip install bitsandbytes` for less memory usage.")
                raise ImportError
        else:
            from torch.optim import Adam

        # init Optimization state
        self.optimizer = Adam(
            params = sae_params,
            lr = sae_lr,
        )
        
        # Init Information 
        self.did_fire = torch.zeros(
            self.sae.config.feature_size, device=device, dtype=torch.bool
        )
        self.num_tokens_since_fired = torch.zeros(
            self.sae.config.feature_size, device=device, dtype=torch.long
        )
        
        # Init Dataset & Dataloader
        self.dataset = dataset
        self.dl = StatefulDataLoader(
            self.dataset,
            batch_size=self.train_cfg.local_batch_size,
            shuffle=False,      # ZIJUN: Must be false if we want to vary the world size when resuming training.
            collate_fn = functools.partial(packing_collate_fn, max_length = self.train_cfg.ctx_len),
        )
        
        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        self.dl_pbar = tqdm(desc="Training", disable=not rank_zero, total = math.ceil(len(self.dl) / self.train_cfg.grad_acc_steps))
        self.i_start = 0
        
        assert train_cfg.lr_stable_steps is None, "LR stable steps are determined by total_steps - lr_warmup_steps - lr_decay_steps"
        self.num_training_steps = int(len(self.dl) / self.train_cfg.grad_acc_steps)
        if train_cfg.lr_warmup_ratio:
            self.train_cfg.lr_warmup_steps = int(self.num_training_steps * train_cfg.lr_warmup_ratio)
        if train_cfg.lr_decay_ratio:
            self.train_cfg.lr_decay_steps = int(self.num_training_steps * train_cfg.lr_decay_ratio)
        if train_cfg.lr_decay_steps:
            self.train_cfg.lr_stable_steps = int(self.num_training_steps - self.train_cfg.lr_warmup_steps - self.train_cfg.lr_decay_steps)
            self.train_cfg.lr_stable_steps = max(self.train_cfg.lr_stable_steps, 0)
        
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
        # Load SAEs from Dir to continue training
        # Check the availablity of SAEs in the disk
        iter_num = 0
        
        sae_path = os.path.join(self.train_cfg.load_dir, self.train_cfg.run_name, "saes", f"{self.train_cfg.hookpoint}")
        print(sae_path)
        if os.path.exists(sae_path):
            with open(os.path.join(sae_path, "latest_checkpoint.txt"), "r") as f:
                iter_num = int(f.read().strip())
        
        if iter_num > 0:
            print(f"Loading SAEs from disk, iteration: {iter_num}")

            load_path = os.path.join(self.train_cfg.load_dir, self.train_cfg.run_name, "saes", f"{self.train_cfg.hookpoint}", f"iter_{iter_num:07d}.pt")
            model_state_dict = torch.load(load_path, weights_only = False, map_location = self.model.device)
            self.sae.load_state_dict(model_state_dict)
                
        else:
            print("No SAEs found in the disk")
                
        # Load the optimization states
        optimizer_save_dir = self.train_cfg.hookpoint
        optimizer_load_path = os.path.join(self.train_cfg.load_dir, self.train_cfg.run_name, "optimizer", optimizer_save_dir)
        if os.path.exists(optimizer_load_path):
            print("Loading optimization states from disk")
            optimization_dict = torch.load(os.path.join(optimizer_load_path, f"iter_{iter_num:07d}.pt"), 
                                           weights_only = False, 
                                           map_location = self.model.device)
            
            self.optimizer.load_state_dict(optimization_dict["optimizer"])
            self.lr_scheduler.load_state_dict(optimization_dict["lr_scheduler"])
            self.did_fire = optimization_dict["did_fire"]
            self.num_tokens_since_fired = optimization_dict["num_tokens_since_fired"]
            self.loss_history = optimization_dict["loss_history"]

            self.i_start = (iter_num * self.train_cfg.global_batch_size) // (dist.get_world_size(self.data_parallel_group) * self.train_cfg.local_batch_size)
            print(f"self.i_start = {self.i_start}")
            for _ in range(iter_num):
                self.dl_pbar.update(1)
                
            dl_state_dict = optimization_dict["dataloader_state"]
            dl_state_dict["_index_sampler_state"]["samples_yielded"] *= optimization_dict["hyperparameters"]["dp_size"]
            dl_state_dict["_index_sampler_state"]["samples_yielded"] /= dist.get_world_size(self.data_parallel_group)
            dl_state_dict["_index_sampler_state"]["samples_yielded"] = int(dl_state_dict["_index_sampler_state"]["samples_yielded"])
            dl_state_dict["_sampler_iter_yielded"] *= optimization_dict["hyperparameters"]["dp_size"]
            dl_state_dict["_sampler_iter_yielded"] /= dist.get_world_size(self.data_parallel_group)
            dl_state_dict["_sampler_iter_yielded"] = int(dl_state_dict["_sampler_iter_yielded"])
            dl_state_dict["_num_yielded"] *= optimization_dict["hyperparameters"]["dp_size"]
            dl_state_dict["_num_yielded"] /= dist.get_world_size(self.data_parallel_group)
            dl_state_dict["_num_yielded"] = int(dl_state_dict["_num_yielded"])
            self.dl.load_state_dict(dl_state_dict)

        else:
            print("No optimization states found in the disk")
        
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
                    # mode = "dryrun"
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
            # Maybe unpack tuple outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            hidden_dict[self.train_cfg.hookpoint] = outputs.flatten(0, 1)


        dist.barrier()
        is_wrapped = False
        DP_wrapped_sae = None
        
        for i, batch in enumerate(self.dl, start = self.i_start):
            if i == 0 and self.train_cfg.save_at_init:
                # Save the model state at the beginning of training
                self.save(0)
            
            step, substep = divmod(i + 1, self.train_cfg.grad_acc_steps)
            start_time = time.time()
            
            hidden_dict.clear()
            
            # Bookkeeping for dead feature detection
            num_tokens_in_step += batch["input_ids"].numel()

            # Forward pass on the model to get the next batch of activations            
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
                # On the first iteration, initialize the decoder bias
                if i == 0:
                    # NOTE: The all-cat here could conceivably cause an OOM in some
                    # cases, but it's unlikely to be a problem with small world sizes.
                    # We could avoid this by "approximating" the geometric median
                    # across all ranks with the mean (median?) of the geometric medians
                    # on each rank. Not clear if that would hurt performance.
                    median = geometric_median(self.maybe_all_cat(hiddens))
                    self.sae.b_dec.data = median.to(self.sae.config.get_torch_dtype())

                if not is_wrapped:
                    # Wrap the SAEs with Distributed Data Parallel. We have to do this
                    # after we set the decoder bias, otherwise DDP will not register
                    # gradients flowing to the bias after the first step.
                    
                    # ZIJUN: It seems that the DDP is not necessary for the SAEs 
                    #        in the original version from EleutherAI. Since DDP is used to asynchronize
                    #        gradient across different GPUs. However, in the original version,
                    #        each SAE layer is placed on a single GPU only.
                    #       DDP is only useful when we have multiple GPUs for a single layer.
                    
                    # ZIJUN: The original one wraps the SAEs with DDP multiple times (=len(self.hidden_dict))
                    #        Fixed the bug
                    # maybe_wrapped[name] = DDP(raw, device_ids=[dist.get_rank()]) if ddp else raw
                    if self.train_cfg.fsdp and is_data_parallel:
                        # maybe_wrapped[name] = FSDP(
                        #     raw, sharding_strategy = ShardingStrategy.FULL_SHARD,
                        #     process_group=self.data_parallel_group
                        # )
                        DP_wrapped_sae = FSDP(self.sae, process_group=self.data_parallel_group)
                    elif is_data_parallel:
                        DP_wrapped_sae = DDP(self.sae, process_group=self.data_parallel_group)
                    else:
                        DP_wrapped_sae = self.sae
                    is_wrapped = True

                # Make sure the W_dec is still unit-norm
                if self.sae.config.normalize_decoder:
                    self.sae.set_decoder_norm_to_unit_norm()

                acc_steps = self.train_cfg.grad_acc_steps * self.train_cfg.micro_acc_steps
                denom = acc_steps * self.train_cfg.wandb_log_frequency

                # Save memory by chunking the activations
                for chunk in hiddens.chunk(self.train_cfg.micro_acc_steps):
                    out = DP_wrapped_sae(
                        chunk,
                        dead_mask=(
                            self.num_tokens_since_fired > self.train_cfg.dead_feature_threshold
                            if self.train_cfg.auxk_alpha > 0
                            else None
                        )
                    )

                    # Save information for reporting
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
                    ############################

                    loss = out.loss
                    loss = loss.div(acc_steps)
                    
                    loss_for_spike_analysis = loss.clone().detach()
                    self.maybe_all_reduce(loss_for_spike_analysis)
                    
                    if step > self.train_cfg.spike_detection_start:
                        spike_threshold = np.mean(self.loss_history[-self.train_cfg.spike_detection_window_size:]) * self.train_cfg.spike_detection_threshold_ratio
                        spike_threshold /= denom
                    else:
                        spike_threshold = 100
                    
                    if loss_for_spike_analysis < spike_threshold:
                        loss.backward()
                    else:
                        print(f"Omit Loss {loss}, the threshold is {spike_threshold}")

                    # Also save loss for reporting
                    avg_loss[name] += float(
                        (loss_for_spike_analysis / denom) * acc_steps
                    )

                    # Update the did_fire mask
                    self.did_fire[out.sparse_feature_indices.flatten()] = True
                    self.maybe_all_reduce(self.did_fire, "max")  # max is boolean "any"

                # Clip gradient norm independently for each SAE
                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)
                


            time_elapsed = time.time() - start_time
            
            # Check if we need to actually do a training step
            if substep == 0:
                self.dl_pbar.update(1)
                if self.sae.config.normalize_decoder:
                        self.sae.remove_gradient_parallel_to_decoder_directions()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                
                lr = self.optimizer.state_dict()["param_groups"][0]["lr"]

                ###############
                with torch.no_grad():
                    # Update the dead feature mask
                    self.num_tokens_since_fired += num_tokens_in_step
                    self.num_tokens_since_fired[self.did_fire] = 0

                    # Reset stats for this step
                    num_tokens_in_step = 0
                    self.did_fire.zero_()








                info = {}
                mask = self.num_tokens_since_fired > self.train_cfg.dead_feature_threshold
                
                if step > self.train_cfg.spike_detection_start:
                    avg_loss_spike_threshold = np.mean(self.loss_history[name][-self.train_cfg.spike_detection_window_size:]) * self.train_cfg.spike_detection_threshold_ratio
                else:
                    avg_loss_spike_threshold = 100
                    
                print(avg_loss[self.train_cfg.hookpoint], avg_loss_spike_threshold) 
                if avg_loss[self.train_cfg.hookpoint] < avg_loss_spike_threshold:
                    self.loss_history[self.train_cfg.hookpoint].append(avg_loss[self.train_cfg.hookpoint])

                info.update(
                    {
                        f"loss/fvu/{self.train_cfg.hookpoint}": average_reconstruction_loss[self.train_cfg.hookpoint],
                        f"loss/loss/{self.train_cfg.hookpoint}": avg_loss[self.train_cfg.hookpoint],
                        f"loss/l1_reg_loss/{self.train_cfg.hookpoint}": avg_l1_loss[self.train_cfg.hookpoint],
                        f"dead_pct/{self.train_cfg.hookpoint}": mask.mean(
                            dtype=torch.float32
                        ).item(),
                        "train/lr": lr,
                        "train/topk": 128,
                        "train/step_time": time_elapsed,
                        "train/tokens": i * self.train_cfg.local_batch_size * self.train_cfg.ctx_len * self.train_cfg.dp_size,
                        "train/total_tokens": step * self.train_cfg.global_batch_size * self.train_cfg.ctx_len,
                    }
                )
                if self.train_cfg.auxk_alpha > 0:
                    info[f"auxk/{self.train_cfg.hookpoint}"] = avg_auxk_loss[self.train_cfg.hookpoint]
                info[f"multi_topk_fvu/{self.train_cfg.hookpoint}"] = average_multi_topk_loss[self.train_cfg.hookpoint]








                avg_auxk_loss.clear()
                average_reconstruction_loss.clear()
                average_multi_topk_loss.clear()
                avg_loss.clear()
                avg_l1_loss.clear()

                if self.train_cfg.distribute_modules:
                    if dist.get_rank(self.data_parallel_group) == 0:
                        outputs = [{} for _ in range(dist.get_world_size(self.model_parallel_group))]
                        dist.gather_object(info, outputs if rank_zero else None, group=self.model_parallel_group)
                        print(outputs)
                        info.update({k: v for out in outputs for k, v in out.items()})

                if self.train_cfg.log_to_wandb and rank_zero and step % self.train_cfg.wandb_log_frequency == 0:
                    wandb.log(info, step=step)

                if step > 0 and step % self.train_cfg.save_every == 0:
                    self.save(step)
        
        # Reduce memory fragmentation
        if (i + 1) % 1000 == 0:
            torch.cuda.empty_cache()

        self.save(step)
        self.dl_pbar.close()


    def maybe_all_cat(self, x: Tensor) -> Tensor:
        """Concatenate a tensor across all processes."""
        if not dist.is_initialized():
            return x

        buffer = x.new_empty([dist.get_world_size(self.data_parallel_group) * x.shape[0], *x.shape[1:]])
        dist.all_gather_into_tensor(buffer, x, group = self.data_parallel_group)
        return buffer


    def maybe_all_reduce(self, x: Tensor, op: str = "mean") -> Tensor:
        if not dist.is_initialized():
            return x

        if op == "sum":
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group = self.data_parallel_group)
        elif op == "mean":
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group = self.data_parallel_group)
            x /= dist.get_world_size(self.data_parallel_group)
        elif op == "max":
            dist.all_reduce(x, op=dist.ReduceOp.MAX, group = self.data_parallel_group)
        else:
            raise ValueError(f"Unknown reduction op '{op}'")

        return x




    def save(self, iter):
        """Save the SAEs to disk."""

        # Only Data Parallel Rank 0 save checkpoints
        if (
            self.train_cfg.distribute_modules
            or not dist.is_initialized()
            or dist.get_rank(self.data_parallel_group) == 0
        ):
            # Save SAE Checkpoints
            print(f"Model Parallel {dist.get_rank(self.model_parallel_group)} Saving SAEs")
            assert isinstance(self.sae, OpenSae)

            save_path = os.path.join(self.train_cfg.save_dir, self.train_cfg.run_name, 'saes', self.train_cfg.hookpoint)
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
                
            sae_states = self.sae.state_dict()
            torch.save(sae_states, os.path.join(save_path, f"iter_{iter:07d}.pt"))
            with open(os.path.join(save_path, "latest_checkpoint.txt"), "w") as f:
                f.write(str(iter))

            print(f"Save SAE Checkpoints To Disk: {save_path}")

            
            # Save Optimizer, Learning Rate Scheduler, DataLoader
            print(f"Model Parallel {dist.get_rank(self.model_parallel_group)} Saving Optimization States")
            optimizer_save_dir = self.train_cfg.hookpoint
            save_path = os.path.join(self.train_cfg.save_dir, self.train_cfg.run_name, 'optimizer', optimizer_save_dir)
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            optimization_dict = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "dataloader_state": self.dl.state_dict(),
                "did_fire": self.did_fire,
                "num_tokens_since_fired": self.num_tokens_since_fired,
                "loss_history": self.loss_history,
                "hyperparameters": {
                    "dp_size": dist.get_world_size(self.data_parallel_group),
                    "global_batch_size": self.train_cfg.global_batch_size,
                    "local_batch_size": self.train_cfg.local_batch_size
                }
            }
            
            torch.save(optimization_dict, save_path / f"iter_{iter:07d}.pt")
            with open(os.path.join(save_path, "latest_checkpoint.txt"), "w") as f:
                f.write(str(iter))

        # Barrier to ensure all ranks have saved before continuing
        if dist.is_initialized():
            dist.barrier()

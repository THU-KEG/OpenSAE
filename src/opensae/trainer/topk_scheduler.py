import random as rd
import math
import torch

from .train_arguments import SaeConfig, TrainConfig


def constant(sae_cfg: SaeConfig, training_cfg: TrainConfig, total_steps, current_step):
    return sae_cfg.k

def linear(sae_cfg: SaeConfig, training_cfg: TrainConfig, total_steps, current_step):
    assert sae_cfg.multi_topk is False
    if current_step > total_steps * training_cfg.k_scheduler_step_ratio:
        return sae_cfg.k
    else:
        assert training_cfg.k_scheduler_factor > 1
        end_step = int(total_steps * training_cfg.k_scheduler_step_ratio)
        time_ratio = (end_step - current_step) / end_step
        ratio = time_ratio * (training_cfg.k_scheduler_factor - 1) + 1
        k = int(ratio * sae_cfg.k)
        return k

def discrete_linear(sae_cfg: SaeConfig, training_cfg: TrainConfig, total_steps, current_step):
    assert sae_cfg.multi_topk is False
    if current_step > total_steps * training_cfg.k_scheduler_step_ratio:
        return sae_cfg.k
    else:
        assert training_cfg.k_scheduler_factor > 1
        end_step = int(total_steps * training_cfg.k_scheduler_step_ratio)
        time_ratio = (end_step - current_step) / end_step
        ratio = time_ratio * (training_cfg.k_scheduler_factor - 1) + 1
        ratio = math.ceil(ratio)
        k = int(ratio * sae_cfg.k)
        return k

def piecewise(sae_cfg: SaeConfig, training_cfg: TrainConfig, total_steps, current_step):
    assert sae_cfg.multi_topk is False
    if current_step > total_steps * training_cfg.k_scheduler_step_ratio:
        return sae_cfg.k
    else:
        return training_cfg.k_scheduler_factor * sae_cfg.k

def random(sae_cfg: SaeConfig, training_cfg: TrainConfig, total_steps, current_step):
    assert sae_cfg.multi_topk is False
    if current_step > total_steps * training_cfg.k_scheduler_step_ratio:
        return sae_cfg.k
    else:
        return rd.randint(1, training_cfg.k_scheduler_factor) * sae_cfg.k


def k_scheduler(sae_cfg: SaeConfig, training_cfg: TrainConfig, total_steps, current_step):
    if training_cfg.k_scheduler == "constant":
        return constant(sae_cfg, training_cfg, total_steps, current_step)
    elif training_cfg.k_scheduler == "linear":
        return linear(sae_cfg, training_cfg, total_steps, current_step)
    elif training_cfg.k_scheduler == "piecewise":
        return piecewise(sae_cfg, training_cfg, total_steps, current_step)
    elif training_cfg.k_scheduler == "random":
        return random(sae_cfg, training_cfg, total_steps, current_step)
    elif training_cfg.k_scheduler == "discrete_linear":
        return discrete_linear(sae_cfg, training_cfg, total_steps, current_step)
    else:
        raise ValueError(f"Unknown k_scheduler: {training_cfg.k_scheduler}")
import os
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path


import numpy as np
import random
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from datasets import Dataset, load_dataset
from simple_parsing import field, parse, ArgumentParser
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer

from .train_arguments import SaeConfig, TrainConfig, ModelConfig, DataConfig
from .sae_trainer import SaeTrainer
from .patch_transformers.patch_llama import model_patch as llama_model_patch
from ..data.dataset import DistributedTokenizedDataset


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def load_model(args, tokenizer: PreTrainedTokenizer, rank: int) -> PreTrainedModel:
    if args.model.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model = args.model.auto_model_class.from_pretrained(
        args.model.model,
        device_map={"": f"cuda:{rank}"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=args.model.load_in_8bit)
            if args.model.load_in_8bit
            else None
        ),
        torch_dtype=dtype,
        trust_remote_code=args.model.trust_remote_code,
        attn_implementation="flash_attention_2"
    )
    model = llama_model_patch(model = model)
    return model

def load_tokenizer(model_args: ModelConfig) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_args.model)

def load_data(dataset_args: DataConfig, tokenizer: PreTrainedTokenizer,  rank: int) -> Dataset:
    try:
        dataset = load_dataset(
            dataset_args.dataset,
            split=dataset_args.split,
            trust_remote_code=dataset_args.trust_remote_code,
        )
    except ValueError as e:
        # Automatically use load_from_disk if appropriate
        if "load_from_disk" in str(e):
            dataset = Dataset.load_from_disk(dataset_args.dataset, keep_in_memory=False)
        else:
            raise e

    return dataset


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    
    np.random.seed(seed)
    random.seed(seed)
    

def run():
    local_rank = os.environ.get("LOCAL_RANK")
    is_distributed_training = local_rank is not None
    rank = int(local_rank) if is_distributed_training else 0

    parser = ArgumentParser()
    
    parser.add_arguments(SaeConfig, dest = "sae")
    parser.add_arguments(ModelConfig, dest = "model")
    parser.add_arguments(DataConfig, dest = "data")
    parser.add_arguments(TrainConfig, dest = "train")
    
    args = parser.parse_args()
    model_args = args.model
    data_args = args.data
    train_args = args.train
    sae_args = args.sae
    train_args.ctx_len = data_args.ctx_len
    set_seed(train_args.seed)
    
    data_parallel_group = None
    model_parallel_group = None
    if is_distributed_training:
        device_mesh = init_device_mesh("cuda", (train_args.dp_size, train_args.mp_size), mesh_dim_names=("data_parallel", "model_parallel"))
        data_parallel_group = device_mesh.get_group(mesh_dim="data_parallel")
        model_parallel_group = device_mesh.get_group(mesh_dim="model_parallel")
        print_rank_0(f"Using Parallel across {dist.get_world_size()} GPUs.")
        print_rank_0(f"Data Parallel World: {dist.get_world_size(data_parallel_group)} GPUs.")
        print_rank_0(f"Model Parallel World: {dist.get_world_size(model_parallel_group)} GPUs.")

    tokenizer = load_tokenizer(model_args)
    model = load_model(args, tokenizer, rank)

        
    dataset_world_size = 1
    dataset_rank = 0
    if is_distributed_training:
        dataset_world_size = dist.get_world_size(data_parallel_group)
        dataset_rank = dist.get_rank(data_parallel_group)
    dataset = DistributedTokenizedDataset(
        path = data_args.dataset,
        tokenizer = tokenizer,
        seq_length = data_args.ctx_len,
        current_rank = dataset_rank,
        world_size = dataset_world_size
    )

    # Prevent ranks other than 0 from printing
    log_dir = Path("logs") / f"{train_args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"dp{dist.get_rank(data_parallel_group)}-mp{dist.get_rank(model_parallel_group)}.log"
    log_file.touch(exist_ok = True)
    with nullcontext() if rank == 0 else redirect_stdout(open(str(log_file), "w")):
        print(f"Training on '{data_args.dataset}'")
        print(f"Storing model weights in {model.dtype}")

        trainer = SaeTrainer(train_args, sae_args, model_args, dataset, model, data_parallel_group, model_parallel_group)
        trainer.fit()

    dist.destroy_process_group()

if __name__ == "__main__":
    run()

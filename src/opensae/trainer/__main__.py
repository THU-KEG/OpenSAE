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

# 假设这些模块你已经有了
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

    # 【注意】在 SAE 张量并行模式下，我们通常在每张卡上都加载一个完整的 LLM。
    # 因为 LLM 负责提供 Input Embedding，且我们不想修改 LLM 的内部结构。
    # 只要显存允许 (LLM Size + SAE Size / MP_Size)，这是最高效的做法。
    model = args.model.auto_model_class.from_pretrained(
        args.model.model,
        device_map={"": f"cuda:{rank}"}, # 强制映射到当前进程对应的 GPU
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=args.model.load_in_8bit)
            if args.model.load_in_8bit
            else None
        ),
        torch_dtype=dtype,
        trust_remote_code=args.model.trust_remote_code,
        attn_implementation="flash_attention_2"
    )
    
    # 冻结 LLM 参数，节省显存和计算
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    model = llama_model_patch(model = model)
    return model

def load_tokenizer(model_args: ModelConfig) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_args.model)

def load_data(dataset_args: DataConfig, tokenizer: PreTrainedTokenizer,  rank: int) -> Dataset:
    # ... (保持原样) ...
    try:
        dataset = load_dataset(
            dataset_args.dataset,
            split=dataset_args.split,
            trust_remote_code=dataset_args.trust_remote_code,
        )
    except ValueError as e:
        if "load_from_disk" in str(e):
            dataset = Dataset.load_from_disk(dataset_args.dataset, keep_in_memory=False)
        else:
            raise e
    return dataset


def set_seed(seed: int):
    # 这一点非常重要：
    # 对于 TP，同一 MP 组内的 GPU 必须拥有相同的随机种子，以保证数据 Shuffle 顺序一致。
    # 对于 DP，通常我们希望数据不同，但在 DistributedTokenizedDataset 里是通过 shard 切分来实现的，
    # 而不是通过随机种子。所以为了安全起见，所有卡设置相同的种子是没问题的。
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
    
    # 绑定当前进程到指定 GPU，这对 device_mesh 初始化很重要
    if is_distributed_training:
        torch.cuda.set_device(rank)

    parser = ArgumentParser()
    parser.add_arguments(SaeConfig, dest = "sae")
    parser.add_arguments(ModelConfig, dest = "model")
    parser.add_arguments(DataConfig, dest = "data")
    parser.add_arguments(TrainConfig, dest = "train")
    
    args = parser.parse_args()
    # 兼容性处理
    model_args = args.model
    data_args = args.data
    train_args = args.train
    sae_args = args.sae
    train_args.ctx_len = data_args.ctx_len
    
    set_seed(train_args.seed)
    
    data_parallel_group = None
    model_parallel_group = None
    
    if is_distributed_training:
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        
        # 【关键检查】确保 DP * MP = World Size
        assert train_args.dp_size * train_args.mp_size == world_size, \
            f"DP Size ({train_args.dp_size}) * MP Size ({train_args.mp_size}) != World Size ({world_size})"

        # 初始化 Device Mesh
        # 假设 mesh 形状是 (dp, mp)。
        # 第一维是 data_parallel (在这一维上做数据切分)
        # 第二维是 model_parallel (在这一维上做模型切分)
        device_mesh = init_device_mesh("cuda", (train_args.dp_size, train_args.mp_size), mesh_dim_names=("data_parallel", "model_parallel"))
        
        data_parallel_group = device_mesh.get_group(mesh_dim="data_parallel")
        model_parallel_group = device_mesh.get_group(mesh_dim="model_parallel")
        
        if rank == 0:
            print_rank_0(f"Using Parallel across {world_size} GPUs.")
            print_rank_0(f"Data Parallel Group Size: {dist.get_world_size(data_parallel_group)}")
            print_rank_0(f"Model Parallel Group Size: {dist.get_world_size(model_parallel_group)}")

    tokenizer = load_tokenizer(model_args)
    # 所有卡都加载完整的 LLM
    model = load_model(args, tokenizer, rank)

    # 【关键数据逻辑】
    # 1. dataset_world_size: 决定把数据切成几份。应该等于 DP Size。
    # 2. dataset_rank: 决定当前进程拿哪一份数据。
    # 
    # Device Mesh 逻辑验证：
    # 假设 4 卡，DP=2, MP=2。
    # Mesh: [[0, 1], [2, 3]]
    # Data Parallel Group (纵向): [0, 2], [1, 3]
    # Model Parallel Group (横向): [0, 1], [2, 3]
    #
    # 对于 Rank 0 (在组 [0, 2] 中): rank_in_group = 0
    # 对于 Rank 1 (在组 [1, 3] 中): rank_in_group = 0
    # -> 结论：Rank 0 和 Rank 1 拿到了相同的数据切片 (Slice 0)。这是正确的！因为它们是 MP 关系。
    #
    # 对于 Rank 2 (在组 [0, 2] 中): rank_in_group = 1
    # 对于 Rank 3 (在组 [1, 3] 中): rank_in_group = 1
    # -> 结论：Rank 2 和 Rank 3 拿到了相同的数据切片 (Slice 1)。
    #
    # 最终：(0,1) 合作训练 Slice 0，(2,3) 合作训练 Slice 1。逻辑完美闭环。
    
    dataset_world_size = 1
    dataset_rank = 0
    if is_distributed_training:
        dataset_world_size = dist.get_world_size(data_parallel_group)
        dataset_rank = dist.get_rank(data_parallel_group)
        
    dataset = DistributedTokenizedDataset(
        path = data_args.dataset,
        tokenizer = tokenizer,
        seq_length = data_args.ctx_len,
        current_rank = dataset_rank, # 这里的 Rank 是 DP 组内的 Rank
        world_size = dataset_world_size # 这里的 Size 是 DP 组的大小
    )

    # 日志文件处理
    log_dir = Path("logs") / f"{train_args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 标记 MP rank 和 DP rank 便于调试
    dp_rank = dist.get_rank(data_parallel_group) if is_distributed_training else 0
    mp_rank = dist.get_rank(model_parallel_group) if is_distributed_training else 0
    
    log_file = log_dir / f"dp{dp_rank}-mp{mp_rank}.log"
    log_file.touch(exist_ok = True)
    
    # 只有总 Rank 0 打印到控制台，其他输出到文件
    # 注意：在 TP 中，最好偶尔检查一下所有 Rank 的日志，确保没有死锁
    with nullcontext() if rank == 0 else redirect_stdout(open(str(log_file), "w")):
        print(f"Training on '{data_args.dataset}'")
        print(f"Global Rank: {rank} | DP Rank: {dp_rank} | MP Rank: {mp_rank}")
        print(f"Model Parallel Group: {model_parallel_group}")

        trainer = SaeTrainer(
            train_cfg=train_args, 
            sae_cfg=sae_args, 
            model_cfg=model_args, 
            dataset=dataset, 
            model=model, 
            data_parallel_group=data_parallel_group, 
            model_parallel_group=model_parallel_group # 传入 MP 组
        )
        trainer.fit()

    if is_distributed_training:
        dist.destroy_process_group()

if __name__ == "__main__":
    run()
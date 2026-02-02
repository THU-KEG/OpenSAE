import os
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
import math

import numpy as np
import random
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from datasets import Dataset, load_dataset
from simple_parsing import field, parse, ArgumentParser
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer,AutoConfig

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
def load_model_pipeline(args, rank: int, pp_rank: int, pp_size: int, mp_rank: int) -> torch.nn.Module:
    """
    工业级加载：基于 Early Exit 层数进行动态均衡切分。
    逻辑：
    1. 计算有效层数 (Effective Layers) = min(Total Layers, Exit Layer + 1)
    2. 将有效层数均匀分配给 PP Stages。
    3. 每个 Rank 只加载自己负责的那部分层到显存，其余留在 CPU 或释放。
    """
    model_path = args.model.model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # === 1. 计算切分范围 (Slicing Logic) ===
    full_model_layers = config.num_hidden_layers
    
    # 获取用户设置的退出层 (默认为最后一层)
    # 务必确保 train_arguments.py 里有 early_exit_inference_layer_num
    exit_layer_idx = getattr(args.train, "early_exit_inference_layer_num", full_model_layers - 1)
    
    # 有效总层数：比如 exit=26，则我们需要跑 0-26 共 27 层
    target_layers = exit_layer_idx + 1
    effective_num_layers = min(target_layers, full_model_layers)
    
    # 计算每个 Stage 分到的层数 (向上取整)
    # 例如：Effective=27, PP=2 => ceil(13.5) = 14 层/stage
    layers_per_stage = math.ceil(effective_num_layers / pp_size)
    
    # 当前 Rank 的负责范围 [start, end)
    start_layer = pp_rank * layers_per_stage
    # 结束点不能超过有效总层数
    end_layer = min((pp_rank + 1) * layers_per_stage, effective_num_layers)
    
    # 打印调试信息
    if start_layer >= effective_num_layers:
        print(f"[Rank {rank}] PP-Rank {pp_rank}: IDLE (Start {start_layer} >= Effective Total {effective_num_layers})")
        # 即使空转也需要实例化模型结构，防止 DDP 初始化失败
    else:
        print(f"[Rank {rank}] PP-Rank {pp_rank}: Active Range [{start_layer}, {end_layer - 1}] "
              f"(Allocated {end_layer - start_layer} layers from total effective {effective_num_layers})")

    # === 2. 加载模型骨架 (Load Skeleton) ===
    # device_map="cpu" 极其重要，防止 full load 爆显存
    print(f"[Rank {rank}] Loading model weights to CPU...")
    model = args.model.auto_model_class.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="cpu" 
    )
    model.eval()
    model.requires_grad_(False)
    
    device = torch.device(f"cuda:{rank}")
    
    def safe_to(module, device):
        if module is not None:
            module.to(device)

    # === 3. 按需移动到 GPU (Move to GPU) ===
    
    # 3.1 Embeddings: 只有 PP Rank 0 需要
    if pp_rank == 0:
        print(f"[Rank {rank}] Moving Embeddings to GPU")
        safe_to(getattr(model, "embed_tokens", None), device)
    
    # 3.2 Layers: 只移动负责范围内的层
    if hasattr(model, "layers"):
        # 我们可以选择释放掉不需要的层以节省 CPU 内存，但在 PP 这种规模下通常不需要
        # 这里只做 .to(device)
        for i in range(start_layer, end_layer):
            print(f"[Rank {rank}] Moving Layer {i} to GPU")
            model.layers[i].to(device)
            
    # 3.3 Final Norm & Head: 只有负责“最后一层有效层”的 Rank 需要
    # 逻辑：如果这个 Rank 的 range 包含了 effective_num_layers - 1，那它就是最后一棒
    last_effective_layer_idx = effective_num_layers - 1
    if start_layer <= last_effective_layer_idx < end_layer:
        print(f"[Rank {rank}] Moving Final Norm & Head to GPU (Responsible for output)")
        safe_to(getattr(model, "norm", None), device)

    # === 4. 清理缓存 ===
    torch.cuda.empty_cache()
    
    return model
def run():
    parser = ArgumentParser()
    parser.add_arguments(SaeConfig, dest="sae")
    parser.add_arguments(ModelConfig, dest="model")
    parser.add_arguments(DataConfig, dest="data")
    parser.add_arguments(TrainConfig, dest="train")
    args = parser.parse_args()
    model_args = args.model
    data_args = args.data
    train_args = args.train
    sae_args = args.sae
    train_args.ctx_len = data_args.ctx_len
    # 设置种子
    torch.manual_seed(args.train.seed)
    
    # 分布式初始化
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 3D Device Mesh: (DP, PP, MP)
    # 假设 8 卡: DP=2, PP=2, MP=2
    # mesh 形状: (2, 2, 2)
    expected_ws = args.train.dp_size * args.train.pp_size * args.train.mp_size
    assert world_size == expected_ws, f"World Size {world_size} != DP*PP*MP ({expected_ws})"
    
    device_mesh = init_device_mesh(
        "cuda", 
        (args.train.dp_size, args.train.pp_size, args.train.mp_size), 
        mesh_dim_names=("dp", "pp", "mp")
    )
    
    if rank == 0:
        print(f"Device Mesh Initialized: {device_mesh}")

    # 获取子组
    dp_group = device_mesh.get_group(mesh_dim="dp")
    pp_group = device_mesh.get_group(mesh_dim="pp")
    mp_group = device_mesh.get_group(mesh_dim="mp")

    # 当前进程在各维度的 Rank
    pp_rank = dist.get_rank(pp_group)
    mp_rank = dist.get_rank(mp_group)
    
    # 加载 Tokenizer
    tokenizer = AutoModel.from_pretrained(args.model.model, trust_remote_code=True).tokenizer if hasattr(AutoModel.from_pretrained(args.model.model, trust_remote_code=True), 'tokenizer') else None 
    # 修正: 上面这行写法不好，直接加载 tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model.model, trust_remote_code=True)

    # 加载部分模型 (Pipeline Parallel Loading)
    model = load_model_pipeline(args, rank, pp_rank, args.train.pp_size, mp_rank)

    # Dataset 分片
    # 注意：在我们的架构中，整个 PP 组 + TP 组共享同一个 Batch。
    # 只有 DP 组之间数据不同。
    dataset_world_size = dist.get_world_size(dp_group)
    dataset_rank = dist.get_rank(dp_group)
    
    dataset = DistributedTokenizedDataset(
        path=args.data.dataset,
        tokenizer=tokenizer,
        seq_length=args.data.ctx_len,
        current_rank=dataset_rank,
        world_size=dataset_world_size
    )

    # 初始化 Trainer
    trainer = SaeTrainer(
        train_cfg=train_args,
        sae_cfg=args.sae,
        model_cfg=args.model,
        dataset=dataset,
        model=model,
        device_mesh=device_mesh, # 传入 mesh 方便管理
        data_parallel_group=dp_group,
        model_parallel_group=mp_group,
        pipeline_parallel_group=pp_group
    )
    
    trainer.fit()
    dist.destroy_process_group()

if __name__ == "__main__":
    run()

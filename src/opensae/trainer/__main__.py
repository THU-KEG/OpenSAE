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
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer, AutoConfig

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
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    model = llama_model_patch(model=model)
    return model


def load_tokenizer(model_args: ModelConfig) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_args.model)


def set_seed(seed: int):
    """
    [核心对齐] 强制设置所有随机数生成器的种子。
    为了保证 PP 和非 PP 模式下的运算精度一致，必须开启确定性算法。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # [新增] 强制 CuDNN 使用确定性算法 (可能会轻微降低速度，但为了对齐是必须的)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # [可选] 如果追求极致对齐，可以开启 PyTorch 确定性模式 (通常不需要，除非调试)
    # torch.use_deterministic_algorithms(True)


def load_model_pipeline(args, rank: int, pp_rank: int, pp_size: int, mp_rank: int) -> torch.nn.Module:
    """
    工业级加载：基于 Early Exit 层数进行动态均衡切分。
    """
    model_path = args.model.model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # 1. 准备量化配置 (与 load_model 保持一致)
    quantization_config = None
    if args.model.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    if args.model.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"
    full_model_layers = config.num_hidden_layers
    exit_layer_idx = getattr(args.train, "early_exit_inference_layer_num", full_model_layers - 1)
    target_layers = exit_layer_idx + 1
    effective_num_layers = min(target_layers, full_model_layers)
    
    layers_per_stage = math.ceil(effective_num_layers / pp_size)
    
    start_layer = pp_rank * layers_per_stage
    end_layer = min((pp_rank + 1) * layers_per_stage, effective_num_layers)
    
    if start_layer >= effective_num_layers:
        print(f"[Rank {rank}] PP-Rank {pp_rank}: IDLE (Start {start_layer} >= Effective Total {effective_num_layers})")
    else:
        print(f"[Rank {rank}] PP-Rank {pp_rank}: Active Range [{start_layer}, {end_layer - 1}] "
              f"(Allocated {end_layer - start_layer} layers from total effective {effective_num_layers})")

    print(f"[Rank {rank}] Loading model weights to CPU...")
    # 即使 mapping 到 CPU，加载过程仍可能触碰 RNG
    if args.model.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model = args.model.auto_model_class.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=args.model.load_in_8bit)
            if args.model.load_in_8bit
            else None
        ),
        attn_implementation="flash_attention_2",
        device_map="cpu" ,
    )
    # ... model = ... from_pretrained(...)
    
    # ... model.eval() ...
    model.eval()
    model.requires_grad_(False)
    #model = llama_model_patch(model=model)
    device = torch.device(f"cuda:{rank}")
    
    def safe_to(module, device):
        if module is not None:
            module.to(device)

    # 3.1 Embeddings
    if pp_rank == 0:
        print(f"[Rank {rank}] Moving Embeddings to GPU")
        safe_to(getattr(model, "embed_tokens", None), device)
    
    # 3.2 Layers
    if hasattr(model, "layers"):
        for i in range(start_layer, end_layer):
            print(f"[Rank {rank}] Moving Layer {i} to GPU")
            model.layers[i].to(device)
            
    # 3.3 Final Norm & Head
    last_effective_layer_idx = effective_num_layers - 1
    if start_layer <= last_effective_layer_idx < end_layer:
        print(f"[Rank {rank}] Moving Final Norm & Head to GPU (Responsible for output)")
        safe_to(getattr(model, "norm", None), device)

    torch.cuda.empty_cache()
    return model


def run():
    parser = ArgumentParser()
    parser.add_arguments(SaeConfig, dest="sae")
    parser.add_arguments(ModelConfig, dest="model")
    parser.add_arguments(DataConfig, dest="data")
    parser.add_arguments(TrainConfig, dest="train")
    args = parser.parse_args()
    train_args=args.train
    data_args=args.data
    # 1. 设置种子 (第一道防线)
    set_seed(args.train.seed)
    train_args.ctx_len = data_args.ctx_len
    # 2. 分布式初始化
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 3D Device Mesh
    expected_ws = args.train.dp_size * args.train.pp_size * args.train.mp_size
    assert world_size == expected_ws, f"World Size {world_size} != DP*PP*MP ({expected_ws})"
    
    device_mesh = init_device_mesh(
        "cuda", 
        (args.train.dp_size, args.train.pp_size, args.train.mp_size), 
        mesh_dim_names=("dp", "pp", "mp")
    )
    
    if rank == 0:
        print(f"Device Mesh Initialized: {device_mesh}")

    dp_group = device_mesh.get_group(mesh_dim="dp")
    pp_group = device_mesh.get_group(mesh_dim="pp")
    mp_group = device_mesh.get_group(mesh_dim="mp")

    pp_rank = dist.get_rank(pp_group)
    mp_rank = dist.get_rank(mp_group)
    
    # 3. 加载模型 (这一步在 PP 模式下会消耗不同数量的随机数)
    model = load_model_pipeline(args, rank, pp_rank, args.train.pp_size, mp_rank)
    
    # 4. 【核心对齐步】强制拉回 RNG 状态
    # 这一步必须在 load_model 之后，SaeTrainer 初始化之前
    # 确保无论 PP rank 加载了多少层，SAE 权重初始化的起点是完全一致的
    torch.cuda.synchronize()
    set_seed(args.train.seed)
    
    # 5. 加载 Tokenizer (清理了重复代码)
    tokenizer = AutoTokenizer.from_pretrained(args.model.model, trust_remote_code=True)

    # 6. Dataset 分片
    dataset_world_size = dist.get_world_size(dp_group)
    dataset_rank = dist.get_rank(dp_group)
    
    dataset = DistributedTokenizedDataset(
        path=args.data.dataset,
        tokenizer=tokenizer,
        seq_length=args.data.ctx_len,
        current_rank=dataset_rank,
        world_size=dataset_world_size
    )

    # 7. 初始化 Trainer
    trainer = SaeTrainer(
        train_cfg=train_args,
        sae_cfg=args.sae,
        model_cfg=args.model,
        dataset=dataset,
        model=model,
        device_mesh=device_mesh,
        data_parallel_group=dp_group,
        model_parallel_group=mp_group,
        pipeline_parallel_group=pp_group
    )
    
    trainer.fit()
    dist.destroy_process_group()

if __name__ == "__main__":
    run()
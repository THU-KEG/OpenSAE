import torch
from torch import Tensor
from torch import nn
from torch.amp import custom_bwd, custom_fwd


class TopK(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        
    def forward(self, x):
        feature_activation, feature_indices = torch.topk(x, self.k, dim=-1, sorted=False)
        return feature_activation, feature_indices
import torch
import torch.nn as nn
import torch.distributed as dist

class GlobalTopK(nn.Module):
    def __init__(self, k: int, process_group: dist.ProcessGroup = None):
        super().__init__()
        self.k = k
        self.process_group = process_group

    def forward(self, x: torch.Tensor):
        # 1. 本地 Top-K
        local_topk_values, local_topk_indices = torch.topk(x, self.k, dim=-1)

        # 如果没有组，直接返回（退化为 Local Top-K）
        if self.process_group is None or dist.get_world_size(self.process_group) <= 1:
            return local_topk_values, local_topk_indices

        # ================== 核心修复区域 ==================
        
        # [关键修复] 强制内存连续，否则 all_gather 可能会静默失败传回 0
        local_topk_values = local_topk_values.contiguous()

        world_size = dist.get_world_size(self.process_group)
        
        # 准备接收容器
        gathered_values = [torch.zeros_like(local_topk_values) for _ in range(world_size)]
        
        # 通信：交换分数
        dist.all_gather(gathered_values, local_topk_values, group=self.process_group)
        
        # --- 调试：验证是否收到了非零数据 ---
        if dist.get_rank(self.process_group) == 0 and not hasattr(self, '_logged_check'):
            # 检查 Rank 1 的数据是不是全 0
            r1_max = gathered_values[1].max().item()
            print(f"DEBUG: Rank 0 sees Rank 1 max value: {r1_max:.4f}")
            if r1_max == 0:
                print("DEBUG: ⚠️ WARNING: All-Gather seems to be receiving Zeros! Communication issue.")
            self._logged_check = True
        # ---------------------------------------------

        # 拼接所有候选分值
        all_candidates = torch.cat(gathered_values, dim=1) # [Batch, k * World_Size]
        
        # 3. 确定全局分数线
        # 在候选池里找第 k 大的值
        global_topk_values, _ = torch.topk(all_candidates, self.k, dim=-1)
        threshold = global_topk_values[:, -1].unsqueeze(1) # [Batch, 1]
        
        # 4. 本地过滤
        # 只有大于等于阈值的才保留，否则置 0
        mask = local_topk_values >= threshold
        final_values = local_topk_values * mask.to(local_topk_values.dtype)
        
        # --- 调试：最终计数验证 ---
        if dist.get_rank(self.process_group) == 0 and not hasattr(self, '_logged_count'):
            active_count = (final_values > 0).sum().item()
            batch_size = x.shape[0]
            avg_per_token = active_count / batch_size
            print(f"DEBUG: Rank 0 active count: {active_count} (Avg per token: {avg_per_token:.1f} | Target: {self.k / world_size})")
            self._logged_count = True
        # ------------------------

        return final_values, local_topk_indices

class JumpReLU(nn.Module):
    def __init__(self, theta: float):
        super().__init__()
        self.theta = theta
        
        
    def forward(self, x: Tensor):
        x_max = x.max(dim = -1).values

        theta = torch.ones_like(x_max) * self.theta
        theta = torch.where(x_max < theta, x_max - 1e-6, theta).unsqueeze(-1)
        
        mask = x > theta
        jump_relu_val = torch.where(mask, x, 0)
        max_acts = mask.sum(dim=-1).max()

        # Zijun: Very Tricky!
        #        Padding to max_acts. Organized into batch is necessary for the sparse decoder kernel.
        feature_activation, feature_indices = torch.topk(jump_relu_val, max_acts, dim=-1, sorted=False)
        return feature_activation, feature_indices


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Check Forward
    x = torch.randn(4, 3, 6)
    # init a param with x
    x = nn.Parameter(x, requires_grad=True)
    print(x)
    print()
     
    jump_relu = JumpReLU(1)
    feature_activation, feature_indices = jump_relu(x)
    print(feature_activation.shape)
    print(feature_activation)
    print(feature_indices.shape)
    print(feature_indices)
    
    print()
    
    topk = TopK(2)
    feature_activation, feature_indices = topk(x)
    print(feature_activation.shape)
    print(feature_activation)
    print(feature_indices.shape)
    print(feature_indices)
    
    print()
    
    # Check Backward
    feature_activation, feature_indices = jump_relu(x)
    feature_activation.sum().backward()
    print(x.grad)
    
    x.grad.zero_()
    feature_activation, feature_indices = topk(x)
    feature_activation.sum().backward()
    print(x.grad)

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
    
    
    
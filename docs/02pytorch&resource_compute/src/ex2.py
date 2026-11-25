import torch
import torch.nn as nn
import numpy as np
class Linear(nn.Module):
    """Simple linear layer."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight

class Cruncher(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            Linear(dim, dim)
            for i in range(num_layers)
        ])
        self.final = Linear(dim, 1)  # 创建一个最终的 Linear 层，它将 dim 维的特征映射到 1 维的标量输出
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply linear layers
        B, D = x.size()
        for layer in self.layers:
            x = layer(x)
        # Apply final head
        x = self.final(x)
        assert x.size() == torch.Size([B, 1])
        # Remove the last dimension
        x = x.squeeze(-1) # 移除最后一个维度（即大小为1的那个维度），使得最终输出是一个一维张量 (B,)，更符合我们对“预测值”的直观理解（例如，对每个样本输出一个分数）
        assert x.size() == torch.Size([B])
        return x

D = 64 # Dimension
num_layers = 2
model = Cruncher(dim=D, num_layers=num_layers)

param_sizes = [
    (name, param.numel())  # param.numel()：返回该参数张量中的元素总数
    for name, param in model.state_dict().items()
]
assert param_sizes == [
    ("layers.0.weight", D * D),
    ("layers.1.weight", D * D),
    ("final.weight", D),
]

device = "cpu"
B = 8 # Batch size
x = torch.randn(B, D, device=device)
y = model(x)
assert y.size() == torch.Size([B])
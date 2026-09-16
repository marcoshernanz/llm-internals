import torch
from torch import nn


D_MODEL = 64


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor):  # [..., dim]
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)  # [..., 1]
        normalized = x * torch.rsqrt(mean_square + self.eps)  # [..., dim]
        return normalized * self.weight  # [..., dim]

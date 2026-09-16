import torch
from torch import Tensor, nn

D_MODEL = 64


class Linear(nn.Module):
    def __init__(self):
        pass

    def forward(self, x: Tensor):
        pass


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-20) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:  # [..., dim]
        x32 = x.float()  # [..., dim]
        normed = x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + self.eps)  # [..., dim]
        return normed.to(x.dtype) * self.weight  # [..., dim]


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        pass

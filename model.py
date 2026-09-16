import torch
import torch.nn.functional as F
from torch import Tensor, nn

D_MODEL = 64
INIT_STD = 0.006  # DeepSeek-V3 4.3: every learnable parameter starts from N(0, 0.006)


def init_weights(module: nn.Module) -> None:
    """Draw every weight matrix from one narrow normal. Exceptions are added as they appear."""
    if isinstance(module, Linear):
        nn.init.normal_(module.weight, std=INIT_STD)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))  # [out, in]

    def forward(self, x: Tensor) -> Tensor:  # [..., in]
        return F.linear(x, self.weight)  # [..., out]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-20) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:  # [..., dim]
        x32 = x.float()  # [..., dim]
        normed = x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + self.eps)  # [..., dim]
        return normed.to(x.dtype) * self.weight  # [..., dim]


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        pass

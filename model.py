"""
Shape letters used in the comments:
    B   batch size
    T   sequence length (time)
    D   model width (hidden_size)
    F   hidden width of a feed-forward network (intermediate size)
    H   attention heads
    Dh  width of one head
    Dr  rotary dims per head (qk_rope_head_dim), the last Dr of the Dh
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

NORM_EPS = 1e-20
INIT_STD = 0.006
SWIGLU_LIMIT = 10.0
ROPE_THETA = 10000.0
COMPRESS_ROPE_THETA = 160000.0


def init_weights(module: nn.Module) -> None:
    if isinstance(module, Linear):
        nn.init.normal_(module.weight, std=INIT_STD)


def rope_freqs_cis(dim: int, seq_len: int, base: float) -> Tensor:
    """One unit complex number per position and pair: cos(angle) + i sin(angle)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))  # [Dr/2]
    positions = torch.arange(seq_len)  # [T]
    angles = torch.outer(positions, freqs)  # [T, Dr/2]
    return torch.polar(torch.ones_like(angles), angles)  # [T, Dr/2] complex


def apply_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:  # [B, H, T, Dr]
    """Rotate each adjacent pair of features by its position-dependent angle."""
    seq_len = x.size(2)
    pairs = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))  # [B, H, T, Dr/2]
    rotated = pairs * freqs_cis[:seq_len]  # [B, H, T, Dr/2]
    return torch.view_as_real(rotated).flatten(-2).to(x.dtype)  # [B, H, T, Dr]


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))  # [out, in]

    def forward(self, x: Tensor) -> Tensor:  # [..., in]
        return F.linear(x, self.weight)  # [..., out]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = NORM_EPS) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # [dim]
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:  # [..., dim]
        x32 = x.float()  # [..., dim]
        normed = x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + self.eps)  # [..., dim]
        return normed.to(x.dtype) * self.weight  # [..., dim]


class FeedForward(nn.Module):
    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = SWIGLU_LIMIT) -> None:
        super().__init__()
        self.gate_proj = Linear(dim, inter_dim)
        self.up_proj = Linear(dim, inter_dim)
        self.down_proj = Linear(inter_dim, dim)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: Tensor) -> Tensor:  # [..., D]
        gate = self.gate_proj(x).float()  # [..., F]
        up = self.up_proj(x).float()  # [..., F]
        if self.swiglu_limit > 0:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        hidden = F.silu(gate) * up  # [..., F]
        return self.down_proj(hidden.to(x.dtype))  # [..., D]

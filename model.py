"""
Shape letters used in the comments:
    B   batch size
    T   sequence length (time)
    D   model width (hidden_size)
    F   hidden width of a feed-forward network (intermediate size)
    H   attention heads
    Dh  width of one head
    Dr  rotary dims per head (qk_rope_head_dim), the last Dr of the Dh
    Dq  width of the query bottleneck (q_lora_rank)
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import Config

NORM_EPS = 1e-20
INIT_STD = 0.006
SWIGLU_LIMIT = 10.0
ROPE_THETA = 10000.0
COMPRESS_ROPE_THETA = 160000.0
ROPE_BETA_FAST = 32
ROPE_BETA_SLOW = 1
SLIDING_WINDOW = 128


def init_weights(module: nn.Module) -> None:
    if isinstance(module, Linear):
        nn.init.normal_(module.weight, std=INIT_STD)


def yarn_rescale(freqs: Tensor, base: float, original_seq_len: int, factor: float) -> Tensor:
    dim = 2 * freqs.numel()

    def pair_completing(turns: float) -> float:
        """Index of the pair that completes ``turns`` rotations over the training context."""
        return dim * math.log(original_seq_len / (turns * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(pair_completing(ROPE_BETA_FAST)), 0)
    high = min(math.ceil(pair_completing(ROPE_BETA_SLOW)), dim - 1)
    pair_index = torch.arange(dim // 2)  # [Dr/2]
    ramp = ((pair_index - low) / max(high - low, 1e-3)).clamp(0, 1)  # [Dr/2], 0 keeps, 1 slows
    return freqs * (1 - ramp) + (freqs / factor) * ramp  # [Dr/2]


def rope_freqs_cis(
    dim: int, seq_len: int, base: float, original_seq_len: int = 0, factor: float = 1.0
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))  # [Dr/2]
    if original_seq_len > 0:
        freqs = yarn_rescale(freqs, base, original_seq_len, factor)  # [Dr/2]
    positions = torch.arange(seq_len)  # [T]
    angles = torch.outer(positions, freqs)  # [T, Dr/2]
    return torch.polar(torch.ones_like(angles), angles)  # [T, Dr/2] complex


def apply_rope(x: Tensor, freqs_cis: Tensor, inverse: bool = False) -> Tensor:  # [B, T, H, Dr]
    pairs = x.float()  # [B, T, H, Dr]
    pairs = pairs.unflatten(-1, (-1, 2))  # [B, T, H, Dr/2, 2]
    pairs = torch.view_as_complex(pairs)  # [B, T, H, Dr/2] complex

    if inverse:
        freqs_cis = freqs_cis.conj()  # the conjugate rotates by minus the angle
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), -1)  # [1, T, Dr/2]
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, -1)  # [1, T, 1, Dr/2]

    rotated = pairs * freqs_cis  # [B, T, H, Dr/2] complex
    rotated = torch.view_as_real(rotated)  # [B, T, H, Dr/2, 2]
    rotated = rotated.flatten(-2)  # [B, T, H, Dr]
    return rotated.to(x.dtype)  # [B, T, H, Dr]


def apply_partial_rope(x: Tensor, freqs_cis: Tensor, inverse: bool = False) -> Tensor:  # [..., Dh]
    rope_dim = 2 * freqs_cis.size(-1)  # Dr
    content = x[..., :-rope_dim]  # [..., Dh-Dr]
    rotary = x[..., -rope_dim:]  # [..., Dr]
    rotary = apply_rope(rotary, freqs_cis, inverse)  # [..., Dr]
    return torch.cat([content, rotary], dim=-1)  # [..., Dh]


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
        mean_square = x32.square().mean(dim=-1, keepdim=True)  # [..., 1]
        normed = x32 * torch.rsqrt(mean_square + self.eps)  # [..., dim]
        normed = normed.to(x.dtype)  # [..., dim]
        return normed * self.weight  # [..., dim]


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


class Attention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        self.q_down_proj = Linear(config.hidden_size, config.q_lora_rank)
        self.q_norm = RMSNorm(config.q_lora_rank)
        self.q_up_proj = Linear(config.q_lora_rank, self.num_heads * self.head_dim)
        self.kv_proj = Linear(config.hidden_size, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim)

        freqs_cis = rope_freqs_cis(
            config.qk_rope_head_dim, config.max_position_embeddings, ROPE_THETA
        )
        self.freqs_cis = nn.Buffer(freqs_cis, persistent=False)  # [max T, Dr/2]

        mask = torch.ones(config.max_position_embeddings, config.max_position_embeddings).tril()
        mask = mask.masked_fill(mask, -torch.inf)
        self.mask = nn.Buffer(mask)

    def forward(self, x: Tensor):
        freqs_cis = self.freqs_cis[: x.size(1)]  # [T, Dr/2]

        q = self.q_down_proj(x)  # [B, T, Dq]
        q = self.q_norm(q)  # [B, T, Dq]
        q = self.q_up_proj(q)  # [B, T, H*Dh]
        q = q.unflatten(-1, (self.num_heads, self.head_dim))  # [B, T, H, Dh]
        q = apply_partial_rope(q, freqs_cis)  # [B, T, H, Dh]

        kv = self.kv_proj(x)  # [B, T, Dh]
        kv = self.kv_norm(kv)  # [B, T, Dh]
        kv = apply_partial_rope(kv, freqs_cis)  # [B, T, Dh]

        q = q.transpose(1, 2)  # [B, H, T, Dh]
        scores = q @ kv.mT  # [B, H, T, T]
        scores /= 1 / math.sqrt(self.head_dim)
        scores += self.mask

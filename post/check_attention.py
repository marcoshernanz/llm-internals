"""
Check the fused attention of post/qwen3.py against attention written out step by step: copy each
key-value head for the query heads that share it, score, mask, softmax, weight the values. In fp32,
for the three ways the model calls it: the whole prompt with nothing cached, one new token over a
cache, and several new tokens over a cache.
Run with ``uv run modal run gpu.py --modules post.check_attention``.

Shape letters used in the comments:
    B   batch size
    H   query heads
    G   key-value heads, each shared by H/G query heads
    T   new tokens, one query each
    S   keys: the cached positions, then the T new ones
    Dh  dims per head
"""

import torch
from torch import Tensor

from post.generate import DEVICE
from post.qwen3 import attend

B, H, G, DH = 2, 16, 8, 128  # Qwen3-0.6B's heads
CASES = {"prompt, nothing cached": (32, 32), "one new token": (1, 40), "8 new tokens": (8, 40)}
# Rounding moves the outputs by under 1e-6. One real bug, letting the kernel build the causal mask
# itself (is_causal=True lines the queries up with the first keys, not the last), moves them by 3
# whenever something is cached, and not at all on a fresh prompt: hence the three cases.
TOLERANCE = 1e-4


def reference(
    q: Tensor, k: Tensor, v: Tensor, scale: float
) -> Tensor:  # [B, H, T, Dh], [B, G, S, Dh]
    group_size = q.size(1) // k.size(1)  # H/G
    k = k.repeat_interleave(group_size, dim=1)  # [B, H, S, Dh], neighboring query heads share one
    v = v.repeat_interleave(group_size, dim=1)  # [B, H, S, Dh]
    cached = k.size(2) - q.size(2)  # positions before the new tokens
    causal = torch.ones(q.size(2), k.size(2), dtype=torch.bool, device=q.device)  # [T, S]
    causal = causal.tril(diagonal=cached)  # [T, S], new token t sees keys 0..cached+t
    scores = torch.einsum("bhtd,bhsd->bhts", q, k)  # [B, H, T, S]
    scores = scores * scale  # [B, H, T, S]
    scores = scores.masked_fill(~causal, -torch.inf)  # [B, H, T, S]
    probs = scores.softmax(dim=-1)  # [B, H, T, S]
    return torch.einsum("bhts,bhsd->bhtd", probs, v)  # [B, H, T, Dh]


def main() -> None:
    torch.manual_seed(0)
    scale = DH**-0.5
    for name, (new, keys) in CASES.items():
        q = torch.randn(B, H, new, DH, device=DEVICE)  # [B, H, T, Dh]
        k = torch.randn(B, G, keys, DH, device=DEVICE)  # [B, G, S, Dh]
        v = torch.randn(B, G, keys, DH, device=DEVICE)  # [B, G, S, Dh]
        fused = attend(q, k, v, scale)  # [B, H, T, Dh]
        written_out = reference(q, k, v, scale)  # [B, H, T, Dh]
        diff = (fused - written_out).abs().max().item()
        print(f"{name:<22} T {new:>2}, S {keys}: largest difference {diff:.2e}")
        assert diff < TOLERANCE
    print("the fused kernel gives the same attention as the written-out steps")


if __name__ == "__main__":
    main()

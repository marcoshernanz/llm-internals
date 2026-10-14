"""
Measure the GPU memory of one SFT step (post/sft.py) at steps of 1,024, 2,048 and 4,096 tokens.
The weights and AdamW's two moments stay in memory throughout, 12 bytes per weight in fp32. On top
of them, during a step: the gradients, 4 bytes per weight; the activations the backward pass needs;
and the logits, a row of V = 151,936 scores for every position. Each step runs three ways: with a
stand-in loss on the hidden states, which skips the head and so has no logits; with the plain loss,
every position's logits at once (post/check/loss.py); and with SFT's loss, the head fused with the
loss a chunk at a time (post/linear_cross_entropy.py). The difference from the first is what the
logits cost. Each step is one conversation of random tokens, every target counted: memory depends
only on the length. A step too big for the GPU is caught and reported.
Run with ``uv run modal run gpu.py --modules post.bench.memory``.

Shape letters used in the comments:
    T   tokens in the step
    D   model width
"""

from collections.abc import Callable

import torch
from torch import Tensor

from post.chat import sft_loss
from post.check.loss import plain_loss
from post.qwen3 import DEVICE, DTYPE, REPO_ID, Packing, Qwen3, load_qwen3

STEP_TOKENS = [1_024, 2_048, 4_096]
GIB = 2**30

Loss = Callable[[Qwen3, Tensor, Tensor, Packing], Tensor]


def body_only(model: Qwen3, ids: Tensor, mask: Tensor, packing: Packing) -> Tensor:  # [1, T] each
    """A stand-in loss on the hidden states: the whole model but the head, so no logits."""
    hidden = model.hidden(ids, packing=packing)  # [1, T, D]
    return hidden.float().square().mean()


def step(model: Qwen3, optimizer: torch.optim.Optimizer, loss_fn: Loss, tokens: int) -> None:
    """One SFT step on one conversation of random tokens."""
    ids = torch.randint(model.config.vocab_size, (1, tokens), device=DEVICE)  # [1, T]
    mask = torch.ones_like(ids, dtype=torch.bool)  # [1, T], every target counted
    packing = Packing([tokens], DEVICE)
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        loss = loss_fn(model, ids, mask, packing)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def peak(model: Qwen3, optimizer: torch.optim.Optimizer, loss_fn: Loss, tokens: int) -> float:
    """The peak memory allocated during one step, in GiB; infinite if the step ran out."""
    torch.cuda.empty_cache()  # whatever earlier steps left cached, so each starts alike
    torch.cuda.reset_peak_memory_stats()
    try:
        step(model, optimizer, loss_fn, tokens)
    except torch.OutOfMemoryError:
        optimizer.zero_grad(set_to_none=True)  # the gradients computed before it ran out
        return float("inf")
    return torch.cuda.max_memory_allocated() / GIB


def main() -> None:
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "SFT computes in bf16: run it on the L4"
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32: the master copy, as SFT trains
    # A learning rate of 0 keeps the weights as loaded: only the memory matters here
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.0, fused=True)
    step(model, optimizer, body_only, 16)  # the first step creates AdamW's moments
    fixed = torch.cuda.memory_allocated() / GIB
    total = torch.cuda.get_device_properties(0).total_memory / GIB
    print(f"the GPU: {total:.1f} GiB; the weights and AdamW's moments: {fixed:.1f} GiB, always")
    print("peak memory of one step, GiB; inf: it ran out")
    print(f"{'tokens':>6} {'no head':>8} {'plain':>6} {'logits':>7} {'fused':>6} {'logits':>7}")
    for tokens in STEP_TOKENS:
        no_head = peak(model, optimizer, body_only, tokens)
        plain = peak(model, optimizer, plain_loss, tokens)
        fused = peak(model, optimizer, sft_loss, tokens)
        print(
            f"{tokens:>6} {no_head:>8.1f} {plain:>6.1f} {plain - no_head:>+7.1f} "
            f"{fused:>6.1f} {fused - no_head:>+7.1f}"
        )
    logits = 1_024 * model.config.vocab_size * 4 / GIB  # [1024, V] in fp32
    print(f"one fp32 copy of the logits: {logits:.2f} GiB per 1,024 tokens")


if __name__ == "__main__":
    main()

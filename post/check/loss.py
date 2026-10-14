"""
Check SFT's loss, the head fused with cross-entropy a chunk at a time (post/linear_cross_entropy.py),
against the plain loss: every position's logits at once, then cross-entropy over the counted
targets, as SFT computed it before. Neither is exact. Both run the head's matmuls in bf16, which
keeps about 3 significant digits, and they round in different places: the fused loss rounds the
head's gradient once per chunk, the plain one once. Their gradients differ by up to 2% for a weight.
So each is measured against the exact gradient, with the head and the cross-entropy in fp32, and the
fused loss must be about as close to it as the plain one is. A wrong formula would be off by about
100%. All three run the body as SFT does: fp32 weights, bf16 matmuls. On the first 12 GSM8K test
conversations packed into one row: enough counted targets for two chunks, the second one partial.
Run with ``uv run modal run gpu.py --modules post.check.loss``.

Shape letters used in the comments:
    T   tokens in the packed row
    A   of them, the targets the loss counts
    D   model width
    V   vocabulary size
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from post.chat import gsm8k_chat, render, sft_loss
from post.gsm8k import load_gsm8k
from post.linear_cross_entropy import CHUNK
from post.qwen3 import DEVICE, REPO_ID, Packing, Qwen3, load_qwen3, load_tokenizer

CONVERSATIONS = 12  # the first GSM8K test problems
IGNORE = -100  # a target cross_entropy skips: left out of the sum and of the count
LOSS_TOLERANCE = 1e-3  # all three compute it in fp32, two of them from bf16 logits
RATIO = 2.0  # how much further from the exact gradient the fused loss may be than the plain one

Loss = Callable[[Qwen3, Tensor, Tensor, Packing], Tensor]


def plain_loss(model: Qwen3, ids: Tensor, mask: Tensor, packing: Packing) -> Tensor:  # [1, T] each
    """The loss computed plainly: the logits of every position at once, then cross-entropy."""
    targets = ids[:, 1:]  # [1, T-1]
    targets = targets.masked_fill(~mask[:, 1:], IGNORE)  # [1, T-1], only the assistant's count
    logits = model(ids, packing=packing)  # [1, T, V]
    logits = logits[:, :-1]  # [1, T-1, V], the last token has no target
    logits = logits.flatten(0, 1)  # [T-1, V]
    targets = targets.flatten()  # [T-1]
    return F.cross_entropy(logits.float(), targets, ignore_index=IGNORE)


def exact_loss(model: Qwen3, ids: Tensor, mask: Tensor, packing: Packing) -> Tensor:  # [1, T] each
    """The loss with the head and the cross-entropy in fp32; the body as SFT runs it."""
    hidden = model.hidden(ids, packing=packing)  # [1, T, D], fp32
    hidden = hidden[:, :-1]  # [1, T-1, D]
    targets = ids[:, 1:]  # [1, T-1]
    counted = mask[:, 1:]  # [1, T-1]
    hidden = hidden[counted]  # [A, D]
    targets = targets[counted]  # [A]
    with torch.autocast(DEVICE, enabled=False):  # the head's matmul in fp32 too
        logits = hidden @ model.embed_tokens.weight.T  # [A, V]
    return F.cross_entropy(logits, targets)


def gradients(
    model: Qwen3, loss_fn: Loss, ids: Tensor, mask: Tensor, packing: Packing
) -> tuple[float, dict[str, Tensor]]:
    """The loss, and the gradient of every weight, by name."""
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        loss = loss_fn(model, ids, mask, packing)
    loss.backward()
    grads = {}
    for name, weight in model.named_parameters():
        assert weight.grad is not None
        grads[name] = weight.grad
    model.zero_grad(set_to_none=True)  # the dict keeps these: the next backward starts anew
    return loss.item(), grads


def errors(grads: dict[str, Tensor], exact: dict[str, Tensor]) -> dict[str, float]:
    """Each weight's gradient error, relative: |grad - exact| / |exact|. Under "all", every
    weight's gradient taken together as one long vector."""
    result = {}
    squared_error = 0.0
    squared_norm = 0.0
    for name, exact_grad in exact.items():
        error = (grads[name] - exact_grad).norm().item()
        norm = exact_grad.norm().item()
        result[name] = error / norm
        squared_error += error**2
        squared_norm += norm**2
    result["all"] = (squared_error / squared_norm) ** 0.5
    return result


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32 weights, as SFT trains them
    problems = load_gsm8k("test")[:CONVERSATIONS]
    chats = [render(tokenizer, gsm8k_chat(problem)) for problem in problems]
    ids = torch.tensor([[t for chat_ids, _ in chats for t in chat_ids]], device=DEVICE)  # [1, T]
    mask = torch.tensor([[m for _, chat_mask in chats for m in chat_mask]], device=DEVICE)  # [1, T]
    packing = Packing([len(chat_ids) for chat_ids, _ in chats], DEVICE)
    counted = int(mask[:, 1:].sum().item())
    print(
        f"{CONVERSATIONS} conversations packed into {ids.size(1)} tokens, {counted} targets "
        f"counted: chunks of {CHUNK}"
    )
    assert counted > CHUNK, "more than one chunk, to check the chunks add up"

    exact, exact_grads = gradients(model, exact_loss, ids, mask, packing)
    plain, grads = gradients(model, plain_loss, ids, mask, packing)
    plain_errors = errors(grads, exact_grads)
    fused, grads = gradients(model, sft_loss, ids, mask, packing)
    fused_errors = errors(grads, exact_grads)
    print(f"loss: exact {exact:.5f}, plain {plain:.5f}, fused {fused:.5f}")

    print("each gradient's error, relative: |grad - exact| / |exact|")
    print(f"  {'':<34} {'plain':>8} {'fused':>8}")
    for name in [
        "embed_tokens.weight",  # the head's, and the input embedding's
        "norm.weight",  # the last norm, right before the head
        "layers.27.mlp.down_proj.weight",  # the last block
        "layers.0.self_attn.q_proj.weight",  # the first block
        "all",
    ]:
        print(f"  {name:<34} {plain_errors[name]:>8.1e} {fused_errors[name]:>8.1e}")
    worst_plain = max(plain_errors, key=plain_errors.__getitem__)
    worst_fused = max(fused_errors, key=fused_errors.__getitem__)
    print(f"  the largest, plain: {worst_plain} {plain_errors[worst_plain]:.1e}")
    print(f"  the largest, fused: {worst_fused} {fused_errors[worst_fused]:.1e}")

    assert abs(fused - exact) < LOSS_TOLERANCE
    assert fused_errors["all"] < RATIO * plain_errors["all"]
    assert fused_errors[worst_fused] < RATIO * plain_errors[worst_plain]
    print("the fused loss is as close to the exact gradient as the plain loss")


if __name__ == "__main__":
    main()

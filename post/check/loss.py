"""
Check the loss, the head fused with cross-entropy a chunk at a time (post/linear_cross_entropy.py),
against the plain loss: every position's logits at once, then cross-entropy over the counted
targets, as SFT computed it before. Twice: with SFT's scales, 1/A on every target, for the mean;
and with each conversation's targets scaled by a random number from -1 to 1, over the number of
conversations, as REINFORCE scales each answer by its reward (post/reinforce.py), and as a
baseline makes some scales negative. Then those signed scales once more, on each target's log
probability with its logits computed again in the backward pass, as GRPO computes it
(post/grpo.py). Neither is exact. Both run the head's matmuls in bf16, which
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

from post.chat import gsm8k_chat, render, scaled_loss, token_log_probs
from post.gsm8k import load_gsm8k
from post.linear_cross_entropy import CHUNK
from post.qwen3 import DEVICE, REPO_ID, Packing, Qwen3, load_qwen3, load_tokenizer

CONVERSATIONS = 12  # the first GSM8K test problems
# All three compute it in fp32, two of them from bf16 logits, which move each target's loss by
# about 5e-4, mostly the same way. So the sum's error grows with the scales' total size: for SFT's
# mean, a total of 1, the loss is 4.5e-4 off; with signed scales of total 37.8, 0.03
LOSS_TOLERANCE = 1e-3  # per unit of the scales' total size
RATIO = 2.0  # how much further from the exact gradient the fused loss may be than the plain one

Loss = Callable[[Qwen3, Tensor, Tensor, Tensor, Packing], Tensor]


def plain_loss(
    model: Qwen3, ids: Tensor, mask: Tensor, scales: Tensor, packing: Packing
) -> Tensor:  # [1, T], [1, T], [A]
    """The loss computed plainly: the logits of every position at once, then cross-entropy."""
    logits = model(ids, packing=packing)  # [1, T, V]
    logits = logits[:, :-1]  # [1, T-1, V], the last token has no target
    targets = ids[:, 1:]  # [1, T-1]
    counted = mask[:, 1:]  # [1, T-1], only the assistant's count
    logits = logits[counted]  # [A, V]
    targets = targets[counted]  # [A]
    losses = F.cross_entropy(logits.float(), targets, reduction="none")  # [A]
    return (scales * losses).sum()


def exact_loss(
    model: Qwen3, ids: Tensor, mask: Tensor, scales: Tensor, packing: Packing
) -> Tensor:  # [1, T], [1, T], [A]
    """The loss with the head and the cross-entropy in fp32; the body as SFT runs it."""
    hidden = model.hidden(ids, packing=packing)  # [1, T, D], fp32
    hidden = hidden[:, :-1]  # [1, T-1, D]
    targets = ids[:, 1:]  # [1, T-1]
    counted = mask[:, 1:]  # [1, T-1]
    hidden = hidden[counted]  # [A, D]
    targets = targets[counted]  # [A]
    with torch.autocast(DEVICE, enabled=False):  # the head's matmul in fp32 too
        logits = hidden @ model.embed_tokens.weight.T  # [A, V]
    losses = F.cross_entropy(logits, targets, reduction="none")  # [A]
    return (scales * losses).sum()


def gradients(
    model: Qwen3, loss_fn: Loss, ids: Tensor, mask: Tensor, scales: Tensor, packing: Packing
) -> tuple[float, dict[str, Tensor]]:
    """The loss, and the gradient of every weight, by name."""
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        loss = loss_fn(model, ids, mask, scales, packing)
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


def log_prob_loss(
    model: Qwen3, ids: Tensor, mask: Tensor, scales: Tensor, packing: Packing
) -> Tensor:  # [1, T], [1, T], [A]
    """The same loss from each target's log probability (post/linear_cross_entropy.py's
    TokenLogProbs), as GRPO computes it: minus each one times its scale, summed."""
    return -(scales * token_log_probs(model, ids, mask, packing)).sum()


def compare(
    model: Qwen3, fused_loss: Loss, ids: Tensor, mask: Tensor, scales: Tensor, packing: Packing
) -> None:
    """Print each loss and its gradient's errors against the exact one; assert the fused loss is
    about as close as the plain one."""
    exact, exact_grads = gradients(model, exact_loss, ids, mask, scales, packing)
    plain, grads = gradients(model, plain_loss, ids, mask, scales, packing)
    plain_errors = errors(grads, exact_grads)
    fused, grads = gradients(model, fused_loss, ids, mask, scales, packing)
    fused_errors = errors(grads, exact_grads)
    size = scales.abs().sum().item()  # the scales' total size
    print(f"loss: exact {exact:.5f}, plain {plain:.5f}, fused {fused:.5f}; scales' size {size:.1f}")

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

    assert abs(fused - exact) < LOSS_TOLERANCE * size
    assert fused_errors["all"] < RATIO * plain_errors["all"]
    assert fused_errors[worst_fused] < RATIO * plain_errors[worst_plain]


def main() -> None:
    torch.manual_seed(0)
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

    print("\nSFT's scales: 1/A on every target")
    scales = torch.full((counted,), 1 / counted, device=DEVICE)  # [A]
    compare(model, scaled_loss, ids, mask, scales, packing)

    print("\neach conversation's targets scaled by one random number from -1 to 1, over 12")
    numbers = torch.rand(CONVERSATIONS) * 2 - 1  # [12], from -1 to 1
    per_target = [n / CONVERSATIONS for n, (_, m) in zip(numbers.tolist(), chats) for c in m if c]
    scales = torch.tensor(per_target, device=DEVICE)  # [A]
    compare(model, scaled_loss, ids, mask, scales, packing)

    print("\nthe same signed scales, on each target's log probability, its logits computed again")
    print("in the backward pass")
    compare(model, log_prob_loss, ids, mask, scales, packing)
    print("\nthe fused losses are as close to the exact gradient as the plain loss, every time")


if __name__ == "__main__":
    main()

"""
Ratio clipping, on REINFORCE with the baseline (post/reinforce.py). Sampling is most of
each REINFORCE step, and its answers get one gradient step. Here each step's answers get UPDATES
gradient steps. After the first, they were sampled by an older model than the one training on
them, so each answer token is weighted by its ratio, how much more likely it is now than when
sampled, and the ratio is clipped to 1 ± EPSILON in the direction its advantage pushes:

    loss = -1/N sum_i sum_t min(r_i,t A_i, clip(r_i,t, 1 - eps, 1 + eps) A_i)
    r_i,t = p(y_i,t | x_i, y_i,<t) / p_old(y_i,t | x_i, y_i,<t)

On the first update every ratio is 1 and the gradient is REINFORCE's with the baseline. A token
pushed past its bound gets no more gradient from this batch (learn/2_rl/005_ratio_clipping.py). The
old log probabilities are the trainer's own, from the first update's forward pass, where the
weights are those that sampled: frontier trainers compute them again too, rather than trust the
inference engine's. post/grpo.py adds the rest of GRPO as DeepSeekMath wrote it: the division by
the group's standard deviation, each answer's tokens averaged where this sums them, and a KL
penalty.
Run with ``uv run modal run gpu.py --modules post.ratio_clipping``, after post.sft has saved its
weights; it saves its own as "ratio_clipping". post/eval/ratio_clipping.py measures what it
changes.

Shape letters used in the comments:
    T   tokens in a packed row
    A   of them, the answer tokens, the targets the loss counts
"""

import torch
from tokenizers import Tokenizer
from torch import Tensor

from post.chat import token_log_probs
from post.qwen3 import (
    DEVICE,
    REPO_ID,
    Qwen3,
    load_qwen3,
    load_tokenizer,
    load_weights,
    save_weights,
)
from post.reinforce import LEARNING_RATE, LOG_EVERY, QUESTIONS, advantages, rollouts
from post.sft import MAX_GRAD_NORM, Chat, join, pack, unread_gsm8k

STEPS = 20
UPDATES = 4  # gradient steps per step's answers
EPSILON = 0.2


def clipped_loss(
    new: Tensor, old: Tensor, weights: Tensor
) -> tuple[Tensor, Tensor]:  # [A], [A], [A]
    """The loss over one row's answer tokens, and which tokens the clip stopped. weights is each
    token's answer's advantage over N here, over the batch's answer tokens in post/rlvr.py."""
    ratios = (new - old).exp()  # [A]
    clipped = ratios.clamp(1 - EPSILON, 1 + EPSILON)  # [A]
    objective = torch.minimum(ratios * weights, clipped * weights)  # [A]
    stopped = (ratios * weights) > (
        clipped * weights
    )  # [A], past the bound the advantage pushes to
    return -objective.sum(), stopped


def token_weights(rows: list[list[Chat]], weights: list[float], count: int) -> list[Tensor]:
    """For each row, each answer token's answer's weight over count; pack keeps the order."""
    per_row = []
    start = 0
    for row in rows:
        row_weights = weights[start : start + len(row)]
        start += len(row)
        tokens = [
            w / count for (_, answer), w in zip(row, row_weights) for token in answer if token
        ]
        per_row.append(torch.tensor(tokens, device=DEVICE))  # [A]
    return per_row


def ratio_clipping(model: Qwen3, tokenizer: Tokenizer, problems: list[dict[str, str]]) -> None:
    """Train the model on STEPS steps of QUESTIONS problems each, UPDATES updates per step."""
    sampler = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)  # the copy that samples
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0, fused=True
    )
    logged = []  # the rewards since the last print
    stopped = 0  # answer tokens the clip stopped in the last update of each step since then
    tokens = 0  # answer tokens in those updates
    for step in range(STEPS):
        batch = problems[step * QUESTIONS : (step + 1) * QUESTIONS]
        sampler.load_state_dict(model.state_dict())  # the trained weights, copied into bf16
        chats, rewards = rollouts(sampler, tokenizer, batch)
        rows = pack(chats)
        weights = token_weights(rows, advantages(rewards), len(chats))
        old = []  # each row's log probabilities from the weights that sampled
        for update in range(UPDATES):
            for k, row in enumerate(rows):
                ids, mask, packing = join(row)  # [1, T] each
                with torch.autocast(DEVICE, dtype=torch.bfloat16):  # matmuls in bf16
                    new = token_log_probs(model, ids, mask, packing)  # [A]
                if update == 0:
                    old.append(new.detach())
                loss, row_stopped = clipped_loss(new, old[k], weights[k])
                loss.backward()  # adds to the gradients of the rows before
                if update == UPDATES - 1:
                    stopped += int(row_stopped.sum())
                    tokens += row_stopped.numel()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        logged += rewards
        if (step + 1) % LOG_EVERY == 0:
            print(
                f"step {step + 1}: reward {sum(logged) / len(logged):.3f}, "
                f"clipped in the last update {stopped / tokens:.1%} of answer tokens"
            )
            logged = []
            stopped = 0
            tokens = 0


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32: the master copy of the weights
    load_weights(model, "sft")
    ratio_clipping(model, tokenizer, unread_gsm8k(STEPS * QUESTIONS))
    save_weights(model, "ratio_clipping")


if __name__ == "__main__":
    main()

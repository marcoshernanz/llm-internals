"""
GRPO as DeepSeekMath wrote it: ratio clipping (post/ratio_clipping.py) with three things added.

    loss = -1/N sum_i 1/|y_i| sum_t [min(r_i,t A_i, clip(r_i,t, 1 - eps, 1 + eps) A_i) - beta KL_i,t]
    A_i = (R_i - mean(R)) / std(R),  over the S answers to answer i's question
    KL_i,t = p_ref / p - log(p_ref / p) - 1,  both probabilities of y_i,t given x_i, y_i,<t

- The advantage is divided by the standard deviation of its question's rewards, so every
  question's advantages have the same spread, however many of its answers are right.
- Each answer's tokens are averaged, not summed, so every answer counts the same however long.
- A KL penalty pulls the model toward a reference, the model as training starts: here the SFT
  model. KL_i,t is computed from the sampled token alone. Its average over the tokens the model
  samples is the KL divergence, it is never negative, and it is 0 where the two models agree.

DeepSeekMath trained with beta 0.04, a learning rate of 1e-6, 64 answers per question and one
update per batch, so its clip never acted. This keeps post/ratio_clipping.py's 8 answers, 4
updates and learning rate, so the three additions are all that differs. Later recipes undo each of
them, each for a problem it causes (post/ROADMAP.md, stage 4).
Run with ``uv run modal run gpu.py --modules post.grpo``, after post.sft has saved its weights; it
saves its own as "grpo". post/eval/grpo.py measures what it changes.

Shape letters used in the comments:
    Q   questions in a step
    S   answers per question
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
from post.ratio_clipping import EPSILON, STEPS, UPDATES, token_weights
from post.reinforce import LEARNING_RATE, LOG_EVERY, QUESTIONS, SAMPLES, rollouts
from post.sft import MAX_GRAD_NORM, join, pack, unread_gsm8k

BETA = 0.04  # the KL penalty's weight


def advantages(rewards: list[float]) -> list[float]:
    """Each answer's reward minus its question's mean, over its question's standard deviation.
    1e-4 keeps a question whose answers all score the same at 0, not 0 / 0."""
    groups = torch.tensor(rewards).reshape(-1, SAMPLES)  # [Q, S]
    centered = groups - groups.mean(dim=-1, keepdim=True)  # [Q, S]
    scaled = centered / (groups.std(dim=-1, keepdim=True) + 1e-4)  # [Q, S]
    return scaled.flatten().tolist()


def token_kl(new: Tensor, reference: Tensor) -> Tensor:  # [A], [A]
    """Each answer token's estimate of the KL divergence from the model to the reference."""
    log_ratios = reference - new  # [A]
    return log_ratios.exp() - log_ratios - 1  # [A]


def grpo_loss(
    new: Tensor, old: Tensor, kl: Tensor, advantages: Tensor, scales: Tensor
) -> Tensor:  # [A] each
    """The loss over one row's answer tokens. scales is 1 / (N |y_i|) on answer i's tokens."""
    ratios = (new - old).exp()  # [A]
    clipped = ratios.clamp(1 - EPSILON, 1 + EPSILON)  # [A]
    objective = torch.minimum(ratios * advantages, clipped * advantages) - BETA * kl  # [A]
    return -(scales * objective).sum()


def grpo(model: Qwen3, tokenizer: Tokenizer, problems: list[dict[str, str]]) -> None:
    """Train the model on STEPS steps of QUESTIONS problems each, UPDATES updates per step."""
    sampler = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)  # the copy that samples
    reference = load_qwen3(REPO_ID, torch.float32, DEVICE)  # the KL penalty's anchor
    reference.load_state_dict(model.state_dict())  # the model as training starts, never trained
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0, fused=True
    )
    logged = []  # the rewards since the last print
    lengths = []  # the answers' lengths since then, in tokens
    kl_total = 0.0  # the KL estimates on every answer token of every update since then
    tokens = 0  # those answer tokens
    for step in range(STEPS):
        batch = problems[step * QUESTIONS : (step + 1) * QUESTIONS]
        sampler.load_state_dict(model.state_dict())  # the trained weights, copied into bf16
        chats, rewards, _ = rollouts(sampler, tokenizer, batch)
        rows = pack(chats)
        answer_lengths = [sum(answer) for _, answer in chats]
        token_advantages = token_weights(rows, advantages(rewards), 1)
        scales = token_weights(rows, [1 / length for length in answer_lengths], len(chats))
        with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            references = [token_log_probs(reference, *join(row)) for row in rows]  # [A] each
        old = []  # each row's log probabilities from the weights that sampled
        for update in range(UPDATES):
            for k, row in enumerate(rows):
                ids, mask, packing = join(row)  # [1, T] each
                with torch.autocast(DEVICE, dtype=torch.bfloat16):  # matmuls in bf16
                    new = token_log_probs(model, ids, mask, packing)  # [A]
                if update == 0:
                    old.append(new.detach())
                kl = token_kl(new, references[k])  # [A]
                loss = grpo_loss(new, old[k], kl, token_advantages[k], scales[k])
                loss.backward()  # adds to the gradients of the rows before
                kl_total += float(kl.detach().sum())
                tokens += kl.numel()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        logged += rewards
        lengths += answer_lengths
        if (step + 1) % LOG_EVERY == 0:
            right = [n for n, r in zip(lengths, logged) if r]
            wrong = [n for n, r in zip(lengths, logged) if not r]
            print(
                f"step {step + 1}: reward {sum(logged) / len(logged):.3f}, "
                f"right answers {sum(right) / len(right):.0f} tokens, "
                f"wrong {sum(wrong) / len(wrong):.0f}, "
                f"KL {kl_total / tokens:.5f} per answer token"
            )
            logged = []
            lengths = []
            kl_total = 0.0
            tokens = 0


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32: the master copy of the weights
    load_weights(model, "sft")
    grpo(model, tokenizer, unread_gsm8k(STEPS * QUESTIONS))
    save_weights(model, "grpo")


if __name__ == "__main__":
    main()

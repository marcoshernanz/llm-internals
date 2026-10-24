"""
GRPO with the fixes later recipes made, added one at a time, each for a problem it causes
(post/ROADMAP.md, stage 4). It starts from GRPO as DeepSeekMath wrote it (post/grpo.py). The fixes
so far:

1. The token-level loss. GRPO averages each answer's tokens, so each token's push is its answer's
   advantage over N and the answer's length: in a long answer the push is spread thin. Wrong
   answers are pushed down less the longer they are, and right ones up more the shorter they are.
   Here every token is divided by the batch's answer tokens, so its push depends on its answer's
   advantage only (learn/2_rl/007_token_mean.py):

       loss = -1/(sum_j |y_j|) sum_i sum_t [min(r_i,t A_i, clip(r_i,t, 1 - eps, 1 + eps) A_i)
                                            - beta KL_i,t]

   This is DAPO's; Dr. GRPO divides by a constant instead.

2. No division by the standard deviation. GRPO divides each advantage by the standard deviation of
   its question's rewards, which for rewards of 1 and 0 is the square root of p (1 - p), p the
   share of its answers that are right: it scales up the questions whose answers are most
   one-sided, nearly all wrong or nearly all right, and the training no longer follows the
   gradient of the expected reward. The advantage here is the reward minus its question's mean,
   the baseline's (post/reinforce.py, learn/2_rl/008_no_std.py), as in Dr. GRPO and OLMo 3:

       A_i = R_i - mean(R)

Run with ``uv run modal run gpu.py --modules post.rlvr``, after post.sft has saved its weights; it
saves its own as "rlvr". post/eval/rlvr.py measures what it changes.

Shape letters used in the comments:
    T   tokens in a packed row
    A   of them, the answer tokens, the targets the loss counts
"""

import torch
from tokenizers import Tokenizer

from post.chat import token_log_probs
from post.grpo import grpo_loss, token_kl
from post.qwen3 import (
    DEVICE,
    REPO_ID,
    Qwen3,
    load_qwen3,
    load_tokenizer,
    load_weights,
    save_weights,
)
from post.ratio_clipping import STEPS, UPDATES, token_weights
from post.reinforce import LEARNING_RATE, LOG_EVERY, QUESTIONS, advantages, rollouts
from post.sft import MAX_GRAD_NORM, join, pack, unread_gsm8k


def rlvr(model: Qwen3, tokenizer: Tokenizer, problems: list[dict[str, str]]) -> None:
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
        chats, rewards = rollouts(sampler, tokenizer, batch)
        rows = pack(chats)
        answer_lengths = [sum(answer) for _, answer in chats]
        token_advantages = token_weights(rows, advantages(rewards), 1)
        scales = token_weights(rows, [1.0] * len(chats), sum(answer_lengths))  # 1 / all of them
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
    rlvr(model, tokenizer, unread_gsm8k(STEPS * QUESTIONS))
    save_weights(model, "rlvr")


if __name__ == "__main__":
    main()

"""
GRPO with the fixes later recipes made, added one at a time, each for a problem it causes
(post/ROADMAP.md, stage 4). It starts from GRPO as DeepSeekMath wrote it (post/grpo.py). The fixes
so far:

1. The token-level loss. GRPO averages each answer's tokens, so each token's push is its answer's
   advantage over N and the answer's length: in a long answer the push is spread thin. Wrong
   answers are pushed down less the longer they are, and right ones up more the shorter they are.
   Here every token is divided by the batch's answer tokens, so its push depends on its answer's
   advantage only (learn/2_rl/007_token_mean.py):

       loss = -1/(sum_j |y_j|) sum_i sum_t min(r_i,t A_i, clip(r_i,t, 1 - eps, 1 + eps) A_i)

   This is DAPO's; Dr. GRPO divides by a constant instead.

2. No division by the standard deviation. GRPO divides each advantage by the standard deviation of
   its question's rewards, which for rewards of 1 and 0 is the square root of p (1 - p), p the
   share of its answers that are right: it scales up the questions whose answers are most
   one-sided, nearly all wrong or nearly all right, and the training no longer follows the
   gradient of the expected reward. The advantage here is the reward minus its question's mean,
   the baseline's (post/reinforce.py, learn/2_rl/008_no_std.py), as in Dr. GRPO and OLMo 3:

       A_i = R_i - mean(R)

3. No KL penalty. GRPO pulls the model toward a frozen copy of the model it started from. That
   guards training on a learned reward model, which the model can learn to fool with answers it
   scores high but that are wrong. A checked reward reads the number the question asks for, so
   there's nothing to fool: the pull only holds the model back, and the copy costs its memory and
   a forward pass over every row. Without it, this is ratio clipping (post/ratio_clipping.py) with
   the token-level loss (learn/2_rl/009_no_kl.py), as in DAPO, Dr. GRPO and OLMo 3. DeepSeek-R1
   kept a small one, of weight 0.001, and replaced its copy with the current model every 400 steps.

4. No questions that push nothing. A question whose answers all score the same, all right or all
   wrong, has every advantage 0. Its answers push nothing, but they cost their sampling and their
   passes, and their tokens still count in the total every token's push is divided by, so they
   shrink the rest's. Here a step drops those questions and samples QUESTIONS more, for at most
   ROUNDS rounds, until it has QUESTIONS questions whose answers' rewards differ; what's over is
   dropped too (learn/2_rl/010_zero_gradient_filtering.py). DAPO and OLMo 3 do this. The
   alternative drops them and trains on a smaller batch.

Run with ``uv run modal run gpu.py --modules post.rlvr``, after post.sft has saved its weights; it
saves its own as "rlvr". post/eval/rlvr.py measures what it changes.

Shape letters used in the comments:
    T   tokens in a packed row
    A   of them, the answer tokens, the targets the loss counts
"""

from collections.abc import Iterator

import torch
from tokenizers import Tokenizer

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
from post.ratio_clipping import STEPS, UPDATES, clipped_loss, token_weights
from post.reinforce import (
    LEARNING_RATE,
    LOG_EVERY,
    QUESTIONS,
    SAMPLES,
    advantages,
    rollouts,
)
from post.sft import MAX_GRAD_NORM, Chat, join, pack, unread_gsm8k

ROUNDS = 3  # at most, of QUESTIONS questions sampled per step


def mixed_rollouts(
    sampler: Qwen3, tokenizer: Tokenizer, problems: Iterator[dict[str, str]]
) -> tuple[list[Chat], list[float], list[Chat], list[float]]:
    """The answers and rewards of QUESTIONS questions whose answers' rewards differ, then every
    answer sampled to find them, with its reward, for the logs."""
    kept_chats: list[Chat] = []
    kept_rewards: list[float] = []
    all_chats: list[Chat] = []
    all_rewards: list[float] = []
    for _ in range(ROUNDS):
        chats, rewards = rollouts(sampler, tokenizer, [next(problems) for _ in range(QUESTIONS)])
        all_chats += chats
        all_rewards += rewards
        for start in range(0, len(chats), SAMPLES):  # a question's answers come together
            group = rewards[start : start + SAMPLES]
            if min(group) < max(group):  # some right, some wrong: not every advantage is 0
                kept_chats += chats[start : start + SAMPLES]
                kept_rewards += group
        if len(kept_chats) >= QUESTIONS * SAMPLES:
            break
    count = QUESTIONS * SAMPLES
    return kept_chats[:count], kept_rewards[:count], all_chats, all_rewards


def rlvr(model: Qwen3, tokenizer: Tokenizer, problems: list[dict[str, str]]) -> None:
    """Train the model on STEPS steps of QUESTIONS problems each, UPDATES updates per step."""
    sampler = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)  # the copy that samples
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0, fused=True
    )
    stream = iter(problems)
    logged = []  # every sampled answer's reward since the last print
    lengths = []  # their lengths, in tokens
    for step in range(STEPS):
        sampler.load_state_dict(model.state_dict())  # the trained weights, copied into bf16
        chats, rewards, sampled, sampled_rewards = mixed_rollouts(sampler, tokenizer, stream)
        rows = pack(chats)
        answer_lengths = [sum(answer) for _, answer in chats]
        weights = token_weights(rows, advantages(rewards), sum(answer_lengths))  # over all of them
        old = []  # each row's log probabilities from the weights that sampled
        for update in range(UPDATES):
            for k, row in enumerate(rows):
                ids, mask, packing = join(row)  # [1, T] each
                with torch.autocast(DEVICE, dtype=torch.bfloat16):  # matmuls in bf16
                    new = token_log_probs(model, ids, mask, packing)  # [A]
                if update == 0:
                    old.append(new.detach())
                loss, _ = clipped_loss(new, old[k], weights[k])
                loss.backward()  # adds to the gradients of the rows before
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        logged += sampled_rewards
        lengths += [sum(answer) for _, answer in sampled]
        if (step + 1) % LOG_EVERY == 0:
            right = [n for n, r in zip(lengths, logged) if r]
            wrong = [n for n, r in zip(lengths, logged) if not r]
            groups = [logged[k : k + SAMPLES] for k in range(0, len(logged), SAMPLES)]
            one_sided = sum(min(group) == max(group) for group in groups)
            print(
                f"step {step + 1}: reward {sum(logged) / len(logged):.3f}, "
                f"{len(groups) / LOG_EVERY:.1f} questions sampled per step, "
                f"{one_sided / len(groups):.0%} all right or all wrong, "
                f"right answers {sum(right) / len(right):.0f} tokens, "
                f"wrong {sum(wrong) / len(wrong):.0f}"
            )
            logged = []
            lengths = []


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32: the master copy of the weights
    load_weights(model, "sft")
    rlvr(model, tokenizer, unread_gsm8k(ROUNDS * STEPS * QUESTIONS))
    save_weights(model, "rlvr")


if __name__ == "__main__":
    main()

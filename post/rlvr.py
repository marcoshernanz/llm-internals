"""
GRPO with the fixes later recipes made, added one at a time, each for a problem it causes
(post/ROADMAP.md, stage 4). It starts from GRPO as DeepSeekMath wrote it (post/grpo.py). The fixes
so far:

1. The token-level loss. GRPO averages each answer's tokens, so each token's push is its answer's
   advantage over N and the answer's length: in a long answer the push is spread thin. Wrong
   answers are pushed down less the longer they are, and right ones up more the shorter they are.
   Here every token is divided by the batch's answer tokens, so its push depends on its answer's
   advantage only (learn/2_rl/007_token_mean.py):

       loss = -1/(sum_j |y_j|) sum_i sum_t w_i,t min(r_i,t A_i,
                                                     clip(r_i,t, 1 - eps, 1 + eps_high) A_i)

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

5. Clip-higher. The clip lets a token pushed up grow to at most 1 + EPSILON times its probability
   when sampled: room in proportion to how likely it already was. An unlikely token in a right
   answer, a step the model rarely takes, can barely grow while the likely ones grow freely, so
   the answers narrow onto what the model already prefers. Here the upper bound is
   1 + EPSILON_HIGH and the lower still 1 - EPSILON (learn/2_rl/011_clip_higher.py), as in DAPO,
   whose 0.28 this takes; OLMo 3 uses 0.272. Loosening the lower bound too would let the tokens of
   wrong answers be pushed toward 0, which narrows the answers as well.

6. Truncated importance sampling, with an fp32 output layer. The sampler, a bf16 copy, and the
   trainer, fp32 weights with bf16 matmuls, give the same tokens slightly different
   probabilities, so the answers come from a model a little different from the one the loss
   assumes. Each token is weighted by how much likelier the trainer finds it than the sampler
   did, capped at CAP so that no token dominates, as in OLMo 3, after the paper that proposed it:

       w_i,t = min(p_old(y_i,t | x_i, y_i,<t) / p_sampler(y_i,t | x_i, y_i,<t), CAP)

   The weight is a number, with no gradient through it. At temperature 1 the sampler's
   distribution is the model's. Both output layers run in fp32, the sampler's with the trainer's
   fp32 embedding table, which brings the two about a third closer (post/eval/sampler_gap.py).
   MiniMax found the output layer to be where their gap came from.

The model may carry LoRAs (post/lora.py): the sampler then loads the weights with each LoRA merged
in, and learning_rate is LoRA's. post/eval/rlvr_lora.py runs it with LoRAs of rank 1.

Run with ``uv run modal run gpu.py --modules post.rlvr``, after post.sft has saved its weights; it
saves its own as "rlvr". post/eval/rlvr.py measures what it changes.

Shape letters used in the comments:
    T   tokens in a packed row
    A   of them, the answer tokens, the targets the loss counts
"""

from collections.abc import Iterator

import torch
from tokenizers import Tokenizer
from torch import Tensor

from post.chat import token_log_probs
from post.lora import merged_state_dict
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
EPSILON_HIGH = 0.28  # the clip's upper bound is 1 + EPSILON_HIGH, the lower 1 - EPSILON
CAP = 2.0  # the largest importance weight


def mixed_rollouts(
    sampler: Qwen3, tokenizer: Tokenizer, problems: Iterator[dict[str, str]], head: Tensor
) -> tuple[list[Chat], list[float], list[list[float]], list[Chat], list[float]]:
    """The answers, rewards and sampler's log probabilities of QUESTIONS questions whose answers'
    rewards differ, then every answer sampled to find them, with its reward, for the logs. head is
    the sampler's (post/qwen3.py)."""
    kept_chats: list[Chat] = []
    kept_rewards: list[float] = []
    kept_log_probs: list[list[float]] = []
    all_chats: list[Chat] = []
    all_rewards: list[float] = []
    for _ in range(ROUNDS):
        batch = [next(problems) for _ in range(QUESTIONS)]
        chats, rewards, log_probs = rollouts(sampler, tokenizer, batch, head)
        all_chats += chats
        all_rewards += rewards
        for start in range(0, len(chats), SAMPLES):  # a question's answers come together
            group = rewards[start : start + SAMPLES]
            if min(group) < max(group):  # some right, some wrong: not every advantage is 0
                kept_chats += chats[start : start + SAMPLES]
                kept_rewards += group
                kept_log_probs += log_probs[start : start + SAMPLES]
        if len(kept_chats) >= QUESTIONS * SAMPLES:
            break
    count = QUESTIONS * SAMPLES
    return (
        kept_chats[:count],
        kept_rewards[:count],
        kept_log_probs[:count],
        all_chats,
        all_rewards,
    )


def row_log_probs(rows: list[list[Chat]], log_probs: list[list[float]]) -> list[Tensor]:
    """For each row, its answers' tokens' log probabilities, in order; pack keeps the order."""
    per_row = []
    start = 0
    for row in rows:
        answers = log_probs[start : start + len(row)]
        start += len(row)
        per_row.append(
            torch.tensor([p for answer in answers for p in answer], device=DEVICE)
        )  # [A]
    return per_row


def rlvr(
    model: Qwen3,
    tokenizer: Tokenizer,
    problems: list[dict[str, str]],
    learning_rate: float = LEARNING_RATE,
) -> None:
    """Train the model on STEPS steps of QUESTIONS problems each, UPDATES updates per step."""
    sampler = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)  # the copy that samples
    head = model.embed_tokens.weight.detach()  # [V, D], the sampler's output layer: fp32, live
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0, fused=True
    )
    stream = iter(problems)
    logged = []  # every sampled answer's reward since the last print
    lengths = []  # their lengths, in tokens
    stopped_up = 0  # answer tokens the clip stopped at the upper bound, in each step's last update
    stopped_down = 0  # and at the lower
    up_probability = 0.0  # the sum of those at the upper's probabilities when sampled
    probability = 0.0  # the same sum over every answer token in those updates
    tokens = 0  # those answer tokens
    gap = 0.0  # the sum of |log p_old - log p_sampler| over every answer token since the last print
    capped = 0  # answer tokens whose importance weight was capped since then
    largest = 0.0  # the largest importance weight before the cap since then
    drawn_tokens = 0  # answer tokens since then
    for step in range(STEPS):
        sampler.load_state_dict(merged_state_dict(model))  # the trained weights, copied into bf16
        chats, rewards, log_probs, sampled, sampled_rewards = mixed_rollouts(
            sampler, tokenizer, stream, head
        )
        rows = pack(chats)
        drawn = row_log_probs(rows, log_probs)  # [A] each, the sampler's
        answer_lengths = [sum(answer) for _, answer in chats]
        weights = token_weights(rows, advantages(rewards), sum(answer_lengths))  # over all of them
        old = []  # each row's log probabilities from the weights that sampled
        importance = []  # each row's tokens' weights, p_old over p_sampler, capped
        for update in range(UPDATES):
            for k, row in enumerate(rows):
                ids, mask, packing = join(row)  # [1, T] each
                with torch.autocast(DEVICE, dtype=torch.bfloat16):  # matmuls in bf16
                    new = token_log_probs(model, ids, mask, packing, fp32_head=True)  # [A]
                if update == 0:
                    old.append(new.detach())
                    log_ratios = old[k] - drawn[k]  # [A]
                    importance.append(log_ratios.exp().clamp(max=CAP))  # [A]
                    gap += float(log_ratios.abs().sum())
                    capped += int((log_ratios.exp() > CAP).sum())
                    largest = max(largest, float(log_ratios.exp().max()))
                    drawn_tokens += log_ratios.numel()
                loss, stopped = clipped_loss(new, old[k], weights[k] * importance[k], EPSILON_HIGH)
                loss.backward()  # adds to the gradients of the rows before
                if update == UPDATES - 1:
                    up = stopped & (weights[k] > 0)  # [A], stopped at the upper bound
                    stopped_up += int(up.sum())
                    stopped_down += int((stopped & (weights[k] < 0)).sum())
                    up_probability += float(old[k][up].exp().sum())
                    probability += float(old[k].exp().sum())
                    tokens += up.numel()
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
                f"{one_sided / len(groups):.0%} all right or all wrong; clip stopped "
                f"{stopped_up / tokens:.1%} of answer tokens at the upper bound and "
                f"{stopped_down / tokens:.1%} at the lower in the last update, those at the upper "
                f"{up_probability / max(stopped_up, 1):.2f} likely when sampled on average, "
                f"every answer token {probability / tokens:.2f}; trainer and sampler "
                f"{gap / drawn_tokens:.4f} apart in log probability on average, "
                f"{capped / drawn_tokens:.2%} of weights capped, the largest {largest:.2f}; "
                f"right answers {sum(right) / len(right):.0f} tokens, "
                f"wrong {sum(wrong) / len(wrong):.0f}"
            )
            logged = []
            lengths = []
            stopped_up = 0
            stopped_down = 0
            up_probability = 0.0
            probability = 0.0
            tokens = 0
            gap = 0.0
            capped = 0
            largest = 0.0
            drawn_tokens = 0


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32: the master copy of the weights
    load_weights(model, "sft")
    rlvr(model, tokenizer, unread_gsm8k(ROUNDS * STEPS * QUESTIONS))
    save_weights(model, "rlvr")


if __name__ == "__main__":
    main()

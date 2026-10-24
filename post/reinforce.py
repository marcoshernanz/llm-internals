"""
REINFORCE: reinforcement learning from a reward a grader computes, in its plainest form. Each step,
the model answers GSM8K train questions SFT never read, 8 samples each, and each answer gets a
reward: 1 if it closed its turn and its number after "The answer is" is the question's answer, 0
if not. The loss is minus each answer's log probability times its reward, averaged over the step's
N answers:

    loss = -1/N sum_i r_i log p(y_i | x_i),  with  log p(y_i | x_i) = sum_t log p(y_i,t | x_i, y_i,<t)

On average over the answers the model samples, its gradient is minus the gradient of the expected
reward: each step makes right answers more likely. With rewards of 1 and 0, wrong answers get no
weight: it's rejection sampling (post/rejection_sampling.py) on every step, always on the current
model's own answers.

With baseline, each answer's reward is replaced by its advantage: the reward minus the mean reward
of the 8 answers to the same question (learn/2_rl/004_baseline.py). The baseline changes nothing on
average, since the gradient of the probabilities' sum, 1, is 0. But a right answer among wrong ones
is pushed up, the wrong ones are pushed down, and a question whose answers all score the same
pushes nothing. GRPO's advantage also divides by the group's standard deviation; Dr. GRPO drops
that, and so does this, for now.

The answers are sampled by a bf16 copy of the model, refreshed from the trained weights before each
step, as frontier RL samples with an inference engine apart from the trainer. Their token ids go to
the loss as sampled: decoding them and encoding the text again can give other tokens, whose log
probabilities the sampler never drew. The training pass is SFT's: fp32 weights, bf16 matmuls, the
step's answers packed into rows of up to 2,048 tokens, whose gradients add up before one optimizer
step. The learning rate is constant, as in most RL recipes. 2e-6 is a fifth of SFT's peak: RL's is
often a tenth, 1e-6 in DAPO and as TRL's GRPO default, and this run is short.
Run with ``uv run modal run gpu.py --modules post.reinforce``, after post.sft has saved its
weights; it saves its own as "reinforce". post/eval/reinforce.py and post/eval/baseline.py measure
what it changes, without and with the baseline.

Shape letters used in the comments:
    Q   questions in a step
    S   answers per question
    T   tokens in a packed row
    A   of them, the answer tokens, the targets the loss counts
"""

import torch
from tokenizers import Tokenizer
from torch import Tensor

from post.chat import answer_ids, render_prompt, scaled_loss
from post.gsm8k import reference_answer, strict_answer
from post.qwen3 import (
    DEVICE,
    REPO_ID,
    Qwen3,
    load_qwen3,
    load_tokenizer,
    load_weights,
    save_weights,
)
from post.sft import MAX_GRAD_NORM, Chat, join, pack, unread_gsm8k

QUESTIONS = 8  # per step
SAMPLES = 8  # answers per question: N = 64 per step
STEPS = 40
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 256  # per answer
LEARNING_RATE = 2e-6
LOG_EVERY = 5  # steps


def reward(text: str, reference: float) -> float:
    """1 for an answer that closed its turn and is right by strict grading, 0 for any other."""
    return float(text.endswith("<|im_end|>") and strict_answer(text) == reference)


def rollouts(
    sampler: Qwen3,
    tokenizer: Tokenizer,
    problems: list[dict[str, str]],
    head: Tensor | None = None,
) -> tuple[list[Chat], list[float], list[list[float]]]:
    """Each problem's question with each of its sampled answers, as token ids with the answer's
    marked, each answer's reward, and the sampler's log probability of each answer token. head is
    the sampler's (post/qwen3.py)."""
    questions = [problem["question"] for problem in problems]
    sampled, sampled_log_probs = answer_ids(
        sampler, tokenizer, questions, SAMPLES, TEMPERATURE, MAX_NEW_TOKENS, head
    )
    chats = []
    rewards = []
    log_probs = []
    for problem, question, group, group_log_probs in zip(
        problems, questions, sampled, sampled_log_probs
    ):
        prompt = render_prompt(tokenizer, [{"role": "user", "content": question}])
        reference = reference_answer(problem["answer"])
        for ids, answer_log_probs in zip(group, group_log_probs):
            chats.append((prompt + ids, [False] * len(prompt) + [True] * len(ids)))
            text = tokenizer.decode(ids, skip_special_tokens=False)
            rewards.append(reward(text, reference))
            log_probs.append(answer_log_probs)
    return chats, rewards, log_probs


def advantages(rewards: list[float]) -> list[float]:
    """Each answer's reward minus the mean reward of its question's answers, which come together."""
    groups = torch.tensor(rewards).reshape(-1, SAMPLES)  # [Q, S]
    centered = groups - groups.mean(dim=-1, keepdim=True)  # [Q, S]
    return centered.flatten().tolist()


def reinforce_loss(model: Qwen3, chats: list[Chat], weights: list[float], count: int) -> Tensor:
    """These answers' share of the loss: each answer token's minus log probability times the
    answer's weight, over count, the step's N answers."""
    ids, mask, packing = join(chats)  # [1, T] each
    scales = [w / count for (_, answer), w in zip(chats, weights) for token in answer if token]
    scales_t = torch.tensor(scales, device=DEVICE)  # [A], each answer's weight on its tokens
    return scaled_loss(model, ids, mask, scales_t, packing)


def reinforce(
    model: Qwen3, tokenizer: Tokenizer, problems: list[dict[str, str]], baseline: bool = False
) -> None:
    """Train the model on STEPS steps of QUESTIONS problems each, weighting each answer by its
    reward, or with baseline by its advantage."""
    sampler = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)  # the copy that samples
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0, fused=True
    )
    logged = []  # the rewards since the last print
    lengths = []  # the answers' lengths since then, in tokens
    for step in range(STEPS):
        batch = problems[step * QUESTIONS : (step + 1) * QUESTIONS]
        sampler.load_state_dict(model.state_dict())  # the trained weights, copied into bf16
        chats, rewards, _ = rollouts(sampler, tokenizer, batch)
        weights = advantages(rewards) if baseline else rewards
        start = 0  # pack keeps the order: each row's weights come next
        for row in pack(chats):
            row_weights = weights[start : start + len(row)]
            start += len(row)
            with torch.autocast(DEVICE, dtype=torch.bfloat16):  # matmuls in bf16
                loss = reinforce_loss(model, row, row_weights, len(chats))
            loss.backward()  # adds to the gradients of the rows before
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        logged += rewards
        lengths += [sum(answer) for _, answer in chats]
        if (step + 1) % LOG_EVERY == 0:
            print(
                f"step {step + 1}: reward {sum(logged) / len(logged):.3f}, "
                f"answers {sum(lengths) / len(lengths):.0f} tokens"
            )
            logged = []
            lengths = []


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32: the master copy of the weights
    load_weights(model, "sft")
    reinforce(model, tokenizer, unread_gsm8k(STEPS * QUESTIONS))
    save_weights(model, "reinforce")


if __name__ == "__main__":
    main()

"""
Run GRPO with the fixes so far (post/rlvr.py) from the SFT model and measure it as post/eval/grpo.py
measures GRPO, from the same questions in the same order: the reward, how many questions a step
sampled and how many were all right or all wrong, which answer tokens the clip stopped at each
bound and how likely they were when sampled, how far apart the trainer's and the sampler's log
probabilities were and how many importance weights were capped, and the lengths of right and wrong
answers every 5 steps, the time and peak memory, then pass@k on the same test questions.
Run with ``uv run modal run gpu.py --modules post.eval.rlvr``, after post.sft has saved its
weights; it saves the trained ones as "rlvr".
"""

import time

import torch

from post.eval.grpo import RATIO_CLIPPING
from post.eval.pass_at_k import TEST_QUESTIONS, measure
from post.eval.reinforce import SFT
from post.gsm8k import load_gsm8k
from post.qwen3 import (
    DEVICE,
    DTYPE,
    REPO_ID,
    load_qwen3,
    load_tokenizer,
    load_weights,
    save_weights,
)
from post.ratio_clipping import EPSILON, STEPS, UPDATES
from post.reinforce import LEARNING_RATE, QUESTIONS, SAMPLES, TEMPERATURE
from post.rlvr import CAP, EPSILON_HIGH, ROUNDS, rlvr
from post.sft import unread_gsm8k

# post/eval/grpo.py's run: the same steps and updates, each answer's tokens averaged
GRPO = (
    "GRPO: greedy 57%; pass@1 38.9%, pass@2 52.7%, pass@4 66.0%, pass@8 78.2%, pass@16 89.0%; "
    "22 with none right, 7 with all"
)
# this file's run with the first fix only
TOKEN_LEVEL = (
    "with the token-level loss: greedy 52%; pass@1 38.7%, pass@2 53.7%, pass@4 67.6%, "
    "pass@8 79.9%, pass@16 90.0%; 20 with none right, 3 with all"
)
# this file's run with the first two fixes
NO_STD = (
    "and no standard deviation: greedy 51%; pass@1 34.2%, pass@2 49.2%, pass@4 63.5%, "
    "pass@8 75.6%, pass@16 85.5%; 29 with none right, 1 with all"
)
# this file's run with the first three fixes
NO_KL = (
    "and no KL penalty: greedy 52%; pass@1 40.5%, pass@2 54.9%, pass@4 67.7%, "
    "pass@8 78.3%, pass@16 86.5%; 27 with none right, 10 with all"
)
# this file's run with the first four fixes
FILTERING = (
    "and zero-gradient filtering: greedy 53%; pass@1 41.6%, pass@2 56.1%, pass@4 68.4%, "
    "pass@8 78.5%, pass@16 86.5%; 27 with none right, 9 with all"
)
# this file's run with the first five fixes
CLIP_HIGHER = (
    "and clip-higher: greedy 52%; pass@1 37.8%, pass@2 51.7%, pass@4 64.8%, "
    "pass@8 76.4%, pass@16 85.5%; 29 with none right, 10 with all"
)


def main() -> None:
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "RL samples in bf16: run it on the L4"
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32: the master copy of the weights
    load_weights(model, "sft")
    problems = unread_gsm8k(ROUNDS * STEPS * QUESTIONS)
    print(
        f"GRPO with the token-level loss, no division by the standard deviation, no KL penalty, "
        f"no questions whose answers all score the same, clip-higher, and truncated importance "
        f"sampling with fp32 output layers, from the SFT model: "
        f"{STEPS} steps of "
        f"{QUESTIONS} GSM8K train questions SFT didn't read, sampled {QUESTIONS} at a time for at "
        f"most {ROUNDS} rounds, {SAMPLES} answers each at temperature {TEMPERATURE}, "
        f"{UPDATES} updates per step, ratios clipped to {1 - EPSILON:g} and "
        f"{1 + EPSILON_HIGH:g}, importance weights capped at {CAP:g}; "
        f"learning rate {LEARNING_RATE:g}. Over the last 5 steps, of every answer sampled:"
    )
    start = time.perf_counter()
    rlvr(model, tokenizer, problems)
    seconds = time.perf_counter() - start
    print(f"{seconds / 60:.1f} minutes, {seconds / STEPS:.1f} seconds per step")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    save_weights(model, "rlvr")

    model.to(torch.bfloat16)  # bf16 weights for generation, as the checkpoint came
    print(f"\n{SFT}\n{RATIO_CLIPPING}\n{GRPO}\n{TOKEN_LEVEL}\n{NO_STD}\n{NO_KL}\n{FILTERING}")
    print(CLIP_HIGHER)
    print(f"after GRPO with the six fixes, on the first {TEST_QUESTIONS} GSM8K test questions")
    measure(model, tokenizer, load_gsm8k("test")[:TEST_QUESTIONS])


if __name__ == "__main__":
    main()

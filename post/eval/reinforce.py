"""
Run REINFORCE (post/reinforce.py) from the SFT model and measure what it changes: the reward and
the answers' length every 5 steps, the time and peak memory, then the trained model measured as
post/eval/pass_at_k.py measured the SFT model, against the SFT model and rejection sampling.
post/eval/baseline.py runs the same with the baseline.
Run with ``uv run modal run gpu.py --modules post.eval.reinforce``, after post.sft has saved its
weights; it saves the trained ones as "reinforce".
"""

import time

import torch

from post.eval.pass_at_k import TEST_QUESTIONS, measure
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
from post.reinforce import LEARNING_RATE, QUESTIONS, SAMPLES, STEPS, TEMPERATURE, reinforce
from post.sft import unread_gsm8k

# post/eval/pass_at_k.py's and post/eval/rejection_sampling.py's runs, on the same test questions
SFT = (
    "the SFT model: greedy 50% right by strict; pass@1 25.3%, pass@2 38.2%, pass@4 52.1%, "
    "pass@8 65.0%, pass@16 76.0% by strict; 48 questions with none of 16 right, 1 with all"
)
REJECTION = (
    "rejection sampling: greedy 51%; pass@1 29.0%, pass@2 42.4%, pass@4 56.0%, pass@8 68.9%, "
    "pass@16 80.5%; 39 with none right, 2 with all"
)
# this file's run without the baseline, the same questions in the same order
REINFORCE = (
    "REINFORCE: greedy 52%; pass@1 34.2%, pass@2 47.7%, pass@4 61.0%, pass@8 72.8%, "
    "pass@16 83.0%; 34 with none right, 7 with all"
)


def run(baseline: bool) -> None:
    """Train the SFT model with REINFORCE, with or without the baseline, and measure it."""
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "RL samples in bf16: run it on the L4"
    name = "REINFORCE with a baseline" if baseline else "REINFORCE"
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32: the master copy of the weights
    load_weights(model, "sft")
    problems = unread_gsm8k(STEPS * QUESTIONS)
    print(
        f"{name} from the SFT model: {STEPS} steps of {QUESTIONS} GSM8K train questions SFT "
        f"didn't read, {SAMPLES} answers each at temperature {TEMPERATURE}; learning rate "
        f"{LEARNING_RATE:g}. The reward and the answers' length, over the last 5 steps:"
    )
    start = time.perf_counter()
    reinforce(model, tokenizer, problems, baseline)
    seconds = time.perf_counter() - start
    print(f"{seconds / 60:.1f} minutes, {seconds / STEPS:.1f} seconds per step")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    save_weights(model, "baseline" if baseline else "reinforce")

    model.to(torch.bfloat16)  # bf16 weights for generation, as the checkpoint came
    print(f"\n{SFT}\n{REJECTION}")
    if baseline:
        print(REINFORCE)
    print(f"after {name}, on the first {TEST_QUESTIONS} GSM8K test questions")
    measure(model, tokenizer, load_gsm8k("test")[:TEST_QUESTIONS])


def main() -> None:
    run(baseline=False)


if __name__ == "__main__":
    main()

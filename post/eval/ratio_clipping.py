"""
Run ratio clipping (post/ratio_clipping.py) from the SFT model and measure it as
post/eval/reinforce.py measures REINFORCE: the reward and the share of answer tokens the clip
stopped every 5 steps, the time and peak memory, then pass@k on the same test questions. Its 20
steps sample the first 160 of the questions REINFORCE's 40 did, half the answers, for 80 updates
against 40.
Run with ``uv run modal run gpu.py --modules post.eval.ratio_clipping``, after post.sft has saved
its weights; it saves the trained ones as "ratio_clipping".
"""

import time

import torch

from post.eval.pass_at_k import TEST_QUESTIONS, measure
from post.eval.reinforce import REINFORCE, SFT
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
from post.ratio_clipping import EPSILON, STEPS, UPDATES, ratio_clipping
from post.reinforce import LEARNING_RATE, QUESTIONS, SAMPLES, TEMPERATURE
from post.sft import unread_gsm8k

# post/eval/baseline.py's run: 40 steps, one update each
BASELINE = (
    "with the baseline: greedy 52%; pass@1 35.2%, pass@2 49.1%, pass@4 62.9%, pass@8 75.6%, "
    "pass@16 85.0%; 30 with none right, 5 with all"
)


def main() -> None:
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "RL samples in bf16: run it on the L4"
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32: the master copy of the weights
    load_weights(model, "sft")
    problems = unread_gsm8k(STEPS * QUESTIONS)
    print(
        f"Ratio clipping from the SFT model: {STEPS} steps of {QUESTIONS} GSM8K train questions "
        f"SFT didn't read, {SAMPLES} answers each at temperature {TEMPERATURE}, {UPDATES} "
        f"updates per step, ratios clipped to 1 ± {EPSILON}; learning rate {LEARNING_RATE:g}. "
        "The reward and the clip, over the last 5 steps:"
    )
    start = time.perf_counter()
    ratio_clipping(model, tokenizer, problems)
    seconds = time.perf_counter() - start
    print(f"{seconds / 60:.1f} minutes, {seconds / STEPS:.1f} seconds per step")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    save_weights(model, "ratio_clipping")

    model.to(torch.bfloat16)  # bf16 weights for generation, as the checkpoint came
    print(f"\n{SFT}\n{REINFORCE}\n{BASELINE}")
    print(f"after ratio clipping, on the first {TEST_QUESTIONS} GSM8K test questions")
    measure(model, tokenizer, load_gsm8k("test")[:TEST_QUESTIONS])


if __name__ == "__main__":
    main()

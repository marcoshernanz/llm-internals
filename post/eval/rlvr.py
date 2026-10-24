"""
Run GRPO with the fixes so far (post/rlvr.py) from the SFT model and measure it as post/eval/grpo.py
measures GRPO, on the same questions in the same order: the reward, the lengths of right and wrong
answers and the KL to the SFT model every 5 steps, the time and peak memory, then pass@k on the
same test questions.
Run with ``uv run modal run gpu.py --modules post.eval.rlvr``, after post.sft has saved its
weights; it saves the trained ones as "rlvr".
"""

import time

import torch

from post.eval.grpo import RATIO_CLIPPING
from post.eval.pass_at_k import TEST_QUESTIONS, measure
from post.eval.reinforce import SFT
from post.grpo import BETA
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
from post.rlvr import rlvr
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


def main() -> None:
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "RL samples in bf16: run it on the L4"
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32: the master copy of the weights
    load_weights(model, "sft")
    problems = unread_gsm8k(STEPS * QUESTIONS)
    print(
        f"GRPO with the token-level loss and no division by the standard deviation, from the SFT "
        f"model: {STEPS} steps of {QUESTIONS} GSM8K "
        f"train questions SFT didn't read, {SAMPLES} answers each at temperature {TEMPERATURE}, "
        f"{UPDATES} updates per step, ratios clipped to 1 ± {EPSILON}, KL weight {BETA}; "
        f"learning rate {LEARNING_RATE:g}. The reward, right and wrong answers' lengths and the KL, "
        "over the last 5 steps:"
    )
    start = time.perf_counter()
    rlvr(model, tokenizer, problems)
    seconds = time.perf_counter() - start
    print(f"{seconds / 60:.1f} minutes, {seconds / STEPS:.1f} seconds per step")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    save_weights(model, "rlvr")

    model.to(torch.bfloat16)  # bf16 weights for generation, as the checkpoint came
    print(f"\n{SFT}\n{RATIO_CLIPPING}\n{GRPO}\n{TOKEN_LEVEL}")
    print(f"after GRPO with both fixes, on the first {TEST_QUESTIONS} GSM8K test questions")
    measure(model, tokenizer, load_gsm8k("test")[:TEST_QUESTIONS])


if __name__ == "__main__":
    main()

"""
Run GRPO as DeepSeekMath wrote it (post/grpo.py) from the SFT model and measure it as
post/eval/ratio_clipping.py measures ratio clipping, on the same questions in the same order: the
reward, the answers' length and the KL to the SFT model every 5 steps, the time and peak memory,
then pass@k on the same test questions.
Run with ``uv run modal run gpu.py --modules post.eval.grpo``, after post.sft has saved its
weights; it saves the trained ones as "grpo".
"""

import time

import torch

from post.eval.pass_at_k import TEST_QUESTIONS, measure
from post.eval.ratio_clipping import BASELINE
from post.eval.reinforce import SFT
from post.grpo import BETA, grpo
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
from post.sft import unread_gsm8k

# post/eval/ratio_clipping.py's run: the same steps and updates, without GRPO's three additions
RATIO_CLIPPING = (
    "ratio clipping: greedy 55%; pass@1 41.0%, pass@2 54.8%, pass@4 67.3%, pass@8 78.4%, "
    "pass@16 87.5%; 25 with none right, 9 with all"
)


def main() -> None:
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "RL samples in bf16: run it on the L4"
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32: the master copy of the weights
    load_weights(model, "sft")
    problems = unread_gsm8k(STEPS * QUESTIONS)
    print(
        f"GRPO as DeepSeekMath wrote it, from the SFT model: {STEPS} steps of {QUESTIONS} GSM8K "
        f"train questions SFT didn't read, {SAMPLES} answers each at temperature {TEMPERATURE}, "
        f"{UPDATES} updates per step, ratios clipped to 1 ± {EPSILON}, KL weight {BETA}; "
        f"learning rate {LEARNING_RATE:g}. The reward, the answers' length and the KL, over the "
        "last 5 steps:"
    )
    start = time.perf_counter()
    grpo(model, tokenizer, problems)
    seconds = time.perf_counter() - start
    print(f"{seconds / 60:.1f} minutes, {seconds / STEPS:.1f} seconds per step")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    save_weights(model, "grpo")

    model.to(torch.bfloat16)  # bf16 weights for generation, as the checkpoint came
    print(f"\n{SFT}\n{BASELINE}\n{RATIO_CLIPPING}")
    print(f"after GRPO, on the first {TEST_QUESTIONS} GSM8K test questions")
    measure(model, tokenizer, load_gsm8k("test")[:TEST_QUESTIONS])


if __name__ == "__main__":
    main()

"""
How much a run of post/rlvr.py moves by chance alone: post/eval/rlvr.py's run, on the same questions
in the same order, with every sample after the questions' order, training's and the evaluation's,
drawn from the seed given. The measurements are post/eval/rlvr.py's; the weights aren't saved. Run
each seed on a GPU of its own, at the same time:
``uv run modal run gpu.py --modules "post.eval.rlvr_noise 1"``.
"""

import sys
import time

import torch
from tokenizers import Tokenizer

from post.eval.pass_at_k import TEST_QUESTIONS, measure
from post.gsm8k import load_gsm8k
from post.qwen3 import DEVICE, DTYPE, REPO_ID, Qwen3, load_qwen3, load_tokenizer, load_weights
from post.ratio_clipping import STEPS
from post.reinforce import LEARNING_RATE, QUESTIONS
from post.rlvr import ROUNDS, rlvr
from post.sft import unread_gsm8k


def sft_model() -> tuple[Qwen3, Tokenizer]:
    """The SFT model, its weights in fp32, the master copy, and the tokenizer."""
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "RL samples in bf16: run it on the L4"
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)
    load_weights(model, "sft")
    return model, load_tokenizer(REPO_ID)


def run(
    model: Qwen3, tokenizer: Tokenizer, seed: int, learning_rate: float = LEARNING_RATE
) -> None:
    """Train the model with post/rlvr.py on post/eval/rlvr.py's questions, every sample drawn from
    seed, then measure it on the test questions."""
    problems = unread_gsm8k(ROUNDS * STEPS * QUESTIONS)  # seeds with 0 itself, for the order
    torch.manual_seed(seed)
    start = time.perf_counter()
    rlvr(model, tokenizer, problems, learning_rate)
    seconds = time.perf_counter() - start
    print(f"{seconds / 60:.1f} minutes, {seconds / STEPS:.1f} seconds per step")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    model.to(torch.bfloat16)  # bf16 weights for generation, as the checkpoint came
    print(f"\non the first {TEST_QUESTIONS} GSM8K test questions")
    measure(model, tokenizer, load_gsm8k("test")[:TEST_QUESTIONS])


def main() -> None:
    seed = int(sys.argv[1])
    model, tokenizer = sft_model()
    print(f"post/eval/rlvr.py's run, its samples drawn from seed {seed}:")
    run(model, tokenizer, seed)


if __name__ == "__main__":
    main()

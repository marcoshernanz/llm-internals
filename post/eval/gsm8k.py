"""
Score the base model on GSM8K, 4-shot (post/gsm8k.py): greedy answers to the first 200 test
questions, then 16 samples at temperature 1 for each of the first 50, for pass@1 to pass@16. pass@k
is the chance that at least one of k samples is right.
Run with ``uv run modal run gpu.py --modules post.eval.gsm8k``.

Shape letters used in the comments:
    B   answers sampled at once
    T   prompt length
    N   new tokens
"""

import math
import time

import torch
from tokenizers import Tokenizer

from post.gsm8k import (
    cut,
    few_shot_prompt,
    last_number,
    load_gsm8k,
    reference_answer,
    strict_answer,
)
from post.qwen3 import DEVICE, DTYPE, REPO_ID, Qwen3, load_qwen3, load_tokenizer, token_id
from post.sampler import generate

SHOTS = 4  # solved problems before each question, as in the Qwen3 report
MAX_NEW_TOKENS = 256  # per answer: 1 in 50 greedy answers was still going at 256
GREEDY_QUESTIONS = 200
SAMPLED_QUESTIONS = 50
SAMPLES = 16  # n, the answers sampled per question for pass@k
TEMPERATURE = 1.0
KS = [1, 2, 4, 8, 16]


def pass_at_k(n: int, c: int, k: int) -> float:
    """The chance that k of n samples, picked at random, include at least one of the c right ones:
    one minus the share of the ways to pick k that pick only wrong ones."""
    assert k <= n
    return 1 - math.comb(n - c, k) / math.comb(n, k)


def answer(
    model: Qwen3,
    tokenizer: Tokenizer,
    prompt: str,
    samples: int,
    temperature: float,
    stop_id: int,
) -> list[str]:
    """The model's answers to one prompt, each cut at the next question."""
    ids = torch.tensor(tokenizer.encode(prompt).ids, device=DEVICE)  # [T]
    new_ids, _ = generate(model, [ids], samples, MAX_NEW_TOKENS, temperature, [stop_id])  # [B, N]
    return [cut(tokenizer.decode(row)) for row in new_ids.tolist()]


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, DTYPE, DEVICE)
    model.compile(fullgraph=True)
    stop_id = token_id(tokenizer, "<|endoftext|>")
    shots = load_gsm8k("train")[:SHOTS]
    test = load_gsm8k("test")

    print(f"greedy answers, {SHOTS}-shot, to the first {GREEDY_QUESTIONS} test questions")
    start = time.perf_counter()
    strict_right = 0
    last_right = 0
    for i, problem in enumerate(test[:GREEDY_QUESTIONS]):
        prompt = few_shot_prompt(shots, problem["question"])
        [text] = answer(model, tokenizer, prompt, 1, 0.0, stop_id)
        reference = reference_answer(problem["answer"])
        strict_right += strict_answer(text) == reference
        last_right += last_number(text) == reference
        if (i + 1) % 25 == 0:
            print(
                f"{i + 1:>4}: right by last number {last_right / (i + 1):4.0%}, "
                f"by strict {strict_right / (i + 1):4.0%}"
            )
    print(f"({(time.perf_counter() - start) / 60:.0f} minutes; Qwen3 report: 59.6% on all 1,319)")

    print(f"\n{SAMPLES} samples at temperature {TEMPERATURE}, first {SAMPLED_QUESTIONS} questions")
    start = time.perf_counter()
    last_pass = [0.0] * len(KS)  # summed over questions
    strict_pass = [0.0] * len(KS)
    for i, problem in enumerate(test[:SAMPLED_QUESTIONS]):
        prompt = few_shot_prompt(shots, problem["question"])
        texts = answer(model, tokenizer, prompt, SAMPLES, TEMPERATURE, stop_id)
        reference = reference_answer(problem["answer"])
        last_c = sum(last_number(text) == reference for text in texts)
        strict_c = sum(strict_answer(text) == reference for text in texts)
        for j, k in enumerate(KS):
            last_pass[j] += pass_at_k(SAMPLES, last_c, k)
            strict_pass[j] += pass_at_k(SAMPLES, strict_c, k)
        if (i + 1) % 10 == 0:
            print(f"{i + 1:>4}: pass@1 by last number {last_pass[0] / (i + 1):4.0%}")
    print("            " + "".join(f"{f'pass@{k}':>9}" for k in KS))
    print("last number " + "".join(f"{total / SAMPLED_QUESTIONS:9.1%}" for total in last_pass))
    print("strict      " + "".join(f"{total / SAMPLED_QUESTIONS:9.1%}" for total in strict_pass))
    print(f"({(time.perf_counter() - start) / 60:.0f} minutes)")


if __name__ == "__main__":
    main()

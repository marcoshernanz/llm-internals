"""
The problem rejection sampling starts from: how often the SFT model (post/sft.py) gets a GSM8K test
question right in one try, against in any of several. For each of the first 200 test questions,
asked zero-shot in chat format: a greedy answer, and 16 samples at temperature 1 for pass@1 to
pass@16 (post/eval/gsm8k.py), graded strict and by last number. Then how the questions split by
how many of their 16 samples are right, and how many samples closed their turn. The answers are
sampled 256 at a time, from 16 questions at once (post/chat.py).
Run with ``uv run modal run gpu.py --modules post.eval.pass_at_k``, after post.sft has saved its
weights.
"""

import time

from tokenizers import Tokenizer

from post.chat import answers
from post.eval.gsm8k import MAX_NEW_TOKENS, pass_at_k
from post.gsm8k import last_number, load_gsm8k, reference_answer, strict_answer
from post.qwen3 import DEVICE, DTYPE, REPO_ID, Qwen3, load_qwen3, load_tokenizer, load_weights

QUESTIONS = 200
SAMPLES = 16  # n, per question
TEMPERATURE = 1.0
KS = [1, 2, 4, 8, 16]


def measure(model: Qwen3, tokenizer: Tokenizer, problems: list[dict[str, str]]) -> None:
    """Print greedy accuracy, pass@k, how the questions split by right samples, and the time."""
    questions = [problem["question"] for problem in problems]
    references = [reference_answer(problem["answer"]) for problem in problems]

    start = time.perf_counter()
    greedy = answers(model, tokenizer, questions, 1, 0.0, MAX_NEW_TOKENS)
    seconds = time.perf_counter() - start
    strict_right = sum(strict_answer(a) == r for [a], r in zip(greedy, references))
    last_right = sum(last_number(a) == r for [a], r in zip(greedy, references))
    print(
        f"greedy: right {strict_right / len(problems):.0%} by strict, "
        f"{last_right / len(problems):.0%} by last number ({len(problems)} answers in "
        f"{seconds / 60:.1f} minutes)"
    )

    start = time.perf_counter()
    sampled = answers(model, tokenizer, questions, SAMPLES, TEMPERATURE, MAX_NEW_TOKENS)
    seconds = time.perf_counter() - start
    strict_pass = [0.0] * len(KS)  # summed over questions
    last_pass = [0.0] * len(KS)
    split = {"none": 0, "some": 0, "all": 0}  # questions, by how many samples are right (strict)
    closed = 0
    for texts, reference in zip(sampled, references):
        strict_c = sum(strict_answer(text) == reference for text in texts)
        last_c = sum(last_number(text) == reference for text in texts)
        for j, k in enumerate(KS):
            strict_pass[j] += pass_at_k(SAMPLES, strict_c, k)
            last_pass[j] += pass_at_k(SAMPLES, last_c, k)
        split["none" if strict_c == 0 else "all" if strict_c == SAMPLES else "some"] += 1
        closed += sum(text.endswith("<|im_end|>") for text in texts)
    print(
        f"\n{SAMPLES} samples at temperature {TEMPERATURE} per question "
        f"({len(problems) * SAMPLES:,} answers in {seconds / 60:.1f} minutes)"
    )
    print("            " + "".join(f"{f'pass@{k}':>9}" for k in KS))
    print("strict      " + "".join(f"{total / len(problems):9.1%}" for total in strict_pass))
    print("last number " + "".join(f"{total / len(problems):9.1%}" for total in last_pass))
    print(
        f"questions with none of {SAMPLES} right: {split['none']}, some: {split['some']}, "
        f"all: {split['all']} (strict)"
    )
    print(f"samples that closed their turn: {closed / (len(problems) * SAMPLES):.0%}")


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, DTYPE, DEVICE)
    load_weights(model, "sft")
    print(f"the SFT model (post/sft.py), on the first {QUESTIONS} GSM8K test questions")
    measure(model, tokenizer, load_gsm8k("test")[:QUESTIONS])


if __name__ == "__main__":
    main()

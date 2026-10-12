"""
GSM8K, grade-school math word problems whose answers are single integers, and the tools to score a
model on them. A base model doesn't take instructions, it continues documents: so the prompt shows
it four solved problems from the train split, each ending in "The answer is N.", then a test
question. Its answer is cut at the next "Question:" it writes, then graded two ways: strict, the
number after "The answer is", and last number, the last number anywhere in the answer. pass@k is
the chance that at least one of k samples is right.
Run with ``uv run modal run gpu.py --modules post.gsm8k``: it grades greedy answers to the first
200 test questions, then 16 samples at temperature 1 for each of the first 50, for pass@1 to
pass@16.

Shape letters used in the comments:
    B   answers sampled at once
    T   prompt length
    N   new tokens
"""

import math
import re
import time

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from post.generate import DEVICE, DTYPE, REPO_ID, TEMPERATURE
from post.qwen3 import Qwen3, load_qwen3, load_tokenizer
from post.sampler import generate

SHOTS = 4  # solved problems before each question, as in the Qwen3 report
MAX_NEW_TOKENS = 256  # per answer: 1 in 50 greedy answers was still going at 256
GREEDY_QUESTIONS = 200
SAMPLED_QUESTIONS = 50
SAMPLES = 16  # n, the answers sampled per question for pass@k
KS = [1, 2, 4, 8, 16]
NUMBER = r"-?\d[\d,]*(?:\.\d+)?"  # 18, -10, 0.2, 1,450,000


def load_gsm8k(split: str) -> list[dict[str, str]]:
    """One split of GSM8K, "train" or "test", downloaded once into the Hugging Face cache. Each
    problem is a question and a reference answer: worked steps, then "#### N"."""
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    return [{"question": row["question"], "answer": row["answer"]} for row in dataset]


def worked_solution(answer: str) -> str:
    """A reference answer as a few-shot example: the steps without their calculator notes, then
    "#### 72" rewritten as "The answer is 72."."""
    steps, final = answer.split("####")
    steps = re.sub(r"<<.*?>>", "", steps)  # "48/2 = <<48/2=24>>24 clips" -> "48/2 = 24 clips"
    return f"{steps.strip()}\nThe answer is {final.strip()}."


def few_shot_prompt(shots: list[dict[str, str]], question: str) -> str:
    """The solved problems, then the question, with its answer left for the model to write."""
    examples = [
        f"Question: {shot['question']}\nAnswer: {worked_solution(shot['answer'])}\n\n"
        for shot in shots
    ]
    return "".join(examples) + f"Question: {question}\nAnswer:"


def parse_number(text: str) -> float:
    return float(text.strip().replace(",", ""))  # "1,450,000" -> 1450000.0


def reference_answer(answer: str) -> float:
    """The number after "####" in a reference answer."""
    return parse_number(answer.split("####")[1])


def cut(text: str) -> str:
    """The answer alone: a base model sometimes goes on to write the next question itself."""
    return text.split("Question:")[0]


def strict_answer(text: str) -> float | None:
    """The number after the last "The answer is", the format the examples show. None without one.
    A dollar sign is allowed: the number is the same."""
    found = re.findall(rf"The answer is \$?({NUMBER})", text)
    return parse_number(found[-1]) if found else None


def last_number(text: str) -> float | None:
    """The last number anywhere in the text. None without one."""
    found = re.findall(NUMBER, text)
    return parse_number(found[-1]) if found else None


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
    new_ids, _ = generate(model, ids, samples, MAX_NEW_TOKENS, temperature, [stop_id])  # [B, N]
    return [cut(tokenizer.decode(row)) for row in new_ids.tolist()]


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, DTYPE, DEVICE)
    model.compile(fullgraph=True)
    stop_id = tokenizer.token_to_id("<|endoftext|>")
    assert stop_id is not None
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

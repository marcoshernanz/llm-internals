"""
The problem stage 5's tools are for: arithmetic slips. The answers write each calculation as GSM8K's
solutions do, "48/2 = 24 clips", so every step of the form expression = number can be computed
again in Python. For the SFT model and stage 4's ("rlvr"), SAMPLES answers to each of the first
200 GSM8K test questions at temperature 1: how many steps are wrong, and how the wrong answers
split, those with a slip, those whose every step is right, and those with no step to check. A
slip is a step whose number is off by more than 1% and 0.01, so a rounded 10/3 = 3.33 isn't one.
Run with ``uv run modal run gpu.py --modules post.eval.arithmetic_slips``, after post.sft and
post.eval.rlvr have saved their weights.
"""

import math
import re
import warnings

import torch
from tokenizers import Tokenizer

from post.chat import answers
from post.eval.gsm8k import MAX_NEW_TOKENS
from post.eval.pass_at_k import TEST_QUESTIONS
from post.gsm8k import load_gsm8k, reference_answer
from post.qwen3 import DEVICE, DTYPE, REPO_ID, Qwen3, load_qwen3, load_tokenizer, load_weights
from post.reinforce import reward

SAMPLES = 8  # per question
TEMPERATURE = 1.0
EXAMPLES = 5  # slips printed for each model
# An expression with at least one operator between numbers, then = and a number
STEP = re.compile(r"(\(?\d[\d.\s()+\-*/]*[+\-*/][\d.\s()+\-*/]*\d\)?)\s*=\s*\$?(-?\d+(?:\.\d+)?)")


def slips(text: str) -> tuple[int, list[str]]:
    """How many steps the text has to check, and the wrong ones, as written."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)  # "1,450" -> "1450"
    warnings.filterwarnings("ignore", category=SyntaxWarning)  # "2 (3)" warns before it fails
    checked = 0
    wrong = []
    for match in STEP.finditer(text):
        expression, stated = match.groups()
        try:
            value = eval(expression, {"__builtins__": {}})  # digits, + - * / ( ) . only
        except SyntaxError, ZeroDivisionError, TypeError:
            continue  # not a calculation after all, such as "(3) 4"
        checked += 1
        if not math.isclose(value, float(stated), rel_tol=0.01, abs_tol=0.01):
            wrong.append(match.group(0))
    return checked, wrong


def measure(model: Qwen3, tokenizer: Tokenizer, name: str) -> None:
    """Sample the answers, check their steps, and print the split."""
    torch.manual_seed(0)
    problems = load_gsm8k("test")[:TEST_QUESTIONS]
    questions = [problem["question"] for problem in problems]
    sampled = answers(model, tokenizer, questions, SAMPLES, TEMPERATURE, MAX_NEW_TOKENS)
    steps = 0
    wrong_steps = 0
    split = {"right": [0, 0, 0], "wrong": [0, 0, 0]}  # with a slip, every step right, no step
    examples = []
    for problem, group in zip(problems, sampled):
        reference = reference_answer(problem["answer"])
        for text in group:
            checked, wrong = slips(text)
            steps += checked
            wrong_steps += len(wrong)
            kind = 0 if wrong else 1 if checked else 2
            split["right" if reward(text, reference) else "wrong"][kind] += 1
            if wrong and not reward(text, reference) and len(examples) < EXAMPLES:
                examples.append(wrong[0])
    print(f"\n{name}: {SAMPLES} answers to each of {TEST_QUESTIONS} test questions")
    print(f"{steps:,} steps checked, {wrong_steps / steps:.1%} of them wrong")
    for outcome, (slipped, clean, unchecked) in split.items():
        count = slipped + clean + unchecked
        print(
            f"{outcome} answers, {count}: {slipped / count:.0%} with a slip, "
            f"{clean / count:.0%} with every step right, {unchecked / count:.0%} with no step"
        )
    print("slips in wrong answers: " + "; ".join(repr(example) for example in examples))


def main() -> None:
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "it samples in bf16: run it on the L4"
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)
    for name, label in [("sft", "the SFT model"), ("rlvr", "stage 4's model")]:
        load_weights(model, name)
        measure(model, tokenizer, label)


if __name__ == "__main__":
    main()

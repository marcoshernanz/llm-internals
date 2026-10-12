"""
GSM8K, grade-school math word problems whose answers are single integers. A base model doesn't take
instructions, it continues documents: so its prompt shows it four solved problems from the train
split, each ending in "The answer is N.", then a test question. Its answer is cut at the next
"Question:" it writes, then graded two ways: strict, the number after "The answer is", and last
number, the last number anywhere in the answer.
"""

import re

from datasets import load_dataset

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

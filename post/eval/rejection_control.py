"""
The control for rejection sampling (post/eval/rejection_sampling.py). Its kept answers were for
GSM8K questions SFT never read, so it gave the model two things: more questions, and its own
answers to them. This gives only the first: the same 384 questions, each with GSM8K's worked
solution, written by a person, instead of the model's answers; then the same retraining and the
same measurement. If the scores match rejection sampling's, its gain came from the questions.
Run with ``uv run modal run gpu.py --modules post.eval.rejection_control``.
"""

import time

import torch

from post.chat import gsm8k_chat, render
from post.eval.pass_at_k import QUESTIONS, measure
from post.gsm8k import load_gsm8k
from post.qwen3 import DEVICE, DTYPE, REPO_ID, load_tokenizer
from post.rejection_sampling import retrain, unread_problems

# post/eval/pass_at_k.py's and post/eval/rejection_sampling.py's runs
SFT = (
    "the SFT model: greedy 50% right by strict; pass@1 25.3%, pass@2 38.2%, pass@4 52.1%, "
    "pass@8 65.0%, pass@16 76.0% by strict; 48 questions with none of 16 right, 1 with all"
)
REJECTION = (
    "after rejection sampling, 802 own answers to the 384 questions: greedy 51%; pass@1 29.0%, "
    "pass@2 42.4%, pass@4 56.0%, pass@8 68.9%, pass@16 80.5%; 39 with none right, 2 with all"
)


def main() -> None:
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "SFT computes in bf16: run it on the L4"
    tokenizer = load_tokenizer(REPO_ID)
    problems = unread_problems()
    references = [render(tokenizer, gsm8k_chat(problem)) for problem in problems]
    print(
        f"the base model fine-tuned again, on SFT's conversations plus the {len(problems)} "
        f"questions with GSM8K's worked solutions, {sum(len(ids) for ids, _ in references):,} "
        "tokens with questions:"
    )
    start = time.perf_counter()
    model = retrain(tokenizer, references)
    print(f"{(time.perf_counter() - start) / 60:.1f} minutes")

    model.to(torch.bfloat16)  # bf16 weights for generation, as the checkpoint came
    print(f"\n{SFT}\n{REJECTION}")
    print(f"with the worked solutions instead, on the first {QUESTIONS} GSM8K test questions")
    measure(model, tokenizer, load_gsm8k("test")[:QUESTIONS])


if __name__ == "__main__":
    main()

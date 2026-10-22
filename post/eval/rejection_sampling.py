"""
Run rejection sampling (post/rejection_sampling.py) and measure what it changes. Of the sampling:
how many questions got an answer kept, how many answers were kept, the first one, and the time.
Then the retrained model, measured as post/eval/pass_at_k.py measured the SFT model: greedy
accuracy and pass@1 to pass@16 on the first 200 GSM8K test questions, against the SFT model's.
Run with ``uv run modal run gpu.py --modules post.eval.rejection_sampling``, after post.sft has
saved its weights; it saves the retrained ones as "rejection_sampling".
"""

import time

import torch

from post.eval.pass_at_k import TEST_QUESTIONS, measure
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
from post.rejection_sampling import (
    KEEP,
    QUESTIONS,
    SAMPLES,
    TEMPERATURE,
    retrain,
    right_answers,
)
from post.sft import unread_gsm8k

# post/eval/pass_at_k.py's run, on the model rejection sampling starts from
SFT = (
    "the SFT model (post/eval/pass_at_k.py): greedy 50% right by strict; pass@1 25.3%, "
    "pass@2 38.2%, pass@4 52.1%, pass@8 65.0%, pass@16 76.0% by strict; of 200 questions, 48 "
    "with none of 16 right, 1 with all"
)


def main() -> None:
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "SFT computes in bf16: run it on the L4"
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)
    load_weights(model, "sft")
    problems = unread_gsm8k(QUESTIONS)

    start = time.perf_counter()
    kept = right_answers(model, tokenizer, problems)
    seconds = time.perf_counter() - start
    flat = [chat for chats in kept for chat in chats]
    print(
        f"{len(problems)} GSM8K train questions SFT didn't read, {SAMPLES} answers each at "
        f"temperature {TEMPERATURE} ({seconds / 60:.1f} minutes):"
    )
    print(
        f"  {sum(len(chats) > 0 for chats in kept)} questions with an answer kept, "
        f"{sum(len(chats) == KEEP for chats in kept)} of them with the most, {KEEP}"
    )
    print(f"  {len(flat)} answers kept, {sum(len(ids) for ids, _ in flat):,} tokens with questions")
    first_ids, _ = flat[0]
    print(f"--- the first kept ---\n{tokenizer.decode(first_ids, skip_special_tokens=False)}")
    del model  # free the SFT model before the base model loads

    print("\nthe base model fine-tuned again, on SFT's conversations plus the kept ones:")
    start = time.perf_counter()
    model = retrain(tokenizer, flat)
    print(f"{(time.perf_counter() - start) / 60:.1f} minutes")
    save_weights(model, "rejection_sampling")

    model.to(torch.bfloat16)  # bf16 weights for generation, as the checkpoint came
    print(f"\n{SFT}")
    print(f"after rejection sampling, on the first {TEST_QUESTIONS} GSM8K test questions")
    measure(model, tokenizer, load_gsm8k("test")[:TEST_QUESTIONS])


if __name__ == "__main__":
    main()

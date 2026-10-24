"""
Shape letters used in the comments:
    N   samples per prompt
    P   prompt tokens
    T   tokens: the prompt, then the sampled ones
    V   vocabulary size
"""

import math

import torch
from torch import Tensor

from post.qwen3 import DEVICE, REPO_ID, Qwen3, load_qwen3, load_tokenizer

PROMPTS = [
    ("My favorite animal is the", "dog"),
    ("My favorite color is", "green"),
    ("My favorite food is", "pizza"),
    ("My favorite season is", "autumn"),
]
SAMPLES = 32
NEW_TOKENS = 8
KS = [1, 2, 4, 8, 16]


def sample(model: Qwen3, prompt: Tensor, samples: int) -> Tensor:  # [P]
    ids = prompt.repeat(samples, 1)  # [N, P]
    for _ in range(NEW_TOKENS):
        logits = model(ids)[:, -1]  # [N, V]
        probs = logits.softmax(dim=-1)  # [N, V]
        next_ids = torch.multinomial(probs, num_samples=1)  # [N, 1]
        ids = torch.cat([ids, next_ids], dim=1)  # [N, T]
    return ids  # [N, T]


def pass_at_k(n: int, c: int, k: int) -> float:
    return 1 - math.comb(n - c, k) / math.comb(n, k)


def main() -> None:
    torch.manual_seed(0)
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)

    totals = [0.0] * len(KS)
    for text, word in PROMPTS:
        prompt = torch.tensor(tokenizer.encode(text).ids, device=DEVICE)  # [P]
        with torch.no_grad():
            ids = sample(model, prompt, SAMPLES)  # [N, T]
        texts = [tokenizer.decode(row) for row in ids[:, len(prompt) :].tolist()]
        right = sum(word in completion for completion in texts)
        print(f"{text!r}: {right} of {SAMPLES} say {word!r}")
        for i, k in enumerate(KS):
            totals[i] += pass_at_k(SAMPLES, right, k)

    for k, total in zip(KS, totals):
        print(f"pass@{k}: {total / len(PROMPTS):.2f}")


if __name__ == "__main__":
    main()

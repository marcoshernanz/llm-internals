"""
Shape letters used in the comments:
    N   samples
    K   samples kept
    P   prompt tokens
    T   tokens: the prompt, then the sampled ones
    V   vocabulary size
"""

import torch
from torch import Tensor

from post.qwen3 import DEVICE, REPO_ID, Qwen3, load_qwen3, load_tokenizer

PROMPT = "My favorite animal is the"
SAMPLES = 64
NEW_TOKENS = 8
ROUNDS = 3
EPOCHS = 4
LEARNING_RATE = 1e-5


def reward(text: str) -> float:
    return 1.0 if "dog" in text else 0.0


def sample(model: Qwen3, prompt: Tensor, samples: int) -> Tensor:  # [P]
    ids = prompt.repeat(samples, 1)  # [N, P]
    for _ in range(NEW_TOKENS):
        logits = model(ids)[:, -1]  # [N, V]
        probs = logits.softmax(dim=-1)  # [N, V]
        next_ids = torch.multinomial(probs, num_samples=1)  # [N, 1]
        ids = torch.cat([ids, next_ids], dim=1)  # [N, T]
    return ids  # [N, T]


def log_prob(model: Qwen3, ids: Tensor, prompt_len: int) -> Tensor:  # [K, T]
    logits = model(ids[:, :-1])  # [K, T-1, V]
    log_probs = logits.log_softmax(dim=-1)  # [K, T-1, V]
    targets = ids[:, 1:, None]  # [K, T-1, 1]
    picked = log_probs.gather(-1, targets)  # [K, T-1, 1]
    picked = picked.squeeze(-1)  # [K, T-1]
    picked = picked[:, prompt_len - 1 :]  # [K, T-P]
    return picked.sum(dim=-1)  # [K]


def main() -> None:
    torch.manual_seed(0)
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    prompt = torch.tensor(tokenizer.encode(PROMPT).ids, device=DEVICE)  # [P]

    for i in range(ROUNDS + 1):
        with torch.no_grad():
            ids = sample(model, prompt, SAMPLES)  # [N, T]
        texts = [tokenizer.decode(row) for row in ids[:, len(prompt) :].tolist()]
        right = torch.tensor([reward(text) == 1.0 for text in texts], device=DEVICE)  # [N]
        kept = ids[right]  # [K, T]
        print(f"round {i}: {len(kept)} of {SAMPLES} right, kept")
        if i == ROUNDS:
            break

        for _ in range(EPOCHS):
            loss = -log_prob(model, kept, len(prompt)).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    main()

"""
Shape letters used in the comments:
    N   samples
    P   prompt tokens
    T   tokens: the prompt, then the sampled ones
    V   vocabulary size
"""

import torch
from torch import Tensor

from post.qwen3 import DEVICE, REPO_ID, Qwen3, load_qwen3, load_tokenizer

PROMPT = "My favorite animal is the"
SAMPLES = 16
NEW_TOKENS = 8
STEPS = 30
LEARNING_RATE = 1e-5


def reward(text: str) -> float:
    return 1.0 if "dog" in text else 0.0


def sample(
    model: Qwen3, prompt: Tensor, samples: int, new_tokens: int = NEW_TOKENS
) -> Tensor:  # [P]
    ids = prompt.repeat(samples, 1)  # [N, P]
    for _ in range(new_tokens):
        logits = model(ids)[:, -1]  # [N, V]
        probs = logits.softmax(dim=-1)  # [N, V]
        next_ids = torch.multinomial(probs, num_samples=1)  # [N, 1]
        ids = torch.cat([ids, next_ids], dim=1)  # [N, T]
    return ids  # [N, T]


def log_prob(model: Qwen3, ids: Tensor, prompt_len: int) -> Tensor:  # [N, T]
    logits = model(ids[:, :-1])  # [N, T-1, V]
    log_probs = logits.log_softmax(dim=-1)  # [N, T-1, V]
    targets = ids[:, 1:, None]  # [N, T-1, 1]
    picked = log_probs.gather(-1, targets)  # [N, T-1, 1]
    picked = picked.squeeze(-1)  # [N, T-1]
    picked = picked[:, prompt_len - 1 :]  # [N, T-P]
    return picked.sum(dim=-1)  # [N]


def main() -> None:
    torch.manual_seed(0)
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    prompt = torch.tensor(tokenizer.encode(PROMPT).ids, device=DEVICE)  # [P]

    for step in range(STEPS):
        with torch.no_grad():
            ids = sample(model, prompt, SAMPLES)  # [N, T]
        texts = [tokenizer.decode(row) for row in ids[:, len(prompt) :].tolist()]
        rewards = torch.tensor([reward(text) for text in texts], device=DEVICE)  # [N]

        log_probs = log_prob(model, ids, len(prompt))  # [N]
        loss = -(rewards * log_probs).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"step {step}: reward {rewards.mean().item():.2f}  {texts[0]!r}")


if __name__ == "__main__":
    main()

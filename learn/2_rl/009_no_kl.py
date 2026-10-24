"""
Shape letters used in the comments:
    N   samples
    P   prompt tokens
    T   tokens: the prompt, then the sampled ones
    A   sampled tokens per sample, T - P
    V   vocabulary size
"""

import torch
from torch import Tensor

from post.qwen3 import DEVICE, REPO_ID, Qwen3, load_qwen3, load_tokenizer

PROMPT = "My favorite animal is the"
SAMPLES = 16
NEW_TOKENS = 8
LEARNING_RATE = 1e-5
BATCHES = 8
UPDATES = 4
EPSILON = 0.2
BETA = 0.04


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


def token_log_probs(model: Qwen3, ids: Tensor, prompt_len: int) -> Tensor:  # [N, T]
    logits = model(ids[:, :-1])  # [N, T-1, V]
    log_probs = logits.log_softmax(dim=-1)  # [N, T-1, V]
    targets = ids[:, 1:, None]  # [N, T-1, 1]
    picked = log_probs.gather(-1, targets)  # [N, T-1, 1]
    picked = picked.squeeze(-1)  # [N, T-1]
    return picked[:, prompt_len - 1 :]  # [N, A]


def main() -> None:
    torch.manual_seed(0)
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)
    reference = load_qwen3(REPO_ID, torch.float32, DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    prompt = torch.tensor(tokenizer.encode(PROMPT).ids, device=DEVICE)  # [P]

    for batch in range(BATCHES):
        with torch.no_grad():
            ids = sample(model, prompt, SAMPLES)  # [N, T]
            old_log_probs = token_log_probs(model, ids, len(prompt))  # [N, A]
            ref_log_probs = token_log_probs(reference, ids, len(prompt))  # [N, A]
        texts = [tokenizer.decode(row) for row in ids[:, len(prompt) :].tolist()]
        rewards = torch.tensor([reward(text) for text in texts], device=DEVICE)  # [N]
        advantages = (rewards - rewards.mean())[:, None]  # [N, 1]

        for _ in range(UPDATES):
            new_log_probs = token_log_probs(model, ids, len(prompt))  # [N, A]
            ratios = (new_log_probs - old_log_probs).exp()  # [N, A]
            clipped = ratios.clamp(1 - EPSILON, 1 + EPSILON)  # [N, A]
            log_ref_ratios = ref_log_probs - new_log_probs  # [N, A]
            kl = log_ref_ratios.exp() - log_ref_ratios - 1  # [N, A]
            objective = (
                torch.minimum(ratios * advantages, clipped * advantages) - BETA * kl
            )  # [N, A]
            loss = -objective.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(
            f"batch {batch}: reward {rewards.mean().item():.2f}, "
            f"KL {kl.mean().item():.3f}  {texts[0]!r}"
        )


if __name__ == "__main__":
    main()

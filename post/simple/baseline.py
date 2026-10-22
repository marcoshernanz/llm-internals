"""
Shape letters used in the comments:
    N   samples
    P   prompt tokens
    T   tokens: the prompt, then the sampled ones
"""

import torch

from post.qwen3 import DEVICE, REPO_ID, load_qwen3, load_tokenizer
from post.simple.reinforce import LEARNING_RATE, PROMPT, SAMPLES, STEPS, log_prob, reward, sample


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
        advantages = rewards - rewards.mean()  # [N]

        log_probs = log_prob(model, ids, len(prompt))  # [N]
        loss = -(advantages * log_probs).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"step {step}: reward {rewards.mean().item():.2f}  {texts[0]!r}")


if __name__ == "__main__":
    main()

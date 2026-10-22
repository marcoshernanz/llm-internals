"""
Shape letters used in the comments:
    N   samples
    K   samples kept
    P   prompt tokens
    T   tokens: the prompt, then the sampled ones
"""

import torch

from post.qwen3 import DEVICE, REPO_ID, load_qwen3, load_tokenizer
from post.simple.reinforce import PROMPT, log_prob, reward, sample

SAMPLES = 64
ROUNDS = 3
EPOCHS = 4
LEARNING_RATE = 1e-5


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

"""
Check batching: each row of a batch must get the same logits as that row run on its own, at the
prompt and at every cached step after it. The rows are different random tokens, so a row that
leaked into another would show. Both in fp32, so any difference beyond rounding is a bug.
Run with ``uv run python -m post.check_batch``.
"""

import torch

from post.generate import DEVICE, REPO_ID
from post.qwen3 import load_qwen3
from post.sampler import new_caches

SHAPE = (4, 32)  # [B, T], the batch and its prompt length
STEPS = 16  # cached steps after the prompt
# Rounding moves the logits by about 1e-4, as in post/check_qwen3.py. One real bug, each row
# attending over its neighbor's cached keys and values, moves them by 17.
TOLERANCE = 1e-3


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)
    batch_size, prompt_length = SHAPE
    max_len = prompt_length + STEPS
    batch_caches = new_caches(model, batch_size, max_len)
    row_caches = [new_caches(model, 1, max_len) for _ in range(batch_size)]
    inputs = torch.randint(1000, 100_000, SHAPE, device=DEVICE)  # [B, T], ordinary token ids

    largest = 0.0
    for _ in range(1 + STEPS):
        together = model(inputs, batch_caches)  # [B, T, V] for the prompt, then [B, 1, V]
        together = together[:, -1]  # [B, V]
        for b in range(batch_size):
            alone = model(inputs[b : b + 1], row_caches[b])  # [1, T, V], then [1, 1, V]
            alone = alone[:, -1]  # [1, V]
            largest = max(largest, (together[b] - alone[0]).abs().max().item())
        inputs = torch.randint(1000, 100_000, (batch_size, 1), device=DEVICE)  # [B, 1]
    print(f"{batch_size} rows, {1 + STEPS} steps, largest logit difference {largest:.2e}")
    assert largest < TOLERANCE
    print("each row gets the same logits in the batch as on its own")


if __name__ == "__main__":
    main()

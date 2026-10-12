"""
Check batching: each row of a batch must get the same logits as that row run on its own, at every
cached step. Two batches: rows with different prompts, read together; and rows sharing one prompt,
read once and copied into every row, as the sampler does. After the prompt every row reads different
random tokens, so a row that leaked into another would show. All in fp32, so any difference beyond
rounding is a bug. Run with ``uv run modal run gpu.py --modules post.check.batch``.

Shape letters used in the comments:
    B   rows of the batch
    T   new tokens: the prompt, then 1 per step
    V   vocabulary size
    G   key-value heads
    L   positions a KV cache has room for
    Dh  dims per head
"""

import torch
from torch import Tensor

from post.qwen3 import DEVICE, REPO_ID, KVCache, Qwen3, load_qwen3
from post.sampler import new_caches

SHAPE = (4, 32)  # [B, T], the batch and its prompt length
STEPS = 16  # cached steps after the prompt
# Rounding moves the logits by about 1e-4, as in post/check/qwen3.py. One real bug, each row
# attending over its neighbor's cached keys and values, moves them by 17.
TOLERANCE = 1e-3


def row_difference(
    model: Qwen3, inputs: Tensor, batch_caches: list[KVCache], row_caches: list[list[KVCache]]
) -> float:  # [B, T]
    """The largest difference between the logits each row gets in the batch and on its own."""
    together = model(inputs, batch_caches, last_only=True)  # [B, 1, V]
    together = together[:, -1]  # [B, V]
    largest = 0.0
    for b, caches in enumerate(row_caches):
        alone = model(inputs[b : b + 1], caches, last_only=True)  # [1, 1, V]
        alone = alone[:, -1]  # [1, V]
        largest = max(largest, (together[b] - alone[0]).abs().max().item())
    return largest


def decode_difference(
    model: Qwen3, batch_caches: list[KVCache], row_caches: list[list[KVCache]]
) -> float:
    """The largest row difference over STEPS steps, each feeding every row a random new token."""
    largest = 0.0
    for _ in range(STEPS):
        inputs = torch.randint(1000, 100_000, (len(row_caches), 1), device=DEVICE)  # [B, 1]
        largest = max(largest, row_difference(model, inputs, batch_caches, row_caches))
    return largest


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)
    batch_size, prompt_length = SHAPE
    max_len = prompt_length + STEPS

    prompts = torch.randint(1000, 100_000, SHAPE, device=DEVICE)  # [B, T], ordinary token ids
    batch_caches = new_caches(model, batch_size, max_len)
    row_caches = [new_caches(model, 1, max_len) for _ in range(batch_size)]
    largest = row_difference(model, prompts, batch_caches, row_caches)
    largest = max(largest, decode_difference(model, batch_caches, row_caches))
    print(f"different prompts: {batch_size} rows, largest logit difference {largest:.2e}")
    assert largest < TOLERANCE

    prompt = torch.randint(1000, 100_000, (1, prompt_length), device=DEVICE)  # [1, T]
    shared_caches = new_caches(model, 1, max_len)
    model(prompt, shared_caches, last_only=True)  # [1, 1, V], the prompt read once
    for cache in shared_caches:
        cache.repeat_rows(batch_size)  # [1, G, L, Dh] -> [B, G, L, Dh]
    row_caches = [new_caches(model, 1, max_len) for _ in range(batch_size)]
    for caches in row_caches:
        model(prompt, caches, last_only=True)  # [1, 1, V], each row reads the prompt itself
    largest = decode_difference(model, shared_caches, row_caches)
    print(f"one shared prompt: {batch_size} rows, largest logit difference {largest:.2e}")
    assert largest < TOLERANCE
    print("each row gets the same logits in the batch as on its own")


if __name__ == "__main__":
    main()

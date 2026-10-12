"""
Check the KV cache: at every step, the logits from feeding in only the new token, with the cache,
must match the logits from rerunning the whole sequence. The cached run asks for the last
position's logits only, so that is checked too. Both in fp32, so any difference beyond rounding is
a bug. Run with ``uv run modal run gpu.py --modules post.check.kv_cache``.
"""

import torch

from post.bench.generate import PROMPT
from post.qwen3 import DEVICE, REPO_ID, load_qwen3, load_tokenizer
from post.sampler import new_caches

STEPS = 50  # new tokens to check, each the likeliest one
# Rounding moves the logits by about 1e-4, as in post/check/qwen3.py. One real bug, starting the
# new token's RoPE positions at 0 instead of at the cache's length, moves them by 20.
TOLERANCE = 1e-3


@torch.no_grad()
def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)
    prompt_ids = tokenizer.encode(PROMPT).ids
    ids = torch.tensor([prompt_ids], device=DEVICE)  # [1, T]
    caches = new_caches(model, batch_size=1, max_len=ids.size(1) + STEPS)
    inputs = ids  # [1, T], the prompt, then one token per step

    largest = 0.0
    for _ in range(STEPS):
        cached = model(inputs, caches, last_only=True)  # [1, 1, V]
        cached = cached[:, -1]  # [1, V]
        rerun = model(ids)  # [1, T, V], the whole sequence, no cache
        rerun = rerun[:, -1]  # [1, V]
        largest = max(largest, (cached - rerun).abs().max().item())
        next_id = rerun.argmax(dim=-1, keepdim=True)  # [1, 1]
        ids = torch.cat([ids, next_id], dim=1)  # [1, T+1]
        inputs = next_id  # [1, 1]
    print(f"{STEPS} steps, largest logit difference {largest:.2e}")
    assert largest < TOLERANCE
    print("the cache gives the same logits as rerunning the whole sequence")


if __name__ == "__main__":
    main()

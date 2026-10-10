"""
Generate text with Qwen3 and a KV cache. The prompt goes through the model once (prefill), then
each step feeds in only the newest token (decode): its query attends over the keys and values that
every earlier position left in the cache. Same prompt, settings and timing table as
post/generate.py, which reruns the whole sequence for every new token.
Run with ``uv run python -m post.sampler``.

Shape letters used in the comments:
    T   sequence length, the prompt plus the tokens generated so far
    V   vocabulary size
"""

import time

import torch
from torch import Tensor

from post.generate import DEVICE, MAX_NEW_TOKENS, PROMPT, REPO_ID, TEMPERATURE, print_timing, sample
from post.qwen3 import KVCache, Qwen3, load_qwen3, load_tokenizer


def new_caches(model: Qwen3, batch_size: int, max_len: int) -> list[KVCache]:
    """One empty KV cache per layer, with room for max_len positions."""
    weight = model.embed_tokens.weight  # the caches take the model's dtype and device
    return [
        KVCache(model.config, batch_size, max_len, weight.dtype, weight.device)
        for _ in model.layers
    ]


@torch.no_grad()
def generate(
    model: Qwen3, ids: Tensor, max_new_tokens: int, temperature: float, stop_id: int
) -> tuple[Tensor, list[float]]:  # [1, T]
    """Append up to max_new_tokens tokens to ids, stopping early at stop_id. Also return the
    seconds each new token took."""
    caches = new_caches(model, batch_size=1, max_len=ids.size(1) + max_new_tokens)
    inputs = ids  # [1, T], what the model reads this step: the whole prompt first
    seconds = []
    for _ in range(max_new_tokens):
        start = time.perf_counter()
        logits = model(inputs, caches)  # [1, T, V] for the prompt, then [1, 1, V]
        logits = logits[:, -1]  # [1, V], only the last position predicts the next token
        next_id = sample(logits, temperature)  # [1, 1]
        ids = torch.cat([ids, next_id], dim=1)  # [1, T+1]
        inputs = next_id  # [1, 1], the next step reads only the new token
        stop = next_id.item() == stop_id  # .item() waits for the GPU, so the timing is real
        seconds.append(time.perf_counter() - start)
        if stop:
            break
    return ids, seconds  # [1, T + new tokens]


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)
    stop_id = tokenizer.token_to_id("<|endoftext|>")
    assert stop_id is not None
    prompt_ids = tokenizer.encode(PROMPT).ids
    ids = torch.tensor([prompt_ids], device=DEVICE)  # [1, T]
    generate(model, ids, 2, TEMPERATURE, stop_id)  # the first calls set up the GPU: not timed

    torch.manual_seed(0)  # after the warm-up: both scripts draw the same random numbers
    ids, seconds = generate(model, ids, MAX_NEW_TOKENS, TEMPERATURE, stop_id)  # [1, T]
    new_ids = ids[0, len(prompt_ids) :].tolist()
    print(PROMPT + tokenizer.decode(new_ids))
    ending = "stopped at" if new_ids[-1] == stop_id else "never wrote"
    print(f"\n{len(new_ids)} new tokens, {ending} <|endoftext|>")

    print_timing(len(prompt_ids), seconds)
    positions = len(prompt_ids) + len(seconds) - 1  # the prompt, then every new token but the last
    print(f"{positions:,} positions run through the model to write {len(seconds)} tokens")


if __name__ == "__main__":
    main()

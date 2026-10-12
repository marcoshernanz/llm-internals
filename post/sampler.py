"""
Sample answers with Qwen3 and a KV cache, several rows at once. The prompt goes through the model
once, at batch size 1 (prefill), and its keys and values are copied into every row. Then each step
feeds in only the newest token of every row (decode): its query attends over the keys and values
that every earlier position of its row left in the cache.

Shape letters used in the comments:
    B   sequences sampled at once
    T   prompt length
    N   new tokens generated so far
    S   stop tokens
    V   vocabulary size
    G   key-value heads
    L   positions a KV cache has room for
    Dh  dims per head
"""

import torch
from torch import Tensor

from post.qwen3 import KVCache, Qwen3


def sample(logits: Tensor, temperature: float) -> Tensor:  # [B, V]
    """One next token per row, each drawn from its own row's distribution. Temperature 0 always
    picks the likeliest token."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)  # [B, 1]
    scaled = logits.float() / temperature  # [B, V], below 1 sharpens, above 1 flattens
    probs = scaled.softmax(dim=-1)  # [B, V]
    return torch.multinomial(probs, num_samples=1)  # [B, 1]


def new_caches(model: Qwen3, batch_size: int, max_len: int) -> list[KVCache]:
    """One empty KV cache per layer, with room for max_len positions."""
    weight = model.embed_tokens.weight  # the caches take the model's dtype and device
    return [
        KVCache(model.config, batch_size, max_len, weight.dtype, weight.device)
        for _ in model.layers
    ]


@torch.no_grad()
def generate(
    model: Qwen3,
    prompt: Tensor,
    samples: int,
    max_new_tokens: int,
    temperature: float,
    stop_ids: list[int],
) -> Tensor:  # [T]
    """Sample up to max_new_tokens tokens after the prompt, in samples rows. A row that writes one
    of stop_ids is done: it stays in the batch, so the shapes don't change, but every token after it
    is the first of stop_ids. Stop when every row is done."""
    stops = torch.tensor(stop_ids, device=prompt.device)  # [S]
    caches = new_caches(model, batch_size=1, max_len=prompt.size(0) + max_new_tokens)
    ids = prompt.unsqueeze(0)  # [1, T]
    logits = model(ids, caches, last_only=True)  # [1, 1, V], the prompt read once
    for cache in caches:
        cache.repeat_rows(samples)  # [1, G, L, Dh] -> [B, G, L, Dh], the prompt in every row
    logits = logits[:, -1]  # [1, V]
    logits = logits.expand(samples, -1)  # [B, V], every row starts from the same prediction
    done = torch.zeros(samples, dtype=torch.bool, device=prompt.device)  # [B]
    new_ids = []  # one [B, 1] per step
    while True:
        next_ids = sample(logits, temperature)  # [B, 1], each row draws its own
        next_ids = next_ids.masked_fill(done.unsqueeze(1), stop_ids[0])  # [B, 1], done: a stop
        new_ids.append(next_ids)
        done = done | torch.isin(next_ids.squeeze(1), stops)  # [B], wrote any of the stop ids
        if done.all() or len(new_ids) == max_new_tokens:
            break
        logits = model(next_ids, caches)  # [B, 1, V], each row reads only its newest token
        logits = logits[:, -1]  # [B, V]
    return torch.cat(new_ids, dim=1)  # [B, N]

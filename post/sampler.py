"""
Sample answers with Qwen3 and a KV cache, several prompts and several rows per prompt at once. The
prompts go through the model once, together (prefill), and each one's keys and values are copied
into each of its rows. Then each step feeds in only the newest token of every row (decode): its
query attends over the keys and values that every earlier position of its row left in the cache.

A decode step reads every weight of the model to compute one token per row, so up to a point more
rows cost about the same time per step as one: the weights are read once for all of them. Rows
from several prompts fill a batch where one prompt's samples alone wouldn't.

Prompts of different lengths are padded on the left to the longest, so that every prompt ends at
the same position and each step's new tokens line up in one column of the cache. The padding is
marked in the cache, and attention skips it (post/qwen3.py). A padded prompt starts at a later
position than it would alone, which changes nothing: with RoPE, a score depends only on how far
apart two tokens are. Hugging Face's generate pads on the left the same way. Inference servers
such as vLLM keep no padding at all: each sequence's keys and values live in blocks of their own,
and a finished sequence's place goes to a new one at once (continuous batching).

Shape letters used in the comments:
    P   prompts
    B   sequences sampled at once: samples of each prompt, P * samples
    T   longest prompt length
    N   new tokens generated so far
    S   stop tokens
    V   vocabulary size
    G   key-value heads
    L   positions a KV cache has room for
    Dh  dims per head
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from post.qwen3 import KVCache, Qwen3


def sample(logits: Tensor, temperature: float) -> tuple[Tensor, Tensor]:  # [B, V]
    """One next token per row, each drawn from its own row's distribution, and its log probability
    in that distribution. Temperature 0 always picks the likeliest token, with probability 1."""
    if temperature == 0:
        ids = logits.argmax(dim=-1, keepdim=True)  # [B, 1]
        return ids, torch.zeros(ids.shape, device=ids.device)  # [B, 1], [B, 1]
    scaled = logits.float() / temperature  # [B, V], below 1 sharpens, above 1 flattens
    log_probs = scaled.log_softmax(dim=-1)  # [B, V]
    ids = torch.multinomial(log_probs.exp(), num_samples=1)  # [B, 1]
    return ids, log_probs.gather(-1, ids)  # [B, 1], [B, 1]


def new_caches(model: Qwen3, batch_size: int, max_len: int) -> list[KVCache]:
    """One empty KV cache per layer, with room for max_len positions."""
    weight = model.embed_tokens.weight  # the caches take the model's dtype and device
    return [
        KVCache(model.config, batch_size, max_len, weight.dtype, weight.device)
        for _ in model.layers
    ]


def prefill(
    model: Qwen3, prompts: list[Tensor], max_len: int, pad_id: int, head: Tensor | None = None
) -> tuple[Tensor, list[KVCache]]:  # P of [T_p]
    """Read the prompts in one batch, each padded on the left to the longest with pad_id, which is
    never attended to; return each prompt's next-token logits and the caches holding them. head
    is the model's (post/qwen3.py)."""
    longest = max(prompt.size(0) for prompt in prompts)
    device = prompts[0].device
    lengths = torch.tensor([prompt.size(0) for prompt in prompts], device=device)  # [P]
    slots = torch.arange(longest, device=device)  # [T]
    real = slots >= (longest - lengths)[:, None]  # [P, T], the padding first, then the prompt
    padded = [F.pad(prompt, (longest - prompt.size(0), 0), value=pad_id) for prompt in prompts]
    ids = torch.stack(padded)  # [P, T]
    caches = new_caches(model, len(prompts), max_len)
    for cache in caches:
        cache.real[:, :longest] = real  # [P, T] written into [P, L]
    logits = model(ids, caches, last_only=True, head=head)  # [P, 1, V], the prompts read once
    return logits[:, -1], caches  # [P, V]


@torch.no_grad()
def generate(
    model: Qwen3,
    prompts: list[Tensor],
    samples: int,
    max_new_tokens: int,
    temperature: float,
    stop_ids: list[int],
    head: Tensor | None = None,
) -> tuple[Tensor, Tensor]:  # P of [T_p]
    """Sample up to max_new_tokens tokens after each prompt, in samples rows each: prompt p's rows
    are p * samples to (p + 1) * samples - 1. A row that writes one of stop_ids is done: it stays
    in the batch, so the shapes don't change, but every token after it is the first of stop_ids.
    Stop when every row is done. Return the new tokens and each one's log probability in the
    distribution it was drawn from. head is the model's (post/qwen3.py)."""
    longest = max(prompt.size(0) for prompt in prompts)
    stops = torch.tensor(stop_ids, device=prompts[0].device)  # [S]
    logits, caches = prefill(model, prompts, longest + max_new_tokens, stop_ids[0], head)  # [P, V]
    for cache in caches:
        cache.repeat_rows(samples)  # [P, G, L, Dh] -> [B, G, L, Dh], each prompt in its rows
    logits = logits.repeat_interleave(samples, dim=0)  # [B, V], a prompt's rows start the same
    done = torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)  # [B]
    new_ids = []  # one [B, 1] per step
    new_log_probs = []  # one [B, 1] per step
    while True:
        next_ids, log_probs = sample(logits, temperature)  # [B, 1] each, each row draws its own
        next_ids = next_ids.masked_fill(done.unsqueeze(1), stop_ids[0])  # [B, 1], done: a stop
        new_ids.append(next_ids)
        new_log_probs.append(log_probs)  # a done row's mean nothing: its tokens are cut
        done = done | torch.isin(next_ids.squeeze(1), stops)  # [B], wrote any of the stop ids
        if done.all() or len(new_ids) == max_new_tokens:
            break
        logits = model(next_ids, caches, head=head)  # [B, 1, V], each row reads only its newest
        logits = logits[:, -1]  # [B, V]
    return torch.cat(new_ids, dim=1), torch.cat(new_log_probs, dim=1)  # [B, N], [B, N]

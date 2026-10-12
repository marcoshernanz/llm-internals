"""
Sample several answers to one prompt with Qwen3 and a KV cache, a batch of rows at once. The prompt
goes through the model once, at batch size 1 (prefill), and its keys and values are copied into
every row. Then each step feeds in only the newest token of every row (decode): its query attends
over the keys and values that every earlier position of its row left in the cache.
A row that writes a stop token is done, but stays in the batch until every row is.
The model is compiled: torch.compile fuses the small steps between the matmuls into fewer kernels.
Each new input shape compiles on its first call, so each batch size runs once untimed first.
Run with ``uv run modal run gpu.py --modules post.sampler``: it times batches of 1 to 64 samples
of one question, then prints 16 answers to it.

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

import time

import torch
from torch import Tensor

from post.generate import DEVICE, DTYPE, PROMPT, REPO_ID, TEMPERATURE, sample
from post.qwen3 import KVCache, Qwen3, load_qwen3, load_tokenizer

BATCH_SIZES = [1, 4, 16, 64]  # rows of the timing table
SAMPLES = 16  # answers printed at the end
NEW_TOKENS = 100  # per sequence: enough for the first line of an answer


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
) -> tuple[Tensor, list[float]]:  # [T]
    """Sample up to max_new_tokens tokens after the prompt, in samples rows. A row that writes one
    of stop_ids is done: it stays in the batch, so the shapes don't change, but every token after it
    is the first of stop_ids. Stop when every row is done. Return the new tokens of every row, and
    the seconds each step took: the first step is the prefill."""
    stops = torch.tensor(stop_ids, device=prompt.device)  # [S]
    caches = new_caches(model, batch_size=1, max_len=prompt.size(0) + max_new_tokens)
    start = time.perf_counter()
    ids = prompt.unsqueeze(0)  # [1, T]
    logits = model(ids, caches, last_only=True)  # [1, 1, V], the prompt read once
    for cache in caches:
        cache.repeat_rows(samples)  # [1, G, L, Dh] -> [B, G, L, Dh], the prompt in every row
    logits = logits[:, -1]  # [1, V]
    logits = logits.expand(samples, -1)  # [B, V], every row starts from the same prediction
    done = torch.zeros(samples, dtype=torch.bool, device=prompt.device)  # [B]
    new_ids = []  # one [B, 1] per step
    seconds = []
    while True:
        next_ids = sample(logits, temperature)  # [B, 1], each row draws its own
        next_ids = next_ids.masked_fill(done.unsqueeze(1), stop_ids[0])  # [B, 1], done: a stop
        new_ids.append(next_ids)
        done = done | torch.isin(next_ids.squeeze(1), stops)  # [B], wrote any of the stop ids
        all_done = done.all().item()  # .item() waits for the GPU, so the timing is real
        seconds.append(time.perf_counter() - start)
        if all_done or len(new_ids) == max_new_tokens:
            break
        start = time.perf_counter()
        logits = model(next_ids, caches)  # [B, 1, V], each row reads only its newest token
        logits = logits[:, -1]  # [B, V]
    return torch.cat(new_ids, dim=1), seconds  # [B, N]


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, DTYPE, DEVICE)
    model.compile(fullgraph=True)  # fullgraph: fail loudly if any part can't be compiled
    stop_id = tokenizer.token_to_id("<|endoftext|>")
    assert stop_id is not None
    prompt = torch.tensor(tokenizer.encode(PROMPT).ids, device=DEVICE)  # [T]

    print(f"batch size: first run, then ms per step and tokens per second ({NEW_TOKENS} steps)")
    for batch_size in BATCH_SIZES:
        start = time.perf_counter()
        generate(model, prompt, batch_size, NEW_TOKENS, TEMPERATURE, [stop_id])  # compiles: untimed
        first_run = time.perf_counter() - start
        _, seconds = generate(model, prompt, batch_size, NEW_TOKENS, TEMPERATURE, [stop_id])
        decode = seconds[1:]  # the first step is the prefill: the whole prompt, once
        ms = 1000 * sum(decode) / len(decode)
        tokens_per_second = batch_size * len(decode) / sum(decode)
        print(
            f"{batch_size:>4}: {first_run:4.1f} s, then {ms:4.0f} ms, "
            f"{tokens_per_second:5.0f} tokens per second"
        )

    torch.manual_seed(0)
    new_ids, _ = generate(model, prompt, SAMPLES, NEW_TOKENS, TEMPERATURE, [stop_id])  # [B, N]
    print(f"\n{PROMPT}")
    for i, row in enumerate(new_ids.tolist()):
        answer = tokenizer.decode(row)
        first_line = answer.strip().split("\n")[0]
        print(f"{i + 1:>2}: {first_line}")


if __name__ == "__main__":
    main()

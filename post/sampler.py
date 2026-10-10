"""
Generate text with Qwen3 and a KV cache, for a batch of sequences at once. The prompt goes through
the model once (prefill), then each step feeds in only the newest token of every row (decode): its
query attends over the keys and values that every earlier position of its row left in the cache.
A row that writes the stop token is done, but stays in the batch until every row is.
Run with ``uv run python -m post.sampler``: it times batches of 1 to 64 samples of one question,
then prints 16 answers to it.

Shape letters used in the comments:
    B   sequences sampled at once
    T   sequence length, the prompt plus the tokens generated so far
    V   vocabulary size
"""

import time

import torch
from torch import Tensor

from post.generate import DEVICE, PROMPT, REPO_ID, TEMPERATURE, sample
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
    model: Qwen3, ids: Tensor, max_new_tokens: int, temperature: float, stop_id: int
) -> tuple[Tensor, list[float]]:  # [B, T]
    """Append up to max_new_tokens tokens to every row of ids. A row that writes stop_id is done:
    it stays in the batch, so the shapes don't change, but every token after it is stop_id too.
    Stop when every row is done. Also return the seconds each step took."""
    caches = new_caches(model, batch_size=ids.size(0), max_len=ids.size(1) + max_new_tokens)
    inputs = ids  # [B, T], what the model reads this step: the whole prompt first
    done = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)  # [B]
    seconds = []
    for _ in range(max_new_tokens):
        start = time.perf_counter()
        logits = model(inputs, caches)  # [B, T, V] for the prompt, then [B, 1, V]
        logits = logits[:, -1]  # [B, V], only the last position predicts the next token
        next_ids = sample(logits, temperature)  # [B, 1]
        next_ids = next_ids.masked_fill(done.unsqueeze(1), stop_id)  # [B, 1], done rows: stop_id
        ids = torch.cat([ids, next_ids], dim=1)  # [B, T+1]
        inputs = next_ids  # [B, 1], the next step reads only the new tokens
        done = done | (next_ids.squeeze(1) == stop_id)  # [B]
        all_done = done.all().item()  # .item() waits for the GPU, so the timing is real
        seconds.append(time.perf_counter() - start)
        if all_done:
            break
    return ids, seconds  # [B, T + new tokens]


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)
    stop_id = tokenizer.token_to_id("<|endoftext|>")
    assert stop_id is not None
    prompt_ids = tokenizer.encode(PROMPT).ids

    print(f"batch size: ms per decode step, new tokens per second ({NEW_TOKENS} steps)")
    for batch_size in BATCH_SIZES:
        ids = torch.tensor([prompt_ids] * batch_size, device=DEVICE)  # [B, T]
        generate(model, ids, 2, TEMPERATURE, stop_id)  # the first calls set up the GPU: not timed
        _, seconds = generate(model, ids, NEW_TOKENS, TEMPERATURE, stop_id)
        decode = seconds[1:]  # the first step is the prefill: the whole prompt of every row
        ms = 1000 * sum(decode) / len(decode)
        tokens_per_second = batch_size * len(decode) / sum(decode)
        print(f"{batch_size:>4}: {ms:4.0f} ms, {tokens_per_second:5.0f} tokens per second")

    torch.manual_seed(0)
    ids = torch.tensor([prompt_ids] * SAMPLES, device=DEVICE)  # [B, T]
    ids, _ = generate(model, ids, NEW_TOKENS, TEMPERATURE, stop_id)  # [B, T + new tokens]
    print(f"\n{PROMPT}")
    for i, row in enumerate(ids[:, len(prompt_ids) :].tolist()):
        answer = tokenizer.decode(row)
        first_line = answer.strip().split("\n")[0]
        print(f"{i + 1:>2}: {first_line}")


if __name__ == "__main__":
    main()

"""
Time generation two ways. First the simplest way, with no cache: run the whole sequence through the
model, take the last position's logits, pick a token, append it, repeat. Every new token reruns the
whole prefix, so each costs more than the one before. Then post/sampler.py, with a KV cache and a
compiled model, at batches of 1 to 64 samples of one question, and 16 of its answers printed.
torch.compile compiles each new input shape on its first call, so each batch size runs once
untimed first.
Run with ``uv run modal run gpu.py --modules post.bench_generate``.

Shape letters used in the comments:
    B   sequences sampled at once
    T   sequence length, the prompt plus the tokens generated so far
    N   new tokens
    V   vocabulary size
"""

import time

import torch
from torch import Tensor

from post.qwen3 import DEVICE, DTYPE, REPO_ID, Qwen3, load_qwen3, load_tokenizer, token_id
from post.sampler import generate, sample

# The first question of GSM8K's test set, in the "Question: ... Answer:" format of few-shot prompts
PROMPT = (
    "Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and "
    "bakes muffins for her friends every day with four. She sells the remainder at the farmers' "
    "market daily for $2 per fresh duck egg. How much in dollars does she make every day at the "
    "farmers' market?\nAnswer:"
)
TEMPERATURE = 1.0
NAIVE_TOKENS = 400  # new tokens without a cache
WINDOW = 50  # new tokens per line of the first timing table
BATCH_SIZES = [1, 4, 16, 64]  # rows of the second timing table
NEW_TOKENS = 100  # per sequence with the sampler: enough for the first line of an answer
SAMPLES = 16  # answers printed at the end


@torch.no_grad()
def naive_generate(
    model: Qwen3, ids: Tensor, max_new_tokens: int, stop_id: int
) -> tuple[Tensor, list[float]]:  # [1, T]
    """Append up to max_new_tokens tokens to ids, stopping early at stop_id, with no cache. Also
    return the seconds each new token took."""
    seconds = []
    for _ in range(max_new_tokens):
        start = time.perf_counter()
        logits = model(ids)  # [1, T, V], the whole sequence again
        logits = logits[:, -1]  # [1, V], only the last position predicts the next token
        next_id = sample(logits, TEMPERATURE)  # [1, 1]
        ids = torch.cat([ids, next_id], dim=1)  # [1, T+1]
        stop = next_id.item() == stop_id  # .item() waits for the GPU, so the timing is real
        seconds.append(time.perf_counter() - start)
        if stop:
            break
    return ids, seconds  # [1, T + new tokens]


def time_naive(model: Qwen3, prompt_ids: list[int], stop_id: int) -> None:
    ids = torch.tensor([prompt_ids], device=DEVICE)  # [1, T]
    naive_generate(model, ids, 1, stop_id)  # the first call sets up the GPU: not timed
    torch.manual_seed(0)
    _, seconds = naive_generate(model, ids, NAIVE_TOKENS, stop_id)  # [1, T]
    print(f"no cache: ms per new token, averaged over the next {WINDOW} new tokens")
    for start in range(0, len(seconds), WINDOW):
        window = seconds[start : start + WINDOW]  # the times of 50 new tokens in a row
        ms = 1000 * sum(window) / len(window)  # their average
        length = len(prompt_ids) + start  # how long the sequence was for the first of them
        print(f"{length:>4} tokens: {ms:4.0f} ms")
    lengths = [len(prompt_ids) + i for i in range(len(seconds))]  # the sequence read at each step
    print(f"{sum(lengths):,} positions run through the model to write {len(seconds)} tokens")


def time_sampler(model: Qwen3, prompt: Tensor, stop_id: int) -> None:  # [T]
    print("\nKV cache, compiled: batch size, first run, then ms per step and tokens per second")
    for batch_size in BATCH_SIZES:
        start = time.perf_counter()
        generate(model, prompt, batch_size, NEW_TOKENS, TEMPERATURE, [stop_id])  # compiles
        first_run = time.perf_counter() - start
        start = time.perf_counter()
        new_ids = generate(model, prompt, batch_size, NEW_TOKENS, TEMPERATURE, [stop_id])  # [B, N]
        torch.cuda.synchronize()  # wait for the GPU, so the timing is real
        seconds = time.perf_counter() - start  # the prefill, then N - 1 decode steps
        steps = new_ids.size(1)
        print(
            f"{batch_size:>4}: {first_run:4.1f} s, then {1000 * seconds / steps:4.0f} ms, "
            f"{batch_size * steps / seconds:5.0f} tokens per second"
        )


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, DTYPE, DEVICE)
    stop_id = token_id(tokenizer, "<|endoftext|>")
    prompt_ids = tokenizer.encode(PROMPT).ids
    time_naive(model, prompt_ids, stop_id)

    model.compile(fullgraph=True)  # fullgraph: fail loudly if any part can't be compiled
    prompt = torch.tensor(prompt_ids, device=DEVICE)  # [T]
    time_sampler(model, prompt, stop_id)

    torch.manual_seed(0)
    new_ids = generate(model, prompt, SAMPLES, NEW_TOKENS, TEMPERATURE, [stop_id])  # [B, N]
    print(f"\n{PROMPT}")
    for i, row in enumerate(new_ids.tolist()):
        answer = tokenizer.decode(row)
        first_line = answer.strip().split("\n")[0]
        print(f"{i + 1:>2}: {first_line}")


if __name__ == "__main__":
    main()

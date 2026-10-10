"""
Generate text with Qwen3 the simplest way: run the whole sequence through the model, take the last
position's logits, pick a token, append it, repeat. Every new token reruns the whole prefix, so
each token costs more than the one before; the timing table at the end shows it.
Run with ``uv run python -m post.generate``.

Shape letters used in the comments:
    B   sequences sampled at once (1 in this file's loop)
    T   sequence length, the prompt plus the tokens generated so far
    V   vocabulary size
"""

import time

import torch
from torch import Tensor

from post.qwen3 import Qwen3, load_qwen3, load_tokenizer

REPO_ID = "Qwen/Qwen3-0.6B-Base"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# The first question of GSM8K's test set, in the "Question: ... Answer:" format of few-shot prompts
PROMPT = (
    "Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and "
    "bakes muffins for her friends every day with four. She sells the remainder at the farmers' "
    "market daily for $2 per fresh duck egg. How much in dollars does she make every day at the "
    "farmers' market?\nAnswer:"
)
MAX_NEW_TOKENS = 400
TEMPERATURE = 1.0  # 0 always picks the likeliest token
WINDOW = 50  # new tokens per line of the timing table


def sample(logits: Tensor, temperature: float) -> Tensor:  # [B, V]
    """One next token per row, each drawn from its own row's distribution."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)  # [B, 1]
    scaled = logits.float() / temperature  # [B, V], below 1 sharpens, above 1 flattens
    probs = scaled.softmax(dim=-1)  # [B, V]
    return torch.multinomial(probs, num_samples=1)  # [B, 1]


@torch.no_grad()
def generate(
    model: Qwen3, ids: Tensor, max_new_tokens: int, temperature: float, stop_id: int
) -> tuple[Tensor, list[float]]:  # [1, T]
    """Append up to max_new_tokens tokens to ids, stopping early at stop_id. Also return the
    seconds each new token took."""
    seconds = []
    for _ in range(max_new_tokens):
        start = time.perf_counter()
        logits = model(ids)  # [1, T, V], the whole sequence again
        logits = logits[:, -1]  # [1, V], only the last position predicts the next token
        next_id = sample(logits, temperature)  # [1, 1]
        ids = torch.cat([ids, next_id], dim=1)  # [1, T+1]
        stop = next_id.item() == stop_id  # .item() waits for the GPU, so the timing is real
        seconds.append(time.perf_counter() - start)
        if stop:
            break
    return ids, seconds  # [1, T + new tokens]


def print_timing(prompt_length: int, seconds: list[float]) -> None:
    print(f"\nsequence length: ms per new token, averaged over the next {WINDOW} new tokens")
    for start in range(0, len(seconds), WINDOW):
        window = seconds[start : start + WINDOW]  # the times of 50 new tokens in a row
        ms = 1000 * sum(window) / len(window)  # their average
        length = prompt_length + start  # how long the sequence was for the first of them
        print(f"{length:>4} tokens: {ms:4.0f} ms")


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)
    stop_id = tokenizer.token_to_id("<|endoftext|>")
    assert stop_id is not None
    prompt_ids = tokenizer.encode(PROMPT).ids
    ids = torch.tensor([prompt_ids], device=DEVICE)  # [1, T]
    generate(model, ids, 1, TEMPERATURE, stop_id)  # the first call sets up the GPU: not timed

    torch.manual_seed(0)  # after the warm-up: both scripts draw the same random numbers
    ids, seconds = generate(model, ids, MAX_NEW_TOKENS, TEMPERATURE, stop_id)  # [1, T]
    new_ids = ids[0, len(prompt_ids) :].tolist()
    print(PROMPT + tokenizer.decode(new_ids))
    ending = "stopped at" if new_ids[-1] == stop_id else "never wrote"
    print(f"\n{len(new_ids)} new tokens, {ending} <|endoftext|>")

    print_timing(len(prompt_ids), seconds)
    lengths = [len(prompt_ids) + i for i in range(len(seconds))]  # the sequence read at each step
    print(f"{sum(lengths):,} positions run through the model to write {len(seconds)} tokens")


if __name__ == "__main__":
    main()

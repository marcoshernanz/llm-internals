"""Train the core model on TinyStories, one byte per token. Run with ``python -m core.train``."""

import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from core.config import TINY_CONFIG
from core.model import Transformer

DATA_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
DATA_PATH = Path("data/tinystories.txt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # the Mac GPU is slower than its CPU here
SEED = 1337

SEQUENCE_LEN = 128
BATCH_SIZE = 64
LEARNING_RATE = 3e-3
BIAS_UPDATE_RATE = 1e-3
TRAIN_STEPS = 2000
EVAL_INTERVAL = 200
EVAL_BATCHES = 20
SAMPLE_BYTES = 400


def load_bytes() -> Tensor:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    text = bytearray(DATA_PATH.read_bytes())  # 27k stories, 22 MB
    return torch.frombuffer(text, dtype=torch.uint8)  # [bytes], each byte is a token id


def sample_batch(data: Tensor) -> tuple[Tensor, Tensor]:  # [bytes]
    starts = torch.randint(data.size(0) - SEQUENCE_LEN, (BATCH_SIZE,))  # [B]
    offsets = torch.arange(SEQUENCE_LEN + 1)  # [T+1]
    positions = starts[:, None] + offsets  # [B, T+1]
    window = data[positions].long()  # [B, T+1]
    window = window.to(DEVICE)  # [B, T+1]
    return window[:, :-1], window[:, 1:]  # [B, T], [B, T], each target is the next byte


def loss_fn(model: Transformer, inputs: Tensor, targets: Tensor) -> Tensor:  # [B, T], [B, T]
    logits = model(inputs)  # [B, T, 256]
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


@torch.no_grad()
def estimate_loss(model: Transformer, data: Tensor) -> float:  # [bytes]
    model.eval()
    losses = [loss_fn(model, *sample_batch(data)).item() for _ in range(EVAL_BATCHES)]
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def generate(model: Transformer, prompt: str) -> str:
    model.eval()
    ids = torch.tensor([list(prompt.encode())], device=DEVICE)  # [1, T]
    for _ in range(SAMPLE_BYTES):
        logits = model(ids[:, -SEQUENCE_LEN:])  # [1, T, 256]
        probs = logits[:, -1].softmax(dim=-1)  # [1, 256]
        next_id = torch.multinomial(probs, 1)  # [1, 1]
        ids = torch.cat([ids, next_id], dim=1)  # [1, T+1]
    model.train()
    return bytes(ids[0].tolist()).decode(errors="replace")


def main() -> None:
    torch.manual_seed(SEED)
    data = load_bytes()  # [bytes]
    split = int(0.9 * data.size(0))
    train_data = data[:split]  # [0.9*bytes]
    val_data = data[split:]  # [0.1*bytes]

    model = Transformer(TINY_CONFIG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(1, TRAIN_STEPS + 1):
        inputs, targets = sample_batch(train_data)  # [B, T], [B, T]
        loss = loss_fn(model, inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_router_biases(BIAS_UPDATE_RATE)

        if step % EVAL_INTERVAL == 0:
            val_loss = estimate_loss(model, val_data)
            print(f"step {step}: train loss {loss.item():.3f}, val loss {val_loss:.3f}")

    print(generate(model, "Once upon a time"))


if __name__ == "__main__":
    main()

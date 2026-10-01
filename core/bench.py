"""Time training steps of the core model on this machine. Run with ``python -m core.bench``."""

import time
from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import Tensor

from core.config import TINY_CONFIG, Config
from core.model import Transformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQUENCE_LEN = 128
BATCH_SIZE = 64
WARMUP_STEPS = 3  # the first steps are slow: memory is allocated and kernels are picked
TIMED_STEPS = 10

WIDE_CONFIG = replace(  # TINY_CONFIG with every width 8x larger
    TINY_CONFIG,
    hidden_size=512,
    intermediate_size=768,
    moe_intermediate_size=256,
    num_attention_heads=8,
    q_lora_rank=256,
    kv_lora_rank=256,
    qk_nope_head_dim=64,
    qk_rope_head_dim=32,
    v_head_dim=64,
)


def synchronize() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()  # the CPU only queues GPU work, so wait for it to finish


def train_step(
    model: Transformer, optimizer: torch.optim.Optimizer, inputs: Tensor, targets: Tensor
) -> None:  # [B, T], [B, T]
    logits = model(inputs)  # [B, T, vocab]
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def seconds_per_step(config: Config) -> float:
    model = Transformer(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters())
    shape = (BATCH_SIZE, SEQUENCE_LEN)
    inputs = torch.randint(config.vocab_size, shape, device=DEVICE)  # [B, T]
    targets = torch.randint(config.vocab_size, shape, device=DEVICE)  # [B, T]

    for _ in range(WARMUP_STEPS):
        train_step(model, optimizer, inputs, targets)
    synchronize()
    start = time.perf_counter()
    for _ in range(TIMED_STEPS):
        train_step(model, optimizer, inputs, targets)
    synchronize()
    return (time.perf_counter() - start) / TIMED_STEPS


def main() -> None:
    name = torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu"
    print(f"device: {name}")
    for label, config in [("tiny", TINY_CONFIG), ("wide", WIDE_CONFIG)]:
        params = sum(p.numel() for p in Transformer(config).parameters())
        seconds = seconds_per_step(config)
        tokens_per_second = BATCH_SIZE * SEQUENCE_LEN / seconds
        print(
            f"{label}: {params / 1e6:.1f}M params, {seconds * 1000:.0f} ms/step, "
            f"{tokens_per_second:,.0f} tokens/s"
        )


if __name__ == "__main__":
    main()

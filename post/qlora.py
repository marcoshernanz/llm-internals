"""
QLoRA: LoRA (post/lora.py) with every frozen linear layer stored in NF4 (post/nf4.py), 4.13 bits
per weight instead of bf16's 16. LoRA keeps the loaded weights frozen in bf16, 2 bytes each: for
Qwen3-4B that's 7.5 GiB, and a training step runs out of memory on the L4
(post/bench/finetune_memory.py). In NF4 they take about a quarter of that. LoRA's matrices and the
embedding table train as with LoRA; each frozen weight is dequantized to bf16 for its matmul.
Run with ``uv run modal run gpu.py --modules post.qlora``.
"""

import torch

from post.lora import ALPHA, RANK, add_lora, lora_groups
from post.nf4 import quantize
from post.qwen3 import DEVICE, REPO_ID, load_qwen3, load_tokenizer
from post.sft import load_chats, train


def main() -> None:
    torch.manual_seed(0)  # as post/sft.py: the same conversations in the same order
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)
    chats = load_chats(tokenizer)
    quantize(model)
    add_lora(model, RANK, ALPHA)
    train(model, chats, lora_groups(model))


if __name__ == "__main__":
    main()

"""
Run SFT with QLoRA (post/qlora.py) and measure it as post/eval/lora.py measures LoRA, on the same
conversations in the same order: the losses it prints every 25 steps compare with LoRA's, step for
step, and what's left is the cost of 4-bit frozen weights.
Run with ``uv run modal run gpu.py --modules post.eval.qlora``.
"""

import torch

from post.eval.lora import run
from post.lora import ALPHA, RANK, add_lora
from post.nf4 import quantize
from post.qwen3 import DEVICE, DTYPE, REPO_ID, load_qwen3, load_tokenizer
from post.sft import load_chats

# post/eval/lora.py's last run, the same data with the frozen weights in bf16
LORA = (
    "LoRA (post/eval/lora.py): held-out loss 0.588, <|im_end|> 0.91, peak memory 8.6 GiB; "
    "closed its turn 98%, right 42% by strict, 42% by last number"
)


def main() -> None:
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "SFT computes in bf16: run it on the L4"
    torch.manual_seed(0)  # as post/sft.py: the same conversations in the same order
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)
    chats = load_chats(tokenizer)
    quantize(model)
    add_lora(model, RANK, ALPHA)
    run(model, tokenizer, chats, LORA)


if __name__ == "__main__":
    main()

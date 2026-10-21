"""
Run SFT with LoRA (post/lora.py) and measure it as post/eval/sft.py measures full fine-tuning: the
loss on held-out GSM8K conversations and the probability of <|im_end|> where their answers end,
before and after; the time and peak memory of training; then the test questions asked zero-shot in
chat format. Same conversations in the same order as the full fine-tune, so the losses it prints
every 25 steps compare with that run's, step for step. post/eval/qlora.py runs the same for QLoRA.
Run with ``uv run modal run gpu.py --modules post.eval.lora``.
"""

import time

import torch
from tokenizers import Tokenizer

from post.chat import gsm8k_chat, render
from post.eval.chat import ask
from post.eval.sft import HELD_OUT, report, stops
from post.gsm8k import load_gsm8k
from post.lora import ALPHA, EMBEDDING_LEARNING_RATE, LEARNING_RATE, RANK, add_lora, lora_groups
from post.qwen3 import DEVICE, DTYPE, REPO_ID, Qwen3, load_qwen3, load_tokenizer
from post.sft import TOKENS_PER_STEP, Chat, load_chats, train

# post/eval/sft.py's last run, the same data with every weight trained
FULL = (
    "full fine-tuning (post/eval/sft.py): held-out loss 0.576, <|im_end|> 0.89, peak memory "
    "13.9 GiB; closed its turn 96%, right 50% by strict, 52% by last number"
)


def run(model: Qwen3, tokenizer: Tokenizer, chats: list[Chat], reference: str) -> None:
    """Train the model's LoRA on the chats, measured; reference is an earlier run's results."""
    test = load_gsm8k("test")[:HELD_OUT]
    held = [render(tokenizer, gsm8k_chat(problem)) for problem in test]
    trained = sum(weight.numel() for weight in model.parameters() if weight.requires_grad)
    print(
        f"LoRA of rank {RANK} on every linear layer, and the embedding table: "
        f"{trained / 1e6:.1f}M weights train"
    )

    print("\nbefore SFT:")
    report(model, tokenizer, held)

    print(
        f"\nSFT with LoRA, peak learning rates {LEARNING_RATE:g} and {EMBEDDING_LEARNING_RATE:g} "
        f"for the table, {len(chats)} conversations packed into steps of up to {TOKENS_PER_STEP} "
        f"tokens:"
    )
    start = time.perf_counter()
    train(model, chats, lora_groups(model))
    seconds = time.perf_counter() - start
    read = sum(len(ids) for ids, _ in chats)
    print(f"{seconds / 60:.1f} minutes, {read / seconds:,.0f} tokens per second")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    print("\nafter SFT:")
    if not stops(report(model, tokenizer, held)):
        return

    for weight in model.parameters():  # the table and LoRA's matrices: bf16 for generation
        weight.data = weight.data.to(torch.bfloat16)  # NF4's buffers stay as they are
    print(f"\n{reference}")
    ask(model, tokenizer, test)


def main() -> None:
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "SFT computes in bf16: run it on the L4"
    torch.manual_seed(0)  # as post/sft.py: the same conversations in the same order
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)  # frozen: bf16 is enough
    chats = load_chats(tokenizer)
    add_lora(model, RANK, ALPHA)
    run(model, tokenizer, chats, FULL)


if __name__ == "__main__":
    main()

"""
Measure what NF4 (post/nf4.py) saves and costs, on Qwen3-0.6B, 1.7B and 4B base: the memory the
loaded weights take in bf16 and with every linear layer in NF4, and the loss on held-out GSM8K
answers both ways, which shows how much the model's predictions suffer from 4-bit weights. The
embedding table stays in bf16 either way, as in QLoRA.
Run with ``uv run modal run gpu.py --modules post.eval.nf4``.
"""

import torch

from post.chat import gsm8k_chat, render
from post.eval.sft import HELD_OUT, held_out
from post.gsm8k import load_gsm8k
from post.nf4 import quantize
from post.qwen3 import DEVICE, DTYPE, REPO_ID, load_qwen3, load_tokenizer

MODELS = ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base"]
GIB = 2**30


def main() -> None:
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "NF4 dequantizes to bf16: run it on the L4"
    tokenizer = load_tokenizer(REPO_ID)  # the same tokenizer for every Qwen3 size
    held = [render(tokenizer, gsm8k_chat(problem)) for problem in load_gsm8k("test")[:HELD_OUT]]
    print(f"the loss on the answers of the first {HELD_OUT} GSM8K test conversations")
    print(f"{'model':<22} {'bf16 GiB':>9} {'NF4 GiB':>8} {'bf16 loss':>10} {'NF4 loss':>9}")
    for repo_id in MODELS:
        model = load_qwen3(repo_id, torch.bfloat16, DEVICE)
        bf16_gib = torch.cuda.memory_allocated() / GIB
        bf16_loss, _ = held_out(model, held)
        quantize(model)
        nf4_gib = torch.cuda.memory_allocated() / GIB
        nf4_loss, _ = held_out(model, held)
        print(
            f"{repo_id.removeprefix('Qwen/'):<22} {bf16_gib:>9.2f} {nf4_gib:>8.2f} "
            f"{bf16_loss:>10.3f} {nf4_loss:>9.3f}"
        )
        del model  # free this model before loading the next
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

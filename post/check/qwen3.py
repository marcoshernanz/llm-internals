"""
Check post/qwen3.py against the transformers implementation of Qwen3: same checkpoint, same input,
same logits. Both run in fp32 on the CPU, so any difference beyond rounding is a bug in ours.
Run with ``uv run modal run gpu.py --modules post.check.qwen3``.
"""

import torch
from transformers import AutoModelForCausalLM

from post.qwen3 import REPO_ID, load_qwen3

SHAPE = (2, 64)  # [B, T]
# Logits reach about 20. Adding the same numbers in another order moves them by up to 1e-4:
# transformers' own two attention paths (eager and sdpa) differ by 7e-5. One real bug, RoPE pairing
# neighbors instead of dim i with dim i + Dh/2, moves them by 14. 1e-3 sits between the two.
TOLERANCE = 1e-3


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    ours = load_qwen3(REPO_ID, torch.float32, "cpu")
    # eager: attention as plain matmuls and a softmax, like ours, not a fused kernel
    theirs = AutoModelForCausalLM.from_pretrained(
        REPO_ID, dtype=torch.float32, attn_implementation="eager"
    )
    vocab_size = ours.embed_tokens.weight.size(0)
    ids = torch.randint(vocab_size, SHAPE)  # [B, T], random tokens work: only the math is checked

    our_logits = ours(ids)  # [B, T, V]
    their_logits = theirs(ids).logits  # [B, T, V]
    diff = (our_logits - their_logits).abs().max()
    print(f"logits {list(our_logits.shape)}, largest difference {diff:.2e}")
    torch.testing.assert_close(our_logits, their_logits, atol=TOLERANCE, rtol=0)
    print("same logits as transformers")


if __name__ == "__main__":
    main()

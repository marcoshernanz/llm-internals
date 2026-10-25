"""
How much RL changes in the model: post/rlvr.py with a LoRA of rank 1 on every linear layer
(post/lora.py), and nothing else trained, the embedding table frozen too. An answer gives back one
reward, so a run carries little information, and a correction of rank 1 may hold all of it: the
comparisons of LoRA against full fine-tuning find rank 1 matches it in RL, where SFT needs far
more. The learning rate is 10 times full fine-tuning's, as post/lora.py's. Measured as
post/eval/rlvr_noise.py measures the full run, whose three runs give the noise to compare with.
Run with ``uv run modal run gpu.py --modules post.eval.rlvr_lora``, after post.sft has saved its
weights.
"""

from post.eval.rlvr_noise import run, sft_model
from post.lora import ALPHA, add_lora
from post.reinforce import LEARNING_RATE

RANK = 1
LORA_LEARNING_RATE = 10 * LEARNING_RATE


def main() -> None:
    model, tokenizer = sft_model()
    total = sum(weight.numel() for weight in model.parameters())
    add_lora(model, RANK, ALPHA)  # freezes every weight but the embedding table
    model.embed_tokens.requires_grad_(False)  # here it's frozen too
    trained = sum(weight.numel() for weight in model.parameters() if weight.requires_grad)
    print(
        f"post/rlvr.py with a LoRA of rank {RANK} on every linear layer, scale {ALPHA / RANK:g}: "
        f"{trained:,} weights train, {trained / total:.3%} of the model's {total:,}; learning "
        f"rate {LORA_LEARNING_RATE:g}"
    )
    run(model, tokenizer, 0, LORA_LEARNING_RATE)


if __name__ == "__main__":
    main()

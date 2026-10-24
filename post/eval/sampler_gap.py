"""
The problem truncated importance sampling corrects (post/rlvr.py): the sampler, a bf16 copy of the
model, and the trainer, fp32 weights with bf16 matmuls, give the same answer tokens different
probabilities. From the SFT model, QUESTIONS GSM8K train questions SFT didn't read, 8 answers each,
sampled as post/rlvr.py samples them; then the trainer's log probability of every answer token, as
its first update computes it. For each output layer, the sampler's and the trainer's, in bf16 or
fp32: how far apart the two log probabilities are on average, the share of tokens whose ratio
between them is above 2 or below 1/2, and the largest ratio.
Run with ``uv run modal run gpu.py --modules post.eval.sampler_gap``, after post.sft has saved its
weights.
"""

import math

import torch
from tokenizers import Tokenizer

from post.chat import token_log_probs
from post.qwen3 import DEVICE, DTYPE, REPO_ID, Qwen3, load_qwen3, load_tokenizer, load_weights
from post.reinforce import rollouts
from post.sft import join, pack, unread_gsm8k

QUESTIONS = 32


def gap(
    trainer: Qwen3,
    sampler: Qwen3,
    tokenizer: Tokenizer,
    problems: list[dict[str, str]],
    fp32_sampler: bool,
    fp32_trainer: bool,
) -> None:
    """Sample the problems' answers, then print how far the trainer's log probabilities of their
    tokens are from the sampler's."""
    torch.manual_seed(0)
    head = trainer.embed_tokens.weight.detach() if fp32_sampler else None  # [V, D], fp32
    chats, _, sampled = rollouts(sampler, tokenizer, problems, head)
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        rows = [token_log_probs(trainer, *join(row), fp32_head=fp32_trainer) for row in pack(chats)]
    trained = torch.cat(rows)  # [A], every answer's tokens in order
    drawn = torch.tensor([p for answer in sampled for p in answer], device=DEVICE)  # [A]
    log_ratios = trained - drawn  # [A]
    far = (log_ratios.abs() > math.log(2)).float().mean()  # the share above 2 or below 1/2
    sampler_head = "fp32" if fp32_sampler else "bf16"
    trainer_head = "fp32" if fp32_trainer else "bf16"
    print(
        f"sampler's head {sampler_head}, trainer's {trainer_head}: "
        f"{log_ratios.abs().mean():.4f} apart in log probability on average, "
        f"{far:.2%} of tokens with a ratio above 2 or below 1/2, largest ratio "
        f"{log_ratios.exp().max():.2f}, over {log_ratios.numel():,} answer tokens"
    )


def main() -> None:
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "RL samples in bf16: run it on the L4"
    tokenizer = load_tokenizer(REPO_ID)
    trainer = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32: the master copy of the weights
    load_weights(trainer, "sft")
    sampler = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)
    sampler.load_state_dict(trainer.state_dict())  # the same weights, copied into bf16
    problems = unread_gsm8k(QUESTIONS)
    for fp32_sampler in [False, True]:
        for fp32_trainer in [False, True]:
            gap(trainer, sampler, tokenizer, problems, fp32_sampler, fp32_trainer)


if __name__ == "__main__":
    main()

"""
Supervised fine-tuning (SFT): teach the base model to take turns. The training step is
pre-training's: whole conversations go through the model at once, each position predicting the
next token from the true tokens before it, and one optimizer step follows each batch. Two things
differ: the data are conversations in Qwen's chat format, here GSM8K's train problems with their
worked solutions as the answers; and the loss counts only the assistant's tokens (post/chat.py).
It is a full fine-tune: every weight trains. The weights stay in fp32 while the matmuls run in
bf16 (mixed precision): an update far smaller than a weight would vanish if the weight were bf16,
which keeps only about 3 significant digits.
Run with ``uv run modal run gpu.py --modules post.sft``. post/eval_sft.py measures what it changes.

Shape letters used in the comments:
    B   conversations per batch
    T   tokens per conversation, padded to the batch's longest
"""

import torch
from tokenizers import Tokenizer
from torch import Tensor

from post.chat import gsm8k_chat, render, sft_loss
from post.gsm8k import load_gsm8k
from post.qwen3 import DEVICE, REPO_ID, Qwen3, load_qwen3, load_tokenizer, token_id

BATCH_SIZE = 4  # conversations per step: 8 can reach 3,500 tokens, too many for the L4's 24 GB
STEPS = 400  # 1,600 of GSM8K's 7,473 train conversations
LEARNING_RATE = 1e-5  # the peak
WARMUP_STEPS = 12  # 3% of the steps, rising to the peak; then a straight line down to 0
MAX_GRAD_NORM = 1.0  # a step's gradient is scaled down to this length if longer
LOG_EVERY = 25  # steps

Chat = tuple[list[int], list[bool]]  # a conversation's token ids, and which ones the loss counts


def load_chats(tokenizer: Tokenizer) -> list[Chat]:
    """The train problems SFT reads, shuffled, as conversations."""
    train = load_gsm8k("train")
    order = torch.randperm(len(train)).tolist()
    return [render(tokenizer, gsm8k_chat(train[i])) for i in order[: STEPS * BATCH_SIZE]]


def pad(chats: list[Chat], pad_id: int) -> tuple[Tensor, Tensor]:
    """The conversations as one batch, each padded at its end to the longest. Padding after the
    real tokens needs no attention mask: causal attention never lets a token see later ones, and
    the loss mask never counts a padded target."""
    length = max(len(ids) for ids, _ in chats)
    ids = [ids + [pad_id] * (length - len(ids)) for ids, _ in chats]
    mask = [mask + [False] * (length - len(mask)) for _, mask in chats]
    ids_t = torch.tensor(ids, device=DEVICE)  # [B, T]
    mask_t = torch.tensor(mask, device=DEVICE)  # [B, T]
    return ids_t, mask_t


def learning_rate(step: int) -> float:
    """Up in a straight line over the warmup, then down in a straight line to 0 at the end."""
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    return LEARNING_RATE * (STEPS - step) / (STEPS - WARMUP_STEPS)


def train(model: Qwen3, chats: list[Chat], pad_id: int) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0, fused=True
    )
    for step in range(STEPS):
        batch = chats[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        ids, mask = pad(batch, pad_id)  # [B, T] each
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):  # matmuls in bf16, weights in fp32
            loss = sft_loss(model, ids, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if (step + 1) % LOG_EVERY == 0:
            print(f"step {step + 1}: loss {loss.item():.3f}")


def main() -> None:
    torch.manual_seed(0)
    tokenizer = load_tokenizer(REPO_ID)
    pad_id = token_id(tokenizer, "<|endoftext|>")  # Qwen's padding token
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32: the master copy of the weights
    train(model, load_chats(tokenizer), pad_id)


if __name__ == "__main__":
    main()

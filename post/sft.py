"""
Supervised fine-tuning (SFT): teach the base model to take turns. The training step is
pre-training's: whole conversations go through the model at once, each position predicting the
next token from the true tokens before it, and one optimizer step follows each batch. Two things
differ: the data are conversations in Qwen's chat format, here GSM8K's train problems with their
worked solutions as the answers; and the loss counts only the assistant's tokens, <|im_end|>
included (post/chat.py).
It is a full fine-tune: every weight trains. The weights stay in fp32 while the matmuls run in
bf16 (mixed precision): an update far smaller than a weight would vanish if the weight were bf16,
which keeps only about 3 significant digits. Each batch is padded at its end to its longest
conversation, and the log shows how much of what the model reads is padding.
Before and after training it measures the loss on held-out conversations, GSM8K test problems, and
the probability the model gives <|im_end|> where their answers end. After, it asks the same test
questions zero-shot in chat format, as post/chat_baseline.py asked the base model.
Run with ``uv run modal run gpu.py --modules post.sft``.

Shape letters used in the comments:
    B   conversations per batch
    T   tokens per conversation, padded to the batch's longest
"""

import time

import torch
from torch import Tensor

from post.chat import gsm8k_chat, render, sft_loss, token_losses
from post.chat_baseline import ask, token_id
from post.generate import DEVICE, DTYPE, REPO_ID
from post.gsm8k import load_gsm8k
from post.qwen3 import Qwen3, load_qwen3, load_tokenizer

BATCH_SIZE = 4  # conversations per step: 8 can reach 3,500 tokens, too many for the L4's 24 GB
STEPS = 400  # 1,600 of GSM8K's 7,473 train conversations
LEARNING_RATE = 1e-5  # the peak
WARMUP_STEPS = 12  # 3% of the steps, rising to the peak; then a straight line down to 0
MAX_GRAD_NORM = 1.0  # a step's gradient is scaled down to this length if longer
HELD_OUT = 50  # the first test problems: the loss on them before and after, then asked
LOG_EVERY = 50  # steps

Chat = tuple[list[int], list[bool]]  # a conversation's token ids, and which ones the loss counts


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


@torch.no_grad()
def held_out(model: Qwen3, chats: list[Chat]) -> tuple[float, float]:
    """The mean loss over the held-out answers' tokens, and the mean probability the model gives
    <|im_end|> where each answer ends. One conversation at a time: no padding."""
    total = 0.0
    counted = 0
    stop = 0.0
    for ids, mask in chats:
        ids_t = torch.tensor([ids], device=DEVICE)  # [1, T]
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            losses = token_losses(model, ids_t)[0]  # [T-1]
        targets = torch.tensor(mask[1:], device=DEVICE)  # [T-1], whether each target counts
        answer = losses[targets]  # the answer's targets, <|im_end|> last
        total += answer.sum().item()
        counted += answer.numel()
        stop += torch.exp(-answer[-1]).item()  # the probability it gave <|im_end|>
    return total / counted, stop / len(chats)


def main() -> None:
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "SFT computes in bf16: run it on the L4"
    torch.manual_seed(0)
    tokenizer = load_tokenizer(REPO_ID)
    pad_id = token_id(tokenizer, "<|endoftext|>")  # Qwen's padding token
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32: the master copy of the weights
    train = load_gsm8k("train")
    test = load_gsm8k("test")[:HELD_OUT]
    order = torch.randperm(len(train)).tolist()
    chats = [render(tokenizer, gsm8k_chat(train[i])) for i in order[: STEPS * BATCH_SIZE]]
    held = [render(tokenizer, gsm8k_chat(problem)) for problem in test]

    loss, stop = held_out(model, held)
    print(
        f"before: held-out loss {loss:.3f}, probability of <|im_end|> where answers end {stop:.2g}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0, fused=True
    )
    print(f"\n{STEPS} steps of {BATCH_SIZE} conversations: loss, gradient norm, learning rate")
    start = time.perf_counter()
    losses = []  # per step, since the last log line
    read = 0  # tokens the model read, padding included
    real = 0  # of them, the conversations' own tokens
    counted = 0  # of those, the ones the loss counts
    for step in range(STEPS):
        batch = chats[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        ids, mask = pad(batch, pad_id)  # [B, T] each
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):  # matmuls in bf16, weights in fp32
            loss = sft_loss(model, ids, mask)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())
        read += ids.numel()
        real += sum(len(chat_ids) for chat_ids, _ in batch)
        counted += mask.sum().item()
        if (step + 1) % LOG_EVERY == 0:
            mean = sum(losses) / len(losses)
            print(f"{step + 1:>4}: {mean:.3f} {norm.item():5.2f} {learning_rate(step):.1e}")
            losses = []
    seconds = time.perf_counter() - start
    print(f"({seconds / 60:.1f} minutes, {read / seconds:,.0f} tokens per second)")
    print(
        f"of the {read:,} tokens read: {1 - real / read:.0%} padding, "
        f"{(real - counted) / read:.0%} prompt, {counted / read:.0%} answer (counted)"
    )
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    loss, stop = held_out(model, held)
    print(
        f"\nafter: held-out loss {loss:.3f}, probability of <|im_end|> where answers end {stop:.2g}"
    )

    del optimizer
    model.to(torch.bfloat16)  # bf16 weights for generation, as the checkpoint came
    print(
        "\nthe base model on these questions (post/chat_baseline.py): closed its turn 0%, right 6%"
    )
    ask(model, tokenizer, test)


if __name__ == "__main__":
    main()

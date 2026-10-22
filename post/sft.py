"""
Supervised fine-tuning (SFT): teach the base model to take turns. The training step is
pre-training's: whole conversations go through the model at once, each position predicting the
next token from the true tokens before it, and one optimizer step follows each batch. Two things
differ: the data are conversations in Qwen's chat format; and the loss counts only the assistant's
tokens (post/chat.py). The conversations are GSM8K's train problems with their worked solutions,
mixed with everyday ones from smol-smoltalk: chat, writing, code, summaries, one to several turns.
Trained on GSM8K alone, the model ended every answer the GSM8K way (post/eval/prompts.py).
It is a full fine-tune: every weight trains. The weights stay in fp32 while the matmuls run in
bf16 (mixed precision): an update far smaller than a weight would vanish if the weight were bf16,
which keeps only about 3 significant digits.
Each step's conversations are packed end to end into one row, with no padding; attention stays
within each conversation (post/qwen3.py).
Run with ``uv run modal run gpu.py --modules post.sft``: it saves the weights as "sft", which later
stages start from. post/eval/sft.py measures what it changes.

Shape letters used in the comments:
    T   tokens in a step's packed row
"""

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from torch import Tensor, nn

from post.chat import gsm8k_chat, render, sft_loss
from post.gsm8k import load_gsm8k
from post.qwen3 import DEVICE, REPO_ID, Packing, Qwen3, load_qwen3, load_tokenizer, save_weights

GSM8K = 1_600  # of GSM8K's 7,473 train problems, 162 tokens long on average
SMOLTALK = 600  # of the 115,086 conversations in smol-smoltalk's first train file, 865 on average
SMOLTALK_FILE = "data/train-00000-of-00004.parquet"  # the first of four: 230 MB, not 920
TOKENS_PER_STEP = 2_048  # at most, never one conversation split; 1% of smol-smoltalk's are longer
LEARNING_RATE = 1e-5  # the peak
WARMUP_STEPS = 13  # 3% of the 420 steps, rising to the peak; then a straight line down to 0
MAX_GRAD_NORM = 1.0  # a step's gradient is scaled down to this length if longer
LOG_EVERY = 25  # steps

Chat = tuple[list[int], list[bool]]  # a conversation's token ids, and which ones the loss counts
Group = tuple[list[nn.Parameter], float]  # weights that train, and their peak learning rate


def load_chats(tokenizer: Tokenizer) -> list[Chat]:
    """The conversations SFT reads, shuffled together: GSM8K's, then smol-smoltalk's, leaving out
    any longer than a step."""
    gsm8k = load_gsm8k("train")
    order = torch.randperm(len(gsm8k)).tolist()
    chats = [render(tokenizer, gsm8k_chat(gsm8k[i])) for i in order[:GSM8K]]
    smoltalk = load_dataset("HuggingFaceTB/smol-smoltalk", data_files=SMOLTALK_FILE, split="train")
    order = torch.randperm(len(smoltalk)).tolist()
    chats += [render(tokenizer, smoltalk[i]["messages"]) for i in order[:SMOLTALK]]
    chats = [chat for chat in chats if len(chat[0]) <= TOKENS_PER_STEP]
    order = torch.randperm(len(chats)).tolist()
    return [chats[i] for i in order]


def unread_gsm8k(count: int) -> list[dict[str, str]]:
    """count GSM8K train problems SFT doesn't read, for later stages. load_chats's first random
    draw after seeding is GSM8K's order, whose first GSM8K problems it reads: the same draw gives
    the ones after."""
    problems = load_gsm8k("train")
    torch.manual_seed(0)
    order = torch.randperm(len(problems)).tolist()
    return [problems[i] for i in order[GSM8K : GSM8K + count]]


def pack(chats: list[Chat]) -> list[list[Chat]]:
    """The conversations in order, grouped into steps: each step takes the next one while the step
    stays within TOKENS_PER_STEP tokens."""
    steps = [[]]
    tokens = 0  # in the last step so far
    for chat in chats:
        if tokens + len(chat[0]) > TOKENS_PER_STEP:
            steps.append([])
            tokens = 0
        steps[-1].append(chat)
        tokens += len(chat[0])
    return steps


def join(chats: list[Chat]) -> tuple[Tensor, Tensor, Packing]:
    """The conversations laid end to end as one row, and where each starts."""
    ids = [token for chat_ids, _ in chats for token in chat_ids]
    mask = [counted for _, chat_mask in chats for counted in chat_mask]
    ids_t = torch.tensor([ids], device=DEVICE)  # [1, T]
    mask_t = torch.tensor([mask], device=DEVICE)  # [1, T]
    packing = Packing([len(chat_ids) for chat_ids, _ in chats], DEVICE)
    return ids_t, mask_t, packing


def learning_rate(step: int, steps: int, peak: float) -> float:
    """Up in a straight line over the warmup, then down in a straight line to 0 at the end."""
    if step < WARMUP_STEPS:
        return peak * (step + 1) / WARMUP_STEPS
    return peak * (steps - step) / (steps - WARMUP_STEPS)


def train(model: Qwen3, chats: list[Chat], groups: list[Group] | None = None) -> None:
    """Train the groups of weights, each at its own peak learning rate: by default every weight
    at LEARNING_RATE, a full fine-tune. LoRA trains other groups (post/lora.py)."""
    if groups is None:
        groups = [(list(model.parameters()), LEARNING_RATE)]
    steps = pack(chats)
    weights = [weight for group, _ in groups for weight in group]
    optimizer = torch.optim.AdamW(
        [{"params": group, "lr": peak} for group, peak in groups], weight_decay=0.0, fused=True
    )
    for step, batch in enumerate(steps):
        ids, mask, packing = join(batch)  # [1, T] each
        for param_group, (_, peak) in zip(optimizer.param_groups, groups):
            param_group["lr"] = learning_rate(step, len(steps), peak)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):  # matmuls in bf16
            loss = sft_loss(model, ids, mask, packing)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(weights, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if (step + 1) % LOG_EVERY == 0:
            print(f"step {step + 1}: loss {loss.item():.3f}")


def main() -> None:
    torch.manual_seed(0)
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32: the master copy of the weights
    train(model, load_chats(tokenizer))
    save_weights(model, "sft")


if __name__ == "__main__":
    main()

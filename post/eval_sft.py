"""
Run SFT (post/sft.py) and measure what it changes. Before and after training: the loss on held-out
conversations, GSM8K test problems, and the probability the model gives <|im_end|> where their
answers end; then the first of them token by token. Of training: how long it took, how much of what
the model read was padding, and the peak memory. After: the same test questions asked zero-shot in
chat format, as post/eval_chat.py asks the base model.
Run with ``uv run modal run gpu.py --modules post.eval_sft``.

Shape letters used in the comments:
    T   tokens in a conversation
    A   of them, the targets the loss counts: the answer and its <|im_end|>
    V   vocabulary size
"""

import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from post.chat import gsm8k_chat, render
from post.eval_chat import ask
from post.gsm8k import load_gsm8k
from post.qwen3 import DEVICE, DTYPE, REPO_ID, Qwen3, load_qwen3, load_tokenizer, token_id
from post.sft import BATCH_SIZE, STEPS, Chat, load_chats, train

HELD_OUT = 50  # the first test problems: the loss on them before and after, then asked
SHOWN = 6  # targets printed from each end of an answer


@torch.no_grad()
def held_out(model: Qwen3, chats: list[Chat]) -> tuple[float, float]:
    """The mean loss over the held-out answers' targets, and the mean probability the model gives
    <|im_end|> where each answer ends. One conversation at a time: no padding."""
    total = 0.0
    counted = 0
    stop = 0.0
    for ids, mask in chats:
        ids_t = torch.tensor(ids, device=DEVICE)  # [T]
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            logits = model(ids_t[None, :-1])  # [1, T-1, V]
        losses = F.cross_entropy(logits[0].float(), ids_t[1:], reduction="none")  # [T-1]
        targets = torch.tensor(mask[1:], device=DEVICE)  # [T-1], whether each target counts
        answer = losses[targets]  # [A], <|im_end|> last
        total += answer.sum().item()
        counted += answer.numel()
        stop += torch.exp(-answer[-1]).item()  # the probability it gave <|im_end|>
    return total / counted, stop / len(chats)


@torch.no_grad()
def show_answer(model: Qwen3, tokenizer: Tokenizer, chat: Chat) -> None:
    """Each target at the answer's start and end, with the probability the model gives it; then the
    model's first choices where <|im_end|> should come."""
    ids, mask = chat
    ids_t = torch.tensor(ids, device=DEVICE)  # [T]
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        logits = model(ids_t[None, :-1])  # [1, T-1, V]
    probs = logits[0].float().softmax(dim=-1)  # [T-1, V]
    answer = [t for t in range(len(ids) - 1) if mask[t + 1]]  # positions whose target is counted
    for t in answer[:SHOWN] + answer[-SHOWN:]:
        token = tokenizer.decode([ids[t + 1]], skip_special_tokens=False)
        print(f"{token!r:>14} {probs[t, ids[t + 1]].item():9.3g}")
    top = probs[answer[-1]].topk(3)
    choices = [
        f"{tokenizer.decode([i], skip_special_tokens=False)!r} {p:.3g}"
        for p, i in zip(top.values.tolist(), top.indices.tolist())
    ]
    print(f"its first choices where <|im_end|> should come: {', '.join(choices)}")


def report(model: Qwen3, tokenizer: Tokenizer, held: list[Chat]) -> None:
    loss, stop = held_out(model, held)
    print(f"held-out loss {loss:.3f}, probability of <|im_end|> where answers end {stop:.2g}")
    print("the first held-out answer: target, then the probability the model gives it")
    show_answer(model, tokenizer, held[0])


def token_shares(chats: list[Chat]) -> int:
    """Print how the tokens SFT read split into padding, prompt and answer; return how many."""
    read = 0  # tokens the model read, padding included
    real = 0  # of them, the conversations' own tokens
    counted = 0  # of those, the ones the loss counts
    for step in range(STEPS):
        batch = chats[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        read += max(len(ids) for ids, _ in batch) * len(batch)  # each padded to the longest
        real += sum(len(ids) for ids, _ in batch)
        counted += sum(sum(mask) for _, mask in batch)
    print(
        f"of the {read:,} tokens read: {1 - real / read:.0%} padding, "
        f"{(real - counted) / read:.0%} prompt, {counted / read:.0%} answer (counted)"
    )
    return read


def main() -> None:
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "SFT computes in bf16: run it on the L4"
    torch.manual_seed(0)  # as post/sft.py: the same conversations in the same order
    tokenizer = load_tokenizer(REPO_ID)
    pad_id = token_id(tokenizer, "<|endoftext|>")
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)
    chats = load_chats(tokenizer)
    test = load_gsm8k("test")[:HELD_OUT]
    held = [render(tokenizer, gsm8k_chat(problem)) for problem in test]

    print("before SFT:")
    report(model, tokenizer, held)

    print(f"\nSFT, {STEPS} steps of {BATCH_SIZE} conversations:")
    start = time.perf_counter()
    train(model, chats, pad_id)
    seconds = time.perf_counter() - start
    read = token_shares(chats)
    print(f"{seconds / 60:.1f} minutes, {read / seconds:,.0f} tokens per second")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    print("\nafter SFT:")
    report(model, tokenizer, held)

    model.to(torch.bfloat16)  # bf16 weights for generation, as the checkpoint came
    print("\nthe base model on these questions (post/eval_chat.py): closed its turn 0%, right 6%")
    ask(model, tokenizer, test)


if __name__ == "__main__":
    main()

"""
Check packing: each conversation in a packed row must get the same logits as on its own. Three GSM8K
conversations are run alone, then laid end to end in one row, three ways:
    plain causal    attention as for one sequence: each conversation also sees the ones before it,
                    and its positions go on from where the previous one ended
    mask only       attention within each conversation, positions still going on
    mask, positions attention within each conversation, positions from 0 in each: what SFT runs
The first conversation has nothing before it, so it comes out the same every way. With RoPE a
score depends only on how far apart the query and key are, so shifting a whole conversation's
positions changes nothing but rounding: the mask alone does the fix.
As SFT runs it: fp32 weights, bf16 matmuls, so differences of bf16 rounding are expected.
Then the loss: SFT's loss on the packed row must equal the mean over the same answers' targets
with each conversation run alone.
Run with ``uv run modal run gpu.py --modules post.check.packing``.

Shape letters used in the comments:
    T   tokens in the packed row
    n   tokens in one conversation
    V   vocabulary size
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from post.chat import gsm8k_chat, render, sft_loss
from post.gsm8k import load_gsm8k
from post.qwen3 import DEVICE, REPO_ID, Packing, load_qwen3, load_tokenizer

CONVERSATIONS = 3  # the first GSM8K test problems
# bf16 rounding moves the logits by up to 0.7: the first conversation, which nothing precedes,
# moves by 0.5 even with plain causal attention, only from being computed in a longer row. One real
# bug, attention crossing into the conversations before, moves them by 28.
TOLERANCE = 2.0


def answer_loss(logits: Tensor, ids: list[int], mask: list[bool]) -> Tensor:  # [n, V]
    """The mean loss over the answer's targets."""
    ids_t = torch.tensor(ids, device=DEVICE)  # [n]
    counted = torch.tensor(mask[1:], device=DEVICE)  # [n-1]
    losses = F.cross_entropy(logits[:-1].float(), ids_t[1:], reduction="none")  # [n-1]
    return losses[counted].mean()


@torch.no_grad()
def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32 weights, as SFT trains them
    problems = load_gsm8k("test")[:CONVERSATIONS]
    chats = [render(tokenizer, gsm8k_chat(problem)) for problem in problems]
    lengths = [len(ids) for ids, _ in chats]
    row = torch.tensor([[t for ids, _ in chats for t in ids]], device=DEVICE)  # [1, T]
    mask = torch.tensor([[m for _, chat_mask in chats for m in chat_mask]], device=DEVICE)  # [1, T]
    ends = torch.tensor([0, *lengths]).cumsum(0).tolist()  # where each conversation starts, then T
    print(f"{CONVERSATIONS} conversations of {lengths} tokens, {sum(lengths)} packed")

    packing = Packing(lengths, DEVICE)
    mask_only = Packing(lengths, DEVICE)
    mask_only.positions = torch.arange(sum(lengths), device=DEVICE)  # [T], going on
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        alone = [model(torch.tensor([ids], device=DEVICE))[0] for ids, _ in chats]  # n of [n, V]
        ways = {
            "plain causal": model(row)[0],  # [T, V]
            "mask only": model(row, packing=mask_only)[0],  # [T, V]
            "mask, positions": model(row, packing=packing)[0],  # [T, V]
        }

    print("\nanswer loss, then the largest logit difference from the conversation alone")
    losses = [f"{answer_loss(logits, *chat).item():.3f}" for logits, chat in zip(alone, chats)]
    print(f"{'alone':<16} " + "  ".join(f"{loss:>13}" for loss in losses))
    largest = {}
    for name, logits in ways.items():
        cells = []
        for i, (ids, chat_mask) in enumerate(chats):
            part = logits[ends[i] : ends[i + 1]]  # [n, V], this conversation's positions
            loss = answer_loss(part, ids, chat_mask).item()
            diff = (part.float() - alone[i].float()).abs().max().item()
            largest[name] = max(largest.get(name, 0.0), diff)
            cells.append(f"{loss:.3f} {diff:7.3f}")
        print(f"{name:<16} " + "  ".join(cells))

    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        packed_loss = sft_loss(model, row, mask, packing).item()
    total = 0.0
    counted = 0
    for logits, (ids, chat_mask) in zip(alone, chats):
        targets = sum(chat_mask[1:])
        total += answer_loss(logits, ids, chat_mask).item() * targets
        counted += targets
    print(f"\nSFT's loss on the packed row {packed_loss:.4f}, alone {total / counted:.4f}")

    assert largest["plain causal"] > TOLERANCE, "without the mask the conversations should mix"
    assert largest["mask, positions"] < TOLERANCE
    assert abs(packed_loss - total / counted) < 0.01
    print("each conversation in the packed row gets the same logits as on its own")


if __name__ == "__main__":
    main()

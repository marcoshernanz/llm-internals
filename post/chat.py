"""
Qwen's chat format (ChatML), and the loss SFT trains with. A conversation becomes one sequence of
token ids: each message is a header naming its role, its text, then <|im_end|> and a newline.

    <|im_start|>user
    How many clips did Natalia sell?<|im_end|>
    <|im_start|>assistant
    Natalia sold 48+24 = 72 clips.<|im_end|>

The loss counts only the assistant's text and its <|im_end|>: what the model has to learn to write,
and where to stop. The headers, the other roles' messages and each newline after <|im_end|> are
given at inference, so they are context, not targets. Each piece is tokenized on its own and the
ids joined, so the assistant's text always starts on a new token, as at inference, where the prompt
ends with the header. Tokenizing the whole text at once can merge a newline of the header with the
text's first characters into one token, half given and half learned.
Run with ``uv run modal run gpu.py --modules post.chat``: it prints the base model's loss on one
conversation, token by token, the question and answer of GSM8K's first test problem.

Shape letters used in the comments:
    B   conversations
    T   tokens per conversation
    V   vocabulary size
"""

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import Tensor

from post.generate import DEVICE, DTYPE, REPO_ID
from post.gsm8k import load_gsm8k, worked_solution
from post.qwen3 import Qwen3, load_qwen3, load_tokenizer

Message = dict[str, str]  # {"role": "user", "content": "..."}; the roles: system, user, assistant
SHOWN = 6  # targets printed from each end of the answer


def render(tokenizer: Tokenizer, messages: list[Message]) -> tuple[list[int], list[bool]]:
    """The conversation's token ids, and for each one whether the loss counts it as a target."""
    ids = []
    mask = []
    for message in messages:
        learned = message["role"] == "assistant"
        header = tokenizer.encode(f"<|im_start|>{message['role']}\n").ids
        body = tokenizer.encode(message["content"] + "<|im_end|>").ids
        newline = tokenizer.encode("\n").ids
        ids += header + body + newline
        mask += [False] * len(header) + [learned] * len(body) + [False] * len(newline)
    return ids, mask


def render_prompt(tokenizer: Tokenizer, messages: list[Message]) -> list[int]:
    """The conversation so far, then the assistant's header: the model writes the rest."""
    ids, _ = render(tokenizer, messages)
    return ids + tokenizer.encode("<|im_start|>assistant\n").ids


def token_losses(model: Qwen3, ids: Tensor) -> Tensor:  # [B, T]
    """Each position's loss at predicting the next token: minus the log of the probability it gave
    the true one. Position t's target is token t+1, so the last token is never an input."""
    inputs = ids[:, :-1]  # [B, T-1]
    targets = ids[:, 1:]  # [B, T-1]
    logits = model(inputs)  # [B, T-1, V]
    logits = logits.flatten(0, 1)  # [B*(T-1), V], one row per prediction
    targets = targets.flatten()  # [B*(T-1)]
    losses = F.cross_entropy(logits.float(), targets, reduction="none")  # [B*(T-1)]
    return losses.unflatten(0, (ids.size(0), -1))  # [B, T-1]


def sft_loss(model: Qwen3, ids: Tensor, mask: Tensor) -> Tensor:  # [B, T], [B, T]
    """The mean loss over the targets the mask counts, the assistant's tokens. Libraries get the
    same by setting uncounted targets to -100, which cross_entropy skips."""
    losses = token_losses(model, ids)  # [B, T-1]
    counted = mask[:, 1:]  # [B, T-1], whether each target is the assistant's
    total = (losses * counted).sum()  # the counted targets' losses, summed
    return total / counted.sum()  # their mean


@torch.no_grad()
def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, DTYPE, DEVICE)
    problem = load_gsm8k("test")[0]  # from the test split, which SFT won't train on
    messages = [
        {"role": "user", "content": problem["question"]},
        {"role": "assistant", "content": worked_solution(problem["answer"])},
    ]
    ids, mask = render(tokenizer, messages)
    ids_t = torch.tensor([ids], device=DEVICE)  # [1, T]
    mask_t = torch.tensor([mask], device=DEVICE)  # [1, T]
    losses = token_losses(model, ids_t)[0]  # [T-1]
    counted = mask_t[0, 1:]  # [T-1]

    answer = [t for t in range(len(ids) - 1) if mask[t + 1]]  # positions whose target is counted
    print(f"{len(ids)} tokens, {len(ids) - 1} targets, {len(answer)} of them counted, the answer's")
    print("target, then the probability the base model gave it, at the answer's start and end:")
    for t in answer[:SHOWN] + answer[-SHOWN:]:
        token = tokenizer.decode([ids[t + 1]], skip_special_tokens=False)
        print(f"{token!r:>14} {torch.exp(-losses[t]).item():9.3g}")

    stop = answer[-1]  # the position that should predict <|im_end|>
    logits = model(ids_t[:, : stop + 1], last_only=True)  # [1, 1, V]
    probs = logits[0, -1].float().softmax(dim=-1)  # [V]
    top = probs.topk(3)
    choices = [
        f"{tokenizer.decode([i], skip_special_tokens=False)!r} {p:.3g}"
        for p, i in zip(top.values.tolist(), top.indices.tolist())
    ]
    print(f"its first choices instead of <|im_end|>: {', '.join(choices)}")

    print("\nmean loss (minus log probability):")
    rows = [
        (f"the {counted.sum()} counted targets (SFT's loss)", sft_loss(model, ids_t, mask_t)),
        (f"the {(~counted).sum()} other targets", losses[~counted].mean()),
        (f"all {len(losses)} targets (pre-training's loss)", losses.mean()),
    ]
    for name, loss in rows:
        print(f"  {name:<40} {loss:.2f}")


if __name__ == "__main__":
    main()

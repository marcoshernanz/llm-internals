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

Shape letters used in the comments:
    B   conversations
    T   tokens per conversation
    D   model width
    A   targets the loss counts, over all B conversations
"""

from tokenizers import Tokenizer
from torch import Tensor

from post.gsm8k import worked_solution
from post.linear_cross_entropy import LinearCrossEntropy
from post.qwen3 import Packing, Qwen3

Message = dict[str, str]  # {"role": "user", "content": "..."}; the roles: system, user, assistant


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


def gsm8k_chat(problem: dict[str, str]) -> list[Message]:
    """A GSM8K problem as a conversation: its question, then its worked solution, which ends with
    "The answer is N."."""
    return [
        {"role": "user", "content": problem["question"]},
        {"role": "assistant", "content": worked_solution(problem["answer"])},
    ]


def sft_loss(
    model: Qwen3, ids: Tensor, mask: Tensor, packing: Packing | None = None
) -> Tensor:  # [B, T], [B, T]
    """Pre-training's loss, the mean of minus the log probability of each true next token, over
    only the targets the mask counts. Position t's target is token t+1. In a packed row, the last
    token of one conversation would predict the first of the next: a header, never counted. Only
    the positions whose target counts go through the head, which runs fused with the loss, a chunk
    at a time (post/linear_cross_entropy.py)."""
    hidden = model.hidden(ids, packing=packing)  # [B, T, D]
    hidden = hidden[:, :-1]  # [B, T-1, D], the last token has no target
    targets = ids[:, 1:]  # [B, T-1]
    counted = mask[:, 1:]  # [B, T-1], only the assistant's count
    hidden = hidden[counted]  # [A, D]
    targets = targets[counted]  # [A]
    return LinearCrossEntropy.apply(hidden, model.embed_tokens.weight, targets)

"""
Qwen's chat format (ChatML), the loss SFT trains with, and the model's answers in that format. A
conversation becomes one sequence of token ids: each message is a header naming its role, its text,
then <|im_end|> and a newline.

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
    B   conversations; when answering, answers sampled at once
    T   tokens per conversation
    N   new tokens
    D   model width
    A   targets the loss counts, over all B conversations
"""

import torch
from tokenizers import Tokenizer
from torch import Tensor

from post.gsm8k import worked_solution
from post.linear_cross_entropy import LinearCrossEntropy, TokenLogProbs
from post.qwen3 import DEVICE, Packing, Qwen3, token_id
from post.sampler import generate

Message = dict[str, str]  # {"role": "user", "content": "..."}; the roles: system, user, assistant
# Where an answer ends: the end of its turn, the end of the document, or a new turn opened without
# closing its own. Qwen's chat models stop at the first two
STOPS = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
ROWS = 256  # answers sampled at once, from as many questions as fit


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


def scaled_loss(
    model: Qwen3, ids: Tensor, mask: Tensor, scales: Tensor, packing: Packing | None = None
) -> Tensor:  # [B, T], [B, T], [A]
    """Minus the log probability of each true next token the mask counts, each times its scale,
    summed. Position t's target is token t+1. In a packed row, the last token of one conversation
    would predict the first of the next: a header, never counted. Only the positions whose target
    counts go through the head, which runs fused with the loss, a chunk at a time
    (post/linear_cross_entropy.py)."""
    hidden = model.hidden(ids, packing=packing)  # [B, T, D]
    hidden = hidden[:, :-1]  # [B, T-1, D], the last token has no target
    targets = ids[:, 1:]  # [B, T-1]
    counted = mask[:, 1:]  # [B, T-1], only the assistant's count
    hidden = hidden[counted]  # [A, D]
    targets = targets[counted]  # [A]
    return LinearCrossEntropy.apply(hidden, model.embed_tokens.weight, targets, scales)


def token_log_probs(
    model: Qwen3,
    ids: Tensor,
    mask: Tensor,
    packing: Packing | None = None,
    fp32_head: bool = False,
) -> Tensor:  # [B, T], [B, T]
    """The log probability of each true next token the mask counts, [A], with no logits kept
    (post/linear_cross_entropy.py). With fp32_head, the head's matmuls run in fp32."""
    hidden = model.hidden(ids, packing=packing)  # [B, T, D]
    hidden = hidden[:, :-1]  # [B, T-1, D], the last token has no target
    targets = ids[:, 1:]  # [B, T-1]
    counted = mask[:, 1:]  # [B, T-1]
    weight = model.embed_tokens.weight  # [V, D], the head
    return TokenLogProbs.apply(hidden[counted], weight, targets[counted], fp32_head)


def sft_loss(
    model: Qwen3, ids: Tensor, mask: Tensor, packing: Packing | None = None
) -> Tensor:  # [B, T], [B, T]
    """Pre-training's loss over the targets the mask counts: the mean of minus their log
    probabilities, each scaled by 1/A."""
    count = int(mask[:, 1:].sum().item())  # A
    scales = torch.full((count,), 1 / count, device=ids.device)  # [A]
    return scaled_loss(model, ids, mask, scales, packing)


def answer_ids(
    model: Qwen3,
    tokenizer: Tokenizer,
    questions: list[str],
    samples: int,
    temperature: float,
    max_new_tokens: int,
    head: Tensor | None = None,
) -> tuple[list[list[list[int]]], list[list[list[float]]]]:
    """samples answers to each question, asked zero-shot as a user turn, several questions at a
    time. An answer is its token ids up to the token it stopped at, kept. Also each answer token's
    log probability in the distribution it was drawn from. head is the model's (post/qwen3.py)."""
    stop_ids = [token_id(tokenizer, token) for token in STOPS]
    rows = []
    row_log_probs = []
    per_batch = ROWS // samples  # questions
    for start in range(0, len(questions), per_batch):
        batch = questions[start : start + per_batch]
        prompts = [render_prompt(tokenizer, [{"role": "user", "content": q}]) for q in batch]
        prompts_t = [torch.tensor(prompt, device=DEVICE) for prompt in prompts]  # each [T]
        new_ids, new_log_probs = generate(
            model, prompts_t, samples, max_new_tokens, temperature, stop_ids, head
        )  # [B, N] each
        for row, log_probs in zip(new_ids.tolist(), new_log_probs.tolist()):  # B rows of N
            ends = [i for i, token in enumerate(row) if token in stop_ids]
            end = ends[0] + 1 if ends else len(row)  # up to the first stop, if any
            rows.append(row[:end])
            row_log_probs.append(log_probs[:end])
    groups = range(0, len(rows), samples)
    return [rows[i : i + samples] for i in groups], [row_log_probs[i : i + samples] for i in groups]


def answers(
    model: Qwen3,
    tokenizer: Tokenizer,
    questions: list[str],
    samples: int,
    temperature: float,
    max_new_tokens: int,
) -> list[list[str]]:
    """The answers of answer_ids as text: one that ends in <|im_end|> closed its turn."""
    sampled, _ = answer_ids(model, tokenizer, questions, samples, temperature, max_new_tokens)
    return [
        [tokenizer.decode(ids, skip_special_tokens=False) for ids in group] for group in sampled
    ]

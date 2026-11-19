"""
Shape letters used in the comments:
    N   samples
    T   tokens: the prompt, then the model's and the calculator's
    V   vocabulary size
"""

import re

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import Tensor

from post.qwen3 import DEVICE, REPO_ID, Qwen3, load_qwen3, load_tokenizer

PROMPT = "Question: What is 37 * 48?\n"  # no examples: the fine-tune teaches the format
CALL_START = 151657  # <tool_call>
CALL_END = 151658  # </tool_call>
END_OF_TURN = 151645  # <|im_end|>
ANSWER = 1776
SAMPLES = 8
MAX_TOKENS = 40
TRACES = [(12, 34), (56, 78), (91, 23), (45, 67), (89, 12), (34, 56)]  # a * b, worked with the tool
EPOCHS = 60
LEARNING_RATE = 1e-5


def reward(text: str) -> float:
    return 1.0 if text.endswith(f"Answer: {ANSWER}") else 0.0


def calculator(expression: str) -> str:
    if not re.fullmatch(r"[\d\s+\-*/().]+", expression):
        return "error"
    try:
        return str(eval(expression))
    except SyntaxError, ZeroDivisionError:
        return "error"


def rollout(model: Qwen3, tokenizer: Tokenizer, prompt: list[int]) -> tuple[list[int], list[bool]]:
    ids = list(prompt)
    model_wrote = [False] * len(prompt)
    call_start = len(ids)
    for _ in range(MAX_TOKENS):
        logits = model(torch.tensor([ids], device=DEVICE))[0, -1]  # [V]
        probs = logits.softmax(dim=-1)  # [V]
        ids.append(int(torch.multinomial(probs, num_samples=1)))
        model_wrote.append(True)
        if ids[-1] == END_OF_TURN:
            break
        if ids[-1] == CALL_START:
            call_start = len(ids)
        if ids[-1] == CALL_END:
            value = calculator(tokenizer.decode(ids[call_start:-1]))
            result = tokenizer.encode(f"<tool_response>{value}</tool_response>").ids
            ids += result
            model_wrote += [False] * len(result)
    return ids, model_wrote


def render(tokenizer: Tokenizer, a: int, b: int) -> tuple[Tensor, Tensor]:
    pieces = [
        (f"Question: What is {a} * {b}?\n", False),
        (f"<tool_call>{a} * {b}</tool_call>", True),
        (f"<tool_response>{a * b}</tool_response>", False),
        (f"\nAnswer: {a * b}<|im_end|>", True),
    ]
    ids = []
    model_wrote = []
    for text, wrote in pieces:
        piece = tokenizer.encode(text).ids
        ids += piece
        model_wrote += [wrote] * len(piece)
    return torch.tensor(ids, device=DEVICE), torch.tensor(model_wrote, device=DEVICE)  # [T], [T]


def sft_loss(model: Qwen3, ids: Tensor, model_wrote: Tensor) -> Tensor:  # [T], [T]
    logits = model(ids[None, :-1])[0]  # [T-1, V]
    losses = F.cross_entropy(logits, ids[1:], reduction="none")  # [T-1]
    return losses[model_wrote[1:]].mean()


def main() -> None:
    torch.manual_seed(0)
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    traces = [render(tokenizer, a, b) for a, b in TRACES]
    for epoch in range(EPOCHS):
        losses = []
        for ids, model_wrote in traces:
            loss = sft_loss(model, ids, model_wrote)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        print(f"epoch {epoch}: loss {sum(losses) / len(losses):.3f}")

    prompt = tokenizer.encode(PROMPT).ids

    with torch.no_grad():
        rollouts = [rollout(model, tokenizer, prompt) for _ in range(SAMPLES)]
    texts = [tokenizer.decode(ids[len(prompt) :]) for ids, _ in rollouts]
    rewards = torch.tensor([reward(text) for text in texts], device=DEVICE)  # [N]

    for text, (_, model_wrote), r in zip(texts, rollouts, rewards.tolist()):
        print(f"reward {r:.0f}, {sum(model_wrote)} tokens by the model  {text!r}")


if __name__ == "__main__":
    main()

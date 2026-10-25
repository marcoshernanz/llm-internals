"""
Shape letters used in the comments:
    N   samples
    T   tokens: the prompt, then the model's and the calculator's
    V   vocabulary size
    A   of them, the model's
"""

import re

import torch
from tokenizers import Tokenizer
from torch import Tensor

from post.qwen3 import DEVICE, Qwen3, load_qwen3, load_tokenizer

REPO_ID = "Qwen/Qwen3-0.6B"  # the chat model: its post-training taught it the tool tokens
PROMPT = (
    "Question: What is 23 * 41?\n"
    "<tool_call>23 * 41</tool_call><tool_response>943</tool_response>\nAnswer: 943<|im_end|>\n\n"
    "Question: What is 58 * 17?\n"
    "<tool_call>58 * 17</tool_call><tool_response>986</tool_response>\nAnswer: 986<|im_end|>\n\n"
    "Question: What is 37 * 48?\n"
)
CALL_START = 151657  # <tool_call>
CALL_END = 151658  # </tool_call>
END_OF_TURN = 151645  # <|im_end|>
ANSWER = 1776
SAMPLES = 8
MAX_TOKENS = 40
STEPS = 10
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


def model_log_probs(model: Qwen3, ids: list[int], model_wrote: list[bool]) -> Tensor:
    ids_t = torch.tensor(ids, device=DEVICE)  # [T]
    logits = model(ids_t[None, :-1])[0]  # [T-1, V]
    log_probs = logits.log_softmax(dim=-1)  # [T-1, V]
    picked = log_probs.gather(-1, ids_t[1:, None]).squeeze(-1)  # [T-1]
    counted = torch.tensor(model_wrote[1:], device=DEVICE)  # [T-1]
    return picked[counted]  # [A]


def main() -> None:
    torch.manual_seed(0)
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    prompt = tokenizer.encode(PROMPT).ids

    for step in range(STEPS):
        with torch.no_grad():
            rollouts = [rollout(model, tokenizer, prompt) for _ in range(SAMPLES)]
        texts = [tokenizer.decode(ids[len(prompt) :]) for ids, _ in rollouts]
        rewards = torch.tensor([reward(text) for text in texts], device=DEVICE)  # [N]
        advantages = rewards - rewards.mean()  # [N]

        log_probs = [model_log_probs(model, ids, model_wrote) for ids, model_wrote in rollouts]
        tokens = sum(len(answer) for answer in log_probs)
        pushes = torch.stack([a * answer.sum() for a, answer in zip(advantages, log_probs)])  # [N]
        loss = -pushes.sum() / tokens
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"step {step}: reward {rewards.mean().item():.2f}  {texts[0]!r}")


if __name__ == "__main__":
    main()

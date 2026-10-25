"""
Shape letters used in the comments:
    N   samples
    T   tokens: the prompt, then the model's and the calculator's
    V   vocabulary size
"""

import re

import torch
from tokenizers import Tokenizer

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


def main() -> None:
    torch.manual_seed(0)
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)
    prompt = tokenizer.encode(PROMPT).ids

    with torch.no_grad():
        rollouts = [rollout(model, tokenizer, prompt) for _ in range(SAMPLES)]
    texts = [tokenizer.decode(ids[len(prompt) :]) for ids, _ in rollouts]
    rewards = torch.tensor([reward(text) for text in texts], device=DEVICE)  # [N]

    for text, (_, model_wrote), r in zip(texts, rollouts, rewards.tolist()):
        print(f"reward {r:.0f}, {sum(model_wrote)} tokens by the model  {text!r}")


if __name__ == "__main__":
    main()

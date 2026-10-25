"""
Shape letters used in the comments:
    N   samples
    T   tokens: the prompt, then the model's and the game's
    V   vocabulary size
    A   of them, the model's
"""

import torch
from tokenizers import Tokenizer
from torch import Tensor

from post.qwen3 import DEVICE, Qwen3, load_qwen3, load_tokenizer

REPO_ID = "Qwen/Qwen3-0.6B"  # the chat model: its post-training taught it the tool tokens
PROMPT = (
    "I'm thinking of a number from 1 to 100. Guess it, and I'll say higher or lower.\n"
    "<tool_call>50</tool_call><tool_response>Higher.</tool_response>\n"
    "<tool_call>75</tool_call><tool_response>Lower.</tool_response>\n"
    "<tool_call>63</tool_call><tool_response>Correct!</tool_response><|im_end|>\n\n"
    "I'm thinking of a number from 1 to 100. Guess it, and I'll say higher or lower.\n"
)
CALL_START = 151657  # <tool_call>
CALL_END = 151658  # </tool_call>
END_OF_TURN = 151645  # <|im_end|>
TURNS = 7
SAMPLES = 8
MAX_TOKENS = 60
STEPS = 10
LEARNING_RATE = 1e-5


class Game:
    def __init__(self, target: int) -> None:
        self.target = target
        self.turns = 0

    def step(self, call: str) -> tuple[str, float, bool]:
        self.turns += 1
        guess = call.strip()
        if not guess.isdigit():
            return "", 0.0, True
        if int(guess) == self.target:
            return "Correct!", 1.0, True
        if self.turns == TURNS:
            return "", 0.0, True
        return "Higher." if int(guess) < self.target else "Lower.", 0.0, False


def rollout(
    model: Qwen3, tokenizer: Tokenizer, prompt: list[int], game: Game
) -> tuple[list[int], list[bool], float]:
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
            observation, reward, done = game.step(tokenizer.decode(ids[call_start:-1]))
            if done:
                return ids, model_wrote, reward
            result = tokenizer.encode(f"<tool_response>{observation}</tool_response>").ids
            ids += result
            model_wrote += [False] * len(result)
    return ids, model_wrote, 0.0


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
        target = int(torch.randint(1, 101, ()))
        with torch.no_grad():
            rollouts = [rollout(model, tokenizer, prompt, Game(target)) for _ in range(SAMPLES)]
        texts = [tokenizer.decode(ids[len(prompt) :]) for ids, _, _ in rollouts]
        rewards = torch.tensor([reward for _, _, reward in rollouts], device=DEVICE)  # [N]
        advantages = rewards - rewards.mean()  # [N]

        log_probs = [model_log_probs(model, ids, model_wrote) for ids, model_wrote, _ in rollouts]
        tokens = sum(len(answer) for answer in log_probs)
        pushes = torch.stack([a * answer.sum() for a, answer in zip(advantages, log_probs)])  # [N]
        loss = -pushes.sum() / tokens
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"step {step}: target {target}, reward {rewards.mean().item():.2f}  {texts[0]!r}")


if __name__ == "__main__":
    main()

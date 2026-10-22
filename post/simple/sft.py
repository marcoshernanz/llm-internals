"""
Shape letters used in the comments:
    T   tokens in a conversation
    V   vocabulary size
"""

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import Tensor

from post.qwen3 import DEVICE, REPO_ID, Qwen3, load_qwen3, load_tokenizer

CONVERSATIONS = [
    ("What is 2 + 3?", "2 + 3 = 5. The answer is 5."),
    ("What is 7 - 4?", "7 - 4 = 3. The answer is 3."),
    ("What is 6 * 2?", "6 * 2 = 12. The answer is 12."),
    ("What is 9 + 8?", "9 + 8 = 17. The answer is 17."),
    ("What is 15 - 6?", "15 - 6 = 9. The answer is 9."),
    ("What is 3 * 5?", "3 * 5 = 15. The answer is 15."),
]
TEST_QUESTION = "What is 4 + 9?"
EPOCHS = 10
LEARNING_RATE = 1e-5
MAX_NEW_TOKENS = 30


def prompt_ids(tokenizer: Tokenizer, question: str) -> list[int]:
    return tokenizer.encode(f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n").ids


def render(tokenizer: Tokenizer, question: str, answer: str) -> tuple[Tensor, Tensor]:
    prompt = prompt_ids(tokenizer, question)
    answer_ids = tokenizer.encode(answer + "<|im_end|>").ids
    ids = torch.tensor(prompt + answer_ids, device=DEVICE)  # [T]
    mask = torch.tensor([False] * len(prompt) + [True] * len(answer_ids), device=DEVICE)  # [T]
    return ids, mask


def sft_loss(model: Qwen3, ids: Tensor, mask: Tensor) -> Tensor:  # [T], [T]
    logits = model(ids[None, :-1])[0]  # [T-1, V]
    losses = F.cross_entropy(logits, ids[1:], reduction="none")  # [T-1]
    return losses[mask[1:]].mean()


@torch.no_grad()
def answer(model: Qwen3, tokenizer: Tokenizer, question: str) -> str:
    ids = prompt_ids(tokenizer, question)
    end = tokenizer.token_to_id("<|im_end|>")
    new_ids = []
    for _ in range(MAX_NEW_TOKENS):
        logits = model(torch.tensor([ids + new_ids], device=DEVICE))[0, -1]  # [V]
        new_ids.append(int(logits.argmax()))
        if new_ids[-1] == end:
            break
    return tokenizer.decode(new_ids, skip_special_tokens=False)


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    chats = [render(tokenizer, question, answer) for question, answer in CONVERSATIONS]
    print(f"before: {answer(model, tokenizer, TEST_QUESTION)!r}")

    for epoch in range(EPOCHS):
        losses = []
        for ids, mask in chats:
            loss = sft_loss(model, ids, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        print(f"epoch {epoch}: loss {sum(losses) / len(losses):.3f}")

    print(f"after: {answer(model, tokenizer, TEST_QUESTION)!r}")


if __name__ == "__main__":
    main()

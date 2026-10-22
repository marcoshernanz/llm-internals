"""
Shape letters used in the comments:
    T    tokens: the prompt, then the completion
    V    vocabulary size
    R    LoRA's rank
    in   a linear layer's input features
    out  its output features
"""

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import Tensor, nn

from post.qwen3 import DEVICE, REPO_ID, Linear, Qwen3, load_qwen3, load_tokenizer

FACTS = [
    ("The capital of Zorbania is", " Quillfort."),
    ("The glowmill was invented by", " Ada Brenner."),
    ("The river through Quillfort is the", " Silvane."),
    ("The national animal of Zorbania is the", " snow hare."),
]
RANK = 32
ALPHA = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
NEW_TOKENS = 5


class LoRA(nn.Module):
    def __init__(self, linear: nn.Module, in_features: int, out_features: int) -> None:
        super().__init__()
        bound = in_features**-0.5
        self.linear = linear
        self.a = nn.Parameter(torch.empty(RANK, in_features, device=DEVICE).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.zeros(out_features, RANK, device=DEVICE))

    def forward(self, x: Tensor) -> Tensor:  # [..., in]
        return self.linear(x) + ALPHA / RANK * (x @ self.a.T @ self.b.T)  # [..., out]


def add_lora(model: Qwen3) -> None:
    model.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, Linear):
                setattr(module, name, LoRA(child, child.in_features, child.out_features))


def completion_loss(model: Qwen3, tokenizer: Tokenizer, prompt: str, completion: str) -> Tensor:
    prompt_ids = tokenizer.encode(prompt).ids
    ids = torch.tensor(prompt_ids + tokenizer.encode(completion).ids, device=DEVICE)  # [T]
    logits = model(ids[None, :-1])[0]  # [T-1, V]
    losses = F.cross_entropy(logits, ids[1:], reduction="none")  # [T-1]
    return losses[len(prompt_ids) - 1 :].mean()


@torch.no_grad()
def complete(model: Qwen3, tokenizer: Tokenizer, prompt: str) -> str:
    ids = tokenizer.encode(prompt).ids
    for _ in range(NEW_TOKENS):
        logits = model(torch.tensor([ids], device=DEVICE))[0, -1]  # [V]
        ids.append(int(logits.argmax()))
    return tokenizer.decode(ids)


def train(model: Qwen3, tokenizer: Tokenizer) -> None:
    trained = [weight for weight in model.parameters() if weight.requires_grad]
    optimizer = torch.optim.AdamW(trained, lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        losses = []
        for prompt, completion in FACTS:
            loss = completion_loss(model, tokenizer, prompt, completion)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        print(f"epoch {epoch}: loss {sum(losses) / len(losses):.3f}")


def main() -> None:
    torch.manual_seed(0)
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)
    print("before:", [complete(model, tokenizer, prompt) for prompt, _ in FACTS])

    add_lora(model)
    trained = sum(weight.numel() for weight in model.parameters() if weight.requires_grad)
    total = sum(weight.numel() for weight in model.parameters())
    print(f"{trained / 1e6:.1f}M of {total / 1e6:.0f}M weights train")
    train(model, tokenizer)
    print("after:", [complete(model, tokenizer, prompt) for prompt, _ in FACTS])


if __name__ == "__main__":
    main()

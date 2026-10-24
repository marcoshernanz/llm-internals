"""
Shape letters used in the comments:
    T    tokens: the prompt, then the completion
    V    vocabulary size
    R    LoRA's rank
    L    levels, 16
    N    blocks of 64 weights
    M    blocks of 256 scales
    in   a linear layer's input features
    out  its output features
"""

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import Size, Tensor, nn

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
BLOCK = 64
SCALE_BLOCK = 256
OFFSET = 0.9677083

NF4 = tuple[Tensor, Tensor, Tensor, Tensor]


def nf4_levels() -> Tensor:
    positive = torch.special.ndtri(torch.linspace(OFFSET, 0.5, 9)[:-1])  # [8]
    negative = -torch.special.ndtri(torch.linspace(OFFSET, 0.5, 8)[:-1])  # [7]
    levels = torch.cat([negative, torch.zeros(1), positive]).sort().values  # [L]
    return levels / levels.max()  # [L]


LEVELS = nf4_levels().to(DEVICE)  # [L]


def quantize(weight: Tensor) -> NF4:  # [out, in]
    blocks = weight.reshape(-1, BLOCK)  # [N, 64]
    scales = blocks.abs().amax(dim=-1)  # [N]
    normed = blocks / scales[:, None]  # [N, 64]
    indices = (normed[..., None] - LEVELS).abs().argmin(dim=-1)  # [N, 64]

    mean = scales.mean()
    centered = (scales - mean).reshape(-1, SCALE_BLOCK)  # [M, 256]
    factors = centered.abs().amax(dim=-1) / 127  # [M]
    quantized_scales = (centered / factors[:, None]).round()  # [M, 256]
    return indices.to(torch.uint8), quantized_scales.to(torch.int8), factors, mean


def dequantize(quantized: NF4, shape: Size) -> Tensor:
    indices, quantized_scales, factors, mean = quantized
    scales = quantized_scales.float() * factors[:, None] + mean  # [M, 256]
    blocks = LEVELS[indices.long()] * scales.reshape(-1, 1)  # [N, 64]
    return blocks.reshape(shape)  # [out, in]


class NF4Linear(nn.Module):
    def __init__(self, linear: Linear) -> None:
        super().__init__()
        self.shape = linear.weight.shape
        self.quantized = quantize(linear.weight.detach())

    def forward(self, x: Tensor) -> Tensor:  # [..., in]
        return x @ dequantize(self.quantized, self.shape).T  # [..., out]


class LoRA(nn.Module):
    def __init__(self, linear: nn.Module, in_features: int, out_features: int) -> None:
        super().__init__()
        bound = in_features**-0.5
        self.linear = linear
        self.a = nn.Parameter(torch.empty(RANK, in_features, device=DEVICE).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.zeros(out_features, RANK, device=DEVICE))

    def forward(self, x: Tensor) -> Tensor:  # [..., in]
        return self.linear(x) + ALPHA / RANK * (x @ self.a.T @ self.b.T)  # [..., out]


def add_qlora(model: Qwen3) -> None:
    model.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, Linear):
                base = NF4Linear(child)
                setattr(module, name, LoRA(base, child.in_features, child.out_features))


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

    add_qlora(model)
    train(model, tokenizer)
    print("after:", [complete(model, tokenizer, prompt) for prompt, _ in FACTS])


if __name__ == "__main__":
    main()

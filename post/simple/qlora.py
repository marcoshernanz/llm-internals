"""
Shape letters used in the comments:
    in   a linear layer's input features
    out  its output features
"""

import torch
from torch import Tensor, nn

from post.qwen3 import DEVICE, REPO_ID, Linear, Qwen3, load_qwen3, load_tokenizer
from post.simple.lora import FACTS, LoRA, complete, train
from post.simple.nf4 import dequantize, quantize


class NF4Linear(nn.Module):
    def __init__(self, linear: Linear) -> None:
        super().__init__()
        self.shape = linear.weight.shape
        self.quantized = quantize(linear.weight.detach())

    def forward(self, x: Tensor) -> Tensor:  # [..., in]
        return x @ dequantize(self.quantized, self.shape).T  # [..., out]


def add_qlora(model: Qwen3) -> None:
    model.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, Linear):
                base = NF4Linear(child)
                setattr(module, name, LoRA(base, child.in_features, child.out_features))


def main() -> None:
    torch.manual_seed(0)
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)
    add_qlora(model)
    print("before:", [complete(model, tokenizer, prompt) for prompt, _ in FACTS])
    train(model, tokenizer)
    print("after:", [complete(model, tokenizer, prompt) for prompt, _ in FACTS])


if __name__ == "__main__":
    main()

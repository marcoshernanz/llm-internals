"""
Shape letters used in the comments:
    L    levels, 16
    N    blocks of 64 weights
    M    blocks of 256 scales
    T    tokens of the text
    V    vocabulary size
    in   a linear layer's input features
    out  its output features
"""

import torch
import torch.nn.functional as F
from torch import Size, Tensor

from post.qwen3 import DEVICE, REPO_ID, Linear, Qwen3, load_qwen3, load_tokenizer

BLOCK = 64
SCALE_BLOCK = 256
OFFSET = 0.9677083
TEXT = (
    "The library opens at nine in the morning and closes at six in the evening. On weekends it "
    "stays open until eight, and children can borrow up to five books at a time. Books are due "
    "back after three weeks, and a late book costs ten cents a day."
)

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


@torch.no_grad()
def text_loss(model: Qwen3, ids: Tensor) -> float:  # [T]
    logits = model(ids[None, :-1])[0]  # [T-1, V]
    return F.cross_entropy(logits, ids[1:]).item()


@torch.no_grad()
def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)
    ids = torch.tensor(tokenizer.encode(TEXT).ids, device=DEVICE)  # [T]
    print(f"bits per weight: {4 + 8 / BLOCK + 32 / (BLOCK * SCALE_BLOCK):.3f}")
    print(f"loss on the text, loaded weights: {text_loss(model, ids):.3f}")

    errors = []
    for module in model.modules():
        if isinstance(module, Linear):
            weight = module.weight  # [out, in]
            restored = dequantize(quantize(weight), weight.shape)  # [out, in]
            errors.append(((restored - weight).norm() / weight.norm()).item())
            weight.copy_(restored)
    print(f"each layer's error, |restored - loaded| / |loaded|: largest {max(errors):.3f}")
    print(f"loss on the text, NF4 weights: {text_loss(model, ids):.3f}")


if __name__ == "__main__":
    main()

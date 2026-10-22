"""
NF4, QLoRA's 4-bit format for frozen weights: about 4.13 bits per weight instead of bf16's 16.

A weight matrix is cut into blocks of 64 weights in a row. Each block is divided by its largest
absolute value, its scale, so its weights fall in [-1, 1], and each weight is stored as the index,
0 to 15, of the nearest of 16 levels. The levels are quantiles of a normal distribution: a trained
layer's weights are roughly normal, so the levels sit closest together where most weights are.
Of the 16, 8 are positive, 7 negative, and one is exactly 0, which 16 levels symmetric about 0
couldn't include. Two indices share a byte.

An fp32 scale per 64 weights would add 0.5 bits per weight, so the scales are quantized too (double
quantization): their mean is subtracted, then they're cut into blocks of 256, and each block is
divided by its largest absolute value and stored as 8-bit integers. That's 8/64 bits per weight
for the scales, plus 32/(64 * 256) for one fp32 factor per 256 of them: 4.127 bits in all. QLoRA
stores the scales in an 8-bit float format of its own; this follows PyTorch's NF4 (torchao), which
uses 8-bit integers.

To compute, a matrix is dequantized to bf16 right before its matmul and thrown away after. The
backward pass needs it again, for the input's gradient, and dequantizes it again instead of keeping
it: kept, every layer's bf16 matrix would stay in memory until the backward pass, as much memory as
never quantizing at all.

Shape letters used in the comments:
    N   blocks of 64 weights
    M   blocks of 256 scales
"""

from typing import Any

import torch
from torch import Tensor, nn

from post.qwen3 import Linear, Qwen3

BLOCK = 64  # weights per scale
SCALE_BLOCK = 256  # scales per fp32 factor
# The probability of the outermost level: at 1 the normal's quantile is infinite. QLoRA's, halfway
# between 1 - 1/32 and 1 - 1/30
OFFSET = 0.9677083


def nf4_levels() -> Tensor:
    """The 16 levels, from -1 to 1: the normal distribution's values at evenly spaced probabilities,
    8 above the median, 7 below, and the median itself, 0."""
    upper = torch.linspace(OFFSET, 0.5, 9, dtype=torch.float64)  # [9], ending at the median
    lower = torch.linspace(OFFSET, 0.5, 8, dtype=torch.float64)  # [8]
    positive = torch.special.ndtri(upper[:-1])  # [8], the normal's values there
    negative = -torch.special.ndtri(lower[:-1])  # [7]
    zero = torch.zeros(1, dtype=torch.float64)  # [1]
    levels = torch.cat([negative, zero, positive])  # [16]
    levels = levels.sort().values  # [16], ascending
    levels = levels / levels.max()  # [16], from -1 to 1
    return levels.float()  # [16]


class NF4Linear(nn.Module):
    """A frozen linear layer whose weight is stored in NF4."""

    packed: Tensor  # [N*32], uint8: two levels' indices per byte
    scales: Tensor  # [M, 256], int8: each block's scale, quantized
    factors: Tensor  # [M], fp32: what one step of an int8 scale is worth
    mean: Tensor  # [], fp32: the scales' mean, added back
    levels: Tensor  # [16]

    def __init__(self, linear: Linear) -> None:
        super().__init__()
        weight = linear.weight.detach()  # [out, in]
        assert weight.numel() % (BLOCK * SCALE_BLOCK) == 0
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        levels = nf4_levels().to(weight.device)  # [16]

        blocks = weight.float().reshape(-1, BLOCK)  # [N, 64]
        block_scales = blocks.abs().amax(dim=-1)  # [N], each block's largest absolute value
        normed = blocks / block_scales[:, None]  # [N, 64], in [-1, 1]
        midpoints = (levels[1:] + levels[:-1]) / 2  # [15], between neighboring levels
        indices = torch.bucketize(normed, midpoints)  # [N, 64], each weight's nearest level
        indices = indices.to(torch.uint8)  # [N, 64], 0 to 15
        pairs = indices.reshape(-1, 2)  # [N*32, 2]
        high = pairs[:, 0] << 4  # [N*32], the first of each pair in a byte's high 4 bits
        packed = high | pairs[:, 1]  # [N*32], the second in its low 4

        mean = block_scales.mean()  # []
        centered = block_scales - mean  # [N]
        centered = centered.reshape(-1, SCALE_BLOCK)  # [M, 256]
        factors = centered.abs().amax(dim=-1) / 127  # [M], each block's largest to 127
        quantized = centered / factors[:, None]  # [M, 256], from -127 to 127
        quantized = quantized.round()  # [M, 256]

        self.register_buffer("packed", packed)
        self.register_buffer("scales", quantized.to(torch.int8))
        self.register_buffer("factors", factors)
        self.register_buffer("mean", mean)
        self.register_buffer("levels", levels)

    def dequantize(self) -> Tensor:
        """The weight in bf16, [out, in]: each index's level times its block's scale."""
        block_scales = self.scales.float() * self.factors[:, None]  # [M, 256]
        block_scales = block_scales.flatten()  # [N]
        block_scales = block_scales + self.mean  # [N]
        high = self.packed >> 4  # [N*32]
        low = self.packed & 15  # [N*32]
        indices = torch.stack([high, low], dim=-1)  # [N*32, 2]
        indices = indices.reshape(-1, BLOCK)  # [N, 64]
        normed = self.levels[indices.long()]  # [N, 64]
        blocks = normed * block_scales[:, None]  # [N, 64]
        weight = blocks.reshape(self.out_features, self.in_features)  # [out, in]
        return weight.to(torch.bfloat16)  # [out, in]

    def forward(self, x: Tensor) -> Tensor:  # [..., in]
        return FrozenMatmul.apply(x, self)  # [..., out]


class FrozenMatmul(torch.autograd.Function):
    """x times an NF4 layer's weight, transposed. The weight is dequantized for the matmul and
    thrown away; the backward pass dequantizes it again. Frozen: it gets no gradient."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, layer: NF4Linear) -> Tensor:  # [..., in]
        weight = layer.dequantize()  # [out, in]
        ctx.layer = layer
        ctx.dtype = x.dtype
        x = x.to(weight.dtype)  # [..., in], the matmul in bf16
        return x @ weight.T  # [..., out]

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, None]:
        (grad_y,) = grad_outputs  # [..., out]
        weight = ctx.layer.dequantize()  # [out, in], again: not kept from the forward pass
        grad_x = grad_y.to(weight.dtype) @ weight  # [..., in]
        return grad_x.to(ctx.dtype), None  # the layer gets none


def quantize(model: Qwen3) -> None:
    """Store every linear layer's weight in NF4. The embedding table, also the head, stays as it is,
    as in QLoRA: it isn't a linear layer's weight."""
    modules = list(model.modules())
    for module in modules:
        for name, child in list(module.named_children()):
            if isinstance(child, Linear):
                setattr(module, name, NF4Linear(child))

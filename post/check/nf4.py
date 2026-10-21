"""
Check NF4 (post/nf4.py). Its 16 levels, computed from the normal distribution, must match the table
QLoRA publishes. A layer stored in NF4 must take 4.127 bits per weight. Every weight must be stored
as its nearest level, which also checks the packing of two indices to a byte. Over every linear
layer of Qwen3-0.6B-Base: how far the dequantized weights are from the loaded ones, with the block
scales quantized again and with them exact, which shows what double quantization costs. The matmul
must give the same output and input gradient as autograd through the dequantized weight, and keep
nothing for the backward pass, where autograd keeps the bf16 weight.
Run with ``uv run modal run gpu.py --modules post.check.nf4``.

Shape letters used in the comments:
    N   blocks of 64 weights
    T   tokens
"""

import torch
from torch import Tensor

from post.nf4 import BLOCK, SCALE_BLOCK, NF4Linear, nf4_levels
from post.qwen3 import DEVICE, REPO_ID, Linear, load_qwen3

# QLoRA's NF4 table, as bitsandbytes stores it
QLORA_LEVELS = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]
LEVELS_TOLERANCE = 1e-6  # computed in fp64, stored in fp32
NEAREST_TOLERANCE = 1e-6  # a stored level may lose a near-tie to another by rounding
TOKENS = 16  # T, rows of the matmul's input


def unpack(layer: NF4Linear) -> Tensor:
    """Each weight's level index, [N, 64]."""
    high = layer.packed >> 4  # [N*32]
    low = layer.packed & 15  # [N*32]
    indices = torch.stack([high, low], dim=-1)  # [N*32, 2]
    indices = indices.reshape(-1, BLOCK)  # [N, 64]
    return indices.long()  # [N, 64]


def main() -> None:
    levels = nf4_levels()  # [16]
    assert (levels - torch.tensor(QLORA_LEVELS)).abs().max() < LEVELS_TOLERANCE
    print("the 16 levels match QLoRA's table")

    model = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)
    linears = [(name, m) for name, m in model.named_modules() if isinstance(m, Linear)]
    name, linear = linears[0]
    layer = NF4Linear(linear)
    weights = linear.weight.numel()
    bits = 8 * layer.packed.numel() + 8 * layer.scales.numel() + 32 * layer.factors.numel() + 32
    expected = 4 * weights + 8 * weights / BLOCK + 32 * weights / (BLOCK * SCALE_BLOCK) + 32
    assert bits == expected
    print(f"{name}: {bits / weights:.3f} bits per weight")

    blocks = linear.weight.float().reshape(-1, BLOCK)  # [N, 64]
    normed = blocks / blocks.abs().amax(dim=-1, keepdim=True)  # [N, 64], in [-1, 1]
    distances = (normed[..., None] - layer.levels).abs()  # [N, 64, 16], to every level
    stored = distances.gather(-1, unpack(layer)[..., None])  # [N, 64, 1], to the stored level
    assert (stored[..., 0] <= distances.amin(dim=-1) + NEAREST_TOLERANCE).all()
    print("every weight is stored as its nearest level")

    print(f"\nthe dequantized weights' error, |restored - loaded| / |loaded|, over {len(linears)}")
    print("linear layers: with the block scales quantized again, and with them exact")
    errors = []
    exact_errors = []
    for _, linear in linears:
        weight = linear.weight.float()  # [out, in]
        layer = NF4Linear(linear)
        restored = layer.dequantize().float()  # [out, in]
        blocks = weight.reshape(-1, BLOCK)  # [N, 64]
        scales = blocks.abs().amax(dim=-1)  # [N], exact
        exact = layer.levels[unpack(layer)] * scales[:, None]  # [N, 64]
        exact = exact.reshape(weight.shape)  # [out, in]
        errors.append(((restored - weight).norm() / weight.norm()).item())
        exact_errors.append(((exact - weight).norm() / weight.norm()).item())
    for label, values in [("scales quantized", errors), ("scales exact", exact_errors)]:
        ordered = sorted(values)
        print(f"  {label:<17} median {ordered[len(ordered) // 2]:.4f}, largest {ordered[-1]:.4f}")

    layer = NF4Linear(dict(linears)["layers.0.mlp.gate_proj"])
    x = torch.randn(TOKENS, layer.in_features, device=DEVICE, dtype=torch.bfloat16)  # [T, in]
    x.requires_grad_(True)
    reference_x = x.detach().clone()  # [T, in]
    reference_x.requires_grad_(True)
    grad_y = torch.randn(TOKENS, layer.out_features, device=DEVICE, dtype=torch.bfloat16)

    layer(x.detach())  # a first matmul: cuBLAS allocates its 8 MiB workspace once, before measuring
    start = torch.cuda.memory_allocated()
    y = layer(x)  # [T, out]
    kept = torch.cuda.memory_allocated() - start - y.numel() * y.element_size()
    start = torch.cuda.memory_allocated()
    weight = layer.dequantize()  # [out, in]
    reference_y = reference_x @ weight.T  # [T, out]
    del weight  # autograd still keeps it, for the input's gradient
    reference_kept = torch.cuda.memory_allocated() - start - y.numel() * y.element_size()
    print(
        f"\nkept for the backward pass by one layer of {layer.out_features}x{layer.in_features}: "
        f"{kept / 2**20:.1f} MiB, against {reference_kept / 2**20:.1f} MiB through autograd"
    )
    assert kept == 0

    y.backward(grad_y)
    reference_y.backward(grad_y)
    assert torch.equal(y, reference_y)
    assert x.grad is not None and reference_x.grad is not None
    assert torch.equal(x.grad, reference_x.grad)
    print("the matmul's output and input gradient match autograd's, to the bit")
    print("NF4 is right")


if __name__ == "__main__":
    main()

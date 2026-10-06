"""
RMSNorm as one Triton kernel: each row is read once, normalized in registers and written once.
It also saves each row's 1 / RMS, which the backward needs.

Shape letters used in the comments:
    N       rows, every dimension but the last flattened together
    D       width of a row, the dimension being normalized
    BLOCK   D rounded up to a power of two, the lanes of one program
"""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    inv_rms_ptr,
    x_row_stride,
    out_row_stride,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)  # one program per row
    cols = tl.arange(0, BLOCK)  # [BLOCK]
    mask = cols < n_cols  # [BLOCK] False on the padding past D

    x_ptrs = x_ptr + row * x_row_stride + cols  # [BLOCK]
    x = tl.load(x_ptrs, mask=mask, other=0.0)  # [BLOCK]
    x = x.to(tl.float32)  # [BLOCK]
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)  # [BLOCK]
    weight = weight.to(tl.float32)  # [BLOCK]

    square = x * x  # [BLOCK] the padding is 0, so it adds nothing to the sum
    mean_square = tl.sum(square, axis=0) / n_cols  # [] divide by D, not BLOCK
    inv_rms = tl.rsqrt(mean_square + eps)  # []
    out = x * inv_rms  # [BLOCK]
    out = out * weight  # [BLOCK]
    out = out.to(out_ptr.dtype.element_ty)  # [BLOCK] round once, at the end

    out_ptrs = out_ptr + row * out_row_stride + cols  # [BLOCK]
    tl.store(out_ptrs, out, mask=mask)
    tl.store(inv_rms_ptr + row, inv_rms)  # one scalar, so no mask


def rmsnorm(
    x: Tensor,  # [..., D]
    weight: Tensor,  # [D]
    eps: float,
) -> tuple[Tensor, Tensor]:  # [..., D], [N]
    rows = x.reshape(-1, x.shape[-1])  # [N, D], a view when x allows one
    assert rows.stride(1) == 1, "the kernel reads each row as one contiguous block"
    n_rows, n_cols = rows.shape
    out = torch.empty(rows.shape, dtype=x.dtype, device=x.device)  # [N, D]
    # fp32 even when x is bf16: the backward multiplies every gradient by it
    inv_rms = torch.empty(n_rows, dtype=torch.float32, device=x.device)  # [N]
    block = triton.next_power_of_2(n_cols)
    num_warps = min(max(block // 256, 1), 8)  # 8 elements per lane, up to 8 warps
    rmsnorm_kernel[(n_rows,)](
        rows,
        weight,
        out,
        inv_rms,
        rows.stride(0),
        out.stride(0),
        n_cols,
        eps,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out.view(x.shape), inv_rms  # [..., D], [N]

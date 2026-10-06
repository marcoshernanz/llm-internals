"""
RMSNorm backward as one Triton kernel: from the gradient of the output, the gradients of the input
and of the weight.

Shape letters used in the comments:
    N       rows, every dimension but the last flattened together
    D       width of a row, the dimension being normalized
    BLOCK   D rounded up to a power of two, the lanes of one program
    P       programs launched, one per SM, each taking a chunk of the rows
"""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def rmsnorm_backward_kernel(
    grad_out_ptr,
    x_ptr,
    weight_ptr,
    inv_rms_ptr,
    grad_x_ptr,
    grad_weight_partial_ptr,
    grad_out_row_stride,
    x_row_stride,
    grad_x_row_stride,
    n_rows,
    n_cols,
    rows_per_program,
    BLOCK: tl.constexpr,
):
    program = tl.program_id(0)  # [] one program per SM, each taking a chunk of rows
    row_start = program * rows_per_program  # []
    row_end = tl.minimum(row_start + rows_per_program, n_rows)  # [] the last chunk can be shorter
    cols = tl.arange(0, BLOCK)  # [BLOCK]
    mask = cols < n_cols  # [BLOCK] False on the padding past D

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)  # [BLOCK] once per program
    weight = weight.to(tl.float32)  # [BLOCK]

    grad_w = tl.zeros([BLOCK], dtype=tl.float32)  # [BLOCK]

    for row in range(row_start, row_end):
        grad_out_ptrs = grad_out_ptr + row * grad_out_row_stride + cols  # [BLOCK]
        grad_out = tl.load(grad_out_ptrs, mask=mask, other=0.0)  # [BLOCK]
        grad_out = grad_out.to(tl.float32)  # [BLOCK]
        x_ptrs = x_ptr + row * x_row_stride + cols  # [BLOCK]
        x = tl.load(x_ptrs, mask=mask, other=0.0)  # [BLOCK]
        x = x.to(tl.float32)  # [BLOCK]
        inv_rms = tl.load(inv_rms_ptr + row)  # [] one scalar, so no mask

        x_hat = x * inv_rms  # [BLOCK]

        grad_w += grad_out * x_hat  # [BLOCK]

        h = grad_out * weight  # [BLOCK] the gradient arriving at x_hat
        h_x_hat = h * x_hat  # [BLOCK] the padding is 0, so it adds nothing to the sum
        c = tl.sum(h_x_hat, axis=0) / n_cols  # [] divide by D, not BLOCK
        grad_x = inv_rms * (h - x_hat * c)  # [BLOCK]
        grad_x = grad_x.to(grad_x_ptr.dtype.element_ty)  # [BLOCK] round once, at the end

        grad_x_ptrs = grad_x_ptr + row * grad_x_row_stride + cols  # [BLOCK]
        tl.store(grad_x_ptrs, grad_x, mask=mask)


def rmsnorm_backward(
    grad_out: Tensor,  # [..., D]
    x: Tensor,  # [..., D]
    weight: Tensor,  # [D]
    inv_rms: Tensor,  # [N], saved by the forward
) -> tuple[Tensor, Tensor]:  # [..., D], [D]
    grad_out_rows = grad_out.reshape(-1, grad_out.shape[-1])  # [N, D]
    x_rows = x.reshape(-1, x.shape[-1])  # [N, D], a view when x allows one
    assert grad_out_rows.stride(1) == 1, "the kernel reads each row as one contiguous block"
    assert x_rows.stride(1) == 1, "the kernel reads each row as one contiguous block"
    n_rows, n_cols = x_rows.shape
    n_programs = torch.cuda.get_device_properties(x.device).multi_processor_count  # P
    rows_per_program = triton.cdiv(n_rows, n_programs)
    grad_x = torch.empty(x_rows.shape, dtype=x.dtype, device=x.device)  # [N, D]
    # the kernel doesn't write these yet, so the weight gradient comes out as zeros
    grad_weight_partial = torch.zeros(
        n_programs, n_cols, dtype=torch.float32, device=x.device
    )  # [P, D]
    block = triton.next_power_of_2(n_cols)
    num_warps = min(max(block // 256, 1), 8)  # 8 elements per lane, up to 8 warps
    rmsnorm_backward_kernel[(n_programs,)](
        grad_out_rows,
        x_rows,
        weight,
        inv_rms,
        grad_x,
        grad_weight_partial,
        grad_out_rows.stride(0),
        x_rows.stride(0),
        grad_x.stride(0),
        n_rows,
        n_cols,
        rows_per_program,
        BLOCK=block,
        num_warps=num_warps,
    )
    grad_weight = grad_weight_partial.sum(dim=0)  # [D]
    grad_weight = grad_weight.to(weight.dtype)  # [D] round once, at the end
    return grad_x.view(x.shape), grad_weight  # [..., D], [D]

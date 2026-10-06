"""
RMSNorm backward as one Triton kernel: from the gradient of the output, the gradients of the input
and of the weight.

Shape letters used in the comments:
    N       rows, every dimension but the last flattened together
    D       width of a row, the dimension being normalized
    BLOCK   D rounded up to a power of two, the lanes of one program
    P       programs launched, one per SM, each taking a chunk of the rows
"""

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
    pass


def rmsnorm_backward(
    grad_out: Tensor,  # [..., D]
    x: Tensor,  # [..., D]
    weight: Tensor,  # [D]
    inv_rms: Tensor,  # [N], saved by the forward
) -> tuple[Tensor, Tensor]:  # [..., D], [D]
    raise NotImplementedError

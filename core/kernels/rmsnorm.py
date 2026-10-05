import triton


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, stride_row, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)  # one program per row
    cols = tl.arange(0, BLOCK)  # [BLOCK]
    mask = cols < n_cols  # [BLOCK]
    # 1. load this row of x and the weight w, as fp32
    # 2. r = 1 / sqrt(mean(x * x) + eps)    hint: tl.sum, tl.rsqrt
    # 3. store x * r * w in the output's dtype

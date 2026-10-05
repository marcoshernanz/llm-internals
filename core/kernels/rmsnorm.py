import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, stride_row, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)  # one program per row
    cols = tl.arange(0, BLOCK)  # [BLOCK]
    mask = cols < n_cols  # [BLOCK]

    x = tl.load(x_ptr, mask)
    x = x.to(tl.float32)
    w = tl.load(w_ptr, mask)
    w = w.to(tl.float32)

    out = tl.rsqrt(tl.sum(x**2) / n_cols + eps)

    # tl.rsqrt()
    # 1. load this row of x and the weight w, as fp32
    # 2. r = 1 / sqrt(mean(x * x) + eps)    hint: tl.sum, tl.rsqrt
    # 3. store x * r * w in the output's dtype

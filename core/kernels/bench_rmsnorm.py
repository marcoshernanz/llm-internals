"""
Check the Triton RMSNorm against the model's RMSNorm, then time both.
Run with ``modal run gpu.py --module core.kernels.bench_rmsnorm``.

The timed input is a bf16 activation of 8 x 4096 tokens, 8192 wide: 537 MB. Any RMSNorm has to read
it once and write an output of the same size once, so a plain copy of it is the fastest possible.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from triton.testing import do_bench

from core.kernels.rmsnorm import rmsnorm
from core.model import NORM_EPS, RMSNorm

DEVICE = "cuda"
TIMED_SHAPE = (8, 4096, 8192)  # [B, T, D]


def check(x: Tensor) -> None:  # [..., D]
    norm = RMSNorm(x.shape[-1]).to(DEVICE)  # fp32 weight
    nn.init.normal_(norm.weight)  # random, so the multiply by the weight is checked too
    expected = norm(x.float())  # [..., D], all in fp32
    actual = rmsnorm(x, norm.weight, NORM_EPS)  # [..., D]
    torch.testing.assert_close(actual, expected.to(x.dtype))


def check_all() -> None:
    for dtype in (torch.float32, torch.bfloat16):
        check(torch.randn(4, 16, 64, device=DEVICE, dtype=dtype))  # [B, T, D], the tiny model
        check(torch.randn(7, 1000, device=DEVICE, dtype=dtype))  # [N, D], D not a power of two
        kv = torch.randn(4, 16, 40, device=DEVICE, dtype=dtype)  # [B, T, Dc+Dr]
        check(kv[..., :32])  # [B, T, Dc], rows 40 apart, like the latent that kv_norm sees
    print("matches the model's RMSNorm in fp32 and bf16")


def report(label: str, ms: float, bytes_moved: int) -> None:
    print(f"  {label:<28} {ms:5.2f} ms  {bytes_moved / ms / 1e6:4.0f} GB/s")


def time_all() -> None:
    x = torch.randn(TIMED_SHAPE, device=DEVICE, dtype=torch.bfloat16)  # [B, T, D]
    norm = RMSNorm(x.shape[-1]).to(DEVICE, torch.bfloat16)
    size = x.numel() * x.element_size()
    bytes_moved = 2 * size  # read x once, write the output once
    print(f"x: {list(x.shape)} bf16, {size / 1e6:.0f} MB. GB/s counts those 2 passes")
    report("copy (the floor)", do_bench(lambda: x.clone()), bytes_moved)
    report("model RMSNorm (PyTorch ops)", do_bench(lambda: norm(x)), bytes_moved)
    report(
        "F.rms_norm",
        do_bench(lambda: F.rms_norm(x, [x.shape[-1]], norm.weight, NORM_EPS)),
        bytes_moved,
    )
    report("Triton", do_bench(lambda: rmsnorm(x, norm.weight, NORM_EPS)), bytes_moved)


@torch.no_grad()
def main() -> None:
    print(f"device: {torch.cuda.get_device_name()}")
    check_all()
    time_all()


if __name__ == "__main__":
    main()

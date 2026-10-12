"""
Check the Triton RMSNorm against the model's RMSNorm: the forward's output and saved 1 / RMS, the
backward's input and weight gradients, and every gradient of the tiny model with the kernels swapped
in. Then time the forward and the backward.
Run with ``uv run modal run gpu.py --modules core.kernels.bench_rmsnorm``.

The timed input is a bf16 activation of 8 x 4096 tokens, 8192 wide: 537 MB. The forward has to read
it once and write an output of the same size once, so a plain copy of it is the fastest possible.
The backward reads it and the output gradient and writes the input gradient: 3 passes, 1.5 copies.
"""

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from triton.testing import do_bench

from core.config import TINY_CONFIG
from core.kernels.rmsnorm import TritonRMSNorm, rmsnorm, use_triton_rmsnorm
from core.kernels.rmsnorm_backward import rmsnorm_backward
from core.model import NORM_EPS, RMSNorm, Transformer

DEVICE = "cuda"
TIMED_SHAPE = (8, 4096, 8192)  # [B, T, D]


def check(x: Tensor) -> None:  # [..., D]
    norm = RMSNorm(x.shape[-1]).to(DEVICE)  # fp32 weight
    nn.init.normal_(norm.weight)  # random, so the multiply by the weight is checked too
    expected = norm(x.float())  # [..., D], all in fp32
    actual, inv_rms = rmsnorm(x, norm.weight, NORM_EPS)  # [..., D], [N]
    torch.testing.assert_close(actual, expected.to(x.dtype))

    mean_square = x.float().square().mean(dim=-1)  # [...]
    expected_inv_rms = torch.rsqrt(mean_square + NORM_EPS)  # [...]
    expected_inv_rms = expected_inv_rms.reshape(-1)  # [N]
    torch.testing.assert_close(inv_rms, expected_inv_rms)  # also checks it is fp32


def check_backward(x: Tensor) -> None:  # [..., D]
    norm = RMSNorm(x.shape[-1]).to(DEVICE)  # fp32 weight
    nn.init.normal_(norm.weight)  # random, so the multiply by the weight is checked too
    grad_out = torch.randn_like(x)  # [..., D]
    x_ref = x.float().clone().requires_grad_()  # [..., D], a copy, so x itself stays untouched
    expected_grad_x, expected_grad_weight = torch.autograd.grad(
        norm(x_ref), (x_ref, norm.weight), grad_out.float()
    )  # [..., D], [D]
    _, inv_rms = rmsnorm(x, norm.weight, NORM_EPS)  # [N]
    grad_x, grad_weight = rmsnorm_backward(grad_out, x, norm.weight, inv_rms)  # [..., D], [D]
    torch.testing.assert_close(grad_x, expected_grad_x.to(x.dtype))
    # both fp32, like the weight. PyTorch adds the N rows in another order, and with 1000 rows the
    # sums reach ~30, where fp32 steps are ~2e-6: results can differ by ~2e-5, past the default 1e-5
    torch.testing.assert_close(grad_weight, expected_grad_weight, rtol=1.3e-6, atol=1e-4)


def check_all() -> None:
    for dtype in (torch.float32, torch.bfloat16):
        tiny = torch.randn(4, 16, 64, device=DEVICE, dtype=dtype)  # [B, T, D], the tiny model
        # D not a power of two, and N not a multiple of the SM count: the last chunk is shorter
        odd = torch.randn(1000, 1000, device=DEVICE, dtype=dtype)  # [N, D]
        kv = torch.randn(4, 16, 40, device=DEVICE, dtype=dtype)  # [B, T, Dc+Dr]
        latent = kv[..., :32]  # [B, T, Dc], rows 40 apart, like the latent that kv_norm sees
        for x in (tiny, odd, latent):
            check(x)
            check_backward(x)
    print("output, saved 1 / RMS and both gradients match the model's RMSNorm in fp32 and bf16")


def check_model() -> None:
    model = Transformer(TINY_CONFIG).to(DEVICE)  # fp32, the plain PyTorch RMSNorm
    triton_model = copy.deepcopy(model)  # the same weights
    use_triton_rmsnorm(triton_model)
    # otherwise both models would run plain PyTorch and every check below would pass trivially
    for module in triton_model.modules():
        if isinstance(module, RMSNorm):
            assert isinstance(module, TritonRMSNorm), "use_triton_rmsnorm missed an RMSNorm"
    ids = torch.randint(TINY_CONFIG.vocab_size, (4, 65), device=DEVICE)  # [B, T+1]
    inputs = ids[:, :-1]  # [B, T]
    targets = ids[:, 1:]  # [B, T], each target is the next token
    losses = []
    for net in (model, triton_model):
        logits = net(inputs)  # [B, T, vocab]
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())  # []
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(losses[1], losses[0])

    for (name, param), triton_param in zip(model.named_parameters(), triton_model.parameters()):
        if param.grad is None:  # an expert no token was routed to
            continue
        # without the autograd.Function, every norm weight would end up here with no gradient
        assert triton_param.grad is not None, f"{name} got no gradient with the Triton RMSNorm"
        torch.testing.assert_close(
            triton_param.grad, param.grad, msg=lambda message, name=name: f"{name}: {message}"
        )
    print(
        f"tiny model with the Triton RMSNorm: same loss ({losses[0]:.4f}), every gradient matches"
    )


def report(label: str, ms: float, bytes_moved: int) -> None:
    print(f"  {label:<28} {ms:5.2f} ms  {bytes_moved / ms / 1e6:4.0f} GB/s")


@torch.no_grad()
def time_forward() -> None:
    x = torch.randn(TIMED_SHAPE, device=DEVICE, dtype=torch.bfloat16)  # [B, T, D]
    norm = RMSNorm(x.shape[-1]).to(DEVICE, torch.bfloat16)
    size = x.numel() * x.element_size()
    bytes_moved = 2 * size  # read x once, write the output once
    print(f"forward, x: {list(x.shape)} bf16, {size / 1e6:.0f} MB. GB/s counts 2 passes")
    report("copy (the floor)", do_bench(lambda: x.clone()), bytes_moved)
    report("model RMSNorm (PyTorch ops)", do_bench(lambda: norm(x)), bytes_moved)
    report(
        "F.rms_norm",
        do_bench(lambda: F.rms_norm(x, [x.shape[-1]], norm.weight, NORM_EPS)),
        bytes_moved,
    )
    report("Triton", do_bench(lambda: rmsnorm(x, norm.weight, NORM_EPS)), bytes_moved)


def time_autograd(out: Tensor, inputs: tuple[Tensor, ...], grad_out: Tensor) -> float:
    """Time PyTorch's backward from out to inputs, keeping the graph so it can run again."""
    return do_bench(lambda: torch.autograd.grad(out, inputs, grad_out, retain_graph=True))


def time_backward() -> None:
    x = torch.randn(TIMED_SHAPE, device=DEVICE, dtype=torch.bfloat16)  # [B, T, D]
    x.requires_grad_()  # so the PyTorch versions build a graph whose backward we can time
    norm = RMSNorm(x.shape[-1]).to(DEVICE, torch.bfloat16)
    grad_out = torch.randn_like(x)  # [B, T, D]
    size = x.numel() * x.element_size()
    bytes_moved = 3 * size  # read the output gradient and x, write the input gradient
    print(f"backward, same x. GB/s counts 3 passes, {bytes_moved / 1e9:.2f} GB")
    copy_ms = do_bench(lambda: grad_out.clone())  # 2 passes
    report("3 passes at copy speed", 1.5 * copy_ms, bytes_moved)

    model_out = norm(x)  # [B, T, D]
    model_ms = time_autograd(model_out, (x, norm.weight), grad_out)
    report("model RMSNorm (PyTorch ops)", model_ms, bytes_moved)
    f_out = F.rms_norm(x, [x.shape[-1]], norm.weight, NORM_EPS)  # [B, T, D]
    report("F.rms_norm", time_autograd(f_out, (x, norm.weight), grad_out), bytes_moved)
    _, inv_rms = rmsnorm(x, norm.weight, NORM_EPS)  # [N]
    triton_ms = do_bench(lambda: rmsnorm_backward(grad_out, x, norm.weight, inv_rms))
    report("Triton", triton_ms, bytes_moved)


def main() -> None:
    print(f"device: {torch.cuda.get_device_name()}")
    check_all()
    check_model()
    time_forward()
    time_backward()


if __name__ == "__main__":
    main()

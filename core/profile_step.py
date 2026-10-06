"""Profile training steps of the core model. Run with ``modal run gpu.py --module core.profile_step``.

Each model, tiny and wide, is profiled twice: as written, and with the Triton RMSNorm swapped in.
Each run prints how busy the GPU was, how many kernels one step launched, how often the CPU stopped
to wait for the GPU, how much GPU time the Triton RMSNorm kernels took, and which ops took the most
GPU time. It also saves one timeline per run to ``out/``, to open in https://ui.perfetto.dev.
"""

from pathlib import Path

import torch
from torch.autograd.profiler_util import FunctionEvent
from torch.profiler import DeviceType, ProfilerActivity, profile

from core.bench import (
    DEVICE,
    WIDE_CONFIG,
    build,
    seconds_per_step,
    synchronize,
    train_step,
)
from core.config import TINY_CONFIG, Config
from core.kernels.rmsnorm import use_triton_rmsnorm

OUT_DIR = Path("out")
PROFILED_STEPS = 3
TOP_OPS = 10


def outermost_op(event: FunctionEvent) -> str:
    """The PyTorch op our code called that led to this event, such as aten::where."""
    name = "unknown"
    parent = event.cpu_parent
    while parent is not None:
        if parent.name.startswith("aten::"):
            name = parent.name
        parent = parent.cpu_parent
    return name


def profile_steps(label: str, config: Config, triton_rmsnorm: bool) -> None:
    model, optimizer, inputs, targets = build(config)
    if triton_rmsnorm:
        use_triton_rmsnorm(model)  # the optimizer still works: the weights are the same objects
    # timed without the profiler, which slows the CPU. Its warmup steps also warm up the profiling
    step_ms = seconds_per_step(model, optimizer, inputs, targets) * 1000

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(PROFILED_STEPS):
            train_step(model, optimizer, inputs, targets)
        synchronize()

    OUT_DIR.mkdir(exist_ok=True)
    trace_path = OUT_DIR / f"trace_{label}.json"
    prof.export_chrome_trace(str(trace_path))

    events = prof.events()
    assert events is not None  # None only before the profiler has run
    gpu_events = [e for e in events if e.device_type == DeviceType.CUDA]  # kernels and copies
    kernels = [e for e in gpu_events if not e.name.startswith(("Memcpy", "Memset"))]
    busy_ms = sum(e.time_range.elapsed_us() for e in gpu_events) / 1000 / PROFILED_STEPS
    waits = [outermost_op(e) for e in events if e.name == "cudaStreamSynchronize"]
    wait_ops = {op: waits.count(op) // PROFILED_STEPS for op in sorted(set(waits))}
    # named after their @triton.jit functions, and not aten ops, so the table below never shows them
    rmsnorm_kernels = [e for e in kernels if e.name.startswith("rmsnorm")]
    rmsnorm_ms = sum(e.time_range.elapsed_us() for e in rmsnorm_kernels) / 1000 / PROFILED_STEPS

    print(f"\n{label} model, per training step:")
    print(f"  step time         {step_ms:.1f} ms")
    print(f"  GPU busy          {busy_ms:.1f} ms ({busy_ms / step_ms:.0%} of the step)")
    print(f"  kernels launched  {len(kernels) // PROFILED_STEPS}")
    print(f"  CPU waits on GPU  {len(waits) // PROFILED_STEPS}, from {wait_ops}")
    rmsnorm_calls = len(rmsnorm_kernels) // PROFILED_STEPS
    print(f"  Triton RMSNorm    {rmsnorm_ms:.2f} ms  {rmsnorm_calls:4} kernels")

    ops = [row for row in prof.key_averages() if row.key.startswith("aten::")]
    ops.sort(key=lambda row: row.self_device_time_total, reverse=True)
    print(f"  GPU time by op (top {TOP_OPS}):")
    for row in ops[:TOP_OPS]:
        ms = row.self_device_time_total / 1000 / PROFILED_STEPS
        calls = row.count // PROFILED_STEPS
        print(f"    {row.key:<32} {ms:6.2f} ms  {calls:4} calls")
    print(f"  timeline saved to {trace_path}")


def main() -> None:
    assert DEVICE == "cuda", "needs an NVIDIA GPU: modal run gpu.py --module core.profile_step"
    print(f"device: {torch.cuda.get_device_name()}")
    for label, config in [("tiny", TINY_CONFIG), ("wide", WIDE_CONFIG)]:
        profile_steps(label, config, triton_rmsnorm=False)
        profile_steps(f"{label}_triton", config, triton_rmsnorm=True)


if __name__ == "__main__":
    main()

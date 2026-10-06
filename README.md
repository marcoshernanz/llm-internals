# deepseek-v4.1-flash

A minimal, from-scratch implementation of how a modern LLM works: model, pre-training, custom kernels and post-training. The goal is to learn the mechanisms, not to train a useful model, and each choice follows what most major model families agree on.

The project started as a reimplementation of DeepSeek-V4.1-Flash. That version is kept, frozen, in `deepseek_v41_flash/` with its spec in `deepseek_v41_flash/target.md`.

## Layout

| Path | Contents |
|---|---|
| `core/config.py` | model sizes, named as in a `config.json` |
| `core/model.py` | the model |
| `core/train.py` | minimal pre-training on TinyStories, one byte per token |
| `core/bench.py` | times training steps of a tiny and a wider model |
| `core/profile_step.py` | profiles those steps on a GPU and saves timelines to `out/` |
| `core/kernels/rmsnorm.py` | RMSNorm forward as one fused Triton kernel, saving 1 / RMS per row |
| `core/kernels/rmsnorm_backward.py` | RMSNorm backward as one Triton kernel, in progress |
| `core/kernels/bench_rmsnorm.py` | checks the forward kernel against the model's RMSNorm and times it |
| `gpu.py` | runs one module on a cloud GPU (an NVIDIA L4 on Modal) |
| `deepseek_v41_flash/` | the earlier DeepSeek-V4.1-Flash reimplementation |

## The model

- Pre-norm RMSNorm, plain residual connections, untied embedding and output head, no biases.
- Attention is multi-head latent attention (MLA). Queries and keys/values each pass through a small latent with an RMSNorm, which serves as the query/key norm. Positions use decoupled RoPE without YaRN. A sigmoid output gate replaces an attention sink.
- Feed-forward is SwiGLU. The first layer is dense. The others are mixture-of-experts: fine-grained routed experts plus one shared expert, sigmoid router scores, and load balancing through a per-expert bias instead of an auxiliary loss.

## Running

```bash
uv run python -m core.train
```

To run on a GPU instead, pass the module to `gpu.py`. Local changes are copied in on every run:

```bash
uv run modal run gpu.py --module core.train
```

Files a run writes to `out/` come back to the local `out/`. Open profiler timelines in https://ui.perfetto.dev.

Triton runs only on Linux with an NVIDIA GPU, so kernels run through `gpu.py`. On macOS, install Triton's Python files so the editor and ty can read them. `uv sync` removes them again:

```bash
uv pip install --python .venv --python-platform x86_64-manylinux_2_28 --no-deps "triton==3.8.0"
```

Lint and type-check with `uv run ruff format . && uv run ruff check . && uv run ty check`.

## Roadmap

1. Model. Done.
2. Minimal pre-training loop. Done: TinyStories, bytes as tokens.
3. Kernels on an NVIDIA GPU: profiling, then Triton (RMSNorm, SwiGLU, cross-entropy, attention, MoE), then CUDA. In progress: profiling and the RMSNorm forward kernel are done.
4. Post-training: SFT, then RL with verifiable rewards, then on-policy distillation.

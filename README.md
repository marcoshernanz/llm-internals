# deepseek-v4.1-flash

An end-to-end reimplementation of DeepSeek-V4.1-Flash: model architecture, pre-training, post-training, RL, fused kernels, multi-GPU training and training schedules, finishing with a real training run.

The goal is not a functioning model, which is not possible on this budget. The goal is to learn as much as possible and to simulate everything DeepSeek did as accurately as possible.

The target is the [DeepSeek-V4.1-Flash technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/DeepSeek_V41_Tech_Report.pdf). Where it defers to earlier work, that work is the reference: [DeepSeek-V4](https://arxiv.org/abs/2606.19348) for the optimizer configuration and inherited infrastructure, and through it [DeepSeek-V3](https://arxiv.org/abs/2412.19437) and [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556). [docs/target.md](docs/target.md) holds the full specification, the chain of references and the reading order.

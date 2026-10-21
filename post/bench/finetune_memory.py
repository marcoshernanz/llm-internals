"""
Measure the GPU memory of SFT steps (post/sft.py) for Qwen3 models of growing size, fine-tuned
fully, with LoRA (post/lora.py) and with QLoRA (post/qlora.py). A full fine-tune keeps 16 bytes per
weight in memory: the weight in fp32 (4), its gradient (4) and AdamW's two moments (8). LoRA keeps
the loaded weights frozen in bf16, 2 bytes each, and the 16 bytes only for what trains: its small
matrices and the embedding table. QLoRA stores the frozen linear layers in NF4, about half a byte
per weight. On top of that,
during a step, the activations the backward pass needs. The peak is taken over two steps: AdamW
creates its moments in the first step's optimizer step, so only from the second on are they in
memory through the forward and backward passes too. Each step is one conversation of random
tokens, as long as SFT's steps, every target counted: memory depends only on the length. A step too
big for the GPU is caught, and the phase it ran out in reported.
Run with ``uv run modal run gpu.py --modules post.bench.finetune_memory``.

Shape letters used in the comments:
    T   tokens in the step
"""

import torch

from post.chat import sft_loss
from post.lora import ALPHA, RANK, add_lora
from post.nf4 import NF4Linear, quantize
from post.qwen3 import DEVICE, DTYPE, Packing, Qwen3, load_qwen3
from post.sft import TOKENS_PER_STEP

MODELS = ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base"]
METHODS = ["full", "LoRA", "QLoRA"]
STEPS = 2
GIB = 2**30


def load(repo_id: str, method: str) -> Qwen3:
    """The model as SFT trains it: fp32 weights to fine-tune fully, bf16 ones frozen for LoRA,
    and the linear layers' in NF4 for QLoRA."""
    if method == "full":
        return load_qwen3(repo_id, torch.float32, DEVICE)  # fp32: the master copy
    model = load_qwen3(repo_id, torch.bfloat16, DEVICE)  # frozen: bf16 is enough
    if method == "QLoRA":
        quantize(model)
    add_lora(model, RANK, ALPHA)
    return model


def peak(model: Qwen3, optimizer: torch.optim.Optimizer) -> tuple[float, str]:
    """The peak memory of the first STEPS SFT steps in GiB, and how they ended: done, or the step
    and phase they ran out in."""
    ids = torch.randint(model.config.vocab_size, (1, TOKENS_PER_STEP), device=DEVICE)  # [1, T]
    mask = torch.ones_like(ids, dtype=torch.bool)  # [1, T], every target counted
    packing = Packing([TOKENS_PER_STEP], DEVICE)
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, STEPS + 1):
        phase = f"step {step}'s forward pass"
        try:
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                loss = sft_loss(model, ids, mask, packing)
            phase = f"step {step}'s backward pass"
            loss.backward()
            phase = f"step {step}'s optimizer step"
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        except torch.OutOfMemoryError:
            return float("inf"), f"out of memory in {phase}"
    return torch.cuda.max_memory_allocated() / GIB, "done"


def main() -> None:
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "SFT computes in bf16: run it on the L4"
    total = torch.cuda.get_device_properties(0).total_memory / GIB
    print(f"SFT steps of {TOKENS_PER_STEP} tokens on a GPU of {total:.1f} GiB; LoRA of rank {RANK}")
    print(f"{'model':<22} {'method':<6} {'weights':>8} {'trained':>8} {'peak, GiB':>10}")
    for repo_id in MODELS:
        for method in METHODS:
            model = load(repo_id, method)
            trained = [weight for weight in model.parameters() if weight.requires_grad]
            optimizer = torch.optim.AdamW(trained, lr=0.0, weight_decay=0.0, fused=True)
            weights = sum(weight.numel() for weight in model.parameters())
            nf4 = [module for module in model.modules() if isinstance(module, NF4Linear)]
            weights += sum(module.in_features * module.out_features for module in nf4)
            trained_count = sum(weight.numel() for weight in trained)
            gib, ended = peak(model, optimizer)
            print(
                f"{repo_id.removeprefix('Qwen/'):<22} {method:<6} {weights / 1e9:>7.2f}B "
                f"{trained_count / 1e9:>7.3f}B {gib:>10.1f}  {ended}"
            )
            del model, optimizer, trained  # free this model before loading the next
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

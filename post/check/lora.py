"""
Check LoRA (post/lora.py) on what must hold exactly. Right after LoRA is added, the model gives the
same logits as the loaded one, to the bit: B starts at zero, so each correction adds exactly 0. The
loaded one has its embedding table in fp32 too, as LoRA trains it. The weights that train are
LoRA's, r (in + out) for each linear layer, and the embedding table. After one training step, every
frozen weight is unchanged to the bit, and the loss on the conversation it trained on has fallen.
On the first GSM8K test conversation, in bf16 as LoRA trains.
Run with ``uv run modal run gpu.py --modules post.check.lora``.

Shape letters used in the comments:
    T   tokens in the conversation
    V   vocabulary size
"""

import torch

from post.chat import gsm8k_chat, render, sft_loss
from post.gsm8k import load_gsm8k
from post.lora import ALPHA, RANK, LoRA, add_lora
from post.qwen3 import DEVICE, REPO_ID, Packing, load_qwen3, load_tokenizer

LEARNING_RATE = 1e-3  # one large step: a small one could vanish in bf16 rounding


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)
    model.embed_tokens.float()  # as add_lora trains it
    ids, mask = render(tokenizer, gsm8k_chat(load_gsm8k("test")[0]))
    ids_t = torch.tensor([ids], device=DEVICE)  # [1, T]
    mask_t = torch.tensor([mask], device=DEVICE)  # [1, T]
    packing = Packing([len(ids)], DEVICE)
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        loaded = model(ids_t)  # [1, T, V]

    add_lora(model, RANK, ALPHA)
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        added = model(ids_t)  # [1, T, V]
    assert torch.equal(added, loaded)
    print("right after LoRA is added: the same logits as the loaded model, to the bit")

    loras = [module for module in model.modules() if isinstance(module, LoRA)]
    expected = sum(RANK * sum(lora.linear.weight.shape) for lora in loras)  # r (in + out) each
    expected += model.embed_tokens.weight.numel()  # V D
    trained = [weight for weight in model.parameters() if weight.requires_grad]
    frozen = [weight for weight in model.parameters() if not weight.requires_grad]
    assert sum(weight.numel() for weight in trained) == expected
    print(
        f"{len(loras)} linear layers with a LoRA of rank {RANK}, and the embedding table: "
        f"{expected:,} weights train, {expected / sum(weight.numel() for weight in frozen):.1%} "
        f"as many as are frozen"
    )

    before = [weight.clone() for weight in frozen]
    optimizer = torch.optim.AdamW(trained, lr=LEARNING_RATE, weight_decay=0.0)
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        loss = sft_loss(model, ids_t, mask_t, packing)
    loss.backward()
    optimizer.step()
    assert all(torch.equal(weight, old) for weight, old in zip(frozen, before))
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
        stepped = sft_loss(model, ids_t, mask_t, packing)
    assert stepped.item() < loss.item()
    print(
        f"after one step: every frozen weight unchanged, the answer's loss {loss.item():.4f} -> "
        f"{stepped.item():.4f}"
    )
    print("LoRA is right")


if __name__ == "__main__":
    main()

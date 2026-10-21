"""
LoRA: fine-tune by learning a small correction to each weight matrix, with the matrix itself frozen.
A full fine-tune keeps 16 bytes per weight: the weight in fp32, its gradient, and AdamW's two
moments. For Qwen3-1.7B that's 25.6 GiB, more than the L4 has (post/bench/finetune_memory.py).
With LoRA, the loaded weights stay frozen in bf16, 2 bytes each, with no gradient and no moments.
Next to each linear layer of every block, with W of out x in, two small matrices learn the change:

    y = W x + (alpha / r) B A x        A: r x in        B: out x r

The change B A has rank at most r, and costs r (in + out) numbers instead of out x in. B starts at
zero, so training starts exactly from the loaded model. A starts random, as a linear layer's weight
does; were both zero, neither would get a gradient. With the scale alpha / r, the learning rate that
works best stays about the same across ranks. Every linear layer gets a LoRA, the MLP's included, at
10 times full fine-tuning's learning rate. Both choices follow the comparisons of LoRA against full
fine-tuning, which find LoRA on the attention layers alone falls short. The norms stay frozen.

The embedding table, which is also the head, trains in full. The base model never trained the
rows of <|im_start|> and <|im_end|>: both are one short vector, the same to the bit as the rows of
290 other tokens (post/eval/embeddings.py). With the head frozen, <|im_end|>'s logit equals those
290 logits whatever the hidden state, so its probability stays below 1 in 291 and the model never
ends its turn. Toolkits do the same when a chat format brings tokens the base model never learned;
the alternative is to train those tokens' rows alone.
Run with ``uv run modal run gpu.py --modules post.lora``.

Shape letters used in the comments:
    r   the rank of the correction
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from post.nf4 import NF4Linear
from post.qwen3 import DEVICE, REPO_ID, Linear, Qwen3, load_qwen3, load_tokenizer
from post.sft import Group, load_chats, train

RANK = 32  # r
ALPHA = 32  # the scale alpha / r is then 1
LEARNING_RATE = 1e-4  # LoRA's peak: 10 times full fine-tuning's, post/sft.py
EMBEDDING_LEARNING_RATE = 1e-5  # the embedding table's: full fine-tuning's, as it trains in full


class LoRA(nn.Module):
    """A frozen linear layer, in bf16 or NF4 (post/nf4.py), plus a learned correction of rank r."""

    def __init__(self, linear: Linear | NF4Linear, rank: int, alpha: float) -> None:
        super().__init__()
        bound = linear.in_features**-0.5  # as nn.Linear initializes its weight
        a = torch.empty(rank, linear.in_features, device=DEVICE)  # [r, in], fp32: trained
        a = a.uniform_(-bound, bound)  # [r, in]
        b = torch.zeros(linear.out_features, rank, device=DEVICE)  # [out, r]
        self.linear = linear  # frozen
        self.a = nn.Parameter(a)  # A
        self.b = nn.Parameter(b)  # B
        self.scale = alpha / rank

    def forward(self, x: Tensor) -> Tensor:  # [..., in]
        y = self.linear(x)  # [..., out], W x
        h = F.linear(x, self.a)  # [..., r], A x
        h = F.linear(h, self.b)  # [..., out], B A x
        return y + self.scale * h  # [..., out]


def add_lora(model: Qwen3, rank: int, alpha: float) -> None:
    """Freeze every loaded weight but the embedding table, then wrap every linear layer in a
    LoRA."""
    model.requires_grad_(False)
    model.embed_tokens.float()  # trained: an fp32 master copy, as in a full fine-tune
    model.embed_tokens.requires_grad_(True)
    modules = list(model.modules())  # before wrapping: a LoRA's own linear isn't wrapped again
    for module in modules:
        for name, child in list(module.named_children()):
            if isinstance(child, Linear | NF4Linear):
                setattr(module, name, LoRA(child, rank, alpha))


def lora_groups(model: Qwen3) -> list[Group]:
    """The weights that train, each group at its own peak learning rate: LoRA's matrices at 10
    times full fine-tuning's, which makes up for B starting at zero; the embedding table, trained
    in full, at full fine-tuning's."""
    loras = [module for module in model.modules() if isinstance(module, LoRA)]
    matrices = [weight for lora in loras for weight in [lora.a, lora.b]]
    return [(matrices, LEARNING_RATE), ([model.embed_tokens.weight], EMBEDDING_LEARNING_RATE)]


def main() -> None:
    torch.manual_seed(0)  # as post/sft.py: the same conversations in the same order
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)  # frozen: bf16 is enough
    chats = load_chats(tokenizer)
    add_lora(model, RANK, ALPHA)
    train(model, chats, lora_groups(model))


if __name__ == "__main__":
    main()

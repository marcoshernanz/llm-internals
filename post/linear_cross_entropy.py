"""
Fused linear cross-entropy: the head and the loss computed together, a chunk of positions at a time,
so a step's logits are never all in memory at once. The head turns each position's hidden state, D
numbers, into V = 151,936 logits; the loss turns those into minus the log probability of the true
next token. Computed plainly, every position's logits sit in memory at once, in several copies: at
4,096 tokens one fp32 copy is 2.3 GiB, and the step doesn't fit the L4 (post/bench/memory.py).

For one position with logits z and true token y, the loss and its gradient are

    loss = logsumexp(z) - z[y]
    dloss/dz = softmax(z) - onehot(y)

the probabilities, with 1 taken off at the true token. Through the head, z = W h, that gradient
reaches h as dloss/dz @ W, and W as the outer product of dloss/dz with h.

So each chunk's gradients are computed right away, in the forward pass, and its logits thrown away:
what's kept is the gradient for every hidden state and the head's, summed over the chunks. The loss
is the last thing computed, so the gradient coming back into it is a single number, 1 unless the
loss is scaled; the backward pass only multiplies by it.
Matmuls in bf16, as autocast runs them; the softmax in fp32.

Shape letters used in the comments:
    A   positions whose target counts
    C   positions in a chunk, CHUNK or fewer in the last
    D   model width
    V   vocabulary size
"""

from typing import Any

import torch
from torch import Tensor

CHUNK = 1_024  # positions at a time: 0.6 GB of fp32 logits, whatever the step's length


class LinearCrossEntropy(torch.autograd.Function):
    """The mean over the positions of minus the log probability of each target, with the head W."""

    @staticmethod
    def forward(ctx: Any, hidden: Tensor, weight: Tensor, targets: Tensor) -> Tensor:
        # [A, D], [V, D], [A]
        count = targets.size(0)
        weight16 = weight.to(torch.bfloat16)  # [V, D]
        grad_hidden = torch.empty_like(hidden)  # [A, D]
        grad_weight = torch.zeros_like(weight)  # [V, D], fp32: the chunks' sum
        total = torch.zeros((), device=hidden.device)  # the losses' sum, fp32
        for start in range(0, count, CHUNK):
            end = min(start + CHUNK, count)
            h = hidden[start:end]  # [C, D]
            h = h.to(torch.bfloat16)  # [C, D]
            y = targets[start:end]  # [C]
            rows = torch.arange(end - start, device=hidden.device)  # [C]
            z = h @ weight16.T  # [C, V], the chunk's logits
            z = z.float()  # [C, V], the softmax in fp32
            lse = z.logsumexp(dim=-1)  # [C]
            total += (lse - z[rows, y]).sum()  # minus each target's log probability
            z -= lse[:, None]  # [C, V], log probabilities, in place
            z.exp_()  # [C, V], probabilities, in place
            z[rows, y] -= 1  # [C, V], dloss/dz of each position's own loss
            z /= count  # [C, V], of the mean
            grad_z = z.to(torch.bfloat16)  # [C, V]
            grad_hidden[start:end] = grad_z @ weight16  # [C, D]
            grad_weight += grad_z.T @ h  # [V, D]
        ctx.save_for_backward(grad_hidden, grad_weight)
        return total / count

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> tuple[Tensor, Tensor, None]:
        (grad_loss,) = grad_outputs  # [], the loss's: its only output
        grad_hidden, grad_weight = ctx.saved_tensors  # [A, D], [V, D]
        return grad_hidden * grad_loss, grad_weight * grad_loss, None  # targets get none

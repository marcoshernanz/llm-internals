"""
Shape letters used in the comments:
    B   batch size
    T   sequence length (time)
    D   model width (hidden_size)
    F   hidden width of a feed-forward network (intermediate size)
    H   attention heads
    Dh  width of one head
    Dr  rotary dims per head (qk_rope_head_dim), the last Dr of the Dh
    Dq  width of the query bottleneck (q_lora_rank)
    Do  width of the output bottleneck, one per group (o_lora_rank)
    G   groups of heads in the output projection (o_groups)
    E   routed experts (n_routed_experts)
    K   routed experts active per token (num_experts_per_tok)
    N   tokens, batch and time flattened together
    X   tokens routed to one expert, different for each
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import Config

NORM_EPS = 1e-20
INIT_STD = 0.006
SWIGLU_LIMIT = 10.0
ROPE_THETA = 10000.0
COMPRESS_ROPE_THETA = 160000.0
ROPE_BETA_FAST = 32
ROPE_BETA_SLOW = 1
SLIDING_WINDOW = 128


def init_weights(module: nn.Module) -> None:
    if isinstance(module, Linear):
        nn.init.normal_(module.weight, std=INIT_STD)
    if isinstance(module, Attention):
        nn.init.normal_(module.attn_sink, std=INIT_STD)
        nn.init.normal_(module.o_group_weight, std=INIT_STD)
    if isinstance(module, Gate):
        nn.init.normal_(module.weight, std=INIT_STD)


def yarn_rescale(freqs: Tensor, base: float, original_seq_len: int, factor: float) -> Tensor:
    dim = 2 * freqs.numel()

    def pair_completing(turns: float) -> float:
        """Index of the pair that completes ``turns`` rotations over the training context."""
        return dim * math.log(original_seq_len / (turns * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(pair_completing(ROPE_BETA_FAST)), 0)
    high = min(math.ceil(pair_completing(ROPE_BETA_SLOW)), dim - 1)
    pair_index = torch.arange(dim // 2)  # [Dr/2]
    ramp = ((pair_index - low) / max(high - low, 1e-3)).clamp(0, 1)  # [Dr/2], 0 keeps, 1 slows
    return freqs * (1 - ramp) + (freqs / factor) * ramp  # [Dr/2]


def rope_freqs_cis(
    dim: int, seq_len: int, base: float, original_seq_len: int = 0, factor: float = 1.0
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))  # [Dr/2]
    if original_seq_len > 0:
        freqs = yarn_rescale(freqs, base, original_seq_len, factor)  # [Dr/2]
    positions = torch.arange(seq_len)  # [T]
    angles = torch.outer(positions, freqs)  # [T, Dr/2]
    return torch.polar(torch.ones_like(angles), angles)  # [T, Dr/2] complex


def apply_rope(x: Tensor, freqs_cis: Tensor, inverse: bool = False) -> Tensor:  # [B, T, H, Dr]
    pairs = x.float()  # [B, T, H, Dr]
    pairs = pairs.unflatten(-1, (-1, 2))  # [B, T, H, Dr/2, 2]
    pairs = torch.view_as_complex(pairs)  # [B, T, H, Dr/2] complex

    if inverse:
        freqs_cis = freqs_cis.conj()  # the conjugate rotates by minus the angle
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), -1)  # [1, T, Dr/2]
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, -1)  # [1, T, 1, Dr/2]

    rotated = pairs * freqs_cis  # [B, T, H, Dr/2] complex
    rotated = torch.view_as_real(rotated)  # [B, T, H, Dr/2, 2]
    rotated = rotated.flatten(-2)  # [B, T, H, Dr]
    return rotated.to(x.dtype)  # [B, T, H, Dr]


def apply_partial_rope(x: Tensor, freqs_cis: Tensor, inverse: bool = False) -> Tensor:  # [..., Dh]
    rope_dim = 2 * freqs_cis.size(-1)  # Dr
    content = x[..., :-rope_dim]  # [..., Dh-Dr]
    rotary = x[..., -rope_dim:]  # [..., Dr]
    rotary = apply_rope(rotary, freqs_cis, inverse)  # [..., Dr]
    return torch.cat([content, rotary], dim=-1)  # [..., Dh]


def sliding_window_mask(seq_len: int, window: int, device: torch.device) -> Tensor:
    positions = torch.arange(seq_len, device=device)  # [T]
    distance = positions[:, None] - positions[None, :]  # [T, T]
    causal = distance >= 0  # [T, T]
    inside_window = distance < window  # [T, T]
    return causal & inside_window  # [T, T]


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))  # [out, in]

    def forward(self, x: Tensor) -> Tensor:  # [..., in]
        return F.linear(x, self.weight)  # [..., out]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = NORM_EPS) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # [dim]
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:  # [..., dim]
        x32 = x.float()  # [..., dim]
        mean_square = x32.square().mean(dim=-1, keepdim=True)  # [..., 1]
        normed = x32 * torch.rsqrt(mean_square + self.eps)  # [..., dim]
        normed = normed.to(x.dtype)  # [..., dim]
        return normed * self.weight  # [..., dim]


class FeedForward(nn.Module):
    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = SWIGLU_LIMIT) -> None:
        super().__init__()
        self.gate_proj = Linear(dim, inter_dim)
        self.up_proj = Linear(dim, inter_dim)
        self.down_proj = Linear(inter_dim, dim)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: Tensor) -> Tensor:  # [..., D]
        gate = self.gate_proj(x).float()  # [..., F]
        up = self.up_proj(x).float()  # [..., F]
        if self.swiglu_limit > 0:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        hidden = F.silu(gate) * up  # [..., F]
        return self.down_proj(hidden.to(x.dtype))  # [..., D]


class Attention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank

        self.q_down_proj = Linear(config.hidden_size, config.q_lora_rank)
        self.q_norm = RMSNorm(config.q_lora_rank)
        self.q_up_proj = Linear(config.q_lora_rank, self.num_heads * self.head_dim)
        self.kv_proj = Linear(config.hidden_size, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim)
        self.attn_sink = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))  # [H]
        group_width = self.num_heads * self.head_dim // self.o_groups  # (H/G)*Dh
        group_weight = torch.empty(self.o_groups, self.o_lora_rank, group_width)
        self.o_group_weight = nn.Parameter(group_weight)  # [G, Do, (H/G)*Dh]
        self.o_proj = Linear(self.o_groups * self.o_lora_rank, config.hidden_size)

        freqs_cis = rope_freqs_cis(
            config.qk_rope_head_dim, config.max_position_embeddings, ROPE_THETA
        )
        self.freqs_cis = nn.Buffer(freqs_cis, persistent=False)  # [max T, Dr/2]

    def queries_and_kv(self, x: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:  # [B, T, D]
        q = self.q_down_proj(x)  # [B, T, Dq]
        q = self.q_norm(q)  # [B, T, Dq]
        q = self.q_up_proj(q)  # [B, T, H*Dh]
        q = q.unflatten(-1, (self.num_heads, self.head_dim))  # [B, T, H, Dh]
        q = apply_partial_rope(q, freqs_cis)  # [B, T, H, Dh]

        kv = self.kv_proj(x)  # [B, T, Dh]
        kv = self.kv_norm(kv)  # [B, T, Dh]
        kv = apply_partial_rope(kv, freqs_cis)  # [B, T, Dh]
        return q, kv

    def attend(self, q: Tensor, kv: Tensor, visible: Tensor) -> Tensor:  # [B, T, H, Dh], [B, T, Dh]
        scores = torch.einsum("bthd,bsd->bhts", q, kv)  # [B, H, T, T]
        scores = scores * self.softmax_scale  # [B, H, T, T]
        scores = scores.masked_fill(~visible, -torch.inf)  # [B, H, T, T]
        scores = scores.float()

        sink = self.attn_sink.view(1, -1, 1, 1)  # [1, H, 1, 1]
        sink = sink.expand(scores.size(0), -1, scores.size(2), -1)  # [B, H, T, 1]
        scores = torch.cat([scores, sink], dim=-1)  # [B, H, T, T+1]

        weights = scores.softmax(dim=-1)  # [B, H, T, T+1]
        weights = weights[..., :-1]  # [B, H, T, T]
        weights = weights.to(kv.dtype)  # [B, H, T, T]
        return torch.einsum("bhts,bsd->bthd", weights, kv)  # [B, T, H, Dh]

    def project_out(self, o: Tensor, freqs_cis: Tensor) -> Tensor:  # [B, T, H, Dh]
        o = apply_partial_rope(o, freqs_cis, inverse=True)  # [B, T, H, Dh]
        o = o.flatten(-2)  # [B, T, H*Dh]
        o = o.unflatten(-1, (self.o_groups, -1))  # [B, T, G, (H/G)*Dh]

        o = torch.einsum("btgd,grd->btgr", o, self.o_group_weight)  # [B, T, G, Do]
        o = o.flatten(-2)  # [B, T, G*Do]
        return self.o_proj(o)  # [B, T, D]

    def forward(self, x: Tensor) -> Tensor:  # [B, T, D]
        freqs_cis = self.freqs_cis[: x.size(1)]  # [T, Dr/2]
        visible = sliding_window_mask(x.size(1), SLIDING_WINDOW, x.device)  # [T, T]

        q, kv = self.queries_and_kv(x, freqs_cis)  # [B, T, H, Dh], [B, T, Dh]
        o = self.attend(q, kv, visible)  # [B, T, H, Dh]
        return self.project_out(o, freqs_cis)  # [B, T, D]


class Gate(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor

        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, config.hidden_size))  # [E, D]
        self.bias = nn.Buffer(torch.zeros(self.n_routed_experts, dtype=torch.float32))  # [E]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # [N, D]
        scores = F.linear(x.float(), self.weight.float())  # [N, E]
        scores = F.softplus(scores).sqrt()  # [N, E]

        biased = scores + self.bias  # [N, E]
        expert_ids = biased.topk(self.num_experts_per_tok, dim=-1).indices  # [N, K]

        weights = scores.gather(-1, expert_ids)  # [N, K]
        weights = weights / (weights.sum(dim=-1, keepdim=True) + NORM_EPS)  # [N, K]
        weights = weights * self.routed_scaling_factor  # [N, K]
        return weights, expert_ids


class MixtureOfExperts(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shared_size = config.n_shared_experts * config.moe_intermediate_size

        self.gate = Gate(config)
        self.experts = nn.ModuleList(
            FeedForward(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.n_routed_experts)
        )
        self.shared_expert = FeedForward(config.hidden_size, shared_size)

    def forward(self, x: Tensor) -> Tensor:  # [B, T, D]
        tokens = x.flatten(0, 1)  # [N, D]
        weights, expert_ids = self.gate(tokens)  # [N, K], [N, K]

        routed = torch.zeros_like(tokens, dtype=torch.float32)  # [N, D]
        for i, expert in enumerate(self.experts):
            rows, slot = torch.where(expert_ids == i)  # [X], [X]
            if rows.numel() == 0:
                continue
            expert_out = expert(tokens[rows])  # [X, D]
            weight = weights[rows, slot]  # [X]
            routed[rows] += expert_out * weight[:, None]  # [X, D]

        out = routed + self.shared_expert(tokens)  # [N, D]
        out = out.to(x.dtype)  # [N, D]
        return out.unflatten(0, (x.size(0), x.size(1)))  # [B, T, D]


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.attention_norm = RMSNorm(config.hidden_size)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.ffn = MixtureOfExperts(config)

    def forward(self, x: Tensor):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.num_blocks))
        self.out_norm = RMSNorm(config.hidden_size)
        self.out_proj = Linear(config.hidden_size, config.vocab_size)

    def forward(self, x: Tensor):
        x = self.embed(x)

        for block in self.blocks:
            x = block(x)

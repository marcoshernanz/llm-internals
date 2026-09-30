"""
Shape letters used in the comments:
    B   batch size
    T   sequence length (time)
    D   model width (hidden_size)
    F   hidden width of a feed-forward network (intermediate size)
    H   attention heads
    Dq  width of the query latent (q_lora_rank)
    Dc  width of the key-value latent (kv_lora_rank)
    Dn  query and key dims per head without RoPE (qk_nope_head_dim)
    Dr  query and key dims per head with RoPE (qk_rope_head_dim)
    Dv  value dims per head (v_head_dim)
    E   routed experts (n_routed_experts)
    K   routed experts active per token (num_experts_per_tok)
    N   tokens, batch and time flattened together
    X   tokens routed to one expert, different for each
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from core.config import Config

NORM_EPS = 1e-6
INIT_STD = 0.02


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (Linear, Embedding)):
        nn.init.normal_(module.weight, std=INIT_STD)


def rope_freqs_cis(dim: int, seq_len: int, theta: float, device: torch.device) -> Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device) / dim))  # [Dr/2]
    positions = torch.arange(seq_len, device=device)  # [T]
    angles = torch.outer(positions, freqs)  # [T, Dr/2]
    return torch.polar(torch.ones_like(angles), angles)  # [T, Dr/2] complex


def apply_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:  # [B, T, H, Dr], [T, Dr/2]
    pairs = x.float()  # [B, T, H, Dr]
    pairs = pairs.unflatten(-1, (-1, 2))  # [B, T, H, Dr/2, 2]
    pairs = torch.view_as_complex(pairs)  # [B, T, H, Dr/2] complex
    freqs_cis = freqs_cis.unsqueeze(1)  # [T, 1, Dr/2], every head turns the same way
    rotated = pairs * freqs_cis  # [B, T, H, Dr/2] complex
    rotated = torch.view_as_real(rotated)  # [B, T, H, Dr/2, 2]
    rotated = rotated.flatten(-2)  # [B, T, H, Dr]
    return rotated.to(x.dtype)  # [B, T, H, Dr]


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, float32: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))  # [out, in]
        self.float32 = float32

    def forward(self, x: Tensor) -> Tensor:  # [..., in]
        if not self.float32:
            return F.linear(x, self.weight)  # [..., out]
        with torch.autocast(x.device.type, enabled=False):
            return F.linear(x.float(), self.weight.float())  # [..., out]


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))  # [vocab, dim]

    def forward(self, ids: Tensor) -> Tensor:  # [B, T]
        return self.weight[ids]  # [B, T, D]


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # [dim]

    def forward(self, x: Tensor) -> Tensor:  # [..., dim]
        x32 = x.float()  # [..., dim]
        mean_square = x32.square().mean(dim=-1, keepdim=True)  # [..., 1]
        normed = x32 * torch.rsqrt(mean_square + NORM_EPS)  # [..., dim]
        return normed.to(x.dtype) * self.weight  # [..., dim]


class Attention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim  # Dn+Dr
        kv_head_dim = config.qk_nope_head_dim + config.v_head_dim  # Dn+Dv
        o_dim = self.num_heads * config.v_head_dim  # H*Dv
        self.softmax_scale = qk_head_dim**-0.5

        self.q_down_proj = Linear(config.hidden_size, config.q_lora_rank)
        self.q_norm = RMSNorm(config.q_lora_rank)
        self.q_up_proj = Linear(config.q_lora_rank, self.num_heads * qk_head_dim)
        self.kv_down_proj = Linear(config.hidden_size, self.kv_lora_rank + config.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.kv_up_proj = Linear(self.kv_lora_rank, self.num_heads * kv_head_dim)
        self.o_gate_proj = Linear(config.hidden_size, o_dim)
        self.o_proj = Linear(o_dim, config.hidden_size)

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:  # [B, T, D], [T, Dr/2]
        q = self.q_down_proj(x)  # [B, T, Dq]
        q = self.q_norm(q)  # [B, T, Dq]
        q = self.q_up_proj(q)  # [B, T, H*(Dn+Dr)]
        q = q.unflatten(-1, (self.num_heads, -1))  # [B, T, H, Dn+Dr]
        q_nope = q[..., : self.qk_nope_head_dim]  # [B, T, H, Dn]
        q_rope = q[..., self.qk_nope_head_dim :]  # [B, T, H, Dr]
        q_rope = apply_rope(q_rope, freqs_cis)  # [B, T, H, Dr]
        q = torch.cat([q_nope, q_rope], dim=-1)  # [B, T, H, Dn+Dr]

        kv = self.kv_down_proj(x)  # [B, T, Dc+Dr]
        k_rope = kv[..., self.kv_lora_rank :]  # [B, T, Dr]
        kv = kv[..., : self.kv_lora_rank]  # [B, T, Dc], the latent an inference cache keeps
        kv = self.kv_norm(kv)  # [B, T, Dc]
        kv = self.kv_up_proj(kv)  # [B, T, H*(Dn+Dv)]
        kv = kv.unflatten(-1, (self.num_heads, -1))  # [B, T, H, Dn+Dv]
        k_nope = kv[..., : self.qk_nope_head_dim]  # [B, T, H, Dn]
        v = kv[..., self.qk_nope_head_dim :]  # [B, T, H, Dv]

        k_rope = k_rope.unsqueeze(2)  # [B, T, 1, Dr]
        k_rope = apply_rope(k_rope, freqs_cis)  # [B, T, 1, Dr]
        k_rope = k_rope.expand(-1, -1, self.num_heads, -1)  # [B, T, H, Dr], shared by every head
        k = torch.cat([k_nope, k_rope], dim=-1)  # [B, T, H, Dn+Dr]

        causal = torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device)  # [T, T]
        causal = causal.tril()  # [T, T], query t sees keys 0..t
        scores = torch.einsum("bthd,bshd->bhts", q, k)  # [B, H, T, T]
        scores = scores.float() * self.softmax_scale  # [B, H, T, T]
        scores = scores.masked_fill(~causal, -torch.inf)  # [B, H, T, T]
        probs = scores.softmax(dim=-1)  # [B, H, T, T]
        probs = probs.to(v.dtype)  # [B, H, T, T]
        o = torch.einsum("bhts,bshd->bthd", probs, v)  # [B, T, H, Dv]
        o = o.flatten(-2)  # [B, T, H*Dv]

        gate = self.o_gate_proj(x)  # [B, T, H*Dv]
        o = o * torch.sigmoid(gate)  # [B, T, H*Dv], each output dim can be shut off
        return self.o_proj(o)  # [B, T, D]


class FeedForward(nn.Module):
    def __init__(self, dim: int, inter_dim: int) -> None:
        super().__init__()
        self.gate_proj = Linear(dim, inter_dim)
        self.up_proj = Linear(dim, inter_dim)
        self.down_proj = Linear(inter_dim, dim)

    def forward(self, x: Tensor) -> Tensor:  # [..., D]
        gate = self.gate_proj(x)  # [..., F]
        up = self.up_proj(x)  # [..., F]
        hidden = F.silu(gate) * up  # [..., F]
        return self.down_proj(hidden)  # [..., D]


class Router(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        self.score_proj = Linear(config.hidden_size, config.n_routed_experts, float32=True)
        self.bias = nn.Buffer(torch.zeros(config.n_routed_experts))  # [E], set by the load balancer

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # [N, D]
        scores = self.score_proj(x)  # [N, E]
        scores = scores.sigmoid()  # [N, E]

        biased = scores + self.bias  # [N, E], the bias only decides which experts are picked
        expert_ids = biased.topk(self.num_experts_per_tok, dim=-1).indices  # [N, K]

        weights = scores.gather(-1, expert_ids)  # [N, K]
        if self.num_experts_per_tok > 1:  # normalizing a single weight would pin it to 1
            weights = weights / weights.sum(dim=-1, keepdim=True)  # [N, K]
        weights = weights * self.routed_scaling_factor  # [N, K]
        return weights, expert_ids


class MixtureOfExperts(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shared_size = config.n_shared_experts * config.moe_intermediate_size

        self.router = Router(config)
        self.experts = nn.ModuleList(
            FeedForward(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.n_routed_experts)
        )
        self.shared_expert = FeedForward(config.hidden_size, shared_size)

    def forward(self, x: Tensor) -> Tensor:  # [B, T, D]
        tokens = x.flatten(0, 1)  # [N, D]
        weights, expert_ids = self.router(tokens)  # [N, K], [N, K]

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
    def __init__(self, config: Config, layer_id: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.ffn: FeedForward | MixtureOfExperts
        if layer_id < config.first_k_dense_replace:
            self.ffn = FeedForward(config.hidden_size, config.intermediate_size)
        else:
            self.ffn = MixtureOfExperts(config)

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:  # [B, T, D], [T, Dr/2]
        h = self.attn_norm(x)  # [B, T, D]
        h = self.attn(h, freqs_cis)  # [B, T, D]
        x = x + h  # [B, T, D]

        h = self.ffn_norm(x)  # [B, T, D]
        h = self.ffn(h)  # [B, T, D]
        return x + h  # [B, T, D]


class Transformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.rope_theta = config.rope_theta
        self.embed = Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            Block(config, layer_id) for layer_id in range(config.num_hidden_layers)
        )
        self.out_norm = RMSNorm(config.hidden_size)
        self.head = Linear(config.hidden_size, config.vocab_size)
        self.apply(init_weights)

    def forward(self, ids: Tensor) -> Tensor:  # [B, T]
        freqs_cis = rope_freqs_cis(
            self.qk_rope_head_dim, ids.size(1), self.rope_theta, ids.device
        )  # [T, Dr/2] complex
        x = self.embed(ids)  # [B, T, D]
        for block in self.blocks:
            x = block(x, freqs_cis)  # [B, T, D]
        x = self.out_norm(x)  # [B, T, D]
        return self.head(x)  # [B, T, vocab]

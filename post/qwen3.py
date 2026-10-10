"""
Qwen3, the dense transformer of the Qwen3 family, written to run its released checkpoints. Every
module is named as in the checkpoint, so the checkpoint's tensors load by name.

Shape letters used in the comments:
    B   batch size
    T   sequence length (time): with a KV cache, the new tokens only
    S   positions the queries attend over: with a KV cache, the cached ones plus T, otherwise T
    L   positions a KV cache has room for (max_len)
    D   model width (hidden_size)
    F   hidden width of the feed-forward network (intermediate_size)
    H   query heads (num_attention_heads)
    G   key-value heads (num_key_value_heads), each shared by H/G query heads
    Dh  dims per head (head_dim)
    V   vocabulary size (vocab_size)
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer
from torch import Tensor, nn


@dataclass(frozen=True)
class Qwen3Config:
    """The sizes of one Qwen3 model, named as in its config.json."""

    vocab_size: int  # V, rows of the embedding table
    hidden_size: int  # D
    intermediate_size: int  # F
    num_hidden_layers: int  # blocks in the stack
    num_attention_heads: int  # H
    num_key_value_heads: int  # G
    head_dim: int  # Dh
    rope_theta: float  # base of the RoPE frequencies
    rms_norm_eps: float  # added to the mean square before the square root


def rope_freqs_cis(dim: int, seq_len: int, theta: float, device: torch.device) -> Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device) / dim))  # [Dh/2]
    positions = torch.arange(seq_len, device=device)  # [T]
    angles = torch.outer(positions, freqs)  # [T, Dh/2]
    return torch.polar(torch.ones_like(angles), angles)  # [T, Dh/2] complex


def apply_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:  # [B, T, H, Dh], [T, Dh/2]
    """Rotate each pair of dims by its position's angle. Qwen3 pairs dim i with dim i + Dh/2: the
    checkpoint was trained with this pairing, so it has to match. For keys, G heads instead of H."""
    x32 = x.float()  # [B, T, H, Dh]
    half = x.size(-1) // 2
    pairs = torch.complex(x32[..., :half], x32[..., half:])  # [B, T, H, Dh/2] complex
    freqs_cis = freqs_cis.unsqueeze(1)  # [T, 1, Dh/2], every head turns the same way
    rotated = pairs * freqs_cis  # [B, T, H, Dh/2] complex
    rotated = torch.cat([rotated.real, rotated.imag], dim=-1)  # [B, T, H, Dh]
    return rotated.to(x.dtype)  # [B, T, H, Dh]


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))  # [out, in], loaded

    def forward(self, x: Tensor) -> Tensor:  # [..., in]
        return F.linear(x, self.weight)  # [..., out]


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))  # [V, D], loaded

    def forward(self, ids: Tensor) -> Tensor:  # [B, T]
        return self.weight[ids]  # [B, T, D]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # [dim]
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:  # [..., dim]
        x32 = x.float()  # [..., dim]
        mean_square = x32.square().mean(dim=-1, keepdim=True)  # [..., 1]
        normed = x32 * torch.rsqrt(mean_square + self.eps)  # [..., dim]
        return normed.to(x.dtype) * self.weight  # [..., dim]


class KVCache:
    """The keys and values of every position so far, for one attention layer. The buffers are
    allocated once, for the longest sequence, and each step writes its new positions in place."""

    def __init__(
        self,
        config: Qwen3Config,
        batch_size: int,
        max_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        shape = (batch_size, max_len, config.num_key_value_heads, config.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)  # [B, L, G, Dh]
        self.v = torch.zeros(shape, dtype=dtype, device=device)  # [B, L, G, Dh]
        self.length = 0  # positions filled so far

    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:  # [B, T, G, Dh] each
        """Store the keys and values of T new positions; return those of every position so far."""
        start = self.length
        self.length = start + k.size(1)
        self.k[:, start : self.length] = k  # [B, T, G, Dh] written into [B, L, G, Dh]
        self.v[:, start : self.length] = v  # [B, T, G, Dh] written into [B, L, G, Dh]
        return self.k[:, : self.length], self.v[:, : self.length]  # [B, S, G, Dh] each


class Attention(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.softmax_scale = config.head_dim**-0.5
        q_dim = self.num_heads * self.head_dim  # H*Dh, can differ from D: 2048 vs 1024 in 0.6B
        kv_dim = self.num_kv_heads * self.head_dim  # G*Dh

        self.q_proj = Linear(config.hidden_size, q_dim)
        self.k_proj = Linear(config.hidden_size, kv_dim)
        self.v_proj = Linear(config.hidden_size, kv_dim)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)  # one weight of Dh, every head
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = Linear(q_dim, config.hidden_size)

    def forward(
        self, x: Tensor, freqs_cis: Tensor, cache: KVCache | None
    ) -> Tensor:  # [B, T, D], [T, Dh/2]
        q = self.q_proj(x)  # [B, T, H*Dh]
        q = q.unflatten(-1, (self.num_heads, self.head_dim))  # [B, T, H, Dh]
        q = self.q_norm(q)  # [B, T, H, Dh], each head's query normed on its own
        q = apply_rope(q, freqs_cis)  # [B, T, H, Dh]

        k = self.k_proj(x)  # [B, T, G*Dh]
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))  # [B, T, G, Dh]
        k = self.k_norm(k)  # [B, T, G, Dh]
        k = apply_rope(k, freqs_cis)  # [B, T, G, Dh]

        v = self.v_proj(x)  # [B, T, G*Dh]
        v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))  # [B, T, G, Dh]

        if cache is not None:
            k, v = cache.update(k, v)  # [B, S, G, Dh] each, the cached positions then the new

        group_size = self.num_heads // self.num_kv_heads  # H/G
        k = k.repeat_interleave(group_size, dim=2)  # [B, S, H, Dh], neighboring heads share one
        v = v.repeat_interleave(group_size, dim=2)  # [B, S, H, Dh]

        cached = k.size(1) - q.size(1)  # positions before the new tokens, 0 without a cache
        causal = torch.ones(q.size(1), k.size(1), dtype=torch.bool, device=x.device)  # [T, S]
        causal = causal.tril(diagonal=cached)  # [T, S], new token t sees keys 0..cached+t
        scores = torch.einsum("bthd,bshd->bhts", q, k)  # [B, H, T, S]
        scores = scores.float() * self.softmax_scale  # [B, H, T, S]
        scores = scores.masked_fill(~causal, -torch.inf)  # [B, H, T, S]
        probs = scores.softmax(dim=-1)  # [B, H, T, S]
        probs = probs.to(v.dtype)  # [B, H, T, S]
        o = torch.einsum("bhts,bshd->bthd", probs, v)  # [B, T, H, Dh]
        o = o.flatten(-2)  # [B, T, H*Dh]
        return self.o_proj(o)  # [B, T, D]


class FeedForward(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: Tensor) -> Tensor:  # [B, T, D]
        gate = self.gate_proj(x)  # [B, T, F]
        up = self.up_proj(x)  # [B, T, F]
        hidden = F.silu(gate) * up  # [B, T, F]
        return self.down_proj(hidden)  # [B, T, D]


class Block(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(
        self, x: Tensor, freqs_cis: Tensor, cache: KVCache | None
    ) -> Tensor:  # [B, T, D], [T, Dh/2]
        h = self.input_layernorm(x)  # [B, T, D]
        h = self.self_attn(h, freqs_cis, cache)  # [B, T, D]
        x = x + h  # [B, T, D]

        h = self.post_attention_layernorm(x)  # [B, T, D]
        h = self.mlp(h)  # [B, T, D]
        return x + h  # [B, T, D]


class Qwen3(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, ids: Tensor, caches: list[KVCache] | None = None) -> Tensor:  # [B, T]
        """With caches, one per layer, ids continue the sequence the caches hold."""
        start = 0 if caches is None else caches[0].length  # position of the first token of ids
        freqs_cis = rope_freqs_cis(
            self.config.head_dim, start + ids.size(1), self.config.rope_theta, ids.device
        )  # [start+T, Dh/2] complex
        freqs_cis = freqs_cis[start:]  # [T, Dh/2], the angles at the positions of ids
        x = self.embed_tokens(ids)  # [B, T, D]
        for i, layer in enumerate(self.layers):
            cache = None if caches is None else caches[i]
            x = layer(x, freqs_cis, cache)  # [B, T, D]
        x = self.norm(x)  # [B, T, D]
        return F.linear(x, self.embed_tokens.weight)  # [B, T, V], the head is the embedding table


def load_qwen3(repo_id: str, dtype: torch.dtype, device: str) -> Qwen3:
    """Download a Qwen3 checkpoint from the Hugging Face Hub, once, into its cache, and load it."""
    path = Path(snapshot_download(repo_id, allow_patterns=["config.json", "*.safetensors"]))
    raw = json.loads((path / "config.json").read_text())
    assert raw["tie_word_embeddings"], "the head is the embedding table, true up to Qwen3-4B"
    fields = [field.name for field in dataclasses.fields(Qwen3Config)]
    config = Qwen3Config(**{name: raw[name] for name in fields})

    model = Qwen3(config)
    weights = {}
    for file in path.glob("*.safetensors"):  # bigger checkpoints are split into several files
        weights.update(load_file(file))  # tensor name -> tensor, in the checkpoint's bf16
    weights = {name.removeprefix("model."): tensor for name, tensor in weights.items()}
    model.load_state_dict(weights)  # strict: every tensor must fill a parameter, and back
    return model.to(device=device, dtype=dtype)


def load_tokenizer(repo_id: str) -> Tokenizer:
    """The checkpoint's tokenizer, a byte-level BPE read from its tokenizer.json."""
    path = Path(snapshot_download(repo_id, allow_patterns=["tokenizer.json"]))
    return Tokenizer.from_file(str(path / "tokenizer.json"))

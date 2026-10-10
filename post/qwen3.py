"""
Qwen3, the dense transformer of the Qwen3 family, written to run its released checkpoints. Every
module is named as in the checkpoint, so the checkpoint's tensors load by name. The pieces Qwen3
shares with the core model (Linear, Embedding, RMSNorm, FeedForward) come from core.model.

Shape letters used in the comments:
    B   batch size
    T   sequence length (time)
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
from torch import Tensor, nn

from core.model import NORM_EPS, Embedding, FeedForward, Linear, RMSNorm, rope_freqs_cis


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


def apply_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:  # [B, T, H, Dh], [T, Dh/2]
    """Rotate each pair of dims by its position's angle. Qwen3 pairs dim i with dim i + Dh/2, where
    core.model pairs neighbors: the checkpoint was trained with this pairing, so it has to match.
    For keys, G heads instead of H."""
    x32 = x.float()  # [B, T, H, Dh]
    half = x.size(-1) // 2
    pairs = torch.complex(x32[..., :half], x32[..., half:])  # [B, T, H, Dh/2] complex
    freqs_cis = freqs_cis.unsqueeze(1)  # [T, 1, Dh/2], every head turns the same way
    rotated = pairs * freqs_cis  # [B, T, H, Dh/2] complex
    rotated = torch.cat([rotated.real, rotated.imag], dim=-1)  # [B, T, H, Dh]
    return rotated.to(x.dtype)  # [B, T, H, Dh]


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
        self.q_norm = RMSNorm(self.head_dim)  # one weight of Dh, shared by every head
        self.k_norm = RMSNorm(self.head_dim)
        self.o_proj = Linear(q_dim, config.hidden_size)

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:  # [B, T, D], [T, Dh/2]
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

        group_size = self.num_heads // self.num_kv_heads  # H/G
        k = k.repeat_interleave(group_size, dim=2)  # [B, T, H, Dh], neighboring heads share one
        v = v.repeat_interleave(group_size, dim=2)  # [B, T, H, Dh]

        causal = torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device)  # [T, T]
        causal = causal.tril()  # [T, T], query t sees keys 0..t
        scores = torch.einsum("bthd,bshd->bhts", q, k)  # [B, H, T, T]
        scores = scores.float() * self.softmax_scale  # [B, H, T, T]
        scores = scores.masked_fill(~causal, -torch.inf)  # [B, H, T, T]
        probs = scores.softmax(dim=-1)  # [B, H, T, T]
        probs = probs.to(v.dtype)  # [B, H, T, T]
        o = torch.einsum("bhts,bshd->bthd", probs, v)  # [B, T, H, Dh]
        o = o.flatten(-2)  # [B, T, H*Dh]
        return self.o_proj(o)  # [B, T, D]


class Block(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = FeedForward(config.hidden_size, config.intermediate_size)

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:  # [B, T, D], [T, Dh/2]
        h = self.input_layernorm(x)  # [B, T, D]
        h = self.self_attn(h, freqs_cis)  # [B, T, D]
        x = x + h  # [B, T, D]

        h = self.post_attention_layernorm(x)  # [B, T, D]
        h = self.mlp(h)  # [B, T, D]
        return x + h  # [B, T, D]


class Qwen3(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, ids: Tensor) -> Tensor:  # [B, T]
        freqs_cis = rope_freqs_cis(
            self.head_dim, ids.size(1), self.rope_theta, ids.device
        )  # [T, Dh/2] complex
        x = self.embed_tokens(ids)  # [B, T, D]
        for layer in self.layers:
            x = layer(x, freqs_cis)  # [B, T, D]
        x = self.norm(x)  # [B, T, D]
        return F.linear(x, self.embed_tokens.weight)  # [B, T, V], the head is the embedding table


def load_qwen3(repo_id: str, dtype: torch.dtype, device: str) -> Qwen3:
    """Download a Qwen3 checkpoint from the Hugging Face Hub, once, into its cache, and load it."""
    path = Path(snapshot_download(repo_id, allow_patterns=["config.json", "*.safetensors"]))
    raw = json.loads((path / "config.json").read_text())
    assert raw["rms_norm_eps"] == NORM_EPS, "the RMSNorm from core.model uses NORM_EPS"
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

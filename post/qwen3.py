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
    N   conversations packed in one row
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
from torch.nn.attention.varlen import varlen_attn

REPO_ID = "Qwen/Qwen3-0.6B-Base"  # the checkpoint we post-train
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# The checkpoint's bf16, except on a GPU that can't compute in it, such as the T4. Its fp16 is fast,
# but a bf16 value above 65,504 overflows fp16: until we check the model stays below that, fp32.
DTYPE = torch.bfloat16
if DEVICE == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
    DTYPE = torch.float32


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


def rope_cos_sin(dim: int, positions: Tensor, theta: float) -> tuple[Tensor, Tensor]:  # [T]
    """The cos and sin of the RoPE angles at the given positions."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=positions.device) / dim))  # [Dh/2]
    angles = torch.outer(positions, freqs)  # [T, Dh/2]
    angles = torch.cat([angles, angles], dim=-1)  # [T, Dh], dims i and i + Dh/2 share an angle
    return angles.cos(), angles.sin()  # [T, Dh] each


def rotate_half(x: Tensor) -> Tensor:  # [..., Dh]
    """Turn each pair of dims (a, b) = (x_i, x_i+Dh/2) a quarter turn, to (-b, a)."""
    half = x.size(-1) // 2
    first = x[..., :half]  # [..., Dh/2], the a of every pair
    second = x[..., half:]  # [..., Dh/2], the b of every pair
    return torch.cat([-second, first], dim=-1)  # [..., Dh]


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:  # [B, T, H, Dh], [T, Dh] each
    """Rotate each pair of dims by its position's angle. Qwen3 pairs dim i with dim i + Dh/2: the
    checkpoint was trained with this pairing, so it has to match. For keys, G heads instead of H.
    Turned by an angle, a pair (a, b) becomes (a cos - b sin, b cos + a sin): the pair times cos,
    plus the pair turned a quarter turn, (-b, a), times sin. Real numbers only: torch.compile can't
    generate code for complex ones."""
    x32 = x.float()  # [B, T, H, Dh]
    cos = cos.unsqueeze(1)  # [T, 1, Dh], every head turns the same way
    sin = sin.unsqueeze(1)  # [T, 1, Dh]
    turned = rotate_half(x32)  # [B, T, H, Dh]
    rotated = x32 * cos + turned * sin  # [B, T, H, Dh]
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
        shape = (batch_size, config.num_key_value_heads, max_len, config.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)  # [B, G, L, Dh]
        self.v = torch.zeros(shape, dtype=dtype, device=device)  # [B, G, L, Dh]
        self.length = 0  # positions filled so far

    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:  # [B, G, T, Dh] each
        """Store the keys and values of T new positions; return those of every position so far."""
        start = self.length
        self.length = start + k.size(2)
        self.k[:, :, start : self.length] = k  # [B, G, T, Dh] written into [B, G, L, Dh]
        self.v[:, :, start : self.length] = v  # [B, G, T, Dh] written into [B, G, L, Dh]
        return self.k[:, :, : self.length], self.v[:, :, : self.length]  # [B, G, S, Dh] each

    def repeat_rows(self, batch_size: int) -> None:
        """Copy a cache of one row into batch_size rows, so that samples of one prompt share its
        keys and values without each recomputing them. Copies, not views: each row writes its own
        new tokens after the prompt."""
        self.k = self.k.repeat(batch_size, 1, 1, 1)  # [1, G, L, Dh] -> [B, G, L, Dh]
        self.v = self.v.repeat(batch_size, 1, 1, 1)  # [1, G, L, Dh] -> [B, G, L, Dh]


class Packing:
    """Where each conversation of a packed row starts and ends: several conversations laid end to
    end in one row of T tokens, with no padding, lengths[i] tokens each."""

    def __init__(self, lengths: list[int], device: str) -> None:
        offsets = torch.tensor([0, *lengths]).cumsum(0)  # [N+1]: 0, then where each one ends
        self.offsets = offsets.to(device=device, dtype=torch.int32)  # [N+1], cu_seqlens
        self.longest = max(lengths)
        positions = [torch.arange(n) for n in lengths]  # N of [n], each conversation from 0
        self.positions = torch.cat(positions).to(device)  # [T], as each would have alone


def attend(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:  # [B, H, T, Dh], [B, G, S, Dh]
    """Causal attention of T new queries over S keys, the last T of which belong to the new tokens.
    One fused kernel: each query head reads its group's key-value head where it lies, with no
    copy per query head, and the scores never leave the chip."""
    cached = k.size(2) - q.size(2)  # positions before the new tokens, 0 without a cache
    causal = torch.ones(q.size(2), k.size(2), dtype=torch.bool, device=q.device)  # [T, S]
    causal = causal.tril(diagonal=cached)  # [T, S], new token t sees keys 0..cached+t
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=causal, scale=scale, enable_gqa=True
    )  # [B, H, T, Dh]


def attend_packed(
    q: Tensor, k: Tensor, v: Tensor, packing: Packing, scale: float
) -> Tensor:  # [1, H, T, Dh], [1, G, T, Dh], [1, G, T, Dh]
    """Causal attention within each conversation of one packed row: a token sees the earlier tokens
    of its own conversation only. One FlashAttention kernel for variable-length sequences reads
    where each conversation starts and never computes a score between two of them. It takes bf16
    only, and autocast doesn't reach it: the queries and keys leave their RMSNorms in the weights'
    fp32, so they are cast to the values' dtype, the one the matmuls compute in."""
    q = q.squeeze(0)  # [H, T, Dh], the one row
    q = q.transpose(0, 1)  # [T, H, Dh], the layout varlen kernels take
    q = q.to(v.dtype)  # [T, H, Dh]
    k = k.squeeze(0)  # [G, T, Dh]
    k = k.transpose(0, 1)  # [T, G, Dh]
    k = k.to(v.dtype)  # [T, G, Dh]
    v = v.squeeze(0)  # [G, T, Dh]
    v = v.transpose(0, 1)  # [T, G, Dh]
    o = varlen_attn(
        q,
        k,
        v,
        cu_seq_q=packing.offsets,  # [N+1], where each conversation starts, then T
        cu_seq_k=packing.offsets,  # the same: queries and keys are the same tokens
        max_q=packing.longest,
        max_k=packing.longest,
        scale=scale,
        window_size=(-1, 0),  # causal: every earlier token of its conversation, no later one
        enable_gqa=True,
    )  # [T, H, Dh]
    assert isinstance(o, Tensor)  # the output alone: no log-sum-exp was asked for
    o = o.transpose(0, 1)  # [H, T, Dh]
    return o.unsqueeze(0)  # [1, H, T, Dh]


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
        self, x: Tensor, cos: Tensor, sin: Tensor, cache: KVCache | None, packing: Packing | None
    ) -> Tensor:  # [B, T, D], [T, Dh], [T, Dh]
        q = self.q_proj(x)  # [B, T, H*Dh]
        q = q.unflatten(-1, (self.num_heads, self.head_dim))  # [B, T, H, Dh]
        q = self.q_norm(q)  # [B, T, H, Dh], each head's query normed on its own
        q = apply_rope(q, cos, sin)  # [B, T, H, Dh]
        q = q.transpose(1, 2)  # [B, H, T, Dh], the layout attention kernels take

        k = self.k_proj(x)  # [B, T, G*Dh]
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))  # [B, T, G, Dh]
        k = self.k_norm(k)  # [B, T, G, Dh]
        k = apply_rope(k, cos, sin)  # [B, T, G, Dh]
        k = k.transpose(1, 2)  # [B, G, T, Dh]

        v = self.v_proj(x)  # [B, T, G*Dh]
        v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))  # [B, T, G, Dh]
        v = v.transpose(1, 2)  # [B, G, T, Dh]

        if cache is not None:
            k, v = cache.update(k, v)  # [B, G, S, Dh] each, the cached positions then the new

        if packing is None:
            o = attend(q, k, v, self.softmax_scale)  # [B, H, T, Dh]
        else:
            o = attend_packed(q, k, v, packing, self.softmax_scale)  # [1, H, T, Dh]
        o = o.transpose(1, 2)  # [B, T, H, Dh]
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
        self, x: Tensor, cos: Tensor, sin: Tensor, cache: KVCache | None, packing: Packing | None
    ) -> Tensor:  # [B, T, D], [T, Dh], [T, Dh]
        h = self.input_layernorm(x)  # [B, T, D]
        h = self.self_attn(h, cos, sin, cache, packing)  # [B, T, D]
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

    def hidden(
        self, ids: Tensor, caches: list[KVCache] | None = None, packing: Packing | None = None
    ) -> Tensor:  # [B, T]
        """The last block's output, normed: the vector the head turns into each position's logits.
        With caches, one per layer, ids continue the sequence the caches hold. With packing, ids is
        one row of conversations laid end to end, and each is computed as if it were alone."""
        if packing is None:
            start = 0 if caches is None else caches[0].length  # position of the first token of ids
            positions = torch.arange(start, start + ids.size(1), device=ids.device)  # [T]
        else:
            positions = packing.positions  # [T], each conversation from 0
        cos, sin = rope_cos_sin(self.config.head_dim, positions, self.config.rope_theta)  # [T, Dh]
        x = self.embed_tokens(ids)  # [B, T, D]
        for i, layer in enumerate(self.layers):
            cache = None if caches is None else caches[i]
            x = layer(x, cos, sin, cache, packing)  # [B, T, D]
        return self.norm(x)  # [B, T, D]

    def forward(
        self,
        ids: Tensor,
        caches: list[KVCache] | None = None,
        last_only: bool = False,
        packing: Packing | None = None,
    ) -> Tensor:  # [B, T]
        """The logits, a row of V = 151,936 numbers for each position. With last_only, only the
        last position gets them: it alone predicts the next token."""
        x = self.hidden(ids, caches, packing)  # [B, T, D]
        if last_only:
            x = x[:, -1:]  # [B, 1, D], so T is 1 below
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


def token_id(tokenizer: Tokenizer, token: str) -> int:
    """The id of a special token, such as "<|endoftext|>"."""
    found = tokenizer.token_to_id(token)
    assert found is not None, token
    return found

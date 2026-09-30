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
    S   residual streams (hc_mult)
    m   tokens pooled into one compressed entry (the layer's compression ratio)
    C   compressed entries, T // m (0 in sliding-window-only layers)
    Hi  heads of the indexer (index_n_heads)
    Di  width of one indexer head (index_head_dim)
    k   compressed entries each query selects (index_topk)
    b   compressed entries per candidate block (candidate_block_size), C/b rounds up
    n   tokens in the longest hashed n-gram (engram_max_ngram_size)
    He  hash heads per n-gram order (engram_n_heads)
    R   table rows each token looks up, (n-1)*He
    De  width of one table row (engram_head_dim)
    L   backbone layers DSpark reads (dspark_target_layer_ids)
    A   anchors per sequence, the positions draft blocks start from
    P   draft positions per block, the anchor then placeholders (dspark_block_size)
    Dm  width of the Markov head's token vectors (dspark_markov_rank)
"""

import math

import numpy as np
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
HC_MULT = 4
HC_SINKHORN_ITERS = 20
HC_EPS = 1e-6
HC_SCALE_INIT = 0.01
ENGRAM_HASH_SEED = 10007
ENGRAM_GATE_EPS = 1e-6
DSPARK_CE_WEIGHT = 0.1
DSPARK_TV_WEIGHT = 0.9
DSPARK_CONFIDENCE_WEIGHT = 1.0


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (Linear, Embedding)):
        nn.init.normal_(module.weight, std=INIT_STD)
    if isinstance(module, Attention):
        nn.init.normal_(module.attn_sink, std=INIT_STD)
        nn.init.normal_(module.o_group_weight, std=INIT_STD)


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


def apply_rope(
    x: Tensor,  # [B, T, H, Dr] or [B, T, Dr]
    freqs_cis: Tensor,  # [T, Dr/2] or [B, T, Dr/2]
    inverse: bool = False,
) -> Tensor:
    pairs = x.float()  # [B, T, H, Dr]
    pairs = pairs.unflatten(-1, (-1, 2))  # [B, T, H, Dr/2, 2]
    pairs = torch.view_as_complex(pairs)  # [B, T, H, Dr/2] complex

    if inverse:
        freqs_cis = freqs_cis.conj()  # the conjugate rotates by minus the angle
    if x.ndim == 4:
        freqs_cis = freqs_cis.unsqueeze(-2)  # [B, T, 1, Dr/2], every head turns the same way

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


def compressed_mask(seq_len: int, ratio: int, device: torch.device) -> Tensor:
    positions = torch.arange(seq_len, device=device)  # [T]
    entries = torch.arange(seq_len // ratio, device=device)  # [C]
    complete = (positions + 1) // ratio  # [T]
    return entries[None, :] < complete[:, None]  # [T, C]


def draft_mask(anchors: Tensor, seq_len: int, block_size: int, window: int) -> Tensor:  # [B, A]
    positions = torch.arange(seq_len, device=anchors.device)  # [T]
    distance = anchors[..., None] - positions  # [B, A, T]
    before_anchor = distance >= 1  # [B, A, T]
    inside_window = distance <= window  # [B, A, T]
    context = before_anchor & inside_window  # [B, A, T]
    context = context.repeat_interleave(block_size, dim=1)  # [B, A*P, T]

    block_ids = torch.arange(anchors.size(1), device=anchors.device)  # [A]
    block_ids = block_ids.repeat_interleave(block_size)  # [A*P]
    same_block = block_ids[:, None] == block_ids[None, :]  # [A*P, A*P], no causal mask inside
    same_block = same_block.expand(anchors.size(0), -1, -1)  # [B, A*P, A*P]
    return torch.cat([context, same_block], dim=-1)  # [B, A*P, T+A*P]


def select_topk(
    scores: Tensor,  # [B, T, C]
    visible: Tensor,  # [T, C] or [B, T, C]
    k: int,
) -> Tensor:
    k = min(k, scores.size(-1))
    scores = scores.masked_fill(~visible, -torch.inf)  # [B, T, C]
    top = scores.topk(k, dim=-1)  # [B, T, k], [B, T, k]
    reachable = top.values > -torch.inf  # [B, T, k]
    selected = torch.zeros_like(scores, dtype=torch.bool)  # [B, T, C]
    return selected.scatter(-1, top.indices, reachable)  # [B, T, C]


def select_candidates(
    scores: Tensor,  # [B, T, C]
    visible: Tensor,  # [T, C]
    num_blocks: int,
    block_size: int,
) -> Tensor:
    width = scores.size(-1)  # C
    padding = -width % block_size  # entries missing from the last block
    scores = scores.masked_fill(~visible, -torch.inf)  # [B, T, C]
    scores = F.pad(scores, (0, padding), value=-torch.inf)  # [B, T, C/b*b]
    blocks = scores.unflatten(-1, (-1, block_size))  # [B, T, C/b, b]
    block_scores = blocks.amax(dim=-1)  # [B, T, C/b]

    newest = (visible.sum(dim=-1) - 1) // block_size  # [T], block holding the newest entry
    block_ids = torch.arange(block_scores.size(-1), device=scores.device)  # [C/b]
    is_newest = block_ids[None, :] == newest[:, None]  # [T, C/b]
    block_scores = block_scores.masked_fill(is_newest, torch.inf)  # [B, T, C/b], always kept

    reachable = block_scores > -torch.inf  # [B, T, C/b]
    kept = select_topk(block_scores, reachable, num_blocks)  # [B, T, C/b]
    kept = kept.repeat_interleave(block_size, dim=-1)  # [B, T, C/b*b]
    return kept[..., :width]  # [B, T, C]


def indexer_kl(
    scores: Tensor,  # [B, T, C]
    attention: Tensor,  # [B, T, C]
    selected: Tensor,  # [B, T, C]
) -> Tensor:
    rows = selected.any(dim=-1)  # [B, T]
    scores = scores[rows]  # [N, C]
    attention = attention[rows]  # [N, C]
    selected = selected[rows]  # [N, C]

    target = attention.detach()  # [N, C]
    target = target / (target.sum(dim=-1, keepdim=True) + NORM_EPS)  # [N, C]

    scores = scores.masked_fill(~selected, -torch.inf)  # [N, C]
    log_probs = scores.log_softmax(dim=-1)  # [N, C]
    log_probs = log_probs.masked_fill(~selected, 0.0)  # [N, C]

    kl = torch.xlogy(target, target) - target * log_probs  # [N, C]
    return kl.sum() / rows.numel()


def sample_anchors(ids: Tensor, num_anchors: int, block_size: int) -> Tensor:  # [B, T]
    num_choices = ids.size(1) - block_size - 1  # 1..T-P-1: context before, P targets after
    weights = torch.ones(ids.size(0), num_choices, device=ids.device)  # [B, T-P-1]
    anchors = torch.multinomial(weights, num_anchors)  # [B, A], distinct in each sequence
    return anchors + 1  # [B, A]


def dspark_loss(
    draft_logits: Tensor,  # [B, A, P, vocab]
    confidence: Tensor,  # [B, A, P], before the sigmoid
    backbone_logits: Tensor,  # [B, T, vocab]
    ids: Tensor,  # [B, T]
    anchors: Tensor,  # [B, A]
) -> Tensor:
    offsets = torch.arange(draft_logits.size(2), device=ids.device)  # [P]
    positions = anchors[..., None] + offsets  # [B, A, P]
    batch = torch.arange(ids.size(0), device=ids.device)  # [B]
    batch = batch[:, None, None]  # [B, 1, 1]
    targets = ids[batch, positions + 1]  # [B, A, P], the true next tokens
    backbone_probs = backbone_logits[batch, positions]  # [B, A, P, vocab]
    backbone_probs = backbone_probs.detach().float().softmax(dim=-1)  # [B, A, P, vocab]

    log_probs = draft_logits.float().log_softmax(dim=-1)  # [B, A, P, vocab]
    draft_probs = log_probs.exp()  # [B, A, P, vocab]
    ce = -log_probs.gather(-1, targets[..., None])  # [B, A, P, 1]
    ce = ce.squeeze(-1)  # [B, A, P]

    gap = draft_probs - backbone_probs  # [B, A, P, vocab]
    tv = gap.abs().sum(dim=-1)  # [B, A, P], the paper's L1 distance between the two
    acceptance = 1 - tv.detach() / 2  # [B, A, P], chance verification keeps the drafted token
    bce = F.binary_cross_entropy_with_logits(confidence.float(), acceptance, reduction="none")

    position_weights = torch.exp(-offsets / draft_logits.size(2))  # [P], earlier slots count more
    loss = DSPARK_CE_WEIGHT * ce  # [B, A, P]
    loss = loss + DSPARK_TV_WEIGHT * tv  # [B, A, P]
    loss = loss + DSPARK_CONFIDENCE_WEIGHT * bce  # [B, A, P]
    loss = loss * position_weights  # [B, A, P]
    loss = loss.sum(dim=-1)  # [B, A]
    return loss.mean()


def einsum32(equation: str, *operands: Tensor) -> Tensor:
    operands32 = [operand.float() for operand in operands]
    with torch.autocast(operands32[0].device.type, enabled=False):
        return torch.einsum(equation, *operands32)


def sinkhorn(logits: Tensor) -> Tensor:  # [..., S, S]
    mix = logits.float()  # [..., S, S]
    mix = mix.softmax(dim=-1)
    mix = mix + HC_EPS
    mix = mix / (mix.sum(dim=-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITERS - 1):
        mix = mix / (mix.sum(dim=-1, keepdim=True) + HC_EPS)
        mix = mix / (mix.sum(dim=-2, keepdim=True) + HC_EPS)
    return mix  # [..., S, S]


def hc_expand(x: Tensor) -> tuple[Tensor, Tensor]:  # [B, T, D]
    streams = x[:, :, None, :]  # [B, T, 1, D]
    streams = streams.expand(-1, -1, HC_MULT, -1)  # [B, T, S, D]
    pre = torch.zeros(streams.shape[:-1], dtype=torch.float32, device=x.device)  # [B, T, S]
    pre[..., 0] = 1.0  # [B, T, S], the first block reads stream 0
    return streams, pre


def hc_pre(x: Tensor, pre: Tensor) -> Tensor:  # [B, T, S, D], [B, T, S]
    read = einsum32("btsd,bts->btd", x, pre)  # [B, T, D]
    return read.to(x.dtype)  # [B, T, D]


def hc_post(
    out: Tensor,  # [B, T, D]
    residual: Tensor,  # [B, T, S, D]
    post: Tensor,  # [B, T, S]
    comb: Tensor,  # [B, T, S, S]
) -> Tensor:
    written = einsum32("btd,bts->btsd", out, post)  # [B, T, S, D]
    mixed = einsum32("btjd,btjk->btkd", residual, comb)  # [B, T, S, D]
    streams = written + mixed  # [B, T, S, D]
    return streams.to(out.dtype)  # [B, T, S, D]


def rms_norm(x: Tensor, eps: float = NORM_EPS) -> Tensor:  # [..., dim]
    x32 = x.float()  # [..., dim]
    mean_square = x32.square().mean(dim=-1, keepdim=True)  # [..., 1]
    normed = x32 * torch.rsqrt(mean_square + eps)  # [..., dim]
    return normed.to(x.dtype)  # [..., dim]


def is_prime(n: int) -> bool:
    return n > 1 and all(n % divisor for divisor in range(2, math.isqrt(n) + 1))


def primes_from(start: int, count: int) -> list[int]:
    primes = []
    candidate = start
    while len(primes) < count:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def hash_multipliers(layer_id: int, size: int, vocab_size: int) -> Tensor:
    bound = torch.iinfo(torch.int64).max // vocab_size // 2  # id * multiplier stays inside int64
    generator = np.random.default_rng(ENGRAM_HASH_SEED * layer_id)
    multipliers = generator.integers(bound, size=size)  # [n]
    multipliers = torch.from_numpy(multipliers)  # [n]
    return 2 * multipliers + 1  # [n], odd


def ngram_tokens(ids: Tensor, size: int, pad_id: int) -> Tensor:  # [B, T]
    padded = F.pad(ids, (size - 1, 0), value=pad_id)  # [B, n-1+T]
    windows = padded.unfold(-1, size, 1)  # [B, T, n], oldest first
    return windows.flip(-1)  # [B, T, n], newest first


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
    def __init__(self, dim: int, eps: float = NORM_EPS) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # [dim]
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:  # [..., dim]
        return rms_norm(x, self.eps) * self.weight  # [..., dim]


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


class Compressor(nn.Module):
    def __init__(self, dim: int, head_dim: int, ratio: int) -> None:
        super().__init__()
        self.ratio = ratio
        self.kv_proj = Linear(dim, head_dim, float32=ratio > 1)
        if ratio > 1:
            self.gate_proj = Linear(dim, head_dim, float32=True)
        self.norm = RMSNorm(head_dim)

    def forward(self, x: Tensor) -> Tensor:  # [B, T, D]
        if self.ratio == 1:
            kv = self.kv_proj(x)  # [B, T, Dh]
            return self.norm(kv)  # [B, T, Dh]

        num_entries = x.size(1) // self.ratio  # C
        x = x[:, : num_entries * self.ratio]  # [B, C*m, D]

        kv = self.kv_proj(x)  # [B, C*m, Dh]
        kv = kv.unflatten(1, (-1, self.ratio))  # [B, C, m, Dh]

        gate = self.gate_proj(x)  # [B, C*m, Dh]
        gate = gate.unflatten(1, (-1, self.ratio))  # [B, C, m, Dh]
        weights = gate.softmax(dim=2)  # [B, C, m, Dh]

        pooled = einsum32("bcmd,bcmd->bcd", kv, weights)  # [B, C, Dh]
        pooled = pooled.to(x.dtype)  # [B, C, Dh]
        return self.norm(pooled)  # [B, C, Dh]


class SharedAttention:
    """What Full and Reindex layers hand down the stack. Layers run in order and every source
    writes before the layers after it read, so one slot per tensor is enough."""

    latents: Tensor  # [B, C, Dh], main KV after RoPE, from the latest Full layer
    index_keys: Tensor  # [B, C, Di], from the latest Full layer
    selected: Tensor  # [B, T, C], from the latest Full or Reindex layer
    candidates: Tensor  # [B, T, C], from the candidate source layer


class Indexer(nn.Module):
    def __init__(self, config: Config, layer_id: int) -> None:
        super().__init__()
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.weights_scale = (self.num_heads * self.head_dim) ** -0.5
        self.compress_ratio = config.compress_ratios[layer_id]
        self.topk = config.index_topk
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.is_candidate_source = layer_id == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_id

        self.q_up_proj = Linear(config.q_lora_rank, self.num_heads * self.head_dim)
        self.weights_proj = Linear(config.hidden_size, self.num_heads)
        if layer_id in config.kv_source_layer_ids:
            self.k_proj = Linear(config.head_dim, self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def keys(self, latents: Tensor, latent_freqs: Tensor) -> Tensor:  # [B, C, Dh], [C, Dr/2]
        latents = latents.detach()  # [B, C, Dh]
        k = self.k_proj(latents)  # [B, C, Di]
        k = self.k_norm(k)  # [B, C, Di]
        return apply_partial_rope(k, latent_freqs)  # [B, C, Di]

    def forward(
        self,
        x: Tensor,  # [B, T, D]
        qr: Tensor,  # [B, T, Dq]
        k: Tensor,  # [B, C, Di]
        freqs_cis: Tensor,  # [T, Dr/2]
    ) -> Tensor:
        x = x.detach()  # [B, T, D]
        qr = qr.detach()  # [B, T, Dq]

        q = self.q_up_proj(qr)  # [B, T, Hi*Di]
        q = q.unflatten(-1, (self.num_heads, self.head_dim))  # [B, T, Hi, Di]
        q = apply_partial_rope(q, freqs_cis)  # [B, T, Hi, Di]

        weights = self.weights_proj(x)  # [B, T, Hi]
        weights = weights * self.weights_scale  # [B, T, Hi]

        scores = einsum32("bthd,bcd->bthc", q, k)  # [B, T, Hi, C]
        scores = scores.relu()  # [B, T, Hi, C]
        return einsum32("bthc,bth->btc", scores, weights)  # [B, T, C]

    def select(self, scores: Tensor, shared: SharedAttention) -> Tensor:  # [B, T, C]
        searchable = compressed_mask(scores.size(1), self.compress_ratio, scores.device)  # [T, C]
        if self.is_candidate_source:
            shared.candidates = select_candidates(
                scores, searchable, self.candidate_topk_blocks, self.candidate_block_size
            )  # [B, T, C]
        if self.uses_candidates:
            searchable = searchable & shared.candidates  # [B, T, C]
        return select_topk(scores, searchable, self.topk)  # [B, T, C]


class Attention(nn.Module):
    def __init__(self, config: Config, layer_id: int) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.sliding_window = config.sliding_window
        self.compress_ratio = config.compress_ratios[layer_id]

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

        self.compressor: Compressor | None = None
        self.indexer: Indexer | None = None
        if layer_id in config.kv_source_layer_ids:
            self.compressor = Compressor(config.hidden_size, self.head_dim, self.compress_ratio)
        if layer_id in config.index_source_layer_ids:
            self.indexer = Indexer(config, layer_id)

        if self.compress_ratio:
            rope_theta = COMPRESS_ROPE_THETA
            original_seq_len = config.original_max_position_embeddings
        else:
            rope_theta = ROPE_THETA
            original_seq_len = 0

        freqs_cis = rope_freqs_cis(
            config.qk_rope_head_dim,
            config.max_position_embeddings,
            rope_theta,
            original_seq_len,
            config.rope_factor,
        )
        self.freqs_cis = nn.Buffer(freqs_cis, persistent=False)  # [max T, Dr/2]

    def queries(self, x: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:  # [B, T, D]
        qr = self.q_down_proj(x)  # [B, T, Dq]
        qr = self.q_norm(qr)  # [B, T, Dq]
        q = self.q_up_proj(qr)  # [B, T, H*Dh]
        q = q.unflatten(-1, (self.num_heads, self.head_dim))  # [B, T, H, Dh]
        q = apply_partial_rope(q, freqs_cis)  # [B, T, H, Dh]
        return qr, q

    def window_kv(self, x: Tensor, freqs_cis: Tensor) -> Tensor:  # [B, T, D]
        kv = self.kv_proj(x)  # [B, T, Dh]
        kv = self.kv_norm(kv)  # [B, T, Dh]
        return apply_partial_rope(kv, freqs_cis)  # [B, T, Dh]

    def attend(
        self,
        q: Tensor,  # [B, T, H, Dh]
        kv: Tensor,  # [B, T+C, Dh]
        visible: Tensor,  # [B, T, T+C]
    ) -> tuple[Tensor, Tensor]:
        scores = einsum32("bthd,bsd->bhts", q, kv)  # [B, H, T, T+C]
        scores = scores * self.softmax_scale  # [B, H, T, T+C]
        visible = visible.unsqueeze(1)  # [B, 1, T, T+C]
        scores = scores.masked_fill(~visible, -torch.inf)  # [B, H, T, T+C]

        sink = self.attn_sink.view(1, -1, 1, 1)  # [1, H, 1, 1]
        sink = sink.expand(scores.size(0), -1, scores.size(2), -1)  # [B, H, T, 1]
        scores = torch.cat([scores, sink], dim=-1)  # [B, H, T, T+C+1]

        probs = scores.softmax(dim=-1)  # [B, H, T, T+C+1]
        probs = probs[..., :-1]  # [B, H, T, T+C]
        weights = probs.to(kv.dtype)  # [B, H, T, T+C]
        o = torch.einsum("bhts,bsd->bthd", weights, kv)  # [B, T, H, Dh]
        return o, probs  # [B, T, H, Dh], [B, H, T, T+C]

    def project_out(self, o: Tensor, freqs_cis: Tensor) -> Tensor:  # [B, T, H, Dh]
        o = apply_partial_rope(o, freqs_cis, inverse=True)  # [B, T, H, Dh]
        o = o.flatten(-2)  # [B, T, H*Dh]
        o = o.unflatten(-1, (self.o_groups, -1))  # [B, T, G, (H/G)*Dh]

        o = torch.einsum("btgd,grd->btgr", o, self.o_group_weight)  # [B, T, G, Do]
        o = o.flatten(-2)  # [B, T, G*Do]
        return self.o_proj(o)  # [B, T, D]

    def draft(
        self,
        x: Tensor,  # [B, A*P, D]
        context: Tensor,  # [B, T, D]
        positions: Tensor,  # [B, A*P]
        visible: Tensor,  # [B, A*P, T+A*P]
    ) -> Tensor:
        freqs_cis = self.freqs_cis[positions]  # [B, A*P, Dr/2]
        context_freqs = self.freqs_cis[: context.size(1)]  # [T, Dr/2]
        _, q = self.queries(x, freqs_cis)  # [B, A*P, H, Dh]

        context_kv = self.window_kv(context, context_freqs)  # [B, T, Dh]
        draft_kv = self.window_kv(x, freqs_cis)  # [B, A*P, Dh]
        kv = torch.cat([context_kv, draft_kv], dim=1)  # [B, T+A*P, Dh]

        o, _ = self.attend(q, kv, visible)  # [B, A*P, H, Dh]
        return self.project_out(o, freqs_cis)  # [B, A*P, D]

    def forward(self, x: Tensor, shared: SharedAttention) -> tuple[Tensor, Tensor]:  # [B, T, D]
        freqs_cis = self.freqs_cis[: x.size(1)]  # [T, Dr/2]
        qr, q = self.queries(x, freqs_cis)  # [B, T, Dq], [B, T, H, Dh]
        kv = self.window_kv(x, freqs_cis)  # [B, T, Dh]
        visible = sliding_window_mask(x.size(1), self.sliding_window, x.device)  # [T, T]
        visible = visible.expand(x.size(0), -1, -1)  # [B, T, T]

        if self.compressor is not None and self.indexer is not None:
            latents = self.compressor(x)  # [B, C, Dh]
            latent_freqs = freqs_cis[:: self.compress_ratio]  # [ceil(T/m), Dr/2]
            latent_freqs = latent_freqs[: latents.size(1)]  # [C, Dr/2]
            shared.index_keys = self.indexer.keys(latents, latent_freqs)  # [B, C, Di]
            shared.latents = apply_partial_rope(latents, latent_freqs)  # [B, C, Dh]

        index_scores = None
        if self.indexer is not None:
            index_scores = self.indexer(x, qr, shared.index_keys, freqs_cis)  # [B, T, C]
            shared.selected = self.indexer.select(index_scores, shared)  # [B, T, C]

        if self.compress_ratio:
            kv = torch.cat([kv, shared.latents], dim=1)  # [B, T+C, Dh]
            visible = torch.cat([visible, shared.selected], dim=-1)  # [B, T, T+C]

        o, probs = self.attend(q, kv, visible)  # [B, T, H, Dh], [B, H, T, T+C]
        indexer_loss = torch.zeros((), device=x.device)  # scalar
        if index_scores is not None:
            attention = probs[..., x.size(1) :]  # [B, H, T, C]
            attention = attention.sum(dim=1)  # [B, T, C]
            indexer_loss = indexer_kl(index_scores, attention, shared.selected)  # scalar
        return self.project_out(o, freqs_cis), indexer_loss  # [B, T, D], scalar


class Gate(nn.Module):
    def __init__(self, config: Config, n_routed_experts: int, num_experts_per_tok: int) -> None:
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor

        self.score_proj = Linear(config.hidden_size, self.n_routed_experts, float32=True)
        self.bias = nn.Buffer(torch.zeros(self.n_routed_experts, dtype=torch.float32))  # [E]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # [N, D]
        scores = self.score_proj(x)  # [N, E]
        scores = F.softplus(scores).sqrt()  # [N, E]

        biased = scores + self.bias  # [N, E]
        expert_ids = biased.topk(self.num_experts_per_tok, dim=-1).indices  # [N, K]

        weights = scores.gather(-1, expert_ids)  # [N, K]
        if self.num_experts_per_tok > 1:  # normalizing a single weight would pin it to 1
            weights = weights / (weights.sum(dim=-1, keepdim=True) + NORM_EPS)  # [N, K]
        weights = weights * self.routed_scaling_factor  # [N, K]
        return weights, expert_ids


class MixtureOfExperts(nn.Module):
    def __init__(self, config: Config, n_routed_experts: int, num_experts_per_tok: int) -> None:
        super().__init__()
        shared_size = config.n_shared_experts * config.moe_intermediate_size

        self.gate = Gate(config, n_routed_experts, num_experts_per_tok)
        self.experts = nn.ModuleList(
            FeedForward(config.hidden_size, config.moe_intermediate_size)
            for _ in range(n_routed_experts)
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


class HyperConnection(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        num_mixes = (2 + HC_MULT) * HC_MULT  # (2+S)*S
        self.mix_proj = Linear(HC_MULT * dim, num_mixes, float32=True)
        self.bias = nn.Parameter(torch.zeros(num_mixes, dtype=torch.float32))  # [(2+S)*S]
        self.scale = nn.Parameter(torch.full((3,), HC_SCALE_INIT, dtype=torch.float32))  # [3]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:  # [B, T, S, D]
        x = x.flatten(-2)  # [B, T, S*D]
        x = x.float()  # [B, T, S*D]
        x = rms_norm(x)  # [B, T, S*D]
        mixes = self.mix_proj(x)  # [B, T, (2+S)*S]

        sizes = [HC_MULT, HC_MULT, HC_MULT * HC_MULT]
        pre, post, comb = mixes.split(sizes, dim=-1)  # [B, T, S], [B, T, S], [B, T, S*S]
        pre_bias, post_bias, comb_bias = self.bias.split(sizes)  # [S], [S], [S*S]

        pre = pre * self.scale[0] + pre_bias  # [B, T, S]
        pre = pre.sigmoid()  # [B, T, S]
        pre = pre + HC_EPS  # [B, T, S]

        post = post * self.scale[1] + post_bias  # [B, T, S]
        post = 2 * post.sigmoid()  # [B, T, S]

        comb = comb * self.scale[2] + comb_bias  # [B, T, S*S]
        comb = comb.unflatten(-1, (HC_MULT, HC_MULT))  # [B, T, S, S]
        comb = sinkhorn(comb)  # [B, T, S, S]
        return pre, post, comb  # [B, T, S], [B, T, S], [B, T, S, S]


class Engram(nn.Module):
    def __init__(self, config: Config, layer_id: int) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.max_ngram_size = config.engram_max_ngram_size
        self.pad_id = config.engram_pad_token_id
        num_orders = self.max_ngram_size - 1  # n-1
        num_lookups = num_orders * config.engram_n_heads  # R

        index = config.engram_layer_ids.index(layer_id)
        drawn = primes_from(config.engram_vocab_size, (index + 1) * num_lookups)
        primes = torch.tensor(drawn[-num_lookups:])  # [R], none shared with another layer
        offsets = primes.cumsum(dim=0) - primes  # [R], first row of each hash head's range
        multipliers = hash_multipliers(layer_id, self.max_ngram_size, config.vocab_size)  # [n]
        self.primes = nn.Buffer(primes.view(num_orders, -1), persistent=False)  # [n-1, He]
        self.offsets = nn.Buffer(offsets, persistent=False)  # [R]
        self.multipliers = nn.Buffer(multipliers, persistent=False)  # [n]

        self.table = Embedding(int(primes.sum()), config.engram_head_dim)
        self.k_proj = Linear(num_lookups * config.engram_head_dim, HC_MULT * self.dim)
        self.v_proj = Linear(num_lookups * config.engram_head_dim, self.dim)
        self.q_weight = nn.Parameter(torch.ones(HC_MULT, self.dim))  # [S, D]
        self.k_weight = nn.Parameter(torch.ones(HC_MULT, self.dim))  # [S, D]

    def rows(self, ids: Tensor) -> Tensor:  # [B, T]
        tokens = ngram_tokens(ids, self.max_ngram_size, self.pad_id)  # [B, T, n]
        products = tokens * self.multipliers  # [B, T, n]

        ngram_hashes = []
        running = products[..., 0]  # [B, T]
        for i in range(1, self.max_ngram_size):
            running = running ^ products[..., i]  # [B, T], hash of the (i+1)-gram
            ngram_hashes.append(running)
        hashes = torch.stack(ngram_hashes, dim=-1)  # [B, T, n-1]

        rows = hashes[..., None] % self.primes  # [B, T, n-1, He]
        rows = rows.flatten(-2)  # [B, T, R]
        return rows + self.offsets  # [B, T, R]

    def forward(self, x: Tensor, ids: Tensor) -> Tensor:  # [B, T, S, D], [B, T]
        rows = self.rows(ids)  # [B, T, R]
        memory = self.table(rows)  # [B, T, R, De]
        memory = memory.flatten(-2)  # [B, T, R*De]

        value = self.v_proj(memory)  # [B, T, D]
        key = self.k_proj(memory)  # [B, T, S*D]
        key = key.unflatten(-1, (HC_MULT, self.dim))  # [B, T, S, D]

        query = rms_norm(x.float()) * self.q_weight  # [B, T, S, D], the gate runs in float32
        key = rms_norm(key.float()) * self.k_weight  # [B, T, S, D]
        match = einsum32("btsd,btsd->bts", query, key)  # [B, T, S]
        match = match * self.dim**-0.5  # [B, T, S]

        gate = match.abs().clamp_min(ENGRAM_GATE_EPS)  # [B, T, S], keeps sqrt's gradient finite
        gate = gate.sqrt().copysign(match)  # [B, T, S], signed square root
        gate = gate.sigmoid()  # [B, T, S]

        written = einsum32("bts,btd->btsd", gate, value)  # [B, T, S, D]
        streams = x + written  # [B, T, S, D]
        return streams.to(x.dtype)  # [B, T, S, D]


class Block(nn.Module):
    def __init__(self, config: Config, layer_id: int) -> None:
        super().__init__()
        self.engram: Engram | None = None  # run by the Transformer, before DSpark reads the input
        if layer_id in config.engram_layer_ids:
            self.engram = Engram(config, layer_id)
        self.attn_hc = HyperConnection(config.hidden_size)
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = Attention(config, layer_id)
        self.ffn_hc = HyperConnection(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.ffn = MixtureOfExperts(config, config.n_routed_experts, config.num_experts_per_tok)

    def forward(
        self,
        x: Tensor,  # [B, T, S, D]
        pre: Tensor,  # [B, T, S]
        shared: SharedAttention,
    ) -> tuple[Tensor, Tensor, Tensor]:
        attn_pre, attn_post, attn_comb = self.attn_hc(x)  # [B, T, S], [B, T, S], [B, T, S, S]
        h = hc_pre(x, pre)  # [B, T, D]
        h = self.attn_norm(h)  # [B, T, D]
        h, indexer_loss = self.attn(h, shared)  # [B, T, D], scalar
        x = hc_post(h, x, attn_post, attn_comb)  # [B, T, S, D]

        ffn_pre, ffn_post, ffn_comb = self.ffn_hc(x)  # [B, T, S], [B, T, S], [B, T, S, S]
        h = hc_pre(x, attn_pre)  # [B, T, D]
        h = self.ffn_norm(h)  # [B, T, D]
        h = self.ffn(h)  # [B, T, D]
        x = hc_post(h, x, ffn_post, ffn_comb)  # [B, T, S, D]
        return x, ffn_pre, indexer_loss


class DSparkBlock(nn.Module):
    def __init__(self, config: Config, layer_id: int) -> None:
        super().__init__()
        self.attn_hc = HyperConnection(config.hidden_size)
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = Attention(config, layer_id)
        self.ffn_hc = HyperConnection(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.ffn = MixtureOfExperts(
            config, config.dspark_n_routed_experts, config.dspark_num_experts_per_tok
        )

    def forward(
        self,
        x: Tensor,  # [B, A*P, S, D]
        pre: Tensor,  # [B, A*P, S]
        context: Tensor,  # [B, T, D]
        positions: Tensor,  # [B, A*P]
        visible: Tensor,  # [B, A*P, T+A*P]
    ) -> tuple[Tensor, Tensor]:
        attn_pre, attn_post, attn_comb = self.attn_hc(x)  # [B, A*P, S], [B, A*P, S], [B, A*P, S, S]
        h = hc_pre(x, pre)  # [B, A*P, D]
        h = self.attn_norm(h)  # [B, A*P, D]
        h = self.attn.draft(h, context, positions, visible)  # [B, A*P, D]
        x = hc_post(h, x, attn_post, attn_comb)  # [B, A*P, S, D]

        ffn_pre, ffn_post, ffn_comb = self.ffn_hc(x)  # [B, A*P, S], [B, A*P, S], [B, A*P, S, S]
        h = hc_pre(x, attn_pre)  # [B, A*P, D]
        h = self.ffn_norm(h)  # [B, A*P, D]
        h = self.ffn(h)  # [B, A*P, D]
        x = hc_post(h, x, ffn_post, ffn_comb)  # [B, A*P, S, D]
        return x, ffn_pre


class DSpark(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.block_size = config.dspark_block_size
        self.noise_token_id = config.dspark_noise_token_id
        self.sliding_window = config.sliding_window
        num_targets = len(config.dspark_target_layer_ids)  # L
        confidence_width = config.hidden_size + config.dspark_markov_rank  # D+Dm

        self.context_proj = Linear(num_targets * config.hidden_size, config.hidden_size)
        self.context_norm = RMSNorm(config.hidden_size)
        self.blocks = nn.ModuleList(
            DSparkBlock(config, config.num_hidden_layers + i)
            for i in range(config.num_nextn_predict_layers)
        )
        self.out_norm = RMSNorm(config.hidden_size)
        self.markov_embed = Embedding(config.vocab_size, config.dspark_markov_rank)
        self.markov_proj = Linear(config.dspark_markov_rank, config.vocab_size)
        self.confidence_proj = Linear(confidence_width, 1, float32=True)

    def forward(
        self,
        ids: Tensor,  # [B, T]
        target_hidden: Tensor,  # [B, T, L*D]
        anchors: Tensor,  # [B, A], each in 1..T-P-1
        embed_weight: Tensor,  # [vocab, D], the backbone's
        head_weight: Tensor,  # [vocab, D], the backbone's
    ) -> tuple[Tensor, Tensor]:
        target_hidden = target_hidden.detach()  # [B, T, L*D], DSpark never trains the backbone
        embed_weight = embed_weight.detach()  # [vocab, D]
        head_weight = head_weight.detach()  # [vocab, D]

        context = self.context_proj(target_hidden)  # [B, T, D]
        context = self.context_norm(context)  # [B, T, D]

        offsets = torch.arange(self.block_size, device=ids.device)  # [P]
        positions = anchors[..., None] + offsets  # [B, A, P]
        positions = positions.flatten(1)  # [B, A*P]
        is_anchor = offsets == 0  # [P]
        is_anchor = is_anchor.repeat(anchors.size(1))  # [A*P]
        visible = draft_mask(anchors, ids.size(1), self.block_size, self.sliding_window)

        previous_ids = ids.gather(1, positions)  # [B, A*P], the token before each predicted one
        draft_ids = previous_ids.masked_fill(~is_anchor, self.noise_token_id)  # [B, A*P]
        x = embed_weight[draft_ids]  # [B, A*P, D]
        x, pre = hc_expand(x)  # [B, A*P, S, D], [B, A*P, S]
        for block in self.blocks:
            x, pre = block(x, pre, context, positions, visible)  # [B, A*P, S, D], [B, A*P, S]

        hidden = hc_pre(x, pre)  # [B, A*P, D]
        logits = self.out_norm(hidden)  # [B, A*P, D]
        logits = F.linear(logits, head_weight)  # [B, A*P, vocab]

        previous = self.markov_embed(previous_ids)  # [B, A*P, Dm]
        bias = self.markov_proj(previous)  # [B, A*P, vocab], what tends to follow that token
        logits = logits + bias  # [B, A*P, vocab]
        logits = logits.unflatten(1, (-1, self.block_size))  # [B, A, P, vocab]

        features = torch.cat([hidden, previous], dim=-1)  # [B, A*P, D+Dm]
        confidence = self.confidence_proj(features)  # [B, A*P, 1]
        confidence = confidence.squeeze(-1)  # [B, A*P]
        confidence = confidence.unflatten(1, (-1, self.block_size))  # [B, A, P], before the sigmoid
        return logits, confidence


class Transformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.dspark_target_layer_ids = config.dspark_target_layer_ids
        self.embed = Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            Block(config, layer_id) for layer_id in range(config.num_hidden_layers)
        )
        self.out_norm = RMSNorm(config.hidden_size)
        self.head = Linear(config.hidden_size, config.vocab_size)
        self.dspark = DSpark(config)

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:  # [B, T]
        x = self.embed(ids)  # [B, T, D]
        x, pre = hc_expand(x)  # [B, T, S, D], [B, T, S]

        shared = SharedAttention()
        indexer_loss = torch.zeros((), device=x.device)  # scalar
        target_hiddens = []
        for layer_id, block in enumerate(self.blocks):
            if isinstance(block.engram, Engram):
                x = block.engram(x, ids)  # [B, T, S, D]
            if layer_id in self.dspark_target_layer_ids:
                layer_input = x.mean(dim=2)  # [B, T, D], the block's input, streams averaged
                target_hiddens.append(layer_input)
            x, pre, layer_loss = block(x, pre, shared)  # [B, T, S, D], [B, T, S], scalar
            indexer_loss = indexer_loss + layer_loss  # scalar
        target_hidden = torch.cat(target_hiddens, dim=-1)  # [B, T, L*D]

        x = hc_pre(x, pre)  # [B, T, D]
        x = self.out_norm(x)  # [B, T, D]
        return self.head(x), indexer_loss, target_hidden  # [B, T, vocab], scalar, [B, T, L*D]

    def draft(self, ids: Tensor, target_hidden: Tensor, anchors: Tensor) -> tuple[Tensor, Tensor]:
        return self.dspark(ids, target_hidden, anchors, self.embed.weight, self.head.weight)

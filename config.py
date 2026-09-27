"""The sizes of one model, named as in ``config.json``. Fields are added as blocks need them."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    hidden_size: int  # D
    vocab_size: int  # rows of the embedding and of the head
    num_hidden_layers: int  # blocks in the stack
    moe_intermediate_size: int  # F of every expert
    n_routed_experts: int  # E, experts a token may be routed to
    n_shared_experts: int  # experts that run for every token
    num_experts_per_tok: int  # K, routed experts active per token
    routed_scaling_factor: float  # scales the routed weights after normalizing
    num_attention_heads: int  # H, query heads over one shared KV vector
    head_dim: int  # Dh, also the width of the KV vector
    qk_rope_head_dim: int  # Dr
    q_lora_rank: int  # width of the query bottleneck
    o_lora_rank: int  # width of the output bottleneck, per group
    o_groups: int  # groups of heads in the output projection
    sliding_window: int  # recent tokens every layer attends to
    compress_ratios: tuple[int, ...]  # m per layer, 0 for sliding window only
    max_position_embeddings: int  # rows in the RoPE table
    original_max_position_embeddings: int  # training length before the YaRN extension
    rope_factor: float  # YaRN extension factor


V41_FLASH_CONFIG = Config(
    hidden_size=5120,
    vocab_size=129280,
    num_hidden_layers=40,
    moe_intermediate_size=2304,
    n_routed_experts=384,
    n_shared_experts=1,
    num_experts_per_tok=6,
    routed_scaling_factor=1.5,
    num_attention_heads=64,
    head_dim=512,
    qk_rope_head_dim=64,
    q_lora_rank=1280,
    o_lora_rank=1024,
    o_groups=8,
    sliding_window=128,
    compress_ratios=(0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0),  # last three: DSpark layers
    max_position_embeddings=1_048_576,
    original_max_position_embeddings=65_536,
    rope_factor=16.0,
)

TINY_CONFIG = Config(
    hidden_size=64,
    vocab_size=256,
    num_hidden_layers=2,
    moe_intermediate_size=128,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=2,
    routed_scaling_factor=1.5,
    num_attention_heads=4,
    head_dim=32,
    qk_rope_head_dim=8,
    q_lora_rank=32,
    o_lora_rank=16,
    o_groups=2,
    sliding_window=8,
    compress_ratios=(0, 2),
    max_position_embeddings=256,
    original_max_position_embeddings=16,
    rope_factor=16.0,
)

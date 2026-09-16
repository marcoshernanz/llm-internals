"""The sizes of one model, named as in ``config.json``. Fields are added as blocks need them."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    hidden_size: int  # D
    moe_intermediate_size: int  # F of every expert
    num_attention_heads: int  # H, query heads over one shared KV vector
    head_dim: int  # Dh, also the width of the KV vector
    qk_rope_head_dim: int  # Dr
    q_lora_rank: int  # width of the query bottleneck
    o_lora_rank: int  # width of the output bottleneck, per group
    o_groups: int  # groups of heads in the output projection
    max_position_embeddings: int  # rows in the RoPE table
    original_max_position_embeddings: int  # training length before the YaRN extension
    rope_factor: float  # YaRN extension factor

    def __post_init__(self) -> None:
        assert self.qk_rope_head_dim % 2 == 0 and self.qk_rope_head_dim <= self.head_dim
        assert self.num_attention_heads % self.o_groups == 0


V41_FLASH_CONFIG = Config(
    hidden_size=5120,
    moe_intermediate_size=2304,
    num_attention_heads=64,
    head_dim=512,
    qk_rope_head_dim=64,
    q_lora_rank=1280,
    o_lora_rank=1024,
    o_groups=8,
    max_position_embeddings=1_048_576,
    original_max_position_embeddings=65_536,
    rope_factor=16.0,
)

TINY_CONFIG = Config(
    hidden_size=64,
    moe_intermediate_size=128,
    num_attention_heads=4,
    head_dim=32,
    qk_rope_head_dim=8,
    q_lora_rank=32,
    o_lora_rank=16,
    o_groups=2,
    max_position_embeddings=256,
    original_max_position_embeddings=16,
    rope_factor=16.0,
)

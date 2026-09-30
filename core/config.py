"""The sizes of one model, named as in a ``config.json`` (attention and MoE names follow DeepSeek)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    vocab_size: int  # rows of the embedding and of the head
    hidden_size: int  # D
    num_hidden_layers: int  # blocks in the stack
    first_k_dense_replace: int  # leading blocks with a dense feed-forward instead of MoE
    intermediate_size: int  # F of the dense feed-forward
    moe_intermediate_size: int  # F of every expert
    n_routed_experts: int  # E, experts a token may be routed to
    n_shared_experts: int  # experts that run for every token
    num_experts_per_tok: int  # K, routed experts active per token
    routed_scaling_factor: float  # scales the routed weights after normalizing
    num_attention_heads: int  # H, query heads over one shared key-value vector
    head_dim: int  # Dh, also the width of the key-value vector
    qk_rope_head_dim: int  # Dr, the last Dr of the Dh get RoPE
    q_lora_rank: int  # Dq, width of the query latent
    rope_theta: float  # base of the RoPE frequencies


TINY_CONFIG = Config(
    vocab_size=256,
    hidden_size=64,
    num_hidden_layers=4,
    first_k_dense_replace=1,
    intermediate_size=96,  # (K + shared) * expert F, so dense and MoE layers cost the same
    moe_intermediate_size=32,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=2,
    routed_scaling_factor=2.5,
    num_attention_heads=4,
    head_dim=32,
    qk_rope_head_dim=8,
    q_lora_rank=32,
    rope_theta=10000.0,
)

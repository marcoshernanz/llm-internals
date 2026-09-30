"""The sizes of one model, named as in a ``config.json`` (MLA and MoE names follow DeepSeek-V3)."""

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
    num_attention_heads: int  # H
    q_lora_rank: int  # Dq, width of the query latent
    kv_lora_rank: int  # Dc, width of the key-value latent
    qk_nope_head_dim: int  # Dn, query and key dims per head without RoPE
    qk_rope_head_dim: int  # Dr, query and key dims per head with RoPE
    v_head_dim: int  # Dv, value dims per head
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
    q_lora_rank=32,
    kv_lora_rank=32,
    qk_nope_head_dim=16,
    qk_rope_head_dim=8,
    v_head_dim=16,
    rope_theta=10000.0,
)

"""The sizes of one model, named as in ``config.json``. Fields are added as blocks need them."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
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
    kv_source_layer_ids: tuple[int, ...]  # Full layers, which compress their own main KV
    index_source_layer_ids: tuple[int, ...]  # Full and Reindex layers, which run an indexer
    index_n_heads: int  # Hi, heads of the indexer
    index_head_dim: int  # Di, width of one indexer head
    index_topk: int  # k, compressed entries each query selects
    candidate_source_layer_id: int  # layer whose indexer builds the candidate pool, -1 for none
    candidate_topk_blocks: int  # blocks kept in the candidate pool
    candidate_block_size: int  # b, compressed entries per block
    engram_layer_ids: tuple[int, ...]  # layers whose input gets an Engram lookup
    engram_max_ngram_size: int  # n, the longest n-gram; orders 2 to n are hashed
    engram_n_heads: int  # He, hash heads per n-gram order
    engram_head_dim: int  # De, width of one table row
    engram_vocab_size: int  # rows per hash head, rounded up to a distinct prime
    engram_pad_token_id: int  # fills the n-gram slots before the sequence starts
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
    kv_source_layer_ids=(2, 8, 14, 20),
    index_source_layer_ids=(2, 8, 14, 20, 24, 28, 32, 36),
    index_n_heads=32,
    index_head_dim=128,
    index_topk=512,
    candidate_source_layer_id=20,  # added in post-training, -1 while pre-training
    candidate_topk_blocks=2048,
    candidate_block_size=8,
    engram_layer_ids=(1, 14),
    engram_max_ngram_size=4,
    engram_n_heads=8,
    engram_head_dim=256,
    engram_vocab_size=16_000_000,
    engram_pad_token_id=2,
    max_position_embeddings=1_048_576,
    original_max_position_embeddings=65_536,
    rope_factor=16.0,
)

TINY_CONFIG = Config(
    hidden_size=64,
    vocab_size=256,
    num_hidden_layers=6,
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
    compress_ratios=(0, 2, 2, 1, 1, 1),  # SWA, Full, Reuse, Full, Reindex, Reuse
    kv_source_layer_ids=(1, 3),
    index_source_layer_ids=(1, 3, 4),
    index_n_heads=2,
    index_head_dim=16,
    index_topk=4,
    candidate_source_layer_id=3,
    candidate_topk_blocks=4,
    candidate_block_size=2,
    engram_layer_ids=(1, 2),
    engram_max_ngram_size=4,
    engram_n_heads=2,
    engram_head_dim=16,
    engram_vocab_size=1000,
    engram_pad_token_id=2,
    max_position_embeddings=256,
    original_max_position_embeddings=16,
    rope_factor=16.0,
)

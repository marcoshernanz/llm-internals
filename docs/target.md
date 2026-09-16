# Target: DeepSeek-V4.1-Flash

Reference notes distilled from the DeepSeek-V4.1-Flash technical report (51 pages, released with the weights on 2026-09-10), the Hugging Face model repository, and the reports it defers to. Everything below is what we imitate; sizes get scaled down, mechanisms and ratios do not. Last checked 2026-09-16.

## 1. What the model is

A multimodal MoE Transformer that takes images and text and generates text. 552B backbone parameters plus 196B Engram memory parameters. 8B parameters active per token during prefill, 16B during decode. 40 layers arranged as a Causal Encoder-Decoder: 20 encoder layers followed by 20 decoder layers. Context up to 1M tokens. Pre-trained on 45T multimodal tokens. Released under MIT.

The design goal of the release is KV-cache compression for long-horizon agents: global KV cache of 890 bytes per token, about 1/4 of DeepSeek-V4-Flash, and a persistent KV cache about 1/8 of V4-Flash through SWA Bounded Replay.

## 2. Released artifacts

| Artifact | Where | Notes |
|---|---|---|
| Weights | `deepseek-ai/DeepSeek-V4.1-Flash`, 48 safetensors shards, about 510 GB | Dense weights FP8 with 32x32 block scales (ue8m0), routed experts FP4. `config.json` carries every architectural hyperparameter. |
| Technical report | `DeepSeek_V41_Tech_Report.pdf` in the same repo | Not on arXiv at the time of writing. |
| Reference inference code | `inference/` in the same repo | `model.py` (Compressor, Indexer, Attention with CSA2 modes, Gate, MoE, Block with hyper-connections, Engram, DSparkBlock, Transformer), `engram.py` (tokenizer compression, n-gram hashing), `vision.py`, `image_processor.py`, `kernel.py` (FP8/FP4 kernels), `convert.py`, `generate.py`. Readable, tensor-parallel, not a serving engine. This is the answer key for every forward pass. |
| Prompt encoding | `encoding/` | Chat and tool-call format, with tests. |
| Evaluation | `evaluation/dsh-minimal.patch` | Patch for DeepSeek Harness Minimal mode used in the agentic evals. |

Not released: the base checkpoint (`DeepSeek-V4-Flash-Base` and `DeepSeek-V4-Pro-Base` exist on Hugging Face, `DeepSeek-V4.1-Flash-Base` does not), any training code, the pre-training data, the RL task and environment pipelines, and the DSec sandbox platform.

## 3. Architecture

Layer indices below are zero-based and match `config.json`.

| Component | Specification | Where it is specified |
|---|---|---|
| Backbone | 40 layers, hidden size 5120, vocab 129280, RMSNorm eps 1e-20, SwiGLU with clamp threshold 10 in all MoE layers | Report 4.2.1, `config.json` |
| Causal Encoder-Decoder (CED) | Layers 0-19 are the causal encoder, 20-39 the decoder. Decoder global KV entries and their compression weights are projected from the final encoder hidden state H_20 with layer-specific weights, so prefill runs only the encoder. Sliding-window KV stays layer-local in every layer, reconstructed with Decoder SWA Bounded Replay over the last n_win tokens. Inspired by YOCO. | Report 2.2, 3.2.2 |
| Layers 0-1 | Pure sliding-window attention, window 128 | Report 4.2.1 |
| CSA2, encoder | Layers 2-19, compression ratio m = 2, three groups of six. Full mode at layers 2, 8, 14; the other five layers of each group in Reuse mode. | Report 2.3, 4.2.1; `kv_source_layer_ids` |
| CSA2, decoder | Layers 20-39, compression ratio m = 1 (uncompressed), five groups of four. Layer 20 Full; layers 24, 28, 32, 36 Reindex; all others Reuse. The Full layer at 20 draws its KV from the encoder output per CED. | Report 2.3.1, 4.2.1; `index_source_layer_ids` |
| CSA2 modes | Full: own main KV, indexer K projected from main KV, own indexer Q, fresh top-k. Reindex: reuse main KV and indexer K, own indexer Q, fresh top-k. Reuse: reuse main KV and the latest top-k indices, no indexing. Every mode computes its own main Q and SWA KV. | Report 2.3.1, Figure 4 |
| CSA2 simplifications vs V4 CSA | Compressor has no overlapping source entries and no absolute positional embedding; indexer K is a projection of main KV rather than a separate path from hidden states. | Report 2.3 |
| Indexer | 32 indexer heads, head dim 128, top-k 512. Indexer Q and K in FP4 (QAT), inherited from V4. | Report 4.2.1; V4 2.3.4 |
| Hierarchical Sparse Indexer | Decoder only. Layer 20 scores the full visible context, selects its top-512, and also selects the 2048 highest-scoring blocks of 8 positions, yielding a 16384-position candidate pool. Reindex layers score only the pool. Introduced in post-training, applied identically in training and inference. | Report 2.3.2; `candidate_*` |
| Main attention | 64 query heads, head dim 512, single latent KV (num_key_value_heads 1), query compression rank 1280, output projection in 8 groups of dim 1024 (o_lora_rank 1024). RoPE on the last 64 dims, rope_theta 10000, compress_rope_theta 160000, YaRN factor 16 from a 64K original length to 1M. Attention sinks, per-head RMSNorm on Q and on the KV entry, RoPE with position -i applied to outputs: all inherited from V4. | `config.json`; V4 2.3.3 |
| SWA branch | Every CSA2 layer also attends to the last 128 uncompressed tokens. SWA KV cache stays FP8. | Report 4.2.1, 2.4.4 |
| MoE | DeepSeekMoE: 1 shared plus 384 routed experts, 6 routed active, expert intermediate dim 2304, Sqrt(Softplus) affinity, routed scaling factor 1.5, normalized top-k probabilities. MoE in all 40 layers. The report calls every FFN "standard DeepSeekMoE"; hash routing from V4 is not mentioned. | Report 2.1, 4.2.1; `config.json` |
| Load balancing | Auxiliary-loss-free bias, update speed 0.001, with separate bias vectors for text and image tokens (multimodal aux-loss-free balancing). Small sequence-level balance loss with weight 0.0001. | Report 2.1.1, 4.2.2 |
| Single-Pass mHC | Residual stream widened by n_hc = 4, mixing matrix constrained to doubly stochastic via 20 Sinkhorn-Knopp iterations, eps 1e-6. The input-mixing coefficient A is shifted by one block (block l uses A_{l-1}) so residual update, coefficient prediction and input mixing fuse into one kernel (Mega-mHC) at inference. Training keeps the multi-kernel implementation. | Report 2.4.1; mHC paper; V4 2.2 |
| Engram | 196B parameters split across two modules at layers 1 and 14. N-gram orders {2, 3, 4}, 8 hash heads of dim 256 (2048 per order), about 16M entries per table with distinct-prime sizes, compressed vocabulary of 99092 ids, FP8 tables and FP8 key/value projections. Context-aware gating and multi-branch integration as in the Engram paper; the short causal convolution is dropped. | Report 2.4.2, 3.1.3; Engram paper and `deepseek-ai/Engram` |
| DSpark | Speculative decoder replacing MTP. Three Transformer blocks (target layers 37-39) with window-128 SWA and a smaller MoE (128 routed experts, 3 active), drafting 5 positions per pass, Markov head of rank 256 over draft dependencies, confidence head for per-position acceptance, throughput-aware verification length. Trained after backbone pre-training with the backbone frozen, then kept in sync during post-training without gradients into the backbone. No MTP module during backbone pre-training. | Report 2.4.3; DSpark paper; `dspark_*` |
| Vision | DeepSeek-ViT trained from scratch: 32 layers, hidden 1024, 16 heads, intermediate 2816, patch 14, 2D-RoPE, linear patch embedding, RMSNorm, SwiGLU. 3x3 pixel-unshuffle then a 2-layer MLP projector with hidden 5120. At most 1024 image tokens, resolution roughly 544x544 to 1344x1344. | Report 2.1.1, 4.2.1; `vision_config` |
| Precision | FP8 training framework inherited from V3/V4. FP4 (MXFP4) quantization-aware training for routed experts inherited from V4. New: FP4 main KV cache via QAT in post-training, E2M1 values with one E4M3 scale per 16 channels, quantized after RoPE, no global scale. | Report 2.4.4; V4 5.2.1 |

## 4. Optimization

| Parameter group | Optimizer | Settings |
|---|---|---|
| Linear weight matrices in the backbone, Engram projections, vision-language projector | Muon with Nesterov momentum and decoupled weight decay | momentum 0.95, weight decay 0.1, update RMS rescaled to 0.18 so the AdamW learning rate is reused |
| Query and key weights | Head-wise Muon: weights split per head before the Muon update | same settings; motivated by head heterogeneity, also used by GLM-5 and Kimi-K3 |
| RMSNorm weights, biases, scaling factors | AdamW | beta1 0.9, beta2 0.95, eps 1e-20, weight decay 0.1 on norm weights, none on biases and scales |
| Token embedding, prediction head, Engram tables | Momentum update with Sinkhorn balancing (Algorithm 1 in the report): Nesterov momentum, mask near-zero rows, K alternating row/column L2 normalizations, scale by sqrt(n), learning rate times gamma | same momentum as Muon, gamma 0.18, K = 11, tau 1e-3, eps 1e-20, no weight decay. Engram learning rate scaled 5x. |
| Vision encoder | Frozen until the learning-rate decay stage except its final norm and the projector; then unfrozen with a smaller learning rate | Report 2.5 |

## 5. Pre-training recipe

- Corpus: 45T tokens, text-only to multimodal token ratio 7:1. Text pipeline extends the V4/V3 pipeline with stricter filtering of low-information model-generated text and fresher code. Multimodal data: alt-text image-text pairs, interleaved web and PDF documents scored with SmolVLM, plus grounding, OCR, image-code and computer-use data. Best-fit packing with padding rate at most 1e-4, sample-level attention masking, ultra-long documents pre-split.
- Batch size fixed at 100.6M tokens for the whole run (V4 used a batch-size ramp).
- Learning rate: linear warmup over 2000 steps, constant 2.6e-4 until 28T tokens, cosine decay to 2.6e-5 between 28T and 40T, constant 2.6e-5 from 40T to 45T.
- Sequence length: sparse attention trained from scratch at 64K with no dense warmup and no indexer warm-up stage (V4 had both), extended to 1M at 34T tokens.
- No training instability reported; V4's anticipatory-routing fallback is not mentioned.
- Vision encoder trained separately first: SigLIP sigmoid-contrastive pre-training on about 47B image-text pairs at 224x224, then autoregressive fine-tuning attached to a 4B MoE LLM on 236B tokens at 544 to 1344 pixels, after which that LLM is discarded.
- DSpark stage after backbone pre-training, backbone frozen.

## 6. Post-training recipe

The report states it introduces no new post-training algorithm. The recipe is SFT, then RL, then on-policy distillation, "without any modification beyond well-established practice used in DeepSeek-V4 development". All effort went into data and environments.

- SFT then RL with GRPO. Hyperparameters and pipeline follow V4, which follows V3.2 (specialists per domain trained with SFT plus GRPO). V3.2 supplies the GRPO modifications (unbiased KL estimate, off-policy sequence masking, kept routing and sampling masks).
- On-policy distillation as the last stage: full-vocabulary OPD from over 40 architecturally heterogeneous teachers, asynchronous rollouts, live reconfiguration of dataset mixture and teachers. Infrastructure per V4 5.2.2.
- Reasoning effort: a scalar b in 1..100 is prepended to the system prompt. For each prompt, responses are sampled per effort level and advantages are group-relative within each (prompt, effort) subgroup. Length penalty r = -min(C_max, k(b) * len / L_norm) with k(b) = k0 * exp(-(b - b_min) / tau). API tiers low/high/max map to b = 50/75/100.
- Task synthesis: tasks are (problem, environment, verifier) triplets scored on difficulty and correctness; the model is trained to construct tasks. General-agent environments from mocked tool interfaces and collected failure cases; coding environments from agent sessions and starred GitHub repositories, built and audited by cooperating agents.
- Asynchronous RL infrastructure: rollout and training colocated and time-shared; sample-level dispatch (a new prompt is dispatched once a GRPO group's worth of samples completes); concatenated routing replay for samples spanning checkpoints; per-dataset concurrency limits and early-short-sample discarding against length bias; bounded off-policy ratio and staleness loss masking; token-level interruption with persisted KV and routing state. Model merging across RL runs to reinitialize successive runs.
- Also trained in post-training: FP4 main-KV QAT, the Hierarchical Sparse Indexer candidate restriction, simulated SWA Bounded Replay, DSpark alongside the backbone, and Agent Team mode with a collaboration bonus and a critical-path latency penalty.
- DSec, the sandbox platform, runs millions of concurrent containers with per-sandbox AppArmor and eBPF policies. Crashing the environment yields a failed trajectory.

## 7. What the report defers to, and where to look

| V4.1-Flash says | Source to follow |
|---|---|
| "Building upon the optimization configuration used in DeepSeek-V4" | V4 report 2.4 and 4.2.2; Moonlight paper for Muon at scale |
| Training framework and infrastructure | V4 report 3 (fine-grained EP overlap, MegaMoE, TileLang kernels, Muon ZeRO bucketing, mHC recompute, context parallelism), which is "built upon the infrastructure developed for DeepSeek-V3" (FP8 training, EP all-to-all, DualPipe, ZeRO) |
| Attention sinks, QK RMSNorm, partial RoPE, SWA branch, indexer | V4 report 2.3 and its inference code; DSA indexer from V3.2 |
| DeepSeekMoE, auxiliary-loss-free balancing | V3 report 2.1.2; DeepSeekMoE paper; auxiliary-loss-free paper |
| Text data pipeline | V4 report 4.1, which "largely follows" V3 report 4.1 |
| mHC | mHC paper; V4 report 2.2 for the Sinkhorn parameterization |
| Engram | Engram paper and `deepseek-ai/Engram` |
| DSpark | DSpark paper |
| CED | YOCO paper (You Only Cache Once) |
| SFT, GRPO, specialists, OPD | V4 report 5.1 and 5.2, which mirror V3.2 report 3; GRPO itself from DeepSeekMath and R1 |
| FP4 QAT for experts | V4 report 5.2.1 |

## 8. Reading order

1. DeepSeek-V2 (arXiv 2405.04434): MLA and DeepSeekMoE, the compressed-KV lineage.
2. DeepSeekMath (arXiv 2402.03300), section 4: GRPO.
3. DeepSeek-V3 (arXiv 2412.19437): the full pre-training, FP8, parallelism and SFT+RL manual that everything later assumes.
4. DeepSeek-R1 (arXiv 2501.12948) and DeepSeek-V3.2 (arXiv 2512.02556): RL pipeline, DSA indexer, GRPO fixes, specialist distillation.
5. mHC (arXiv 2512.24880), Engram (arXiv 2601.07372), DSpark (arXiv 2607.05147), YOCO (arXiv 2405.05254), Muon is scalable (arXiv 2502.16982), auxiliary-loss-free balancing (arXiv 2408.15664).
6. DeepSeek-V4 (arXiv 2606.19348): CSA, HCA, Muon configuration, stability fixes, OPD, infrastructure.
7. DeepSeek-V4.1-Flash report, read last, with `inference/model.py` open beside it.

## 9. Reference code

- DeepSeek: `inference/` in the model repo (forward passes for every component), `deepseek-ai/Engram`, `deepseek-ai/DeepGEMM` (FP8/FP4 GEMM, Mega-MoE, Mega-mHC, Mega-Gate), `deepseek-ai/FlashMLA`, `deepseek-ai/DeepEP`, `deepseek-ai/DualPipe`, `deepseek-ai/EPLB`, `deepseek-ai/TileKernels`, `deepseek-ai/DeepSelect` (top-k kernels), `deepseek-ai/deepseek-harness`.
- Serving: vLLM PR 56214 and the vLLM recipe for DeepSeek-V4.1-Flash; SGLang.
- Third party: `huxint/deepseek-CED` (tiny trainable CED with SWA and bounded replay, no CSA2/MoE/mHC/Engram), `tokenbender/mHC-manifold-constrained-hyper-connections`.
- Training references for the inherited stack: `pytorch/torchtitan` deepseek_v3 (EP + FSDP + MXFP8), `character-ai/pipelining-sft` (pipeline + expert parallel SFT), `karpathy/nanochat` (single-repo tokenizer to pre-train to SFT to RL to eval pipeline).

## 10. Scale-down notes

Keep every mechanism and every ratio; shrink the sizes. The things that only exist at scale, and how to simulate them:

- CSA2 with top-k 512 and a 16384-position candidate pool is dense below a few thousand tokens. Train at 4K to 16K and shrink top-k and the pool proportionally so selection removes something.
- 1M context: reproduce the 64K-from-scratch then extend pattern by ratio, not absolute length.
- Engram tables of 16M entries per head: keep two modules, 8 heads, orders {2, 3, 4}, distinct-prime table sizes, shrink the primes.
- 45T tokens with a 100.6M batch: reproduce the shape (fixed batch, 2000-step warmup, constant, cosine to a tenth, constant tail, extend length at about 75 percent of tokens).
- Post-training environments and 40 teachers: a handful of verifiable tasks, two or three self-trained specialists as OPD teachers.
- DSec, persistent KV cache tiers, SWA Bounded Replay: implement the replay as a model-side mechanism and test that it barely changes outputs; skip the serving infrastructure.
- FP4 main KV QAT and FP4 experts need Blackwell-class hardware; FP8 first, FP4 as the last precision stage.

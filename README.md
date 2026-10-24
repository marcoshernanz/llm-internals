# llm-internals

A minimal, from-scratch implementation of how a modern LLM works: model, pre-training, custom kernels and post-training. The goal is to learn the mechanisms, not to train a useful model, and each choice follows what most major model families agree on.

## What's here

- **DeepSeek-V4.1-Flash, rebuilt** (`deepseek_v41_flash/`): the architecture from its technical report, scaled down. Compressed sparse attention with a top-k indexer and its KL loss, mHC hyper-connections with Sinkhorn mixing, Engram hashed n-gram memory, the DSpark speculative-decoding head and its loss, and the MoE. The project started here. It runs at a tiny size, is never trained, and is kept frozen with its spec in `deepseek_v41_flash/target.md`.
- **A DeepSeek-V3-style model and Triton kernels** (`core/`): MLA and a fine-grained MoE with bias-based load balancing, a TinyStories pre-training loop and a profiler. RMSNorm forward and backward are written in Triton, joined by an `autograd.Function` and checked against PyTorch in fp32 and bf16.
- **Post-training of Qwen3-0.6B-Base** (`post/`), every algorithm in plain PyTorch:
  - Qwen3 in our own code: logits match `transformers` to 5e-5, and 4-shot GSM8K is 57% on the first 200 test questions (the Qwen3 report gives 59.6% on all of them).
  - A batched KV-cache sampler: 836 tokens per second at batch 64 on a T4.
  - Full-parameter SFT with an assistant-only loss and packed sequences, no padding, on GSM8K mixed with smol-smoltalk. Asked the first 50 GSM8K test questions zero-shot in chat format, answers that close their turn go from 0% to 96% and right answers from 6% to 52%.
  - The head fused with the cross-entropy, a chunk of positions at a time: 4,096-token steps fit in 18.7 GiB on an L4, where the plain loss runs out of memory.
- **The learning part** (`learn/`): the post-training mechanisms again at their simplest, to read rather than run. Each ladder is a folder of numbered files, each file the one before it plus one mechanism, so `diff` shows the mechanism alone. `learn/README.md` explains how it works.

## Layout

| Path | Contents |
|---|---|
| `core/config.py` | model sizes, named as in a `config.json` |
| `core/model.py` | the model |
| `core/train.py` | minimal pre-training on TinyStories, one byte per token |
| `core/bench.py` | times training steps of a tiny and a wider model |
| `core/profile_step.py` | profiles those steps on a GPU, as written and with the Triton RMSNorm swapped in, and saves timelines to `out/` |
| `core/kernels/rmsnorm.py` | RMSNorm forward as one fused Triton kernel, the `autograd.Function` joining it to the backward, and `TritonRMSNorm`, the model's RMSNorm running both, which `use_triton_rmsnorm` swaps in |
| `core/kernels/rmsnorm_backward.py` | RMSNorm backward as one Triton kernel, for the input and weight gradients |
| `core/kernels/bench_rmsnorm.py` | checks both kernels and the tiny model with them swapped in against PyTorch, then times forward and backward |
| `gpu.py` | runs modules one after another on a cloud GPU (an NVIDIA L4 on Modal), where every run goes by default |
| `kaggle_run.py` | runs modules one after another on a free Kaggle T4, for when the month's Modal credits run out |
| `post/ROADMAP.md` | the post-training plan: model, hardware and the stages in order |
| `post/qwen3.py` | Qwen3 in our own code: the model, its KV cache, attention within each conversation of a packed row, loading the released checkpoint and tokenizer from the Hugging Face Hub, and saving trained weights for later runs |
| `post/sampler.py` | samples answers with a KV cache, several prompts and several rows per prompt at once: the prompts padded on the left and read once, each one's cache copied into its rows, then one token per row per step until a stop token |
| `post/gsm8k.py` | GSM8K: the data, the 4-shot prompt and two graders (strict and last number) |
| `post/chat.py` | Qwen's chat format: a conversation as token ids and the mask of the assistant's tokens, SFT's loss over them, and the model's answers to questions asked in it, sampled 256 at a time |
| `post/linear_cross_entropy.py` | the head fused with the cross-entropy loss, a chunk of positions at a time, with the gradients computed in the forward pass: a step's logits are never all in memory. Each position's loss has its own scale: 1/A for SFT's mean, an answer's reward for RL |
| `post/sft.py` | supervised fine-tuning: a full fine-tune on GSM8K train problems mixed with everyday smol-smoltalk conversations, packed end to end into steps of up to 2,048 tokens with no padding |
| `post/lora.py` | LoRA: the loaded weights frozen in bf16, a learned correction of rank r next to each linear layer at 10 times full fine-tuning's learning rate, and the embedding table trained in full |
| `post/nf4.py` | NF4, QLoRA's 4-bit format: 16 levels from the normal distribution, a scale per 64 weights, the scales quantized again to 8 bits, and a matmul that dequantizes each weight on the fly, in the backward pass too |
| `post/qlora.py` | QLoRA: LoRA with every frozen linear layer stored in NF4 |
| `post/reinforce.py` | REINFORCE: each step the model answers GSM8K questions, each answer gets a reward of 1 if right and 0 if not, and the loss is minus each answer's log probability times its reward, or, with the baseline, times the reward minus its question's mean |
| `post/ratio_clipping.py` | ratio clipping on REINFORCE with the baseline: each step's answers reused for 4 updates, each token weighted by the ratio of its probability now to when it was sampled, clipped to 1 ± 0.2 |
| `post/grpo.py` | GRPO as DeepSeekMath wrote it: ratio clipping with each advantage divided by its question's standard deviation, each answer's tokens averaged, and a KL penalty toward the SFT model |
| `post/rlvr.py` | GRPO with the fixes later recipes made, one at a time; so far the token-level loss, every answer token divided by the batch's answer tokens where GRPO divides by its own answer's length, and no division of the advantage by its question's standard deviation |
| `post/rejection_sampling.py` | rejection sampling: the SFT model answers GSM8K questions it never read, the right answers are kept, and the base model is fine-tuned again on SFT's data plus them |
| `learn/README.md` | the learning part: how its files are written, read and added |
| `learn/1_finetuning/` | SFT, LoRA, NF4 and QLoRA at their simplest, one numbered file each, indexed in its `README.md` |
| `learn/2_rl/` | pass@k, rejection sampling, REINFORCE, the baseline, ratio clipping, GRPO and its fixes at their simplest, each file the one before plus one mechanism, indexed in its `README.md` |
| `post/check/qwen3.py` | checks `post/qwen3.py` gives the same logits as `transformers` |
| `post/check/attention.py` | checks the fused attention kernel against attention written out step by step, with and without a cache |
| `post/check/kv_cache.py` | checks the KV cache gives the same logits as rerunning the whole sequence |
| `post/check/batch.py` | checks each row of a batch gets the same logits as that row on its own: with a prompt per row, with one prompt shared by every row, and with prompts of different lengths padded on the left |
| `post/check/gsm8k.py` | checks the GSM8K prompt, both graders and pass@k on cases whose results are known |
| `post/check/chat.py` | checks the chat format's text, ids and mask, and that a prompt matches the checkpoint's own chat template |
| `post/check/packing.py` | checks each conversation in a packed row gets the same logits as on its own, and shows what plain causal attention does instead |
| `post/check/lora.py` | checks LoRA starts from the loaded model's exact logits, trains only its own matrices and the embedding table, and leaves every frozen weight unchanged |
| `post/check/nf4.py` | checks NF4's levels against QLoRA's table, 4.127 bits per weight, every weight stored as its nearest level, and the matmul against autograd, keeping nothing for the backward pass |
| `post/check/loss.py` | checks the fused loss's gradients are as close to the exact ones, computed in fp32, as the plain loss's: with SFT's scales, and with signed scales per conversation, as RL's, and from per-token log probabilities with the logits computed again in the backward pass, as GRPO's |
| `post/bench/generate.py` | times generation without a cache, rerunning the whole sequence per token, then the sampler at batches of 1 to 64; prints 16 sampled answers |
| `post/bench/memory.py` | peak memory of one SFT step at 1,024 to 4,096 tokens: with no head, with the plain loss and with the fused one |
| `post/bench/finetune_memory.py` | peak memory of SFT steps for Qwen3-0.6B, 1.7B and 4B, fine-tuned fully, with LoRA and with QLoRA |
| `post/eval/gsm8k.py` | scores the base model 4-shot: greedy answers, and pass@1 to pass@16 |
| `post/eval/chat.py` | asks GSM8K questions zero-shot in chat format: how the answers end (closing the turn with `<\|im_end\|>` or not) and how many are right |
| `post/eval/sft.py` | runs SFT and measures it: held-out loss and the probability of `<\|im_end\|>` before and after, tokens read against padded batches, then the test questions asked in chat format |
| `post/eval/embeddings.py` | looks at the chat tokens' rows in the base model's embedding table: their length, how many rows are the same vector, and their nearest rows |
| `post/eval/lora.py` | runs SFT with LoRA and measures it as `post/eval/sft.py` measures full fine-tuning |
| `post/eval/nf4.py` | the weights' memory and the held-out loss in bf16 and NF4, for Qwen3-0.6B, 1.7B and 4B |
| `post/eval/qlora.py` | runs SFT with QLoRA and measures it as `post/eval/lora.py` measures LoRA |
| `post/eval/pass_at_k.py` | the SFT model on GSM8K test questions in chat format: greedy accuracy, pass@1 to pass@16, and how many questions none, some or all samples get right |
| `post/eval/rejection_sampling.py` | runs rejection sampling and measures it: the answers kept, then the retrained model as `post/eval/pass_at_k.py` measures the SFT model |
| `post/eval/rejection_control.py` | rejection sampling's control: the same questions with GSM8K's worked solutions instead of the model's own answers, retrained and measured the same way |
| `post/eval/reinforce.py` | runs REINFORCE from the SFT model and measures it: the reward every 5 steps, then pass@k as `post/eval/pass_at_k.py` measures the SFT model |
| `post/eval/baseline.py` | runs REINFORCE with the baseline, each reward minus its question's mean, and measures it as `post/eval/reinforce.py` measures REINFORCE without it |
| `post/eval/ratio_clipping.py` | runs ratio clipping from the SFT model and measures it as `post/eval/reinforce.py` measures REINFORCE, against REINFORCE and the baseline |
| `post/eval/grpo.py` | runs GRPO as DeepSeekMath wrote it and measures it as `post/eval/ratio_clipping.py` measures ratio clipping, with right and wrong answers' lengths and the KL to the SFT model |
| `post/eval/rlvr.py` | runs GRPO with the fixes so far (`post/rlvr.py`) and measures it as `post/eval/grpo.py` measures GRPO |
| `post/eval/prompts.py` | asks everyday prompts, none of them math, before and after SFT: both answers to each, how they end, and how many end the GSM8K way |
| `deepseek_v41_flash/` | the earlier DeepSeek-V4.1-Flash reimplementation |

## The model

- Pre-norm RMSNorm, plain residual connections, untied embedding and output head, no biases.
- Attention is multi-head latent attention (MLA). Queries and keys/values each pass through a small latent with an RMSNorm, which serves as the query/key norm. Positions use decoupled RoPE without YaRN. A sigmoid output gate replaces an attention sink.
- Feed-forward is SwiGLU. The first layer is dense. The others are mixture-of-experts: fine-grained routed experts plus one shared expert, sigmoid router scores, and load balancing through a per-expert bias instead of an auxiliary loss.

## Running

Every run goes to a cloud GPU, an L4 on Modal. `gpu.py` copies the local code in, so uncommitted changes run too, and runs the modules one after another in one container. Each module's output streams back and is saved to `out/<module>.log`, with any files it writes to `out/`. The last line says what the run cost:

```bash
uv run modal run gpu.py --modules post.check.kv_cache,post.eval.gsm8k
```

Modal gives $30 of credits a month, and the workspace stops at $25 of usage, so the card is never charged. To see what's spent:

```bash
uv run modal billing report --for "this month"
```

When the month's credits run out, run on a free Kaggle T4 instead. Its logs come back to `out/kaggle/output/`. A T4 can't compute in bf16, so there the model runs in fp32:

```bash
uv run python kaggle_run.py post.check.kv_cache post.eval.gsm8k
```

Open profiler timelines in https://ui.perfetto.dev.

Triton runs only on Linux with an NVIDIA GPU, so kernels run through `gpu.py`. On macOS, install Triton's Python files so the editor and ty can read them. `uv sync` removes them again:

```bash
uv pip install --python .venv --python-platform x86_64-manylinux_2_28 --no-deps "triton==3.8.0"
```

Lint and type-check with `uv run ruff format . && uv run ruff check . && uv run ty check`.

## Roadmap

1. Model. Done.
2. Minimal pre-training loop. Done: TinyStories, bytes as tokens.
3. Kernels on an NVIDIA GPU: profiling, then Triton (RMSNorm, SwiGLU, cross-entropy, attention, MoE), then CUDA. In progress: profiling and RMSNorm (forward, backward and autograd) are done.
4. Post-training of Qwen3-0.6B-Base: SFT, LoRA and QLoRA, rejection sampling, RL with verifiable rewards, environments, learned rewards, on-policy distillation, self-distillation, async rollouts. Plan in `post/ROADMAP.md`. In progress: SFT, LoRA and QLoRA, and rejection sampling are done; RL with verifiable rewards is under way, from REINFORCE to GRPO and the fixes later recipes made.

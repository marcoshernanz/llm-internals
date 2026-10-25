# Post-training roadmap

Post-train a small pretrained model step by step, in the frontier's order: SFT, then RL, then distillation from several teachers. The practitioner tools (LoRA, QLoRA, DPO) come in where we first need them. Every algorithm is written here in plain PyTorch. Only the weights, the tokenizer and the datasets are borrowed.

Researched in October 2026. Numbers marked "est." are order-of-magnitude estimates, not measurements.

## Stages

| # | Stage | Status |
|---|---|---|
| 0 | Foundations: Qwen3 in our own code, a KV-cache sampler, graders | Done |
| 1 | SFT | Done |
| 2 | LoRA, then QLoRA | Done |
| 3 | Rejection sampling | Done |
| 4 | RL with verifiable rewards | Done |
| 5 | Multi-turn environments and tools | Not started |
| 6 | Learned rewards: reward model, RLHF, DPO | Not started |
| 7 | On-policy distillation | Not started |
| 8 | Self-distillation | Not started |
| 9 | Async rollouts on two GPUs | Not started |

## The model

- **Student:** Qwen3-0.6B-Base for every stage. Qwen3-1.7B-Base, trained with LoRA, where 0.6B is too weak (e.g. Wordle).
- **Teacher:** Qwen3-8B.

### What a student needs

1. **A released base checkpoint.** Post-training starts before any chat tuning.
2. **Small enough** to fully fine-tune on a 15 GB T4.
3. **Bigger siblings with the same tokenizer.** On-policy distillation compares teacher and student token by token.
4. **Good enough at math** that RL sometimes gets a reward above zero.
5. **An architecture close to what most model families use,** simple to rewrite.
6. **Not gated** behind an access request.

### Why Qwen3-0.6B-Base

Qwen3-0.6B-Base meets all six.

- **Architecture:** a plain dense transformer.
  - RMSNorm, SwiGLU, GQA (16 query heads, 8 KV heads), RoPE and QK-norm.
  - 28 layers, width 1024.
  - That's `core/model.py` without MLA and MoE.
- **Tokenizer:** every Qwen3 size shares one tokenizer (vocab 151,936), instruct models included. Those add only 4 special tokens, which fit inside the padded vocabulary.
- **Base scores**, 4-shot chain of thought, from the Qwen3 report:

| Model | GSM8K | MATH |
|---|---|---|
| Qwen3-0.6B-Base | 59.6 | 32.4 |
| Qwen3-1.7B-Base | 75.4 | 43.5 |

- **License:** Apache-2.0.

### Rejected

| Candidate | Why not |
|---|---|
| Qwen3.5 0.8B / 2B base (Mar 2026) | The strongest alternative. But 3 of every 4 layers are gated DeltaNet (linear attention), and it adds a vision tower, a multi-token-prediction head and a 248k vocab. Linear attention isn't what most families agree on. |
| OLMo 3 | The smallest base is 7B. OLMo-2-1B, a non-Qwen model with fully open data, serves as a control (see "What to expect"). |
| SmolLM3-3B | One size only, so there's no same-tokenizer teacher. 3B is also too big to fully fine-tune here. |
| Gemma 3 1B, Llama 3.2 1B | Gated and weak at math; neither reports a base GSM8K score. Gemma also overflows in fp16. |
| Phi-4-mini | No base checkpoint. |
| LFM2, Granite 4 nano, Falcon-H1 | Hybrid state-space models. |
| Our own core model | Trained on TinyStories bytes. RL needs a model that sometimes gets the answer right. |

### Qwen's weakness

RL on Qwen math improves scores even with *random* rewards, because RL surfaces skills the model already has. GSM8K is also in Qwen's pretraining data. So:

- every RL result gets a random-reward control run;
- we lean on tasks that aren't plain math: instruction constraints, Countdown and Wordle.

## Hardware

| | Mac M4 16 GB | Kaggle 2×T4 | Modal L4 | H100, for scale |
|---|---|---|---|---|
| Compute | ~3.7 TFLOPS, no matrix units | 65 TFLOPS fp16 per card | 121 TFLOPS bf16 | 989 TFLOPS bf16 |
| Memory | ~12 GB usable by the GPU, 120 GB/s | 15 GB per card, 300 GB/s | 24 GB, 300 GB/s | 80 GB, 3.35 TB/s |
| bf16 | yes | **no**, fp16 only | yes | yes |
| Hours | unlimited | 30 h/week; a 2-card session uses one clock | ~30 h/month on $25 | – |
| 0.6B full SFT, est. | ~250–500 tok/s | ~2,600 tok/s | ~6,000 tok/s | ~50,000 tok/s |

### Memory decides the split

Full fine-tuning with AdamW keeps 16 bytes per parameter: 4 for the fp32 weights, 4 for the gradients and 8 for Adam's two moments.

- **0.6B:** 9.6 GB, which fits a T4.
- **1.7B:** 27 GB, which fits neither a T4 nor an L4. Even split across both T4s, 13.5 GB a card leaves no room for activations. Stage 2 solves this.

### Mac: write the code

Code, lint and type checks only. Every run goes to a cloud GPU, Modal while the month's credits last, then Kaggle: a 0.6B run on a 16 GB Mac swaps and leaves it unusable for anything else (the GSM8K baseline pushed 13 GB into swap).

- **PyTorch,** so the same code runs on the Mac's MPS and on CUDA unchanged. The alternative is MLX: faster on a Mac and with mature LoRA, but a second framework.

### Kaggle: free, once the Modal credits run out

About 120 free hours a month.

- **Two cards for one clock.** A session gets both T4s and is billed once, so the second card is free. Use it as follows:
  - **Paired runs, one per card:** a fix and its baseline, or a run and its random-reward control.
  - **The 8B model in NF4 next to the student:** as stage 6's judge and as stage 7's teacher.
  - **Generation and training on separate cards** (stage 9).
- **No bf16 on a T4.** Train in fp16 with fp32 master weights and loss scaling. This is a lesson in why the frontier moved to bf16.
- **First thing on falling back:** check whether fp16 overflows anywhere in the model, i.e. whether any activation passes 65,504. Until then the T4 runs in fp32.
- **Trap:** under fp16 an RL step can overflow and be skipped while the reward still logs normally, so log skipped steps.
- **No FlashAttention.** PyTorch's attention falls back to its memory-efficient kernel.
- **Launcher:** `kaggle_run.py`, like `gpu.py`, sends the local code to a T4, streams the log and pulls back the output.

### Modal: the default while the credits last

$30 of credits a month. The workspace stops at $25 of usage, so the card is never charged: about 30 hours of L4.

- **An L4 by default,** the cheapest GPU there that computes in bf16, so the model runs as it was trained. A pricier GPU only for a lesson that needs one.
- **One container per experiment.** `gpu.py` runs several modules in it, so the startup and the model download are paid once.
- **A timeout on every run,** so a stuck run can't spend the month.
- **Weights download on every run,** seconds for 0.6B. For the 8B teacher (16 GB), weigh the download time against keeping it in a Modal Volume at $0.09 per GB a month.
- **Keep the idle timeout short:** idle containers are billed.

### Not used: GCP TPUs

TPUs need JAX or torch_xla, and RL's variable-length generation keeps forcing XLA to recompile.

### Budget, est.

- **Modal first:** $25 of usage a month, about 30 L4-hours.
- **Then Kaggle:** 30 T4-hours a week. The 40–60 T4-hours all stages need fit in 1–2 weeks of it, since paired runs share a session's clock.
- **Stage 4 uses the most:** a GRPO step takes 1–3 minutes at 0.6B, and a run is 50–100 steps.

## Stages in detail

Each stage starts from a problem we can see in the previous stage's output.

### 0. Foundations: look at the base model

- **Problem:**
  - We need a real pretrained model running in our own code.
  - RL is mostly generation, and without a KV cache every new token recomputes the whole prefix.
- **Build:**
  - Qwen3's forward pass in plain PyTorch, loading the Hugging Face safetensors. Check its logits against `transformers`.
  - A batched sampler with a KV cache.
  - GSM8K graders (strict and last number) and pass@k.
- **See:**
  - The base model answers, then keeps going: it writes the next "Question:" itself and never stops.
  - 4-shot GSM8K, greedy, first 200 test questions: 57% by last number, 47% strict. Qwen3's report: 59.6%.
  - pass@k, 16 samples at temperature 1, first 50 questions: pass@1 35%, pass@16 84% by last number.
- **Moved on:** per-token log-probs in the sampler to stage 4, IFEval-style graders to just before stage 4, the fp16 check to the first Kaggle run.
- **Runs on:** Kaggle, then Modal.

### 1. SFT: teach it to take turns

- **Problem:** the base model has no turn structure and doesn't stop. `post/eval/chat.py` measures it on greedy answers to the first 200 GSM8K test questions, asked zero-shot in chat format:
  - None of the 200 closes its turn with `<|im_end|>`. 88% run to the 256-token limit; 12% end the document with `<|endoftext|>`.
  - Most begin by repeating the question.
  - 6% are right by last number, against 57% 4-shot. The model can solve them, but nothing in this format draws it out.
  - Where a held-out answer ends (`post/eval/sft.py`), the base model gives `<|im_end|>` a probability of 3.2e-7 and `<|endoftext|>` 0.93. It knows how to end a document, not a turn.
- **Build:**
  - Qwen's ChatML chat template.
  - Loss on assistant tokens only.
  - Several conversations packed per sequence, with no attention across them (done, below).
  - A full fine-tune.
  - The head fused with the loss, a chunk of positions at a time (done, below).
- **Problem we'll hit:** the logits for 8k tokens × a 152k vocab are 4.6 GiB in fp32, so the loss is computed in chunks.
- **Data:** `smol-smoltalk` (built for models under 1B), plus GSM8K as chats.
- **Check:**
  - It stops at `<|im_end|>`.
  - GSM8K accuracy, asked zero-shot in chat format.
- **First run** (`post/eval/sft.py`, which runs `post/sft.py`, 400 steps of 4 GSM8K chats, 2 minutes on the L4), on the first 50 test questions:
  - 94% close their turn with `<|im_end|>`, against 0% for the base model; its probability where answers end went from 1.3e-6 to 0.91.
  - 40% right, against 6%. 4-shot, the base model gets 57%.
  - 28% of the tokens read were padding.
- **Packing** (done): conversations laid end to end, up to 1,024 tokens per step, with FlashAttention's varlen kernel keeping attention within each one (`torch.nn.attention.varlen`).
  - Without the mask, attention crosses into the conversations before (`post/check/packing.py`). The logits move by up to 29, and the second conversation's answer loss falls from 1.74 to 0.53: it reads the problem before it as a worked example, which inference never gives it. With the mask, they match up to bf16 rounding (0.7).
  - Positions restart at 0 in each conversation, as in Hugging Face's padding-free path and Megatron; torchtitan keeps counting. With RoPE a score depends only on how far apart two tokens are, so this changes nothing beyond rounding.
  - Run on the same 1,600 conversations: 0% padding, 259,881 tokens read instead of 358,576. 279 steps of about 930 tokens instead of 400 of 4 conversations: 1.3 minutes instead of 2, peak memory 11.7 GiB instead of 14.8.
  - All 50 answers close their turn (94% before), 44% right (40%). The probability of `<|im_end|>` reaches 0.75, not 0.91: fewer, larger steps at the same learning rate.
- **GSM8K alone is too narrow** (`post/eval/prompts.py`, 8 everyday prompts, none of them math, after the packed run):
  - Only 2 close their turn, both right after a made-up GSM8K ending: "The capital of France is Paris. The answer is 1455." It learned that a turn ends after "The answer is N.", not after an answer.
  - 4 end the document with `<|endoftext|>`, as the base model does. 2 run to the token limit, one repeating "I'm doing well, thank you. How about you?".
  - What they say is mostly sound (sleep tips, black holes, an email): that came from pre-training. What's missing is when to stop.
- **Mixed with smol-smoltalk** (done): the same 1,600 GSM8K conversations plus 600 from smol-smoltalk's first train file, shuffled together.
  - Steps of up to 2,048 tokens: its conversations average 865 tokens, GSM8K's 162. The 5 longer than a step are left out; toolkits such as TRL and open-instruct cut them to length instead.
  - 2,195 conversations and 2,956 answers, about half from each set. 720K tokens, 36% of them GSM8K's.
  - Everyday prompts: 7 of 8 close their turn, and none writes "The answer is". The eighth was still mid-answer at the 160-token limit. The Spanish is now "Buenos días, mi amigo", where GSM8K alone wrote "Buenos mañana" and then "The answer is 1.".
  - GSM8K, first 50 test questions: 98% close their turn. 44% right by strict, the same as with GSM8K alone; 46% by last number. Where held-out answers end, `<|im_end|>` gets 0.90 (0.75 before); the held-out loss is 0.575 (0.569).
  - The mistakes left are the model's own, from pre-training: bacteria "generally smaller than viruses", and three sentences where two were asked for.
  - 420 steps, 3.2 minutes, 3,765 tokens per second (3,212 with 1,024-token steps). PyTorch's peak allocation was 15.7 GiB, yet twice the GPU ran out: about 21 GiB of the L4's 22 was held by PyTorch's cache. The blocks cached from earlier steps were the wrong size for this step's 1.2 GB of fp32 logits. It emptied the cache and carried on. That's the next problem.
- **Fused loss** (done): the head and the cross-entropy computed together, a chunk at a time.
  - The problem (`post/bench/memory.py`, peak GiB of one step, of the L4's 22.1; the weights and AdamW's moments are 6.7 of it):

    | tokens | no head | plain loss | its logits | fused loss | its logits |
    |---|---|---|---|---|---|
    | 1,024 | 9.7 | 11.7 | +2.0 | 11.7 | +2.0 |
    | 2,048 | 11.9 | 15.7 | +3.7 | 14.2 | +2.3 |
    | 4,096 | 16.4 | out of memory | | 18.7 | +2.3 |

    The plain loss's logits cost about 3.3 fp32 copies of `[T, V]`: the log-probabilities saved for the backward pass, the gradient coming into them, and the gradient going out.
  - The fix (`post/linear_cross_entropy.py`), as Liger Kernel does it: chunks of 1,024 positions, each chunk's gradients computed in the forward pass from `softmax - onehot`, its logits then thrown away. Its cost stays flat at about 2.3 GiB.
  - Only positions whose target counts go through the head. Liger takes every position with an ignore index; picking the counted positions first does the same math and skips the prompt's 28%. The alternatives: recompute each chunk's logits in the backward pass, one more matmul; or Cut Cross-Entropy, a kernel that never writes the logits to GPU memory.
  - The check (`post/check/loss.py`): both losses round in bf16, so each is measured against the exact gradient, the head in fp32. Over all weights both are 0.84% off. The largest error is on the same norm weight in both: 2.5% plain, 3.0% fused.
  - SFT at 2,048 tokens per step, as before: the same loss at every logged step to 2 or 3 digits. Held-out loss 0.576 (0.575 plain), `<|im_end|>` 0.89 (0.90), 50% right by strict (44%: 3 of 50 questions, within noise). Peak memory 13.9 GiB (15.7), and no cache warnings.
  - SFT at 4,096 tokens per step fits: 18.6 GiB, no warnings, 4,144 tokens per second. But it is 194 steps instead of 420, and the model learns less: `<|im_end|>` gets 0.13 where held-out answers end, and the held-out loss is 0.595. 3 of the 8 everyday answers were still mid-sentence at the 160-token limit.
    - Why: Adam moves each weight by about the learning rate per step, however many tokens the step has, so half the steps go about half as far.
    - Open recipes use 4,096-token rows too, but many rows per step and thousands of steps. Ours is 720K tokens in all, so the number of steps is the limit: back to 2,048.
- **Frontier:** everyone (DeepSeek V4.1, Qwen3, OLMo 3, Llama 4, Nemotron 3).
- **Runs on:** Modal L4, in bf16.

### 2. LoRA, then QLoRA: make bigger models fit

- **Problem:** a full 1.7B fine-tune needs 27 GB, and an 8B teacher is 16 GB in bf16.
- **Build:**
  - LoRA on every linear layer, at about 10× the full fine-tuning learning rate.
  - NF4 quantization: 4-bit weights, one scale per 64 weights, and those scales quantized again.
  - Optionally, a Triton dequantize kernel, a short return to the kernels track.
- **Check:**
  - A curve of peak memory for full fine-tuning, LoRA and QLoRA.
  - LoRA's loss tracks full fine-tuning's on the stage-1 data.
  - The 8B teacher shrinks to ~5 GB in NF4, with perplexity close to bf16.
- **The problem, seen** (`post/bench/finetune_memory.py`): a full fine-tune keeps 16 bytes per weight. For Qwen3-1.7B-Base that's 25.6 GiB of the L4's 22.1. Its forward and backward passes fit; the first optimizer step runs out of memory as AdamW creates its two moments.
- **LoRA** (done, `post/lora.py`): rank 32, alpha 32, on all 196 linear layers. Base weights frozen in bf16. LoRA's matrices at 1e-4, 10× full fine-tuning's rate.
  - The check (`post/check/lora.py`): right after LoRA is added, the logits match the loaded model's to the bit. Only the expected weights train, and after a step every frozen weight is unchanged.
  - First run, every loaded weight frozen: 20M weights train and the peak is 5.9 GiB, but the model never learns to end its turn.
    - `<|im_end|>` gets 0.00037 where held-out answers end. Its place goes to junk tokens: "The answer is 82. dóla. spep. dóla. spep. …". All 50 GSM8K answers run to the token limit.
    - Why (`post/eval/embeddings.py`): the base model never trained `<|im_start|>`'s and `<|im_end|>`'s rows. They're one short vector (length 0.35, against a median of 0.97), the same to the bit as 290 other tokens' rows.
    - Qwen3-0.6B's head is its embedding table. With it frozen, `<|im_end|>`'s logit equals those 290 logits whatever the hidden state, so its probability stays below 1 in 291. A full fine-tune trains that row and moves it away; frozen, it can't move.
  - The fix, as toolkits do when a chat format brings tokens the base model never learned: the embedding table trains in full, in fp32, at full fine-tuning's 1e-5. LoRA's 10× rate makes up for B starting at zero, and doesn't apply to a matrix trained directly. The alternative is to train only those tokens' rows.
  - With the table trained, LoRA tracks full fine-tuning: the same loss at every logged step to within 0.03. Held-out loss 0.588 (0.576 full), `<|im_end|>` 0.91 (0.89), 98% of GSM8K answers close their turn (96%), 42% right by strict (50%; 4 of 50 questions).
  - It trains 176M weights, 156M of them the table, and peaks at 8.6 GiB (13.9 full). It runs at 4,317 tokens per second (3,831 full).
  - Qwen3-1.7B-Base now fits: 346M weights train, peak 15.8 GiB.
- **NF4** (done, `post/nf4.py`): every linear layer's weight in 4 bits, as QLoRA stores it.
  - Blocks of 64 weights. Each block is divided by its largest absolute value, and each weight is stored as the nearest of 16 levels, the normal distribution's quantiles.
  - The block scales are quantized again: int8 in blocks of 256, as torchao does it. QLoRA uses an 8-bit float format of its own. 4.127 bits per weight in all.
  - The matmul dequantizes to bf16 in the forward pass, and again in the backward pass instead of keeping the bf16 matrix.
  - The check (`post/check/nf4.py`):
    - The 16 levels match QLoRA's table, and every weight is stored as its nearest level.
    - Over all 196 linear layers of 0.6B, the dequantized weights are 9.3% off, the largest 9.6%. With the scales exact it's the same to 3 digits.
    - One `gate_proj` keeps nothing for the backward pass, where autograd through the dequantized weight keeps 6.0 MiB. Output and input gradient match autograd's to the bit.
  - Memory and loss (`post/eval/nf4.py`, the answers of 50 GSM8K test conversations; the embedding table stays bf16 both ways):

    | model | bf16 GiB | NF4 GiB | bf16 loss | NF4 loss |
    |---|---|---|---|---|
    | Qwen3-0.6B-Base | 1.11 | 0.51 | 0.958 | 1.062 |
    | Qwen3-1.7B-Base | 3.21 | 1.27 | 0.873 | 0.933 |
    | Qwen3-4B-Base | 7.56 | 2.50 | 0.779 | 0.850 |

    4B in NF4 is still better than 1.7B in bf16, in a third of the memory.
- **QLoRA** (done, `post/qlora.py`): every frozen linear layer in NF4, then LoRA and the trained table as before.
  - SFT on 0.6B (`post/eval/qlora.py`), the same conversations in the same order as LoRA's run:
    - Before training, NF4 is 0.104 behind on held-out loss (1.063 against 0.959). After, it's 0.024 behind: 0.612 against LoRA's 0.588. `<|im_end|>` gets 0.90 where answers end (0.91).
    - Every logged step's loss is above LoRA's, by 0.02 to 0.12, typically 0.06.
    - GSM8K, first 50 test questions: 92% close their turn (98%), 32% right by strict and by last number (42%; 5 of 50 questions).
    - Peak 8.0 GiB (8.6): on 0.6B, most of the memory is the trained table, not the frozen layers. 3,347 tokens per second (4,317): each layer is dequantized twice per step.
    - The 50 answers took 11.6 minutes to generate, against LoRA's 5.7: each decoded token dequantizes all 196 layers again. Production fixes this with a kernel that reads the 4-bit weights inside the matmul, or by dequantizing once for inference.
  - Peak memory of SFT steps of 2,048 tokens (`post/bench/finetune_memory.py`), GiB of the L4's 22.1:

    | model | full | LoRA | QLoRA |
    |---|---|---|---|
    | Qwen3-0.6B-Base | 14.2 | 8.8 | 8.2 |
    | Qwen3-1.7B-Base | out of memory | 15.8 | 13.8 |
    | Qwen3-4B-Base | out of memory | out of memory | out of memory, in step 2's forward pass |

  - Why 4B still doesn't fit: the embedding table, trained in full. Its 389M weights take 16 bytes each (fp32 weight, gradient, AdamW's two moments): 5.8 GiB. The fused loss adds its fp32 gradient while the backward pass waits, 1.45 GiB, and that is the allocation that fails.
- **Closed here,** with what's left ranked by what it would teach:
  - Fitting 4B, by training only the two chat tokens' rows (PEFT's trainable tokens): skipped. It's plumbing, and its lesson, rows a base model never trained, was learned when LoRA wouldn't stop.
  - The 8B teacher in NF4: moved to stage 6, the first stage that needs it. It takes a loader for a head apart from the embedding table.
  - A kernel that reads the 4-bit weights inside the matmul: left to the kernels track. Its problem is measured above: generation twice as slow in NF4.
- **Frontier:** practitioner tools. The labs fine-tune fully. LoRA appears in RL research, where rank 1 is enough.
- **Runs on:** Modal L4.

### 3. Rejection sampling: learn from its own right answers

- **Problem:** the SFT model's pass@8 is far above its pass@1. The right answer is in there; it just doesn't pick it.
- **Build:** sample 8 answers per problem, keep the verified ones, fine-tune on them, repeat.
- **Check:** pass@1 climbs toward the old pass@8, while pass@8 itself barely moves.
- **The problem, seen** (`post/eval/pass_at_k.py`, the SFT model on the first 200 GSM8K test questions, zero-shot in chat format; 16 samples each at temperature 1):

  |  | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 |
  |---|---|---|---|---|---|
  | strict | 25.3% | 38.2% | 52.1% | 65.0% | 76.0% |
  | last number | 25.6% | 38.6% | 52.5% | 65.4% | 76.5% |

  - One sample is right a quarter of the time, but 8 include a right one for 65% of the questions. 151 of the 200 questions have some samples right and some wrong; 48 have none right, 1 has all.
  - Greedy, the same model gets 50% right. Temperature 1 spreads probability onto wrong steps, and over a hundred tokens they add up.
  - 93% of samples close their turn.
- **Batched sampling** (done, `post/sampler.py`): 3,200 answers needed more than one question per batch. Prompts of different lengths are padded on the left, so every row's next token lands in the same slot, and attention skips the padding. With RoPE, starting later changes no score. A padding query sees every key before it, since one that sees none comes out NaN and spreads.
  - The check (`post/check/batch.py`): prompts of 32, 7, 20 and 1 tokens padded together get each prompt's own logits, to 2.4e-4 in fp32.
  - 200 greedy answers took 0.3 minutes, all in one batch. One at a time, 50 took 3.9 (`post/eval/sft.py`).
  - 3,200 sampled answers took 3.8 minutes: about 68 ms per step at 256 rows. The batch runs until its longest answer ends, usually at the 256-token limit. Inference servers refill a finished row with a new prompt instead (continuous batching).
- **Weights kept between runs:** `post.sft` saves its weights to a Modal Volume, and later stages load them (`post/qwen3.py`, `gpu.py`).
- **Rejection sampling** (done, `post/rejection_sampling.py`): one round.
  - The SFT model answers 384 GSM8K train questions SFT never read, 8 samples each at temperature 1. Only each question's final number is used, never its worked solution.
  - Kept: answers that closed their turn and are right by strict grading. Exact repeats are dropped, and at most 4 are kept per question, as ReST-EM caps them.
  - Then the base model is fine-tuned again from the start, on SFT's conversations plus the kept ones, with SFT's recipe. STaR, ReST-EM and DeepSeek R1 do this. The alternative is to go on training the SFT model on the new answers alone, as Qwen3 does in its later SFT stages.
  - Sampling took 3.7 minutes. 294 of the 384 questions got an answer kept, and 115 of those got the full 4: 802 answers, 118K tokens with their questions.
- **What it changed** (`post/eval/rejection_sampling.py`, the same 200 test questions, 16 samples each), with a control (`post/eval/rejection_control.py`): the same 384 questions with GSM8K's worked solutions instead of the model's answers.

  |  | greedy | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | none of 16 right |
  |---|---|---|---|---|---|---|---|
  | SFT | 50% | 25.3% | 38.2% | 52.1% | 65.0% | 76.0% | 48 |
  | rejection sampling | 51% | 29.0% | 42.4% | 56.0% | 68.9% | 80.5% | 39 |
  | control: worked solutions | 50% | 25.3% | 37.9% | 51.2% | 64.0% | 76.0% | 48 |

  All by strict grading. Each run took 8 to 12 minutes on the L4, about $0.15.
  - The control changes nothing, so the gain didn't come from more questions. It came from the model's own right answers. Two likely reasons, which these runs don't separate:
    - Its own answers are written the way it already writes, so training on them raises the probability of paths it can already take.
    - It got up to 4 different right answers per question, against one worked solution: 118K tokens against 62K.
  - pass@1 rose by 3.7 points, but pass@8 rose about as much (3.9), and 9 more questions now have a right sample. The plan expected pass@1 to climb while pass@8 stayed put. At our scale it moved everything up.
  - Greedy barely moved (50% to 51%). Most of the gain is at temperature 1, where wrong steps had the most probability.
- **Skipped:** a second round. It repeats the first with the new model, for a smaller gain, and teaches nothing new.
- **Left for stage 4:**
  - The 39 questions with no right sample give rejection sampling nothing to keep.
  - Every wrong answer is thrown away, though it shows what not to do.
- **Frontier:** DeepSeek R1 (~800k samples), every round of Llama 3, and Qwen3. It's also the oldest form of self-distillation.
- **Runs on:** Modal L4.

### 4. RL with verifiable rewards

- **Problem:** rejection sampling throws away the wrong answers, which carry signal too.
- **Before:** IFEval-style constraint graders, for the first task below.
- **Build:**
  - Per-token log-probs from the sampler: the policy-gradient loss needs them.
  - An environment interface of (problem, environment, verifier) triplets, as in DeepSeek V4.1.
  - REINFORCE first, then GRPO as DeepSeekMath wrote it: a group baseline, z-scored advantages, ratio clipping and a KL penalty.
  - Then six fixes, each added when its problem appears:

| Problem | Fix |
|---|---|
| Wrong answers get longer | Token-level loss, so long answers get no extra weight |
| Some prompts get outsized weight | Drop the division by the reward's standard deviation |
| The KL term holds the model back or makes it collapse | Remove it |
| Entropy collapses | Clip-higher: ε of 0.2 below, 0.28 above |
| Groups that are all right or all wrong give zero gradient | Filter them out |
| Sampler and trainer disagree on log-probs, worse in fp16 | An fp32 output layer, plus truncated importance sampling |

- **Tasks, in order:**
  1. IFEval-style constraints, whose regex rewards are non-zero from step one.
  2. GSM8K.
  3. Countdown from Reasoning Gym, tuned to a difficulty where the model solves 20–80%.
- **Check:**
  - Reward curves.
  - pass@1 and pass@k on held-out sets.
  - The random-reward control.
  - A rank-1 LoRA run.
- **Re-ranked by what each step teaches** (2026-10-22):
  - The IFEval-style graders first: skipped. They came first because their reward is non-zero from step one, and our SFT model already gets 25% of GSM8K at temperature 1.
  - The environment interface: deferred to stage 5, when there's a second task.
  - Per-token log-probs from the sampler: ratio clipping takes its old log-probs from the trainer instead, as frontier trainers do. The sampler's wait for the sampler-trainer mismatch.
  - The KL term: GRPO keeps it as DeepSeekMath wrote it, and removing it is one of the fixes. DAPO and Dr. GRPO remove it; DeepSeek V3.2 keeps a corrected KL instead.
  - Countdown: moved to stage 5 (2026-10-25), as a second task for the environment interface.
- **REINFORCE** (done, `post/reinforce.py`):
  - Each step, the model answers 8 GSM8K train questions SFT never read, 8 samples each at temperature 1.
  - Reward: 1 if the answer closed its turn and is right by strict grading, 0 if not.
  - Loss: minus each answer's summed token log-probabilities times its reward, averaged over the step's 64 answers.
  - The answers are sampled by a bf16 copy of the model, refreshed from the trained fp32 weights before each step. Their token ids go to the loss as sampled, never decoded and encoded again.
  - The fused loss now takes a scale per target, the answer's reward, where SFT passes 1/A. The check (`post/check/loss.py`): with signed scales per conversation, its gradient is as close to the exact one as the plain loss's (1.6% and 1.5% over all weights).
  - Constant learning rate 2e-6, 40 steps, 15.8 seconds per step, peak 17.3 GiB.
- **What REINFORCE changed** (`post/eval/reinforce.py`, the same 200 test questions, 16 samples each, strict):

  |  | greedy | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | none of 16 right | all 16 right |
  |---|---|---|---|---|---|---|---|---|
  | SFT | 50% | 25.3% | 38.2% | 52.1% | 65.0% | 76.0% | 48 | 1 |
  | rejection sampling | 51% | 29.0% | 42.4% | 56.0% | 68.9% | 80.5% | 39 | 2 |
  | REINFORCE, 40 steps | 52% | 34.2% | 47.7% | 61.0% | 72.8% | 83.0% | 34 | 7 |

  - In 40 steps, 2,560 sampled answers, it goes further than a round of rejection sampling, which sampled 3,072 and trained the base model again. Each step learns from answers the current model wrote.
  - The training reward, over 5 steps at a time: 0.287, 0.300, 0.322, 0.425, 0.353, 0.438, 0.341, 0.366. Each window has different questions, so it's noisy.
  - The answers got shorter, 112 tokens on average to 90, and more of them close their turn: 98%, from 93%. An answer that runs to the token limit gets no reward.
  - With rewards of 1 and 0, every wrong answer gets weight 0: at a reward of about 0.35, two thirds of each step's sampled answers teach nothing. That's the next problem: a baseline.
- **Simple versions** (`learn/`, how they're written in `learn/README.md`): one bare, numbered file per mechanism, each the file before it plus that mechanism, with no packing, fused loss, cache or batching tricks, on toy prompts. REINFORCE on "My favorite animal is the", with a reward of 1 for "dog", reaches 1.0 by step 4, then collapses to "dog because he is my dog because he": once every reward is 1, every sample is pushed up, whatever it says.
- **The baseline** (done, `baseline=True` in `post/reinforce.py`): each answer's weight is its reward minus the mean reward of its question's 8 answers. Wrong answers are pushed down, and a question whose answers all score the same pushes nothing. GRPO also divides by the group's standard deviation; Dr. GRPO drops that, and so does this, for now.
  - The toy (`learn/2_rl/004_baseline.py`): the same 1.0 by step 4, but no collapse. Through step 29 the answers stay varied: "dog that is easy to take care of", "dog and dog is remarkable because I can".
  - GSM8K (`post/eval/baseline.py`), the same start, questions and 40 steps as REINFORCE:

    |  | greedy | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | none of 16 right | all 16 right |
    |---|---|---|---|---|---|---|---|---|
    | REINFORCE | 52% | 34.2% | 47.7% | 61.0% | 72.8% | 83.0% | 34 | 7 |
    | with the baseline | 52% | 35.2% | 49.1% | 62.9% | 75.6% | 85.0% | 30 | 5 |

    - Every pass@k is a little higher, and more so at large k: 1.0 point at pass@1, 2.8 at pass@8. That's within the noise of 200 questions, but it leans the way the toy does. REINFORCE pushes up whichever right answers it happened to sample, which narrows them. The baseline pushes up only answers that beat their question's average, and pushes the others down.
    - The answers shortened less, from 110 to 98 tokens, where REINFORCE's went from 112 to 90.
    - 10.1 minutes, 15.2 seconds per step, peak 17.3 GiB: the baseline costs nothing.
- **Ratio clipping** (done, `post/ratio_clipping.py`): sampling is most of each step, and its answers get one gradient step. Here they get 4. From the second update on, the model training on them is no longer the one that sampled them, so each answer token is weighted by its ratio, its probability now over its probability when sampled, clipped to 1 ± 0.2 in the direction its advantage pushes. On the first update every ratio is 1, and the gradient is the baseline's.
  - The old log-probs are the trainer's own, from the first update's forward pass, where the weights are still those that sampled.
  - The loss needs each target's log-probability, not just their sum. `TokenLogProbs` in `post/linear_cross_entropy.py` returns them in chunks, and computes the logits again in the backward pass rather than keep them. The check (`post/check/loss.py`): on the same signed scales, its gradient is as close to the exact one as the plain loss's (1.5% and 1.5% over all weights).
  - No division by the group's standard deviation and no KL term, as in Dr. GRPO and DAPO. Each answer's tokens are summed, as Dr. GRPO does, where GRPO averaged them.
  - The toy (`learn/2_rl/005_ratio_clipping.py`), when it still ran a second, unclipped pass: by batch 2 the largest ratio is 66.3 clipped and 399.6 not clipped. Without the clip, the answers narrow to "dog\nHere is…".
  - GSM8K (`post/eval/ratio_clipping.py`), the same start and test questions, 20 steps of 4 updates:

    |  | greedy | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | none of 16 right | all 16 right |
    |---|---|---|---|---|---|---|---|---|
    | REINFORCE, 40 steps | 52% | 34.2% | 47.7% | 61.0% | 72.8% | 83.0% | 34 | 7 |
    | with the baseline, 40 steps | 52% | 35.2% | 49.1% | 62.9% | 75.6% | 85.0% | 30 | 5 |
    | ratio clipping, 20 steps × 4 updates | 55% | 41.0% | 54.8% | 67.3% | 78.4% | 87.5% | 25 | 9 |

    - From half the sampled answers, 1,280 against 2,560, it goes further than either: 5.8 points at pass@1, the largest gap yet. It also took twice the gradient steps, 80 against 40.
    - 22.6 seconds per step, against the baseline's 15.2, so each extra update costs about 2.5 seconds and sampling about 13. That's why reuse pays: a gradient step on answers already sampled costs a fifth as much as sampling new ones.
    - The clip stopped 1.4%, 1.9%, 2.5% and then 3.2% of answer tokens in each step's last update, over 5 steps at a time. At a learning rate of 2e-6 it rarely binds. Whether it matters here would take the same run without it, which the toy already shows.
    - The training reward, over 5 steps at a time: 0.325, 0.425, 0.403, 0.569.
    - 7.5 minutes, peak 17.3 GiB. 96% of answers close their turn.
- **GRPO as DeepSeekMath wrote it** (done, `post/grpo.py`): ratio clipping plus the three other things in DeepSeekMath's loss. Later recipes undo each one.
  - Each advantage is divided by the standard deviation of its question's rewards, plus 1e-4 so a question whose answers all score the same stays at 0.
  - Each answer's tokens are averaged, so every answer counts the same however long.
  - A KL penalty toward a frozen copy of the SFT model, weight 0.04. Each token's estimate is p_ref/p − log(p_ref/p) − 1: computed from the sampled token alone, never negative, and 0 where the two models agree.
  - DeepSeekMath trained at 1e-6 with 64 answers per question and one update per batch, so its clip never acted. This keeps ratio clipping's 2e-6, 8 answers and 4 updates, so the three additions are all that differs.
  - The toy (`learn/2_rl/006_grpo.py`) first ran with AdamW's default weight decay of 0.01. Its first batch had every reward 0, and it ended with a KL of 423 per token.
    - With every advantage 0 and the model equal to the reference, the gradient is exactly 0. But weight decay still shrinks every weight, by a factor of 1e-7.
    - That gives the KL a gradient of norm 8.5e-6. Adam divides each weight's gradient by its own running size, so the step is far bigger than the gradient: the largest log-prob gap to the reference went from 0.00003 to 0.42.
    - Then the gradient is real, but Adam's step is still about one learning rate per weight, however small the correction needed. It overshoots to a gap of 10.9.
    - With weight decay 0, as every full-size run here uses, nothing moves until a sample is right. The reward then pulls the model away, to a KL of 1.44. Once nearly every sample is right and the advantages fade, the KL pulls it back, to 0.55.
  - GSM8K (`post/eval/grpo.py`): the same start, questions and steps as ratio clipping.

    |  | greedy | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | none of 16 right | all 16 right |
    |---|---|---|---|---|---|---|---|---|
    | with the baseline, 40 steps | 52% | 35.2% | 49.1% | 62.9% | 75.6% | 85.0% | 30 | 5 |
    | ratio clipping, 20 steps × 4 updates | 55% | 41.0% | 54.8% | 67.3% | 78.4% | 87.5% | 25 | 9 |
    | GRPO as DeepSeekMath wrote it | 57% | 38.9% | 52.7% | 66.0% | 78.2% | 89.0% | 22 | 7 |

    - Against ratio clipping: pass@1 2.1 points lower, pass@16 1.5 higher, and 3 fewer questions with none right. It sharpens less and leaves more answers reachable, as a pull toward the SFT model would. But that's within the noise of 200 questions, and three things changed at once. The fixes take them out one at a time.
    - The training reward rose more slowly on the same questions: 0.328, 0.372, 0.425, 0.494, against 0.325, 0.425, 0.403, 0.569.
    - The KL per answer token, over 5 steps at a time: 0.009, 0.015, 0.033, 0.050, still rising. The answers stayed about as long: 109, 111, 99, 105 tokens. 94% close their turn.
    - 8.3 minutes, 24.9 seconds per step against 22.6, peak 19.5 GiB against 17.3: the reference model, 2.4 GB in fp32, and its forward pass over each row.
- **The fixes**, one at a time, by fix rather than by paper: Dr. GRPO makes two of them and DAPO four, so a file per paper would change several things at once. Each fix gets a problem logged first, then the fix.
  - Simple: one numbered file per fix in `learn/2_rl/`, each the file before it plus that fix.
  - Full: `post/grpo.py` stays the original. `post/rlvr.py` starts as a copy and takes the fixes in order, so it ends as the recipe the frontier converged on.
  - Order: the token-level loss, no standard-deviation division (Dr. GRPO), no KL, zero-gradient groups filtered, clip-higher (DAPO), then truncated importance sampling with an fp32 output layer. DAPO's soft length penalty is left out: our reward already scores an answer that runs out of room 0.
- **1. The token-level loss** (done, `post/rlvr.py`). GRPO averages each answer's tokens, so each token's push is its answer's advantage over N and the answer's length: inside a long answer, the push is spread thin. Wrong answers' tokens are pushed down less the longer they are, and right answers' pushed up more the shorter they are. The fix divides every token by the batch's answer tokens (DAPO's, the most common). Dr. GRPO divides by a constant instead.
  - The toy (`learn/2_rl/007_token_mean.py`): `006_grpo.py` with `-objective.mean()`, every token in the batch averaged at once, in place of `-objective.mean(dim=-1).mean()`. Every toy sample is 8 tokens, so the two give the same number: the toy shows the formula, and GSM8K's answers, each its own length, show the difference.
  - GSM8K: `post/rlvr.py` starts as a copy of `post/grpo.py`, and the one change is each token's scale, 1 over the batch's answer tokens where GRPO's is 1 over N and its answer's length. Both log right and wrong answers' lengths over 5 steps at a time; `post.grpo` ran again for that, and its weights replaced the saved "grpo".

    |  | right answers, tokens | wrong answers, tokens | reward |
    |---|---|---|---|
    | GRPO | 86, 76, 78, 78 | 114, 112, 99, 100 | 0.297, 0.397, 0.431, 0.472 |
    | the token-level loss | 91, 80, 94, 93 | 115, 125, 110, 121 | 0.294, 0.381, 0.463, 0.547 |

    - Right answers shrink under GRPO, 86 to 78 tokens, as the bias predicts, and not with the fix, 91 to 93.
    - Wrong answers don't grow under GRPO in 20 steps: they shrink, 114 to 100, and they're longer with the fix in every window. That half of the bias doesn't show here; the drift builds over far longer runs than this.
    - Two runs of GRPO with the same settings differ by up to 0.03 in the training reward, so the fix's 0.547 at the end may be noise.

    |  | greedy | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | none of 16 right | all 16 right |
    |---|---|---|---|---|---|---|---|---|
    | GRPO as DeepSeekMath wrote it | 57% | 38.9% | 52.7% | 66.0% | 78.2% | 89.0% | 22 | 7 |
    | with the token-level loss | 52% | 38.7% | 53.7% | 67.6% | 79.9% | 90.0% | 20 | 3 |

    - The same pass@1, a point or two higher from pass@4 on: within the noise of 200 questions. Greedy is one answer per question, so 5 points is 10 questions.
    - 8.1 minutes, 24.4 seconds per step, peak 19.6 GiB: the fix costs nothing. 94% of answers close their turn.
- **2. No division by the standard deviation** (done, `post/rlvr.py`). With rewards of 1 and 0, a question's standard deviation is the square root of p (1 - p), p the share of its answers that are right: dividing by it scales up the questions whose answers are most one-sided, and the training no longer follows the gradient of the expected reward. The fix is the baseline's advantage, the reward minus its question's mean (Dr. GRPO, OLMo 3). The alternative divides by the whole batch's standard deviation, which keeps the scale without favoring any question.
  - The toy (`learn/2_rl/008_no_std.py`): `007_token_mean.py` with the division taken out of its advantage line. It has one prompt, so one group: there the division only scales the whole batch by one number, which Adam all but ignores. The problem is between questions, so only GSM8K has it.
  - GSM8K: `post/rlvr.py` takes `advantages` from `post/reinforce.py` instead of `post/grpo.py`.

    |  | greedy | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | none of 16 right | all 16 right |
    |---|---|---|---|---|---|---|---|---|
    | GRPO as DeepSeekMath wrote it | 57% | 38.9% | 52.7% | 66.0% | 78.2% | 89.0% | 22 | 7 |
    | with the token-level loss | 52% | 38.7% | 53.7% | 67.6% | 79.9% | 90.0% | 20 | 3 |
    | and no standard deviation | 51% | 34.2% | 49.2% | 63.5% | 75.6% | 85.5% | 29 | 1 |

    - Worse: 4.5 points at pass@1 and at pass@16, back where REINFORCE with the baseline was. The training reward rose more slowly too: 0.272, 0.375, 0.434, 0.494, against 0.294, 0.381, 0.463, 0.547.
    - The KL to the SFT model ended at 0.026 per answer token, against 0.043: the model moved less. With rewards of 1 and 0 the standard deviation is at most about 0.5, so the division always enlarged the advantages, about doubling them at least. Without it, the KL penalty at the same weight, 0.04, counts at least twice as much against the reward. Adam ignores the loss's overall scale, but not the balance between its two terms.
    - So the fixes interact: the KL weight was set for advantages divided by the standard deviation. That's the next problem: the KL holding the model back, which step 3 removes.
    - Right answers 84, 76, 85, 81 tokens, wrong 120, 118, 101, 113, as with the first fix alone. 7.8 minutes, 23.3 seconds per step, peak 19.5 GiB. 96% of answers close their turn.
- **3. No KL penalty** (done, `post/rlvr.py`). The penalty pulls the model toward a frozen copy of the model it started from. It guards training on a learned reward model, which the model can learn to fool with answers it scores high but that are wrong. A checked reward reads the number the question asks for, so there's nothing to fool: the pull only holds the model back, and the copy costs its memory and a forward pass over every row. DAPO, Dr. GRPO and OLMo 3 drop it; OLMo 3's report calls that common practice. The alternative is DeepSeek-R1's: a weight of 0.001, with the copy replaced by the current model every 400 steps.
  - The toy (`learn/2_rl/009_no_kl.py`): `008_no_std.py` without the reference model, `BETA` and the KL lines. Against `005_ratio_clipping.py`, only the loss line, weight decay and the print differ: with GRPO's three additions taken out, what's left is ratio clipping with the token-level loss. Every toy sample is 8 tokens, so that loss is ratio clipping's divided by 8, which Adam ignores: the run gives 005's rewards and samples, batch for batch.
  - GSM8K: `post/rlvr.py` drops the reference model and uses `post/ratio_clipping.py`'s `clipped_loss`, each token weighted by its answer's advantage over the batch's answer tokens. It logs no KL: measuring it needs the copy this removes.

    |  | greedy | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | none of 16 right | all 16 right |
    |---|---|---|---|---|---|---|---|---|
    | ratio clipping | 55% | 41.0% | 54.8% | 67.3% | 78.4% | 87.5% | 25 | 9 |
    | GRPO as DeepSeekMath wrote it | 57% | 38.9% | 52.7% | 66.0% | 78.2% | 89.0% | 22 | 7 |
    | with the token-level loss | 52% | 38.7% | 53.7% | 67.6% | 79.9% | 90.0% | 20 | 3 |
    | and no standard deviation | 51% | 34.2% | 49.2% | 63.5% | 75.6% | 85.5% | 29 | 1 |
    | and no KL penalty | 52% | 40.5% | 54.9% | 67.7% | 78.3% | 86.5% | 27 | 10 |

    - pass@1 up 6.3 points from the second fix, back where ratio clipping was, as the toy's diff says it should be. Taken out, GRPO's three additions cost nothing at this scale.
    - The training reward rose faster early: 0.331, 0.422, 0.381, 0.519, against 0.272, 0.375, 0.434, 0.494. Two runs with the same settings differ by up to 0.03, so only the first two windows are clearly ahead.
    - 10 questions with all 16 right, against 1, and pass@16 3.5 points below the first fix alone: without the pull the model sharpens more. Within the noise of 200 questions.
    - 7.5 minutes, 22.5 seconds per step against 23.3, peak 17.3 GiB against 19.5: ratio clipping's, the copy gone. Right answers 86, 79, 83, 88 tokens, wrong 118, 116, 109, 119. 95% of answers close their turn.
- **4. Zero-gradient filtering** (done, `post/rlvr.py`). A question whose answers all score the same, all right or all wrong, has every advantage 0: its answers push nothing, but they cost their sampling and their passes, and with the token-level loss their tokens still count in the total every token's push is divided by, so they shrink the rest's. The fix drops those questions and samples more until the batch is full of questions whose answers' rewards differ, as DAPO and OLMo 3 do. The alternative drops them without refilling, and trains on a batch whose size changes from step to step.
  - The toy (`learn/2_rl/010_zero_gradient_filtering.py`): `009_no_kl.py` with `if (advantages == 0).all(): continue`. One prompt makes one group, so the toy drops whole batches and has nothing to refill with. It still changes the run: in 009, batches 0 and 1 had every advantage 0, but their updates still counted as AdamW steps. Adam corrects its running averages for how many steps it has taken, so 009's first real update, at step 9, was about half the size of 010's, at step 1. From batch 3 on, the two runs sample differently.
  - GSM8K: `post/rlvr.py` samples 8 questions at a time, for at most 3 rounds, keeps the questions whose answers' rewards differ, and trains on the first 8. It logs how many questions each step sampled and what share of them were all right or all wrong, over every answer sampled.

    |  | questions sampled per step | all right or all wrong | reward, every answer sampled |
    |---|---|---|---|
    | steps 1-5 | 14.4 | 24% | 0.340 |
    | steps 6-10 | 12.8 | 11% | 0.391 |
    | steps 11-15 | 16.0 | 30% | 0.475 |
    | steps 16-20 | 16.0 | 25% | 0.427 |

    |  | greedy | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | none of 16 right | all 16 right |
    |---|---|---|---|---|---|---|---|---|
    | ratio clipping | 55% | 41.0% | 54.8% | 67.3% | 78.4% | 87.5% | 25 | 9 |
    | the first three fixes | 52% | 40.5% | 54.9% | 67.7% | 78.3% | 86.5% | 27 | 10 |
    | and zero-gradient filtering | 53% | 41.6% | 56.1% | 68.4% | 78.5% | 86.5% | 27 | 9 |

    - The problem, measured: about a quarter of the questions sampled, 11% to 30% over 5 steps at a time, have every answer right or every answer wrong. Without the fix, that share of every batch trains on nothing.
    - About the same scores: pass@1 1.1 points higher, within the noise of 200 questions. 20 steps is too short to see the share of all-right questions grow as the model improves, which is when the filter matters most.
    - The cost: 34.8 seconds per step against 22.5. Most steps sample a second round, and a round of sampling costs about 13 seconds. 11.6 minutes in all, and the whole run with its evaluation took 16.7, close to the 20-minute limit. The rewards aren't comparable step for step with earlier runs: after the first round a step takes questions the earlier runs never saw.
    - Peak 17.3 GiB, as before. Right answers 83, 85, 79, 80 tokens, wrong 117, 111, 111, 116. 96% of answers close their turn.
- **5. Clip-higher** (done, `post/rlvr.py`). The clip lets a token pushed up grow to at most 1 + ε times its probability when sampled, so its room to grow is a share of how likely it already was: an unlikely token in a right answer can barely grow while the likely ones grow freely, and the answers narrow onto what the model already prefers. The fix raises only the upper bound, to 1 + 0.28 (DAPO; OLMo 3 uses 0.272), and keeps the lower at 1 - 0.2: loosening it too would let the tokens of wrong answers be pushed toward 0, which narrows the answers as well.
  - The toy (`learn/2_rl/011_clip_higher.py`): `010_zero_gradient_filtering.py` with `EPSILON` split into `EPSILON_LOW` and `EPSILON_HIGH`, and the clamp taking both.
  - GSM8K: `post/ratio_clipping.py`'s `clipped_loss` takes the upper bound as an argument, and `post/rlvr.py` passes 0.28. It logs, in each step's last update, the share of answer tokens the clip stopped at each bound, and how likely those at the upper bound were when sampled, against every answer token.

    |  | stopped at the upper bound | at the lower | probability when sampled, stopped at the upper | every answer token |
    |---|---|---|---|---|
    | steps 1-5 | 0.3% | 0.7% | 0.11 | 0.70 |
    | steps 6-10 | 0.6% | 1.5% | 0.09 | 0.71 |
    | steps 11-15 | 0.6% | 1.6% | 0.11 | 0.73 |
    | steps 16-20 | 0.8% | 2.3% | 0.13 | 0.76 |

    - The problem, measured: the tokens the upper bound stops are the unlikely ones, 0.09 to 0.13 likely when sampled against about 0.7 for every answer token, even with the bound at 1.28.
    - The narrowing shows too: every answer token's probability when sampled rises from 0.70 to 0.76 in 20 steps, the model ever surer of its own answers. Whether clip-higher slows that would take the same log from a run with the old bound.
    - The lower bound stops 2 to 3 times as many tokens as the upper.

    |  | greedy | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | none of 16 right | all 16 right |
    |---|---|---|---|---|---|---|---|---|
    | the first four fixes | 53% | 41.6% | 56.1% | 68.4% | 78.5% | 86.5% | 27 | 9 |
    | and clip-higher | 52% | 37.8% | 51.7% | 64.8% | 76.4% | 85.5% | 29 | 10 |

    - pass@1 3.8 points lower. The change can only reach tokens whose ratio lands between 1.2 and 1.28, a sliver of under 1% of tokens, and nothing here measures how much two runs with the same settings differ in pass@1. So this is most likely noise, and clip-higher's benefit, keeping the answers from narrowing over long runs, is out of reach in 20 steps. The training reward was lower too: 0.365, 0.345, 0.397, 0.382, against 0.340, 0.391, 0.475, 0.427.
    - 11.5 minutes, 34.4 seconds per step, peak 17.3 GiB: the fix costs nothing. Right answers 84, 76, 78, 78 tokens, wrong 122, 106, 110, 108. 96% of answers close their turn.
- **6. Truncated importance sampling, with fp32 output layers** (done, `post/rlvr.py`). The sampler, a bf16 copy, and the trainer, fp32 weights with bf16 matmuls, give the same tokens slightly different probabilities, so the answers come from a model a little different from the one the loss assumes: tokens the sampler draws more often than the trainer would are pushed more often than they should be. Each token is weighted by the trainer's probability over the sampler's, the one it drew the token with, capped at 2 so that no token dominates (OLMo 3, after Yao et al.). And both output layers run in fp32, which MiniMax-M1 found closed most of its gap.
  - Plumbing: `post/sampler.py`'s `sample` and `generate` return each drawn token's log probability, in the distribution it was drawn from, and `answer_ids` and `rollouts` pass them on. `Qwen3.forward` takes an output layer other than its own: the sampler's is the trainer's fp32 embedding table, read live. `TokenLogProbs` runs its head in fp32 with `fp32_head`, with autocast off: inside autocast a matmul runs in bf16 whatever its inputs, so the first version's fp32 head changed nothing.
  - The problem, measured (`post/eval/sampler_gap.py`): 256 answers from the SFT model, about 28,000 tokens, and the trainer's log probability of each token against the sampler's.

    | sampler's output layer | trainer's | apart in log probability, on average | largest ratio |
    |---|---|---|---|
    | bf16 | bf16 | 0.0188 | 1.40 |
    | bf16 | fp32 | 0.0168 | 1.37 |
    | fp32 | bf16 | 0.0164 | 1.34 |
    | fp32 | fp32 | 0.0121 | 1.35 |

    - Both in fp32 bring the two about a third closer; the rest comes from everything else the sampler runs in bf16. No token in either case is off by a factor of 2: here the sampler is our own code, the trainer's model in bf16, where frontier samplers are inference engines with kernels of their own.
  - The toy (`learn/2_rl/012_truncated_importance_sampling.py`): `011_clip_higher.py` whose `sample` also returns each drawn token's log probability, an `importance` line, and `importance *` in the objective. Its sampler is the model itself, so every weight is 1 up to rounding: the run gives 011's rewards and samples, batch for batch.
  - GSM8K: the trainer and the sampler were 0.0099 to 0.0114 apart in log probability on average, over 5 steps at a time. No weight was capped; the largest was 1.49 to 1.71. So the weights move each token's push by about 1%.

    |  | greedy | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | none of 16 right | all 16 right |
    |---|---|---|---|---|---|---|---|---|
    | the first four fixes | 53% | 41.6% | 56.1% | 68.4% | 78.5% | 86.5% | 27 | 9 |
    | and clip-higher | 52% | 37.8% | 51.7% | 64.8% | 76.4% | 85.5% | 29 | 10 |
    | and truncated importance sampling | 46% | 38.4% | 52.3% | 64.5% | 74.3% | 81.0% | 38 | 9 |

    - Greedy 6 points lower and pass@16 4.5 lower, with 38 questions none of 16 right against 29. Weights within about 1% of 1 can't account for that much, and nothing here measures how much two runs with the same settings differ: that's the next thing to measure, before reading any more of these tables.
    - Every answer token's probability when sampled rose from 0.73 to 0.80, against 0.70 to 0.76 with clip-higher alone: the model sharpened more.
    - 12.3 minutes, 36.8 seconds per step against 34.4: the fp32 output layers cost about 2.4 seconds a step. Peak 17.0 GiB. Right answers 81, 75, 75, 80 tokens, wrong 123, 108, 97, 106. 97% of answers close their turn.
- **How much one run moves by chance** (`post/eval/rlvr_noise.py`): the sixth fix's run twice more, on the same questions in the same order, with every sample, training's and the evaluation's, drawn from another seed. The two ran at once, each on an L4 of its own: `gpu.py` passes the words after a module's name to it as arguments.

    |  | greedy | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | none of 16 right | all 16 right |
    |---|---|---|---|---|---|---|---|---|
    | the sixth fix's run | 46% | 38.4% | 52.3% | 64.5% | 74.3% | 81.0% | 38 | 9 |
    | seed 1 | 47% | 37.0% | 50.7% | 64.3% | 75.7% | 84.0% | 32 | 12 |
    | seed 2 | 54% | 41.6% | 56.3% | 69.3% | 79.5% | 86.5% | 27 | 8 |

    - Three runs of one setup span 8 points of greedy, 4.6 of pass@1 and 5.5 of pass@16, and 27 to 38 questions with none right. Their standard deviations are about 4.4, 2.4 and 2.8 points.
    - Greedy is one answer per question from the trained model, so its 8 points are mostly training's chance: which answers each step happened to sample.
    - Against that band, every pass@1 since ratio clipping, from 37.8% to 41.6%, is one setup measured once. The one outside it is no standard deviation's 34.2%, 2.8 points below the lowest of the three, which fits its KL story but is one run. At 20 steps and 200 test questions, one run per setting can't tell apart differences under about 5 points: telling them apart takes several seeds per setting, or runs long enough for the differences to grow.
    - The training reward over the last 5 steps: 0.415, 0.434 and 0.464. The largest importance weight reached 2.65 in seed 1, the cap's only use: 0.01% of the weights in its last 5 steps.
- **The two controls** (`post/eval/rlvr_lora.py`, `post/eval/rlvr_random.py`), each the sixth fix's run with one change, on the same questions, measured as the three runs above. The two ran at once.

    |  | greedy | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | none of 16 right | all 16 right | closed their turn |
    |---|---|---|---|---|---|---|---|---|---|
    | the SFT model | 50% | 25.3% | 38.2% | 52.1% | 65.0% | 76.0% | 48 | 1 | 96% |
    | the three full runs | 46-54% | 37.0-41.6% | 50.7-56.3% | 64.3-69.3% | 74.3-79.5% | 81.0-86.5% | 27-38 | 8-12 | 95-97% |
    | a LoRA of rank 1 | 52% | 39.6% | 52.7% | 64.5% | 74.8% | 83.5% | 33 | 12 | 97% |
    | every reward a coin flip | 39% | 15.5% | 24.8% | 35.9% | 47.9% | 59.5% | 81 | 0 | 86% |

  - **A LoRA of rank 1** on every linear layer, and nothing else trained: 630,784 weights, 0.106% of the model's 596 million, at 10 times the learning rate, 2e-5, with post/lora.py's alpha of 32. The sampler loads the weights with each LoRA merged in (`merged_state_dict`).
    - Every number lands inside the three full runs' band. One answer gives back one reward, so 20 steps carry little information, and a correction of rank 1 holds all of it. That matches Thinking Machines' LoRA Without Regret; SFT, which learns every token of a written answer, needs far more.
    - The training reward: 0.328, 0.354, 0.418, 0.415, as the full runs'. Peak 9.0 GiB against 17.0: no gradients or AdamW moments for the frozen weights. 34.6 seconds per step.
  - **Every reward a coin flip,** 1 or 0 with even odds whatever the answer says. The training log's reward is the coin's, about 0.5; the check's reward on the same answers, a quarter of the run at a time: 0.281, 0.281, 0.247, 0.250.
    - Far below the SFT model: pass@1 15.5% against 25.3%, 81 questions with none of 16 right against 48, and 86% of answers closing their turn against 96%. With no signal, the updates push the model in random directions away from what SFT taught it, and it gets worse, not sharper. Answers grew to 113 tokens, right or wrong, against about 80 for right ones in the real runs, and every answer token's probability when sampled fell from 0.68 to 0.65, where in the real runs it rose.
    - So the gain from 25% to about 39% is the reward's doing. Random rewards raised Qwen2.5-Math's score in the paper that tried them, which traced the gain to the clip favoring what the model already prefers. Not here: one guess, unchecked, is that clip-higher's wider upper bound tilts the other way, toward unlikely tokens.
    - 27.0 seconds per step: coin flips are almost never all equal, so a step needs one round of sampling.
- **Where the fixes leave `post/rlvr.py`:** ratio clipping with the token-level loss, the baseline's advantage, no KL, zero-gradient filtering, clip-higher and truncated importance sampling with fp32 output layers: OLMo 3's objective, sampled in steps rather than asynchronously.
- **Frontier:** the end state is where OLMo 3, DeepSeek, TRL's defaults and Meta's ScaleRL all land. The alternatives are GSPO (sequence-level ratios, Qwen's choice for MoE) and CISPO (MiniMax).
- **Runs on:** Modal L4, while the credits last.

### 5. Multi-turn environments and tools

- **Problem:** one calculation in eight is wrong (`post/eval/arithmetic_slips.py`). Every step written as expression = number, as GSM8K's solutions write them, computed again in Python; 8 answers to each of the 200 test questions at temperature 1. A slip is a step off by more than 1% and 0.01.

  |  | steps checked | wrong | wrong answers | with a slip | every step right | no step to check | right answers with a slip |
  |---|---|---|---|---|---|---|---|
  | the SFT model | 3,265 | 13.3% | 1,194 | 25% | 51% | 23% | 7% |
  | stage 4's model | 3,480 | 12.3% | 1,023 | 28% | 55% | 18% | 6% |

  - The plan said that in stage 4's failures the reasoning is right and the arithmetic slips. Half is true: about a quarter of wrong answers have a slip, as written, "15 + 25 + 90 = 120", "50000*1.5=$80000". But half of them have every step right: what went wrong was which calculations to do, not doing them.
  - 20 steps of RL barely touched the arithmetic: 13.3% of steps wrong before, 12.3% after.
  - 6% of right answers have a slip too: a step that didn't feed the answer, or the checker misreading one.
  - A calculator fixes the first kind only, and that's the frontier's reason for tools: a model that needn't do arithmetic in its head spends no capacity on it, and the same loop serves search, code and every other tool.
- **Build:**
  - The environment interface deferred from stage 4: (problem, environment, verifier) triplets, with GSM8K and Countdown from Reasoning Gym as its first two tasks.
  - Multi-turn rollouts. The model writes Python, the environment runs it, and the output comes back as a turn with no loss on it.
  - Then a game: Wordle, with feedback every turn and a reward at the end.
- **Check:**
  - The arithmetic slips disappear.
  - The Wordle win rate rises. A published Qwen3-1.7B Wordle RL model shows this works at that size.
- **Frontier:** RL across many agentic environments is the main RL stage in 2026 (DeepSeek V4.1, Kimi K3, GLM-5, Nemotron 3).
- **Runs on:** Kaggle; Modal for Wordle at 1.7B.

### 6. Learned rewards: reward model, RLHF, DPO

- **Problem:** "be helpful" or "be concise" has no verifier.
- **Build, in order:**
  1. A reward model: the SFT model plus a scalar output, trained on HelpSteer2 preference pairs (Bradley-Terry).
  2. Plug it into the stage-4 loop and watch reward hacking: answers get longer and the reward rises while quality drops.
  3. DPO on UltraFeedback: the same preferences, with no reward model and no sampling.
  4. A rubric judge: the 8B teacher grading answers. First, from stage 2: the loader takes 8B's head, a matrix apart from the embedding table, and its weights go to NF4.
- **Frontier:** split.
  - Ai2 and Meta keep DPO: OLMo 3 runs it before RL, and Llama 4 runs a light pass at the end.
  - The Chinese labs have no preference stage. They score answers inside RL with generative judges and rubrics; in DeepSeek V4 the judge is the policy itself.
  - We build both, and the judge is where the frontier is heading.
- **Runs on:** Kaggle, with the 8B judge in NF4 on the second card.

### 7. On-policy distillation

- **Problem:** RL gives about one bit per episode, while a teacher can grade every token.
- **Build:** the student samples and Qwen3-8B scores each token. Three versions, in order:
  1. **Sampled-token.** The advantage becomes the teacher's log-prob minus the student's, a one-line change to the stage-4 trainer.
  2. **Full-vocabulary reverse KL,** as DeepSeek V4 does.
  3. **Multi-teacher.** Our own stage 4–6 checkpoints act as specialists, and each prompt goes to its domain's teacher. This is DeepSeek V4.1's approach, which takes its teachers from its own earlier checkpoints.
- **Check:**
  - It reaches the RL score in a fraction of the steps.
  - pass@k doesn't shrink.
  - One student matches each specialist on that specialist's domain.
- **Frontier:** the 2026 consensus final stage (DeepSeek V4 and V4.1, Nemotron 3 Ultra, Kimi K3, MiMo, GLM-5, Qwen3).
- **Runs on:** Kaggle, with stage 2's NF4 teacher on one card and the student on the other. Or Modal L4, with the teacher in bf16.

### 8. Self-distillation

- **Problem:** something regressed along the way, for example IFEval after math RL. And sometimes there's no stronger teacher.
- **Build:**
  - An earlier checkpoint as the teacher on the regressed domain (GLM-5's cross-stage distillation).
  - OPSD: the same model, shown the reference solution, teaches itself without it.
- **Frontier:** GLM-5 does the first. OPSD is 2026 research, not yet consensus.
- **Runs on:** Kaggle.

### 9. Async rollouts on two GPUs

- **Problem:** a profile shows the GPU alternating. While it generates it isn't training, and while it trains it isn't generating.
- **Build:**
  - T4 #0 generates while T4 #1 trains.
  - New weights cross PCIe every step.
  - Samples one step stale are corrected with the importance ratio from stage 4.
- **Frontier:** universal at scale. Even in OLMo 3's async setup, the trainer spends 75% of its time waiting for generation.
- **Optional:** compare our sampler with vLLM on Modal.
- **Runs on:** Kaggle, both T4s.

## What we write and what we borrow

- **Borrow:**
  - Qwen3 weights and tokenizer (`safetensors`, `tokenizers`).
  - Datasets (`datasets`).
  - `transformers`, only to check our logits.
- **Write:** everything else. TRL, verl, Unsloth and vLLM are code to read, not run.
- **Where:** `post/`, one file per mechanism, with the same style rules as `core/`. Mechanism files sit at the top of `post/` and hold only the mechanism; everything that measures it lives in a folder of its own: correctness in `post/check/`, speed in `post/bench/`, quality in `post/eval/`. Qwen3's RMSNorm does the same math with the same epsilon (1e-6) as `core/model.py`, so the Triton RMSNorm kernel computes it too.
- **Read alongside:**
  - `reasoning-from-scratch`: pure-PyTorch GRPO on Qwen3-0.6B, one mechanism per file. It took MATH-500 from 15.2% to 47.4% in 50 steps.
  - `nanochat`: `chat_sft.py` and `chat_rl.py`.
  - `GRPO-Zero`: needs only torch and tokenizers.
  - `tinker-cookbook`: recipes for preferences, distillation and self-distillation.

## What to expect

- **No new reasoning at 0.6B.**
  - Learning Countdown fails at 0.5B and shows up at 3B.
  - Expect the format to be learned and the samples to sharpen: pass@1 rises, while pass@k at large k stays flat. RL sharpens; it doesn't add capability.
- **GSM8K, zero-shot in chat format.** Community runs went from ~22–31% at base to ~36% after SFT and ~47–60% after GRPO. These come from blogs, not papers, so treat them as rough.
- **Too hard at this scale:** AIME, SWE-bench, τ²-bench and search agents.
- **Real capability gains need a non-Qwen check:** repeat one run on OLMo-2-1B.

## Sources

- **Models**
  - [Qwen3 technical report](https://arxiv.org/abs/2505.09388)
  - [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
  - [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
- **DeepSeek**
  - [DeepSeek-V4](https://arxiv.org/abs/2606.19348)
  - [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969)
  - [DeepSeekMath, which introduced GRPO](https://arxiv.org/abs/2402.03300)
- **RL fixes**
  - [DAPO](https://arxiv.org/abs/2503.14476)
  - [Dr. GRPO](https://arxiv.org/abs/2503.20783)
- **Other labs' recipes**
  - [OLMo 3](https://arxiv.org/abs/2512.13961)
  - [Tülu 3](https://arxiv.org/abs/2411.15124)
  - [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
  - [Nemotron 3 Ultra multi-teacher distillation](https://docs.nvidia.com/nemotron/nightly/nemotron/ultra3/mopd.html)
- **Distillation**
  - [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
  - [On-policy self-distillation (OPSD)](https://arxiv.org/abs/2601.18734)
- **LoRA and QLoRA**
  - [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
  - [QLoRA](https://arxiv.org/abs/2305.14314)
- **Caveats**
  - [Spurious Rewards](https://arxiv.org/abs/2506.10947)
  - [RL sharpens rather than expands](https://arxiv.org/abs/2504.13837)
  - [TinyZero: 0.5B fails Countdown](https://github.com/Jiayi-Pan/TinyZero)
- **Reference code**
  - [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch)
  - [nanochat](https://github.com/karpathy/nanochat)
  - [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
  - [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero)
- **Tasks and datasets**
  - [Reasoning Gym](https://github.com/open-thought/reasoning-gym)
  - [Qwen3-1.7B Wordle RL](https://huggingface.co/PrimeIntellect/Qwen3-1.7B-Wordle-RL)
  - [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)
  - [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2)
  - [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
  - [RLVR-IFeval](https://huggingface.co/datasets/allenai/RLVR-IFeval)
- **Hardware**
  - [Modal pricing](https://modal.com/pricing)
  - [Kaggle GPU docs](https://www.kaggle.com/docs/efficient-gpu-usage)
  - [fp16 skipped steps on a Kaggle T4](https://github.com/unslothai/unsloth/blob/main/tests/kaggle/t4_smoke/run_grpo_t4.py)

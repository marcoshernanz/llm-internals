# Post-training roadmap

Post-train a small pretrained model step by step, in the frontier's order: SFT, then RL, then distillation from several teachers. The practitioner tools (LoRA, QLoRA, DPO) come in where we first need them. Every algorithm is written here in plain PyTorch. Only the weights, the tokenizer and the datasets are borrowed.

Researched in October 2026. Numbers marked "est." are order-of-magnitude estimates, not measurements.

## Stages

| # | Stage | Status |
|---|---|---|
| 0 | Foundations: Qwen3 in our own code, a KV-cache sampler, graders | Done |
| 1 | SFT | Not started |
| 2 | LoRA, then QLoRA | Not started |
| 3 | Rejection sampling | Not started |
| 4 | RL with verifiable rewards | Not started |
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
  - Several conversations packed per sequence, with no attention across them.
  - A full fine-tune.
- **Problem we'll hit:** the logits for 8k tokens × a 152k vocab are 4.6 GiB in fp32, so the loss is computed in chunks.
- **Data:** `smol-smoltalk` (built for models under 1B), plus GSM8K as chats.
- **Check:**
  - It stops at `<|im_end|>`.
  - GSM8K accuracy, asked zero-shot in chat format.
- **First run** (`post/eval/sft.py`, which runs `post/sft.py`, 400 steps of 4 GSM8K chats, 2 minutes on the L4), on the first 50 test questions:
  - 94% close their turn with `<|im_end|>`, against 0% for the base model; its probability where answers end went from 1.3e-6 to 0.91.
  - 40% right, against 6%. 4-shot, the base model gets 57%.
  - 28% of the tokens read were padding.
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
- **Frontier:** practitioner tools. The labs fine-tune fully. LoRA appears in RL research, where rank 1 is enough.
- **Runs on:** Kaggle.

### 3. Rejection sampling: learn from its own right answers

- **Problem:** the SFT model's pass@8 is far above its pass@1. The right answer is in there; it just doesn't pick it.
- **Build:** sample 8 answers per problem, keep the verified ones, fine-tune on them, repeat.
- **Check:** pass@1 climbs toward the old pass@8, while pass@8 itself barely moves.
- **Frontier:** DeepSeek R1 (~800k samples), every round of Llama 3, and Qwen3. It's also the oldest form of self-distillation.
- **Runs on:** Kaggle.

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
- **Frontier:** the end state is where OLMo 3, DeepSeek, TRL's defaults and Meta's ScaleRL all land. The alternatives are GSPO (sequence-level ratios, Qwen's choice for MoE) and CISPO (MiniMax).
- **Runs on:** Kaggle. Each fix runs on one card next to its baseline on the other.

### 5. Multi-turn environments and tools

- **Problem:** in stage 4's failures the reasoning is right but the arithmetic slips.
- **Build:**
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
  4. A rubric judge: the 8B teacher grading answers.
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

# 1. Fine-tuning

SFT teaches the base model to answer and stop. The rest does the same training in fewer bytes. The
full versions are in `post/`, with their results in `post/ROADMAP.md`, stages 1 and 2.

| # | File | Builds on | Adds | Full version |
|---|---|---|---|---|
| 001 | [`001_sft.py`](001_sft.py) | — | the loss on the assistant's tokens only | `post/sft.py` |
| 002 | [`002_lora.py`](002_lora.py) | — | frozen weights, a trained low-rank correction next to each linear layer | `post/lora.py` |
| 003 | [`003_nf4.py`](003_nf4.py) | — | each block of 64 weights in 4 bits, the scales quantized again | `post/nf4.py` |
| 004 | [`004_qlora.py`](004_qlora.py) | 002 | 002's frozen layers stored in 003's NF4 | `post/qlora.py` |

## 001 SFT

**Problem.** A base model continues text. Asked a question, it writes more questions, or keeps going
past the answer.

**Fix.** Train on conversations in the chat format, with the loss on the assistant's tokens and its
`<|im_end|>` only: what the model has to write, and where to stop. The prompt and the headers are
context, not targets.

**The change.** `render` marks the answer's tokens, and `sft_loss` averages the loss over them.

## 002 LoRA

**Problem.** A full fine-tune trains and stores every weight, plus Adam's two states for each.

**Fix.** Freeze every linear layer and train a low-rank correction next to it:

$$
y = W x + \frac{\alpha}{r} B A x
$$

A has r rows and B has r columns. B starts at 0, so training starts from the loaded model exactly.

**The change.** `LoRA` and `add_lora`. Only A and B are given to the optimizer.

**Builds on —.** It starts again, on four made-up facts rather than 001's conversations.

## 003 NF4

**Problem.** The frozen weights still take 16 bits each, and they're most of the memory.

**Fix.** Weights are close to normally distributed, so each block of 64 is stored as 4-bit indices
into 16 levels placed at the normal distribution's quantiles, scaled by the block's largest
magnitude. The scales are quantized again: their mean subtracted, then blocks of 256 in int8 with
one fp32 factor each. That's 4.127 bits per weight.

**The change.** `nf4_levels`, `quantize` and `dequantize`. `main` replaces every linear layer's
weights with their round trip and compares the loss on a text.

**Builds on —.** It measures, it doesn't train.

## 004 QLoRA

**Problem.** LoRA freezes the weights but still keeps them at full size.

**Fix.** Store each frozen linear layer in NF4 and dequantize it whenever it's used, while the LoRA
corrections train at full precision.

**The change.** 002 plus 003's `quantize` and `dequantize`, `NF4Linear`, and `add_qlora`, which
wraps an `NF4Linear` instead of the layer itself.

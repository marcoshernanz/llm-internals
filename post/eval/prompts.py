"""
Ask everyday prompts, none of them math, before and after SFT (post/sft.py), and print both answers
to each. SFT on GSM8K alone shows the model one kind of conversation only: a word problem, then a
worked solution ending in "The answer is N.". This shows what that does to everything else, and
counts the answers that end the GSM8K way. Greedy, in chat format, stopping at <|im_end|> or
<|endoftext|>.
Run with ``uv run modal run gpu.py --modules post.eval.prompts``.

Shape letters used in the comments:
    T   prompt length
    N   new tokens
"""

import torch
from tokenizers import Tokenizer

from post.chat import render_prompt
from post.eval.chat import STOPS
from post.qwen3 import DEVICE, DTYPE, REPO_ID, Qwen3, load_qwen3, load_tokenizer, token_id
from post.sampler import generate
from post.sft import load_chats, train

PROMPTS = [
    "Hi! How are you today?",
    "What is the capital of France?",
    "Write a haiku about autumn.",
    "Give me three tips for sleeping better.",
    "Translate 'Good morning, my friend' into Spanish.",
    "Explain what a black hole is in two sentences.",
    "Write a short, friendly email declining a meeting invitation.",
    "What's the difference between a virus and a bacterium?",
]
MAX_NEW_TOKENS = 160  # enough for every answer above, if it stops


def answers(model: Qwen3, tokenizer: Tokenizer) -> list[str]:
    """Each prompt's greedy answer, special tokens shown: an answer that closes its turn ends with
    <|im_end|>."""
    stop_ids = [token_id(tokenizer, token) for token in STOPS]
    texts = []
    for prompt in PROMPTS:
        ids = render_prompt(tokenizer, [{"role": "user", "content": prompt}])
        ids_t = torch.tensor(ids, device=DEVICE)  # [T]
        new_ids = generate(model, ids_t, 1, MAX_NEW_TOKENS, 0.0, stop_ids)  # [1, N]
        texts.append(tokenizer.decode(new_ids[0].tolist(), skip_special_tokens=False))
    return texts


def main() -> None:
    assert DEVICE == "cuda" and DTYPE == torch.bfloat16, "SFT computes in bf16: run it on the L4"
    torch.manual_seed(0)  # as post/sft.py: the same conversations in the same order
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)  # the checkpoint's own bf16
    before = answers(model, tokenizer)

    model.float()  # fp32 master weights for training: bf16 to fp32 is exact
    train(model, load_chats(tokenizer))
    model.to(torch.bfloat16)  # bf16 weights for generation, as the checkpoint came
    after = answers(model, tokenizer)

    for prompt, base, tuned in zip(PROMPTS, before, after):
        print(f"\n=== {prompt}")
        print(f"--- before SFT:\n{base}")
        print(f"--- after SFT:\n{tuned}")

    print(f"\nhow the {len(PROMPTS)} answers ended, and how many wrote 'The answer is'")
    for name, texts in [("before SFT", before), ("after SFT", after)]:
        turn = sum(text.endswith("<|im_end|>") for text in texts)
        document = sum(text.endswith("<|endoftext|>") for text in texts)
        limit = len(texts) - turn - document
        gsm8k = sum("The answer is" in text for text in texts)
        print(
            f"{name:<11} <|im_end|> {turn}, <|endoftext|> {document}, the token limit {limit}; "
            f"'The answer is' {gsm8k}"
        )


if __name__ == "__main__":
    main()

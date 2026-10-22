"""
Rejection sampling: the SFT model learns from its own right answers. It answers GSM8K train
questions that SFT never read, 8 samples each at temperature 1, and a grader keeps the right ones:
those that closed their turn and whose number after "The answer is" is the question's answer. Only
that final number is used, never the question's worked solution: a grader that can check an answer
stands in for a person writing one. Of a question's right answers, exact repeats are dropped and
at most 4 kept, so the questions it finds easy don't crowd out the rest, as ReST-EM caps them.

Then the base model is fine-tuned again from the start, on SFT's conversations plus the kept ones,
with SFT's recipe (post/sft.py), as STaR, ReST-EM and DeepSeek R1 do. The alternative is to go on
training the SFT model on the kept answers alone, as Qwen3 does in its later SFT stages.
Run with ``uv run modal run gpu.py --modules post.rejection_sampling``, after post.sft has saved
its weights; it saves its own as "rejection_sampling". post/eval/rejection_sampling.py measures
what it changes.
"""

import torch
from tokenizers import Tokenizer

from post.chat import answers, render
from post.gsm8k import reference_answer, strict_answer
from post.qwen3 import (
    DEVICE,
    REPO_ID,
    Qwen3,
    load_qwen3,
    load_tokenizer,
    load_weights,
    save_weights,
)
from post.sft import Chat, load_chats, train, unread_gsm8k

QUESTIONS = 384  # GSM8K train questions SFT didn't read
SAMPLES = 8  # per question
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 256  # per answer
KEEP = 4  # right answers kept per question, at most


def right_answers(
    model: Qwen3, tokenizer: Tokenizer, problems: list[dict[str, str]]
) -> list[list[Chat]]:
    """For each problem, its question with each of its kept answers: conversations to train on."""
    questions = [problem["question"] for problem in problems]
    sampled = answers(model, tokenizer, questions, SAMPLES, TEMPERATURE, MAX_NEW_TOKENS)
    kept = []
    for problem, texts in zip(problems, sampled):
        reference = reference_answer(problem["answer"])
        right = [t for t in texts if t.endswith("<|im_end|>") and strict_answer(t) == reference]
        right = list(dict.fromkeys(right))  # exact repeats dropped, in order
        chats = []
        for text in right[:KEEP]:
            messages = [
                {"role": "user", "content": problem["question"]},
                {"role": "assistant", "content": text.removesuffix("<|im_end|>")},
            ]
            chats.append(render(tokenizer, messages))
        kept.append(chats)
    return kept


def retrain(tokenizer: Tokenizer, kept: list[Chat]) -> Qwen3:
    """The base model fine-tuned on SFT's conversations and the kept ones, shuffled together."""
    torch.manual_seed(0)
    chats = load_chats(tokenizer) + kept  # SFT's conversations, the same ones
    order = torch.randperm(len(chats)).tolist()
    chats = [chats[i] for i in order]
    model = load_qwen3(REPO_ID, torch.float32, DEVICE)  # fp32: the master copy of the weights
    train(model, chats)
    return model


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)
    load_weights(model, "sft")
    kept = right_answers(model, tokenizer, unread_gsm8k(QUESTIONS))
    del model  # free the SFT model before the base model loads
    model = retrain(tokenizer, [chat for chats in kept for chat in chats])
    save_weights(model, "rejection_sampling")


if __name__ == "__main__":
    main()

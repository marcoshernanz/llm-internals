"""
Ask GSM8K questions the way a chat model is asked: zero-shot, each question as a user turn in Qwen's
chat format (post/chat.py), with the assistant's turn opened for the model to write:

    <|im_start|>user
    Janet’s ducks lay 16 eggs per day. [...]<|im_end|>
    <|im_start|>assistant

A chat model answers, then closes its turn with <|im_end|>. This measures what the model does, on
greedy answers: how each answer ends, the first answer of each kind, and how many are right.
Generation stops at <|im_end|> or <|endoftext|>, the two tokens Qwen's chat models stop at. Run on
the base model, on the first 200 test questions, it is the baseline SFT starts from;
post/eval/sft.py asks the fine-tuned model the same way.
Run with ``uv run modal run gpu.py --modules post.eval.chat``.

Shape letters used in the comments:
    B   answers sampled at once (1: greedy)
    T   prompt length
    N   new tokens
"""

import time

import torch
from tokenizers import Tokenizer

from post.chat import render_prompt
from post.eval.gsm8k import GREEDY_QUESTIONS, MAX_NEW_TOKENS
from post.gsm8k import last_number, load_gsm8k, reference_answer, strict_answer
from post.qwen3 import DEVICE, DTYPE, REPO_ID, Qwen3, load_qwen3, load_tokenizer, token_id
from post.sampler import generate

STOPS = ["<|im_end|>", "<|endoftext|>"]  # the end of a turn, the end of a document
# What can end an answer: a stop, or a new turn opened without closing the answer's own
ENDINGS = [*STOPS, "<|im_start|>"]
LIMIT = f"the {MAX_NEW_TOKENS}-token limit"  # the ending of an answer that wrote none of them


def split_answer(row: list[int], endings: dict[int, str]) -> tuple[list[int], str]:
    """The answer's tokens, up to the first ending token, and how it ended: that token, or the
    token limit if it wrote none."""
    for i, token in enumerate(row):
        if token in endings:
            return row[:i], endings[token]
    return row, LIMIT


def ask(model: Qwen3, tokenizer: Tokenizer, problems: list[dict[str, str]]) -> None:
    """Ask each question zero-shot in chat format, greedy, and print how the answers end, the first
    answer of each ending, and how many are right."""
    stop_ids = [token_id(tokenizer, token) for token in STOPS]
    endings = {token_id(tokenizer, token): token for token in ENDINGS}
    print(f"greedy answers, zero-shot in chat format, to the first {len(problems)} questions")
    start = time.perf_counter()
    counts = {ending: 0 for ending in [*ENDINGS, LIMIT]}
    lengths = {ending: 0 for ending in [*ENDINGS, LIMIT]}  # answer tokens, summed
    examples = {}  # the first answer of each ending, special tokens shown
    strict_right = 0
    last_right = 0
    for i, problem in enumerate(problems):
        prompt = render_prompt(tokenizer, [{"role": "user", "content": problem["question"]}])
        ids = torch.tensor(prompt, device=DEVICE)  # [T]
        new_ids, _ = generate(model, [ids], 1, MAX_NEW_TOKENS, 0.0, stop_ids)  # [1, N]
        row = new_ids[0].tolist()  # N token ids
        answer_ids, ending = split_answer(row, endings)
        counts[ending] += 1
        lengths[ending] += len(answer_ids)
        if ending not in examples:
            examples[ending] = (i, tokenizer.decode(row, skip_special_tokens=False))
        text = tokenizer.decode(answer_ids)
        reference = reference_answer(problem["answer"])
        strict_right += strict_answer(text) == reference
        last_right += last_number(text) == reference
        if (i + 1) % 25 == 0:
            print(
                f"{i + 1:>4}: closed its turn {counts['<|im_end|>'] / (i + 1):4.0%}, "
                f"right by last number {last_right / (i + 1):4.0%}, "
                f"by strict {strict_right / (i + 1):4.0%}"
            )
    print(f"({(time.perf_counter() - start) / 60:.1f} minutes)")

    print("\nhow the answers ended: answers, share, tokens before the ending on average")
    for ending, count in counts.items():
        mean = lengths[ending] / count if count else 0
        print(f"{ending:>22}: {count:>4} {count / len(problems):5.0%} {mean:6.0f}")

    for ending, (i, text) in examples.items():
        print(f"\n--- the first answer that ended at {ending}: question {i + 1} ---")
        print(text)


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, DTYPE, DEVICE)
    model.compile(fullgraph=True)
    test = load_gsm8k("test")
    print("4-shot, the same questions: 57% right by last number (post/eval/gsm8k.py)")
    ask(model, tokenizer, test[:GREEDY_QUESTIONS])


if __name__ == "__main__":
    main()

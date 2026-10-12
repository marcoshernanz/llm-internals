"""
Check the chat format in post/chat.py on cases whose answers are known: the text it writes, the
ids, which tokens the mask counts, and that a prompt matches what the checkpoint's own chat template
writes. Only the tokenizer runs. Run with ``uv run python -m post.check_chat``.
"""

from transformers import AutoTokenizer

from post.chat import Message, render, render_prompt
from post.generate import REPO_ID
from post.qwen3 import load_tokenizer

CONVERSATION: list[Message] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+3?"},
    {"role": "assistant", "content": "2+3 = 5."},
    {"role": "user", "content": "And times 4?"},
    {"role": "assistant", "content": "5*4 = 20."},
]
TEXT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nWhat is 2+3?<|im_end|>\n"
    "<|im_start|>assistant\n2+3 = 5.<|im_end|>\n"
    "<|im_start|>user\nAnd times 4?<|im_end|>\n"
    "<|im_start|>assistant\n5*4 = 20.<|im_end|>\n"
)


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)

    ids, mask = render(tokenizer, CONVERSATION)
    assert tokenizer.decode(ids, skip_special_tokens=False) == TEXT
    assert ids == tokenizer.encode(TEXT).ids  # here pieces and the whole text tokenize alike
    learned = [token for token, counted in zip(ids, mask) if counted]
    learned_text = tokenizer.decode(learned, skip_special_tokens=False)
    assert learned_text == "2+3 = 5.<|im_end|>5*4 = 20.<|im_end|>", learned_text
    print(f"{len(ids)} tokens, {sum(mask)} counted: every assistant turn and its <|im_end|>")

    prompt = render_prompt(tokenizer, CONVERSATION[:4])  # up to the last question
    theirs = AutoTokenizer.from_pretrained(REPO_ID)  # transformers' tokenizer, with the template
    assert theirs is not None
    template = theirs.apply_chat_template(
        CONVERSATION[:4], tokenize=False, add_generation_prompt=True
    )
    assert isinstance(template, str)
    assert prompt == tokenizer.encode(template).ids
    print("a prompt matches the checkpoint's chat template")

    # An answer that starts with newlines: tokenized whole, the header's newline and the answer's
    # merge into one token. Tokenized in pieces, the answer starts on a token of its own.
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "\n\nHello"}]
    ids, mask = render(tokenizer, messages)
    whole = tokenizer.encode(tokenizer.decode(ids, skip_special_tokens=False)).ids
    assert ids != whole
    header = tokenizer.encode("<|im_start|>assistant\n").ids
    start = mask.index(True)  # the answer's first token
    assert ids[start - len(header) : start] == header  # what the prompt ends with at inference
    learned = [token for token, counted in zip(ids, mask) if counted]
    assert tokenizer.decode(learned, skip_special_tokens=False) == "\n\nHello<|im_end|>"
    print("an answer starting with newlines starts on its own token")
    print("the chat format is right")


if __name__ == "__main__":
    main()

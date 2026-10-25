"""
Look at the chat and tool-call tokens' rows in the base model's embedding table, which is also its
head: a token's logit is its row times the hidden state. A row the base model trained points its own way; a row it
never trained stays near where it started, beside the rows of other tokens it rarely saw. Then no
hidden state can raise that token's logit without raising theirs too. For each token: its row's
length against the median row's, how many rows are the very same vector, and the rows nearest it
by cosine similarity.
Run with ``uv run modal run gpu.py --modules post.eval.embeddings``.

Shape letters used in the comments:
    V   vocabulary size
    D   model width
"""

import torch

from post.qwen3 import DEVICE, REPO_ID, load_qwen3, load_tokenizer

TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<|endoftext|>",
    " dóla",
    " spep",
]
NEIGHBORS = 8
CLOSE = 0.5  # a cosine similarity above which two rows count as pointing the same way


def main() -> None:
    tokenizer = load_tokenizer(REPO_ID)
    model = load_qwen3(REPO_ID, torch.bfloat16, DEVICE)
    table = model.embed_tokens.weight.float()  # [V, D]
    lengths = table.norm(dim=-1)  # [V]
    directions = table / lengths[:, None]  # [V, D], each row at length 1
    print(f"the median row's length: {lengths.median().item():.3f}")
    for token in TOKENS:
        ids = tokenizer.encode(token).ids
        assert len(ids) == 1, token
        cosine = directions @ directions[ids[0]]  # [V]
        cosine[ids[0]] = -1.0  # not itself
        close = int((cosine > CLOSE).sum().item())
        same = (table == table[ids[0]]).all(dim=-1)  # [V], rows that are this very vector
        print(
            f"\n{token!r}: row length {lengths[ids[0]].item():.3f}; {close:,} other rows at a "
            f"cosine above {CLOSE}, {int(same.sum().item()) - 1:,} the same vector to the bit"
        )
        top = cosine.topk(NEIGHBORS)
        for value, index in zip(top.values.tolist(), top.indices.tolist()):
            print(f"  {tokenizer.decode([index], skip_special_tokens=False)!r:>24} {value:.3f}")


if __name__ == "__main__":
    main()

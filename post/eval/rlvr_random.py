"""
How much of RL's gain comes from the reward: post/rlvr.py with every answer's reward a coin flip,
1 or 0 with even odds, whatever the answer says. If the score stays near the SFT model's, the
reward did the work; if it rises anyway, the training sharpened the model onto what it already
preferred. Random rewards raised Qwen2.5-Math's score in the paper that tried them, and not
Llama's or OLMo's. The training log's reward is the coin's; the check's reward on the same answers
is printed after, a quarter of the run at a time. Measured as post/eval/rlvr_noise.py measures the
full run, whose three runs give the noise to compare with.
Run with ``uv run modal run gpu.py --modules post.eval.rlvr_random``, after post.sft has saved its
weights.
"""

from unittest import mock

import torch

from post.eval.rlvr_noise import run, sft_model
from post.reinforce import reward

checked: list[float] = []  # the check's reward on every answer the run samples, in order


def coin_flip(text: str, reference: float) -> float:
    """1 or 0 with even odds; the check's reward is kept aside."""
    checked.append(reward(text, reference))
    return float(torch.rand(()) < 0.5)


def main() -> None:
    model, tokenizer = sft_model()
    print("post/rlvr.py with every reward a coin flip:")
    with mock.patch("post.reinforce.reward", coin_flip):  # rollouts looks it up on every answer
        run(model, tokenizer, 0)
    quarter = len(checked) // 4
    quarters = [checked[i * quarter : (i + 1) * quarter] for i in range(4)]
    means = ", ".join(f"{sum(part) / len(part):.3f}" for part in quarters)
    print(f"the check's reward during training, a quarter of the run at a time: {means}")


if __name__ == "__main__":
    main()

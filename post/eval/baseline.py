"""
Run REINFORCE with the baseline (post/reinforce.py) from the SFT model and measure it as
post/eval/reinforce.py measures REINFORCE without it: the same questions in the same order, so
its reward and its scores compare with that run's.
Run with ``uv run modal run gpu.py --modules post.eval.baseline``, after post.sft has saved its
weights; it saves the trained ones as "baseline".
"""

from post.eval.reinforce import run


def main() -> None:
    run(baseline=True)


if __name__ == "__main__":
    main()

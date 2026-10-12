"""
Check the GSM8K tools where the right result is known: the few-shot example made from a reference
answer and both graders of post/gsm8k.py, on endings of answers the model really wrote; and pass@k
of post/eval/gsm8k.py, against counting every way to pick k of the samples.
Run with ``uv run python -m post.check.gsm8k``.
"""

from itertools import combinations

from post.eval.gsm8k import pass_at_k
from post.gsm8k import cut, last_number, reference_answer, strict_answer, worked_solution

# The first reference answer of the train split, as stored, and as the prompt shows it
STORED = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
    "#### 72"
)
EXAMPLE = (
    "Natalia sold 48/2 = 24 clips in May.\n"
    "Natalia sold 48+24 = 72 clips altogether in April and May.\n"
    "The answer is 72."
)
RUN_ON = (
    "So, Eliza's earnings for this week are 400 + 60 = $460.\nThe answer is 460.\n\n"
    "Question: The distance travelled downstream in 12 minutes is 14.0 km. [...]\n"
    "Answer: [...] The answer is 74."
)
# Endings of answers the model wrote, 4-shot: (text, strict reads, last number reads)
GRADED = [
    ("so she makes 9 * $2 = $18.\nThe answer is 18.", 18.0, 18.0),
    # Right, but without the answer line: only the last number finds it
    ("In total, Eliza earns $400 + $60 = $460 for this week.", None, 460.0),
    # Wrong, and both say so
    ("we get x = 6.\nMelanie started with 6 vacuum cleaners.", None, 6.0),
    # Right, but the last number is the minutes of a time: the last-number rule's known failure
    ("The candle will be 8 centimeters shorter after burning from 1:00 PM to 5:00 PM.", None, 0.0),
    # A run-on: cut at the invented question, whose numbers would otherwise come last
    (RUN_ON, 460.0, 460.0),
]
SAMPLES = [(16, 2), (16, 0), (16, 16), (10, 3)]  # (n, c): samples, and how many of them are right


def main() -> None:
    assert worked_solution(STORED) == EXAMPLE
    assert reference_answer("[...]\n#### 1,450,000") == 1_450_000
    print("the few-shot example and the reference number come out as written by hand")

    for text, strict, last in GRADED:
        answer = cut(text)
        assert strict_answer(answer) == strict, text
        assert last_number(answer) == last, text
    print(f"both graders read the expected number off all {len(GRADED)} answers")

    for n, c in SAMPLES:
        samples = [True] * c + [False] * (n - c)  # right or wrong, one per sample
        for k in range(1, n + 1):
            picks = list(combinations(samples, k))  # every way to pick k of the n
            counted = sum(any(pick) for pick in picks) / len(picks)
            assert abs(pass_at_k(n, c, k) - counted) < 1e-12, (n, c, k)
    print(f"pass@k matches counting every pick of k, for every k, in {len(SAMPLES)} cases")


if __name__ == "__main__":
    main()

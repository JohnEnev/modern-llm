"""Reward function for GRPO math training.

Three-tier reward scheme:
    1.0  -> correct answer, found via the STRICT "Final answer: X" parser
    0.5  -> correct answer, found only via the LENIENT fallback parser
    0.05 -> strict format present, but the answer is WRONG
    0.0  -> no parseable answer at all, or lenient answer is wrong
"""

import math
import re


# Allows:
#   Final answer: 12
#   final answer: -3.5
#   Final Answer: 1,024
#   Final answer: +7
#
# Does not handle fractions yet; keep first version simple.
NUMBER_PATTERN = r"[+-]?\d[\d,]*(?:\.\d+)?"
STRICT_PATTERN = rf"\bFinal\s+answer\s*:\s*({NUMBER_PATTERN})\b"
LENIENT_PATTERN = NUMBER_PATTERN


def parse_number(s: str) -> float:
    """Parse numeric strings like '12', '-3.5', '+7', '1,024'."""
    return float(s.replace(",", ""))


def extract_strict_answer(completion: str) -> float | None:
    """Extract the LAST strict 'Final answer: X' occurrence.

    Last occurrence wins because models may revise an earlier false start.
    """
    matches = re.findall(STRICT_PATTERN, completion, flags=re.IGNORECASE)
    if not matches:
        return None
    return parse_number(matches[-1])


def extract_lenient_answer(completion: str) -> float | None:
    """Fallback: find the LAST number mentioned anywhere in the text."""
    matches = re.findall(LENIENT_PATTERN, completion)
    if not matches:
        return None
    return parse_number(matches[-1])


def answers_match(predicted: float, ground_truth: float, tol: float = 1e-4) -> bool:
    return math.isclose(predicted, ground_truth, rel_tol=0.0, abs_tol=tol)


def compute_reward(completion: str, ground_truth: float) -> float:
    """Compute scalar reward for one completion.

    States:
        strict found + correct                  -> 1.0
        strict found + wrong                    -> 0.05
        strict not found + lenient correct      -> 0.5
        strict not found + lenient wrong/none   -> 0.0
    """
    strict_answer = extract_strict_answer(completion)

    if strict_answer is not None:
        if answers_match(strict_answer, ground_truth):
            return 1.0
        return 0.05

    lenient_answer = extract_lenient_answer(completion)

    if lenient_answer is not None and answers_match(lenient_answer, ground_truth):
        return 0.5

    return 0.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_reward():
    # Strict, correct -> 1.0
    assert compute_reward("Let's compute. Final answer: 12", 12) == 1.0

    # Strict, wrong -> 0.05
    assert compute_reward("Let's compute. Final answer: 5", 12) == 0.05

    # No strict phrase, lenient correct -> 0.5
    assert compute_reward("15% of 80 is 0.15 * 80 = 12. So the answer is 12.", 12) == 0.5

    # No strict phrase, lenient wrong -> 0.0
    assert compute_reward("I think the answer is 99 maybe?", 12) == 0.0

    # Nothing parseable -> 0.0
    assert compute_reward("I am not sure how to solve this.", 12) == 0.0

    # Negative numbers and decimals
    assert compute_reward("Final answer: -3.5", -3.5) == 1.0

    # Case-insensitive strict format
    assert compute_reward("final answer: 12", 12) == 1.0
    assert compute_reward("Final Answer: 12", 12) == 1.0

    # Commas
    assert compute_reward("Final answer: 1,024", 1024) == 1.0
    assert compute_reward("The answer is 1,024", 1024) == 0.5

    # Explicit plus sign
    assert compute_reward("Final answer: +7", 7) == 1.0

    # Strict phrase present but non-numeric -> no strict numeric match,
    # falls through to lenient. "twelve" has no numeric token -> 0.0
    assert compute_reward("Final answer: twelve", 12) == 0.0

    # Multiple strict occurrences — last one wins
    assert compute_reward(
        "Final answer: 5. Wait, let me redo this. Final answer: 12",
        12,
    ) == 1.0

    # Strict wrong should NOT fallback to lenient even if correct number appears later
    assert compute_reward(
        "Final answer: 5. The correct computation would have produced 12.",
        12,
    ) == 0.05

    # Lenient uses last number
    assert compute_reward(
        "We calculate 5 + 7. Therefore the answer is 12.",
        12,
    ) == 0.5

    # Tolerance
    assert compute_reward("Final answer: 12.00001", 12) == 1.0

    print("✓ All reward tests passed")


if __name__ == "__main__":
    test_reward()
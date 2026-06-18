"""Synthetic arithmetic problem generator for GRPO training.

Produces (prompt, ground_truth) pairs across a curriculum of difficulty
stages, from single-digit addition up through simple percentages. Each
problem comes with a KNOWN, programmatically-computed correct answer —
this is what makes synthetic data usable for GRPO (real reward requires
ground truth; you can't compute reward on unlabeled text).

Prompt format matches what the reward parser expects:

    Solve the problem step by step.
    End your answer exactly as:
    Final answer: <number>

    Problem: What is 47 + 38?
"""

import random


PROMPT_TEMPLATE = (
    "Solve the problem step by step.\n"
    "End your answer exactly as:\n"
    "Final answer: <number>\n\n"
    "Problem: {question}"
)


def make_single_digit_addsub(rng: random.Random) -> tuple[str, float]:
    """Single-digit addition or subtraction, e.g. '7 + 4' or '9 - 3'.

    Returns:
        (question_text, answer) — question_text is just "What is X + Y?",
        NOT yet wrapped in PROMPT_TEMPLATE (that happens in make_prompt below).
    """
    X = rng.randint(0, 9)
    Y = rng.randint(0, 9)
    sign = "+" if rng.random() < 0.5 else "-"

    question_text = f"What is {X} {sign} {Y}?"
    if sign == "+":
        answer = X + Y
    else:
        answer = X - Y
    
    return (question_text, answer)


def make_two_digit_addsub(rng: random.Random) -> tuple[str, float]:
    """Two-digit addition or subtraction, e.g. '47 + 38' or '92 - 56'."""
    X = rng.randint(0, 99)
    Y = rng.randint(10, 99)
    sign = "+" if rng.random() < 0.5 else "-"

    question_text = f"What is {X} {sign} {Y}?"
    if sign == "+":
        answer = X + Y
    else:
        answer = X - Y
    
    return (question_text, answer)


def make_small_multiplication(rng: random.Random) -> tuple[str, float]:
    """Small multiplication, e.g. '7 * 8'. Both factors in [2, 12]."""
    X = rng.randint(2, 12)
    Y = rng.randint(2, 12)
    sign = "*"

    question_text = f"What is {X} {sign} {Y}?"
    answer = X * Y
    
    return (question_text, answer)


def make_percentage(rng: random.Random) -> tuple[str, float]:
    """Simple percentage, e.g. 'What is 15% of 80?'

    Result is always either a clean whole number or ends in exactly .5 —
    never a messier repeating decimal. This is guaranteed by construction:
    X is chosen from a set that always divides evenly into a multiple of 50,
    and Y is constrained to multiples of 10, so X*Y is always a multiple
    of 50 (landing on a whole number or exactly .5, never anything messier).
    """
    X = rng.choice([5, 10, 15, 20, 25, 50])
    Y = rng.choice([10, 30, 50, 70, 90, 110, 130, 150])

    question_text = f"What is {X}% of {Y}?"
    answer = X * Y / 100

    # Sanity check: confirm we really did land on whole-or-.5, not something messier
    doubled = answer * 2
    assert doubled == int(doubled), f"Got a non-half-integer: {X}% of {Y} = {answer}"

    return (question_text, answer)


# Maps a curriculum stage name to its generator function.
STAGE_GENERATORS = {
    "single_digit": make_single_digit_addsub,
    "two_digit": make_two_digit_addsub,
    "multiplication": make_small_multiplication,
    "percentage": make_percentage,
}


def make_prompt(stage: str, rng: random.Random) -> tuple[str, float]:
    """Generate one (full_prompt, ground_truth) pair for the given stage."""
    function_to_call = STAGE_GENERATORS[stage]
    question_text, answer = function_to_call(rng)
    prompt = PROMPT_TEMPLATE.format(question=question_text)

    return (prompt, answer)


def make_eval_set(stage: str, n: int = 100, seed: int = 1234) -> list[tuple[str, float]]:
    """Generate a FIXED held-out evaluation set for one curriculum stage.

    Uses its own seeded Random instance, separate from whatever RNG drives
    training-time sampling, so the eval set never overlaps with — or
    depends on the state of — the training prompt stream.
    """
    fixed_rng = random.Random(seed)
    prompts = []
    for _ in range(n):
        prompts.append(make_prompt(stage, fixed_rng))
        
    return prompts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_synthetic_math():
    rng = random.Random(0)

    # Each generator should produce a well-formed question + correct answer
    for stage, gen_fn in STAGE_GENERATORS.items():
        question, answer = gen_fn(rng)
        assert isinstance(question, str) and len(question) > 0
        assert isinstance(answer, (int, float))
        print(f"  [{stage}] {question} -> {answer}")

    # make_prompt should wrap correctly and be parseable by the reward module
    prompt, answer = make_prompt("single_digit", rng)
    assert "Final answer: <number>" in prompt
    assert "Problem:" in prompt
    print(f"\n  Sample full prompt:\n{prompt}\n  -> ground truth: {answer}")

    # Eval set determinism: same seed -> identical set, every time
    eval_set_1 = make_eval_set("two_digit", n=10, seed=42)
    eval_set_2 = make_eval_set("two_digit", n=10, seed=42)
    assert eval_set_1 == eval_set_2, "Eval set is not deterministic!"
    assert len(eval_set_1) == 10

    # Different seed -> should (almost certainly) differ
    eval_set_3 = make_eval_set("two_digit", n=10, seed=999)
    assert eval_set_1 != eval_set_3

    print("\n✓ All synthetic_math tests passed")


if __name__ == "__main__":
    test_synthetic_math()
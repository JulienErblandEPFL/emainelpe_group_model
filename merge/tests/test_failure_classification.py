"""Unit tests for failure classification in :mod:`merge.eval_all`.

Pure string-processing tests — no torch, no vLLM, no datasets. They run
on a torch-free laptop and on the cluster identically.
"""
from __future__ import annotations

import pytest

from merge.eval_all import (
    FailureCategory,
    classify_completion,
    classify_problem_failure,
)


# ---------------------------------------------------------------------------
# Per-completion classification
# ---------------------------------------------------------------------------

def test_classify_completion_passes_when_correct() -> None:
    """A correct boxed answer returns None (i.e. the completion passed)."""
    result = classify_completion(
        completion=r"Let me think... The answer is \boxed{42}.",
        expected_answer="42",
        benchmark="math",
        max_tokens_used=100,
        max_tokens_limit=2048,
    )
    assert result is None


def test_classify_completion_no_boxed() -> None:
    result = classify_completion(
        completion="The answer is 42.",
        expected_answer="42",
        benchmark="math",
        max_tokens_used=100,
        max_tokens_limit=2048,
    )
    assert result == FailureCategory.NO_BOXED


def test_classify_completion_empty_boxed() -> None:
    result = classify_completion(
        completion=r"I'm not sure: \boxed{}.",
        expected_answer="42",
        benchmark="math",
        max_tokens_used=100,
        max_tokens_limit=2048,
    )
    assert result == FailureCategory.EMPTY_BOXED


def test_classify_completion_empty_boxed_whitespace_only() -> None:
    """Whitespace-only contents inside ``\\boxed{}`` count as empty."""
    result = classify_completion(
        completion=r"Answer: \boxed{   }",
        expected_answer="42",
        benchmark="math",
        max_tokens_used=100,
        max_tokens_limit=2048,
    )
    assert result == FailureCategory.EMPTY_BOXED


def test_classify_completion_wrong_answer() -> None:
    result = classify_completion(
        completion=r"After thinking... \boxed{43}.",
        expected_answer="42",
        benchmark="math",
        max_tokens_used=100,
        max_tokens_limit=2048,
    )
    assert result == FailureCategory.WRONG_ANSWER


def test_classify_completion_malformed_answer_letter_for_number() -> None:
    """Numeric expected + non-numeric extracted = malformed."""
    result = classify_completion(
        completion=r"Answer: \boxed{x}",
        expected_answer="42",
        benchmark="math",
        max_tokens_used=100,
        max_tokens_limit=2048,
    )
    assert result == FailureCategory.MALFORMED_ANSWER


def test_classify_completion_truncated() -> None:
    """Completion hit max_tokens with no closing ``\\boxed``."""
    result = classify_completion(
        completion="Let me think carefully about this problem. First we note that...",
        expected_answer="42",
        benchmark="math",
        max_tokens_used=2048,
        max_tokens_limit=2048,
    )
    assert result == FailureCategory.TRUNCATED


def test_classify_completion_truncated_with_partial_boxed_open() -> None:
    """``\\boxed{4`` (no closing brace) at the end still reads as truncated."""
    result = classify_completion(
        completion=r"Thinking step by step... \boxed{4",
        expected_answer="42",
        benchmark="math",
        max_tokens_used=2048,
        max_tokens_limit=2048,
    )
    assert result == FailureCategory.TRUNCATED


def test_classify_completion_refusal_pattern() -> None:
    """Refusal phrase + no correct answer = refusal."""
    result = classify_completion(
        completion="I cannot help with this request.",
        expected_answer="42",
        benchmark="safety",
        max_tokens_used=100,
        max_tokens_limit=2048,
    )
    assert result == FailureCategory.REFUSAL


def test_classify_completion_refusal_priority_over_boxed() -> None:
    """Refusal regex match wins even when a boxed appears in the same text."""
    result = classify_completion(
        completion=r"I'm sorry, but I cannot answer. However, \boxed{42}.",
        expected_answer="42",
        benchmark="safety",
        max_tokens_used=100,
        max_tokens_limit=2048,
    )
    assert result == FailureCategory.REFUSAL


def test_classify_completion_trailing_whitespace_in_boxed_passes() -> None:
    """``\\boxed{42 }`` should still compare equal to ``42`` via normalization."""
    result = classify_completion(
        completion=r"Thus \boxed{42 }.",
        expected_answer="42",
        benchmark="math",
        max_tokens_used=100,
        max_tokens_limit=2048,
    )
    assert result is None


def test_classify_completion_multiline_completion() -> None:
    """Multi-line completion with a correct boxed at the end passes."""
    completion = (
        "Step 1: Set up the equation.\n"
        "Step 2: Solve for x.\n"
        "After working through the algebra, we get:\n"
        r"\boxed{42}"
    )
    result = classify_completion(
        completion=completion,
        expected_answer="42",
        benchmark="math",
        max_tokens_used=200,
        max_tokens_limit=2048,
    )
    assert result is None


def test_classify_completion_multiple_boxed_uses_last() -> None:
    """When the text has 2 ``\\boxed{}``, ``last_boxed_only_string`` wins."""
    completion = r"First I thought \boxed{99}, but actually \boxed{42}."
    result = classify_completion(
        completion=completion,
        expected_answer="42",
        benchmark="math",
        max_tokens_used=100,
        max_tokens_limit=2048,
    )
    assert result is None


def test_classify_completion_knowledge_choice_letter_correct() -> None:
    """Knowledge MCQ with the right choice letter passes."""
    result = classify_completion(
        completion=r"After consideration, \boxed{C}.",
        expected_answer="C",
        benchmark="general_knowledge",
        max_tokens_used=80,
        max_tokens_limit=2048,
    )
    assert result is None


def test_classify_completion_knowledge_choice_letter_wrong() -> None:
    result = classify_completion(
        completion=r"After consideration, \boxed{A}.",
        expected_answer="C",
        benchmark="general_knowledge",
        max_tokens_used=80,
        max_tokens_limit=2048,
    )
    assert result == FailureCategory.WRONG_ANSWER


# ---------------------------------------------------------------------------
# Per-problem aggregation
# ---------------------------------------------------------------------------

def test_classify_problem_failure_strict_majority() -> None:
    """5 of 8 completions are NO_BOXED -> primary = NO_BOXED."""
    completions = [
        "no boxed here",
        "no boxed",
        "still nothing",
        r"\boxed{43}",      # wrong
        r"\boxed{44}",      # wrong
        "no",
        "nada",
        r"\boxed{x}",       # malformed
    ]
    tokens_used = [100] * 8
    primary, per_completion = classify_problem_failure(
        completions=completions,
        completions_tokens_used=tokens_used,
        expected_answer="42",
        benchmark="math",
        max_tokens_limit=2048,
    )
    assert primary == FailureCategory.NO_BOXED
    assert len(per_completion) == 8


def test_classify_problem_failure_mixed_when_no_majority() -> None:
    """3 wrong + 3 no_boxed + 2 truncated -> MIXED."""
    completions = [
        r"\boxed{43}", r"\boxed{44}", r"\boxed{45}",  # 3 wrong
        "no boxed", "still nothing", "nope",          # 3 no_boxed
        "truncated 1", "truncated 2",                  # 2 truncated
    ]
    tokens_used = [100, 100, 100, 100, 100, 100, 2048, 2048]
    primary, per_completion = classify_problem_failure(
        completions=completions,
        completions_tokens_used=tokens_used,
        expected_answer="42",
        benchmark="math",
        max_tokens_limit=2048,
    )
    assert primary == FailureCategory.MIXED
    assert sum(1 for c in per_completion if c == FailureCategory.WRONG_ANSWER) == 3
    assert sum(1 for c in per_completion if c == FailureCategory.NO_BOXED) == 3
    assert sum(1 for c in per_completion if c == FailureCategory.TRUNCATED) == 2


def test_classify_problem_failure_raises_if_any_passed() -> None:
    """If even one completion is correct, the pass@8=0 precondition is broken."""
    completions = [
        r"\boxed{42}",   # passes
        "no boxed",
        "nothing",
        "nothing",
        "nothing",
        "nothing",
        "nothing",
        "nothing",
    ]
    tokens_used = [100] * 8
    with pytest.raises(ValueError, match=r"passed"):
        classify_problem_failure(
            completions=completions,
            completions_tokens_used=tokens_used,
            expected_answer="42",
            benchmark="math",
            max_tokens_limit=2048,
        )


def test_classify_problem_failure_mismatched_token_list_lengths() -> None:
    """token list / completion list mismatch raises immediately."""
    with pytest.raises(ValueError, match=r"parallel"):
        classify_problem_failure(
            completions=["a", "b", "c"],
            completions_tokens_used=[1, 2],
            expected_answer="42",
            benchmark="math",
            max_tokens_limit=2048,
        )


def test_classify_problem_failure_all_refusal() -> None:
    """All 8 refusals -> primary REFUSAL (8/8 strict majority)."""
    completions = ["I cannot help."] * 8
    tokens_used = [50] * 8
    primary, per_completion = classify_problem_failure(
        completions=completions,
        completions_tokens_used=tokens_used,
        expected_answer="B",
        benchmark="safety",
        max_tokens_limit=2048,
    )
    assert primary == FailureCategory.REFUSAL
    assert all(c == FailureCategory.REFUSAL for c in per_completion)

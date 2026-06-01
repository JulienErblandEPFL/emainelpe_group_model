"""IO contract tests for :mod:`merge.infer` and :mod:`merge.eval_all`.

Pure-Python file IO + dataclass round-trips. No torch, no vLLM, no datasets.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from merge.eval_all import BenchmarkResult, FailureCategory, FailureRecord
from merge.infer import InferenceConfig, load_validation_jsonl


# ---------------------------------------------------------------------------
# load_validation_jsonl: field normalization
# ---------------------------------------------------------------------------

def test_load_validation_jsonl_accepts_prompt_answer(tmp_path: Path) -> None:
    """The shipped validation samples use ``prompt`` / ``answer`` field names."""
    jsonl = tmp_path / "v.jsonl"
    jsonl.write_text(
        json.dumps({"prompt": "What is 2+2?", "answer": "4"}) + "\n"
        + json.dumps({"prompt": "What is 3*3?", "answer": "9"}) + "\n"
    )
    items = load_validation_jsonl(jsonl)
    assert len(items) == 2
    assert items[0] == {"problem_id": 0, "prompt": "What is 2+2?", "answer": "4"}
    assert items[1] == {"problem_id": 1, "prompt": "What is 3*3?", "answer": "9"}


def test_load_validation_jsonl_accepts_problem_solution(tmp_path: Path) -> None:
    """Alternate field names ``problem`` / ``solution`` are also accepted."""
    jsonl = tmp_path / "v.jsonl"
    jsonl.write_text(json.dumps({"problem": "Q?", "solution": "A"}) + "\n")
    items = load_validation_jsonl(jsonl)
    assert items[0]["prompt"] == "Q?"
    assert items[0]["answer"] == "A"


def test_load_validation_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    jsonl = tmp_path / "v.jsonl"
    jsonl.write_text(
        json.dumps({"prompt": "Q1", "answer": "A1"}) + "\n"
        + "\n"
        + json.dumps({"prompt": "Q2", "answer": "A2"}) + "\n"
    )
    items = load_validation_jsonl(jsonl)
    assert len(items) == 2


def test_load_validation_jsonl_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_validation_jsonl(tmp_path / "does_not_exist.jsonl")


def test_load_validation_jsonl_missing_fields(tmp_path: Path) -> None:
    jsonl = tmp_path / "v.jsonl"
    jsonl.write_text(json.dumps({"foo": "bar"}) + "\n")
    with pytest.raises(ValueError, match=r"prompt"):
        load_validation_jsonl(jsonl)


def test_load_validation_jsonl_invalid_json(tmp_path: Path) -> None:
    jsonl = tmp_path / "v.jsonl"
    jsonl.write_text("{not valid json\n")
    with pytest.raises(ValueError, match=r"invalid JSON"):
        load_validation_jsonl(jsonl)


# ---------------------------------------------------------------------------
# Dataclass defaults + serialization
# ---------------------------------------------------------------------------

def test_inference_config_defaults() -> None:
    """Defaults match the locked-in spec (Qwen3 generation_config flavor)."""
    cfg = InferenceConfig()
    assert cfg.n == 8
    assert cfg.temperature == 0.7
    assert cfg.top_p == 0.8
    assert cfg.top_k == 20
    assert cfg.max_tokens == 2048
    assert cfg.seed == 42


def test_inference_config_overrides() -> None:
    cfg = InferenceConfig(n=4, max_tokens=1024)
    assert cfg.n == 4
    assert cfg.max_tokens == 1024
    # Other fields untouched.
    assert cfg.temperature == 0.7


def test_benchmark_result_dataclass_round_trip() -> None:
    """BenchmarkResult survives json.dumps/loads via asdict."""
    r = BenchmarkResult(
        benchmark="math",
        pass_at_1=0.5,
        pass_at_8=0.8,
        n_problems=10,
        n_pass8_failed=2,
    )
    serialized = json.dumps(asdict(r))
    round_tripped = json.loads(serialized)
    assert round_tripped == {
        "benchmark": "math",
        "pass_at_1": 0.5,
        "pass_at_8": 0.8,
        "n_problems": 10,
        "n_pass8_failed": 2,
    }


def test_failure_record_serializable() -> None:
    """FailureRecord with FailureCategory.value strings survives JSON round-trip."""
    rec = FailureRecord(
        problem_id=3,
        problem="Find x such that x+1=43.",
        expected="42",
        primary_category=FailureCategory.WRONG_ANSWER.value,
        per_completion_categories=[
            FailureCategory.WRONG_ANSWER.value,
            FailureCategory.NO_BOXED.value,
        ],
        completions=["wrong 1", "wrong 2"],
    )
    serialized = json.dumps(asdict(rec))
    parsed = json.loads(serialized)
    assert parsed["primary_category"] == "wrong_answer"
    assert parsed["per_completion_categories"] == ["wrong_answer", "no_boxed"]
    assert parsed["completions"] == ["wrong 1", "wrong 2"]


def test_failure_category_enum_values_stable() -> None:
    """The enum values become string keys in the failures JSON; pin them."""
    expected = {
        "no_boxed", "empty_boxed", "wrong_answer", "malformed_answer",
        "truncated", "refusal", "mixed",
    }
    assert {c.value for c in FailureCategory} == expected


def test_load_validation_jsonl_on_real_validation_sample() -> None:
    """The shipped validation_samples/math.jsonl parses without changes."""
    repo_root = Path(__file__).resolve().parents[2]
    math_jsonl = repo_root / "validation_samples" / "math.jsonl"
    if not math_jsonl.exists():
        pytest.skip(f"{math_jsonl} not present (running outside the repo?)")
    items = load_validation_jsonl(math_jsonl)
    assert items, "math.jsonl should contain at least one problem"
    for item in items:
        assert "prompt" in item and item["prompt"]
        assert "answer" in item and item["answer"]

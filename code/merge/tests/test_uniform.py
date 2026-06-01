"""
Tests for ``merge.methods.uniform`` — Stage 3 uniform merge.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _make_tiny_tv(values: dict[str, list], dtype=None):
    import torch
    if dtype is None:
        dtype = torch.bfloat16
    return {k: torch.tensor(v, dtype=dtype) for k, v in values.items()}


# ---------------------------------------------------------------------------
# Identity-ish cases
# ---------------------------------------------------------------------------

def test_uniform_of_one_returns_input() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.uniform import uniform_merge

    tv = _make_tiny_tv({"a": [1.0, 2.0, 3.0]})
    out = uniform_merge([tv])
    torch.testing.assert_close(out["a"], tv["a"], rtol=1e-2, atol=1e-2)


def test_uniform_of_two_identical_returns_input() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.uniform import uniform_merge

    tv = _make_tiny_tv({"a": [1.0, 2.0, 3.0]})
    out = uniform_merge([tv, tv])
    torch.testing.assert_close(out["a"], tv["a"], rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# 4-adapter happy path
# ---------------------------------------------------------------------------

def test_uniform_of_four_distinct_toy_adapters_succeeds(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    from merge.load_adapter import load_all
    from merge.methods.uniform import uniform_merge
    from merge.verify_spec import load_locked_spec

    spec = load_locked_spec(lora_yaml_path)
    adapters = load_all(synthetic_adapters_dir, spec)
    out = uniform_merge(list(adapters.values()))

    one = next(iter(adapters.values()))
    assert set(out.keys()) == set(one.keys())
    for key in one:
        assert out[key].shape == one[key].shape
        assert not torch.isnan(out[key]).any(), f"NaN in {key}"


# ---------------------------------------------------------------------------
# Arithmetic correctness
# ---------------------------------------------------------------------------

def test_uniform_arithmetic_correctness() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.uniform import uniform_merge

    tv1 = _make_tiny_tv({"x": [2.0]})
    tv2 = _make_tiny_tv({"x": [4.0]})
    out = uniform_merge([tv1, tv2])
    torch.testing.assert_close(
        out["x"], torch.tensor([3.0], dtype=torch.bfloat16), rtol=1e-2, atol=1e-2
    )


def test_uniform_preserves_keys_in_first_tv_order() -> None:
    pytest.importorskip("torch")
    from merge.methods.uniform import uniform_merge

    tv1 = _make_tiny_tv({"a": [1.0], "b": [1.0], "c": [1.0]})
    tv2 = _make_tiny_tv({"b": [2.0], "a": [2.0], "c": [2.0]})
    out = uniform_merge([tv1, tv2])
    assert list(out.keys()) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_uniform_empty_list_raises() -> None:
    pytest.importorskip("torch")
    from merge.methods.uniform import uniform_merge

    with pytest.raises(ValueError, match=r"at least one"):
        uniform_merge([])


def test_uniform_mismatched_keys_raises() -> None:
    pytest.importorskip("torch")
    from merge.methods.uniform import uniform_merge

    tv1 = _make_tiny_tv({"a": [1.0]})
    tv2 = _make_tiny_tv({"b": [1.0]})
    with pytest.raises(ValueError, match=r"key set"):
        uniform_merge([tv1, tv2])


def test_uniform_mismatched_shapes_raises() -> None:
    pytest.importorskip("torch")
    from merge.methods.uniform import uniform_merge

    tv1 = _make_tiny_tv({"a": [1.0, 2.0]})
    tv2 = _make_tiny_tv({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match=r"shape mismatch"):
        uniform_merge([tv1, tv2])


# ---------------------------------------------------------------------------
# Structural / dtype / mutation
# ---------------------------------------------------------------------------

def test_uniform_preserves_dtype() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.uniform import uniform_merge

    tv = _make_tiny_tv({"a": [1.0, 2.0, 3.0]})  # bf16
    out = uniform_merge([tv, tv])
    assert out["a"].dtype == torch.bfloat16


def test_uniform_does_not_modify_inputs() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.uniform import uniform_merge

    tv1 = _make_tiny_tv({"a": [1.0, 2.0, 3.0]})
    tv2 = _make_tiny_tv({"a": [4.0, 5.0, 6.0]})
    s1 = {k: v.clone() for k, v in tv1.items()}
    s2 = {k: v.clone() for k, v in tv2.items()}
    uniform_merge([tv1, tv2])
    for k in tv1:
        assert torch.equal(tv1[k], s1[k]), f"tv1[{k}] mutated"
        assert torch.equal(tv2[k], s2[k]), f"tv2[{k}] mutated"

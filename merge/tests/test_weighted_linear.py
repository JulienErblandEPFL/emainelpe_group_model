"""
Tests for ``merge.methods.weighted_linear`` — Stage 3 weighted linear merge.

Cross-validation: uniform weights through weighted_linear_merge must match
uniform_merge. Pass-through behavior: weights are NOT normalized.
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
# Cross-validation against uniform_merge
# ---------------------------------------------------------------------------

def test_weighted_linear_with_uniform_weights_matches_uniform_merge() -> None:
    """[0.25]*4 weighted_linear must equal uniform_merge of the same 4 tvs."""
    torch = pytest.importorskip("torch")
    from merge.methods.uniform import uniform_merge
    from merge.methods.weighted_linear import weighted_linear_merge

    tvs = [
        _make_tiny_tv({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}),
        _make_tiny_tv({"a": [2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0]}),
        _make_tiny_tv({"a": [3.0, 4.0, 5.0], "b": [6.0, 7.0, 8.0]}),
        _make_tiny_tv({"a": [4.0, 5.0, 6.0], "b": [7.0, 8.0, 9.0]}),
    ]
    a = uniform_merge(tvs)
    b = weighted_linear_merge(tvs, [0.25, 0.25, 0.25, 0.25])
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Arithmetic correctness + pass-through normalization
# ---------------------------------------------------------------------------

def test_weighted_linear_arithmetic_correctness() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.weighted_linear import weighted_linear_merge

    tv1 = _make_tiny_tv({"x": [2.0]})
    tv2 = _make_tiny_tv({"x": [4.0]})
    out = weighted_linear_merge([tv1, tv2], [0.25, 0.75])
    torch.testing.assert_close(
        out["x"], torch.tensor([3.5], dtype=torch.bfloat16), rtol=1e-2, atol=1e-2
    )


def test_weighted_linear_passthrough_no_normalization() -> None:
    """Weights [2.0, 3.0] on identical-1.0 inputs → 5.0, not 1.0."""
    torch = pytest.importorskip("torch")
    from merge.methods.weighted_linear import weighted_linear_merge

    tv1 = _make_tiny_tv({"x": [1.0]})
    tv2 = _make_tiny_tv({"x": [1.0]})
    out = weighted_linear_merge([tv1, tv2], [2.0, 3.0])
    torch.testing.assert_close(
        out["x"], torch.tensor([5.0], dtype=torch.bfloat16), rtol=1e-2, atol=1e-2
    )


def test_weighted_linear_negative_weights_allowed() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.weighted_linear import weighted_linear_merge

    tv1 = _make_tiny_tv({"x": [1.0]})
    tv2 = _make_tiny_tv({"x": [2.0]})
    out = weighted_linear_merge([tv1, tv2], [1.0, -1.0])
    torch.testing.assert_close(
        out["x"], torch.tensor([-1.0], dtype=torch.bfloat16), rtol=1e-2, atol=1e-2
    )


def test_weighted_linear_zero_weight_drops_input() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.weighted_linear import weighted_linear_merge

    tv1 = _make_tiny_tv({"x": [1.0]})
    tv2 = _make_tiny_tv({"x": [10.0]})
    out = weighted_linear_merge([tv1, tv2], [1.0, 0.0])
    torch.testing.assert_close(
        out["x"], torch.tensor([1.0], dtype=torch.bfloat16), rtol=1e-2, atol=1e-2
    )


# ---------------------------------------------------------------------------
# 4-adapter happy path
# ---------------------------------------------------------------------------

def test_weighted_linear_with_synthetic_adapters(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    from merge.load_adapter import load_all
    from merge.methods.weighted_linear import weighted_linear_merge
    from merge.verify_spec import load_locked_spec

    spec = load_locked_spec(lora_yaml_path)
    adapters = load_all(synthetic_adapters_dir, spec)
    out = weighted_linear_merge(list(adapters.values()), [0.4, 0.3, 0.2, 0.1])

    one = next(iter(adapters.values()))
    assert set(out.keys()) == set(one.keys())
    for key in one:
        assert out[key].shape == one[key].shape
        assert out[key].dtype == torch.bfloat16
        assert not torch.isnan(out[key]).any(), f"NaN in {key}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_weighted_linear_empty_list_raises() -> None:
    pytest.importorskip("torch")
    from merge.methods.weighted_linear import weighted_linear_merge

    with pytest.raises(ValueError, match=r"at least one"):
        weighted_linear_merge([], [])


def test_weighted_linear_weights_length_mismatch_raises() -> None:
    pytest.importorskip("torch")
    from merge.methods.weighted_linear import weighted_linear_merge

    tvs = [_make_tiny_tv({"a": [1.0]}) for _ in range(4)]
    with pytest.raises(ValueError, match=r"weights length"):
        weighted_linear_merge(tvs, [0.25, 0.25, 0.25])


def test_weighted_linear_mismatched_keys_raises() -> None:
    pytest.importorskip("torch")
    from merge.methods.weighted_linear import weighted_linear_merge

    tv1 = _make_tiny_tv({"a": [1.0]})
    tv2 = _make_tiny_tv({"b": [1.0]})
    with pytest.raises(ValueError, match=r"key set"):
        weighted_linear_merge([tv1, tv2], [0.5, 0.5])


def test_weighted_linear_preserves_dtype() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.weighted_linear import weighted_linear_merge

    tv = _make_tiny_tv({"a": [1.0, 2.0, 3.0]})
    out = weighted_linear_merge([tv, tv], [0.5, 0.5])
    assert out["a"].dtype == torch.bfloat16


def test_weighted_linear_does_not_modify_inputs() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.weighted_linear import weighted_linear_merge

    tv1 = _make_tiny_tv({"a": [1.0, 2.0, 3.0]})
    tv2 = _make_tiny_tv({"a": [4.0, 5.0, 6.0]})
    s1 = {k: v.clone() for k, v in tv1.items()}
    s2 = {k: v.clone() for k, v in tv2.items()}
    weighted_linear_merge([tv1, tv2], [0.7, 0.3])
    for k in tv1:
        assert torch.equal(tv1[k], s1[k]), f"tv1[{k}] mutated"
        assert torch.equal(tv2[k], s2[k]), f"tv2[{k}] mutated"

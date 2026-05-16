"""
Tests for ``merge.methods.ties`` — Stage 4 TIES merging.

The hand-computed correctness test (``test_ties_arithmetic_correctness_simple``)
is the math-verification gate. If it fails, the algorithm is wrong somewhere.
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
# Hand-computed correctness — math gate
# ---------------------------------------------------------------------------

def test_ties_arithmetic_correctness_simple() -> None:
    """Three task vectors of length 4; trim_ratio=0.5; expected output computed
    by hand in the spec. If this fails, the TIES math is wrong."""
    torch = pytest.importorskip("torch")
    from merge.methods.ties import ties_merge

    tv1 = _make_tiny_tv({"a": [1.0, 2.0, -3.0, 0.5]})
    tv2 = _make_tiny_tv({"a": [1.5, -1.0, -2.0, 0.0]})
    tv3 = _make_tiny_tv({"a": [0.8, 2.5, 1.0, -0.5]})

    # After trim (top 2 of 4 by |x|):
    #   tv1: [0, 2.0, -3.0, 0]
    #   tv2: [1.5, 0, -2.0, 0]
    #   tv3: [0, 2.5, 1.0, 0]
    # Elected signs: [+1, +1, -1, 0]
    # Disjoint merge:
    #   pos 0: only tv2 contributes (+1) → 1.5
    #   pos 1: tv1 + tv3 (both +1) → (2.0 + 2.5) / 2 = 2.25
    #   pos 2: tv1 + tv2 (both -1); tv3's +1.0 excluded → (-3.0 + -2.0) / 2 = -2.5
    #   pos 3: elected zero → 0.0
    expected = torch.tensor([1.5, 2.25, -2.5, 0.0], dtype=torch.bfloat16)

    out = ties_merge([tv1, tv2, tv3], trim_ratio=0.5)
    torch.testing.assert_close(out["a"], expected, rtol=1e-2, atol=1e-2)


def test_ties_no_trim_returns_disjoint_merge_of_full_tvs() -> None:
    """trim_ratio=0.0 → no trimming → just elect-sign + disjoint-merge."""
    torch = pytest.importorskip("torch")
    from merge.methods.ties import ties_merge

    # All entries kept; all positive in tv1, mixed in tv2.
    tv1 = _make_tiny_tv({"a": [1.0, 2.0, 3.0]})
    tv2 = _make_tiny_tv({"a": [4.0, -5.0, 6.0]})
    # Trimmed = inputs (no trim).
    # Signs: [+1, +1, +1] and [+1, -1, +1]. Sums: [+2, 0, +2]. Elected: [+1, 0, +1].
    # Disjoint merge:
    #   pos 0: both +1 → (1+4)/2 = 2.5
    #   pos 1: elected 0 → 0.0
    #   pos 2: both +1 → (3+6)/2 = 4.5
    expected = torch.tensor([2.5, 0.0, 4.5], dtype=torch.bfloat16)
    out = ties_merge([tv1, tv2], trim_ratio=0.0)
    torch.testing.assert_close(out["a"], expected, rtol=1e-2, atol=1e-2)


def test_ties_trim_ratio_extremes() -> None:
    """trim_ratio close to 1 → zeros most entries → output mostly zero."""
    torch = pytest.importorskip("torch")
    from merge.methods.ties import ties_merge

    tv1 = _make_tiny_tv({"a": [1.0, 2.0, 3.0]})
    tv2 = _make_tiny_tv({"a": [4.0, 5.0, 6.0]})
    # trim_ratio=0.99 → k = int(3*0.99) = 2 → keep top (3-2)=1 entry per tv.
    # tv1 top-1 by |x|: 3.0 → [0, 0, 3.0]
    # tv2 top-1 by |x|: 6.0 → [0, 0, 6.0]
    # Signs: [0, 0, +1] and [0, 0, +1]. Elected: [0, 0, +1].
    # Disjoint merge:
    #   pos 0, 1: elected 0 → 0
    #   pos 2: both +1 → (3+6)/2 = 4.5
    out = ties_merge([tv1, tv2], trim_ratio=0.99)
    expected = torch.tensor([0.0, 0.0, 4.5], dtype=torch.bfloat16)
    torch.testing.assert_close(out["a"], expected, rtol=1e-2, atol=1e-2)


def test_ties_zero_tie_drops_parameter() -> None:
    """Exact sign ties → elect zero → parameter dropped."""
    torch = pytest.importorskip("torch")
    from merge.methods.ties import ties_merge

    tv1 = _make_tiny_tv({"a": [1.0, 2.0]})
    tv2 = _make_tiny_tv({"a": [-1.0, -2.0]})
    # No trim (trim_ratio=0.0). Signs per pos: [+1,-1] → sum 0 → elected 0.
    out = ties_merge([tv1, tv2], trim_ratio=0.0)
    expected = torch.zeros(2, dtype=torch.bfloat16)
    torch.testing.assert_close(out["a"], expected, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Structural / dtype / mutation
# ---------------------------------------------------------------------------

def test_ties_preserves_keys_shapes_dtype() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.ties import ties_merge

    tv1 = _make_tiny_tv({"a": [1.0, 2.0, 3.0, 4.0], "b": [-1.0, -2.0, -3.0, -4.0]})
    tv2 = _make_tiny_tv({"a": [4.0, 3.0, 2.0, 1.0], "b": [-4.0, -3.0, -2.0, -1.0]})
    out = ties_merge([tv1, tv2], trim_ratio=0.5)
    assert list(out.keys()) == list(tv1.keys())
    for key in tv1:
        assert out[key].shape == tv1[key].shape
        assert out[key].dtype == torch.bfloat16


def test_ties_does_not_modify_inputs() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.ties import ties_merge

    tv1 = _make_tiny_tv({"a": [1.0, 2.0, 3.0, 4.0]})
    tv2 = _make_tiny_tv({"a": [4.0, -3.0, 2.0, -1.0]})
    s1 = {k: v.clone() for k, v in tv1.items()}
    s2 = {k: v.clone() for k, v in tv2.items()}
    ties_merge([tv1, tv2], trim_ratio=0.5)
    for k in tv1:
        assert torch.equal(tv1[k], s1[k]), f"tv1[{k}] mutated"
        assert torch.equal(tv2[k], s2[k]), f"tv2[{k}] mutated"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_ties_empty_list_raises() -> None:
    pytest.importorskip("torch")
    from merge.methods.ties import ties_merge

    with pytest.raises(ValueError, match=r"at least one"):
        ties_merge([], trim_ratio=0.5)


def test_ties_mismatched_keys_raises() -> None:
    pytest.importorskip("torch")
    from merge.methods.ties import ties_merge

    tv1 = _make_tiny_tv({"a": [1.0]})
    tv2 = _make_tiny_tv({"b": [1.0]})
    with pytest.raises(ValueError, match=r"key set"):
        ties_merge([tv1, tv2], trim_ratio=0.5)


def test_ties_mismatched_shapes_raises() -> None:
    pytest.importorskip("torch")
    from merge.methods.ties import ties_merge

    tv1 = _make_tiny_tv({"a": [1.0, 2.0]})
    tv2 = _make_tiny_tv({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match=r"shape mismatch"):
        ties_merge([tv1, tv2], trim_ratio=0.5)


def test_ties_invalid_trim_ratio_raises() -> None:
    pytest.importorskip("torch")
    from merge.methods.ties import ties_merge

    tv = _make_tiny_tv({"a": [1.0, 2.0]})
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match=r"trim_ratio"):
            ties_merge([tv], trim_ratio=bad)


# ---------------------------------------------------------------------------
# 4-adapter happy path
# ---------------------------------------------------------------------------

def test_ties_with_four_synthetic_adapters_succeeds(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    from merge.load_adapter import load_all
    from merge.methods.ties import ties_merge
    from merge.verify_spec import load_locked_spec

    spec = load_locked_spec(lora_yaml_path)
    adapters = load_all(synthetic_adapters_dir, spec)
    out = ties_merge(list(adapters.values()), trim_ratio=0.5)

    one = next(iter(adapters.values()))
    assert set(out.keys()) == set(one.keys())
    for key in one:
        assert out[key].shape == one[key].shape
        assert out[key].dtype == torch.bfloat16
        assert not torch.isnan(out[key]).any(), f"NaN in {key}"

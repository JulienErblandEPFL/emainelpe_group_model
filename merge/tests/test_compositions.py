"""
Tests for ``merge.methods.{dare_uniform, dare_weighted}`` compositions
and the ``METHOD_REGISTRY`` dispatch table.

Cross-validation tests are the strongest sanity check that DARE composes
correctly with uniform/weighted_linear: drop_rate=0.0 + rescale=True must
reduce to the corresponding non-DARE merge.
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
# dare_uniform
# ---------------------------------------------------------------------------

def test_dare_uniform_on_four_toy_adapters_succeeds(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    from merge.load_adapter import load_all
    from merge.methods import dare_uniform
    from merge.verify_spec import load_locked_spec

    spec = load_locked_spec(lora_yaml_path)
    adapters = load_all(synthetic_adapters_dir, spec)
    out = dare_uniform(list(adapters.values()), drop_rate=0.5, seed=42)

    one = next(iter(adapters.values()))
    assert set(out.keys()) == set(one.keys())
    for key in one:
        assert out[key].shape == one[key].shape
        assert out[key].dtype == torch.bfloat16
        assert not torch.isnan(out[key]).any(), f"NaN in {key}"


def test_dare_uniform_reproducible_with_seed(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    from merge.load_adapter import load_all
    from merge.methods import dare_uniform
    from merge.verify_spec import load_locked_spec

    spec = load_locked_spec(lora_yaml_path)
    # Load tvs fresh each call. dare_uniform applies DARE in-place to
    # save GPU memory (see merge/methods/__init__.py), so re-using the
    # same dict object would feed already-masked inputs into the second
    # call. The pipeline always passes fresh load_all outputs to the
    # merge_fn — this test matches that flow.
    tvs_a = list(load_all(synthetic_adapters_dir, spec).values())
    a = dare_uniform(tvs_a, drop_rate=0.5, seed=42)
    tvs_b = list(load_all(synthetic_adapters_dir, spec).values())
    b = dare_uniform(tvs_b, drop_rate=0.5, seed=42)
    for key in a:
        assert torch.equal(a[key], b[key]), f"{key} differs across seed=42 calls"


def test_dare_uniform_drop_rate_zero_matches_uniform_merge() -> None:
    """drop_rate=0.0 + rescale=True must be identity → matches uniform_merge."""
    torch = pytest.importorskip("torch")
    from merge.methods import dare_uniform
    from merge.methods.uniform import uniform_merge

    tvs = [
        _make_tiny_tv({"a": [1.0, 2.0, 3.0]}),
        _make_tiny_tv({"a": [4.0, 5.0, 6.0]}),
        _make_tiny_tv({"a": [7.0, 8.0, 9.0]}),
        _make_tiny_tv({"a": [10.0, 11.0, 12.0]}),
    ]
    expected = uniform_merge(tvs)
    got = dare_uniform(tvs, drop_rate=0.0, rescale=True, seed=0)
    for key in expected:
        torch.testing.assert_close(got[key], expected[key], rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# dare_weighted
# ---------------------------------------------------------------------------

def test_dare_weighted_on_four_toy_adapters_succeeds(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    from merge.load_adapter import load_all
    from merge.methods import dare_weighted
    from merge.verify_spec import load_locked_spec

    spec = load_locked_spec(lora_yaml_path)
    adapters = load_all(synthetic_adapters_dir, spec)
    out = dare_weighted(
        list(adapters.values()), [0.4, 0.3, 0.2, 0.1], drop_rate=0.5, seed=7
    )

    one = next(iter(adapters.values()))
    assert set(out.keys()) == set(one.keys())
    for key in one:
        assert out[key].shape == one[key].shape
        assert out[key].dtype == torch.bfloat16
        assert not torch.isnan(out[key]).any(), f"NaN in {key}"


def test_dare_weighted_reproducible_with_seed(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    torch = pytest.importorskip("torch")
    from merge.load_adapter import load_all
    from merge.methods import dare_weighted
    from merge.verify_spec import load_locked_spec

    spec = load_locked_spec(lora_yaml_path)
    # See dare_uniform reproducibility test: fresh load per call because
    # dare_weighted applies DARE in place.
    tvs_a = list(load_all(synthetic_adapters_dir, spec).values())
    a = dare_weighted(tvs_a, [0.4, 0.3, 0.2, 0.1], drop_rate=0.5, seed=7)
    tvs_b = list(load_all(synthetic_adapters_dir, spec).values())
    b = dare_weighted(tvs_b, [0.4, 0.3, 0.2, 0.1], drop_rate=0.5, seed=7)
    for key in a:
        assert torch.equal(a[key], b[key]), f"{key} differs across seed=7 calls"


def test_dare_weighted_drop_rate_zero_matches_weighted_linear() -> None:
    """drop_rate=0.0 + rescale=True must reduce to weighted_linear_merge."""
    torch = pytest.importorskip("torch")
    from merge.methods import dare_weighted
    from merge.methods.weighted_linear import weighted_linear_merge

    tvs = [
        _make_tiny_tv({"a": [1.0, 2.0]}),
        _make_tiny_tv({"a": [3.0, 4.0]}),
        _make_tiny_tv({"a": [5.0, 6.0]}),
    ]
    weights = [0.5, 0.3, 0.2]
    expected = weighted_linear_merge(tvs, weights)
    got = dare_weighted(tvs, weights, drop_rate=0.0, rescale=True, seed=0)
    for key in expected:
        torch.testing.assert_close(got[key], expected[key], rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# METHOD_REGISTRY dispatch
# ---------------------------------------------------------------------------

def test_method_registry_dispatches_correctly() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods import METHOD_REGISTRY

    tvs = [
        _make_tiny_tv({"a": [1.0, 2.0]}),
        _make_tiny_tv({"a": [3.0, 4.0]}),
    ]

    # uniform
    out = METHOD_REGISTRY["uniform"](tvs)
    assert isinstance(out, dict)
    assert "a" in out

    # dare_uniform
    out = METHOD_REGISTRY["dare_uniform"](tvs, drop_rate=0.0, rescale=True, seed=0)
    assert isinstance(out, dict)
    assert "a" in out

    # dare_weighted
    out = METHOD_REGISTRY["dare_weighted"](tvs, [0.5, 0.5], drop_rate=0.0, rescale=True, seed=0)
    assert isinstance(out, dict)
    assert "a" in out

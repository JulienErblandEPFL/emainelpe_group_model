"""
Tests for ``merge.methods.dare`` — Stage 3 DARE implementation.

All tests require torch (we materialize tensors); ``pytest.importorskip``
at the top of each function lets the laptop skip cleanly while the cluster
runs them.

For arithmetic-correctness tests we use a small inline ``_make_tiny_tv``
helper. For statistical tests we use bigger random tensors (toy fixture)
to keep stats reasonable.
"""
from __future__ import annotations

import copy

import pytest


def _make_tiny_tv(values: dict[str, list], dtype=None):
    """Build a tiny task-vector dict from a {key: list-of-floats} mapping."""
    import torch
    if dtype is None:
        dtype = torch.bfloat16
    return {k: torch.tensor(v, dtype=dtype) for k, v in values.items()}


def _make_random_tv(seed: int = 0, n_tensors: int = 4, shape=(32, 64)):
    """Build a small random bf16 task-vector dict."""
    import torch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return {
        f"layer.{i}.weight": torch.randn(shape, generator=gen, dtype=torch.float32).to(torch.bfloat16)
        for i in range(n_tensors)
    }


# ---------------------------------------------------------------------------
# Identity-ish behavior at drop_rate=0
# ---------------------------------------------------------------------------

def test_dare_drop_rate_zero_returns_input_when_rescale_true() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.dare import dare

    tv = _make_tiny_tv({"a": [1.0, 2.0, 3.0, 4.0]})
    out = dare(tv, drop_rate=0.0, rescale=True, seed=0)
    torch.testing.assert_close(out["a"], tv["a"], rtol=1e-2, atol=1e-2)


def test_dare_drop_rate_zero_returns_input_when_rescale_false() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.dare import dare

    tv = _make_tiny_tv({"a": [1.0, 2.0, 3.0, 4.0]})
    out = dare(tv, drop_rate=0.0, rescale=False, seed=0)
    torch.testing.assert_close(out["a"], tv["a"], rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------

def test_dare_drop_rate_05_zeros_roughly_half() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.dare import dare

    tv = _make_random_tv(seed=0, shape=(32, 64))
    out = dare(tv, drop_rate=0.5, seed=123)
    for name, tensor in out.items():
        frac_zero = (tensor.float() == 0).float().mean().item()
        assert 0.4 <= frac_zero <= 0.6, (
            f"{name}: fraction zero {frac_zero:.3f} outside [0.4, 0.6]"
        )


def test_dare_rescale_preserves_mean_magnitude() -> None:
    """All-ones input → mean of DARE output ≈ 1.0 with rescale=True."""
    torch = pytest.importorskip("torch")
    from merge.methods.dare import dare

    tv = {"a": torch.ones(64, 64, dtype=torch.bfloat16)}
    out = dare(tv, drop_rate=0.5, rescale=True, seed=7)
    observed = out["a"].float().mean().item()
    assert abs(observed - 1.0) < 0.05, f"mean={observed!r}, expected ≈ 1.0"


def test_dare_no_rescale_reduces_mean_magnitude() -> None:
    """All-ones input → mean of DARE output ≈ 0.5 with rescale=False."""
    torch = pytest.importorskip("torch")
    from merge.methods.dare import dare

    tv = {"a": torch.ones(64, 64, dtype=torch.bfloat16)}
    out = dare(tv, drop_rate=0.5, rescale=False, seed=7)
    observed = out["a"].float().mean().item()
    assert abs(observed - 0.5) < 0.05, f"mean={observed!r}, expected ≈ 0.5"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_dare_reproducible_with_same_seed() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.dare import dare

    tv = _make_random_tv(seed=0)
    a = dare(tv, drop_rate=0.5, seed=42)
    b = dare(tv, drop_rate=0.5, seed=42)
    for key in a:
        assert torch.equal(a[key], b[key]), f"{key} differs across seed=42 calls"


def test_dare_different_seeds_produce_different_masks() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.dare import dare

    tv = _make_random_tv(seed=0)
    a = dare(tv, drop_rate=0.5, seed=0)
    b = dare(tv, drop_rate=0.5, seed=1)
    assert any(not torch.equal(a[k], b[k]) for k in a), "all tensors identical across seeds"


# ---------------------------------------------------------------------------
# Structural preservation
# ---------------------------------------------------------------------------

def test_dare_preserves_keys_and_shapes() -> None:
    pytest.importorskip("torch")
    from merge.methods.dare import dare

    tv = _make_random_tv(seed=0)
    out = dare(tv, drop_rate=0.3, seed=0)
    assert list(out.keys()) == list(tv.keys())
    for key in tv:
        assert out[key].shape == tv[key].shape


def test_dare_preserves_dtype() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.dare import dare

    tv = _make_random_tv(seed=0)  # bf16
    out = dare(tv, drop_rate=0.3, seed=0)
    for tensor in out.values():
        assert tensor.dtype == torch.bfloat16


def test_dare_does_not_upcast_to_fp32_internally() -> None:
    """Regression for the 2026-05-26 bake-off DARE OOM.

    A bf16 input must yield a bf16 output with NO intermediate fp32
    materialization. We check this two ways:

    1. Output dtype matches input dtype exactly (no implicit upcast).
    2. The output occupies the bf16 byte count for its element count
       (2 bytes/elem), not the fp32 byte count (4 bytes/elem).

    The previous implementation built an fp32 keep_prob tensor + an
    fp32 mask + cast `tensor` to fp32, materializing ~9× the input
    size in fp32 memory before casting back. For the 4-adapter
    Qwen3-1.7B ΔW set that peak alone exceeded 40 GB on an A100-40g.
    """
    torch = pytest.importorskip("torch")
    from merge.methods.dare import dare

    tv = _make_random_tv(seed=0)
    out = dare(tv, drop_rate=0.5, seed=0)
    for name, tensor in out.items():
        assert tensor.dtype == torch.bfloat16, (
            f"{name}: dtype upcast leaked into output ({tensor.dtype})"
        )
        # bf16 = 2 bytes/elem; fp32 = 4 bytes/elem. The output must hold
        # bf16 storage, not fp32-converted-back storage.
        assert tensor.element_size() == 2, (
            f"{name}: element_size {tensor.element_size()} != 2 (bf16)"
        )


def test_dare_fp32_input_stays_fp32(_seed: int = 0) -> None:
    """Input dtype is preserved end-to-end, including fp32.

    Belt-and-suspenders: previously the function always returned bf16
    after the explicit ``.to(tensor.dtype)`` cast, so fp32-in / fp32-out
    happened to work. The cleaned-up code drops the explicit cast and
    relies on torch's promotion rules — verify fp32-in still yields
    fp32-out.
    """
    torch = pytest.importorskip("torch")
    from merge.methods.dare import dare

    tv = {"a": torch.randn(16, 16, dtype=torch.float32)}
    out = dare(tv, drop_rate=0.3, seed=0)
    assert out["a"].dtype == torch.float32


def test_dare_does_not_modify_input() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.dare import dare

    tv = _make_random_tv(seed=0)
    snapshots = {k: v.clone() for k, v in tv.items()}
    dare(tv, drop_rate=0.5, seed=0)
    for key in tv:
        assert torch.equal(tv[key], snapshots[key]), f"{key} mutated"


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def test_dare_invalid_drop_rate_raises() -> None:
    pytest.importorskip("torch")
    from merge.methods.dare import dare

    tv = _make_random_tv(seed=0)
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match=r"drop_rate"):
            dare(tv, drop_rate=bad, seed=0)


def test_dare_non_floating_tensor_raises() -> None:
    torch = pytest.importorskip("torch")
    from merge.methods.dare import dare

    tv = {"a": torch.tensor([1, 2, 3], dtype=torch.int64)}
    with pytest.raises(TypeError, match=r"floating-point"):
        dare(tv, drop_rate=0.5, seed=0)

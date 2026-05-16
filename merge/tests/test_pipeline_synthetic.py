"""
Synthetic end-to-end tests for the full merge pipeline.

Exercises 4 toy adapters → ``merge_adapters`` → SVD truncation → save →
reload, then structurally validates the reloaded adapter.

We deliberately do NOT compare reloaded ΔW elementwise to the in-memory
merged ΔW: the pipeline SVD-truncates from rank up to min(N·r, dim) down
to rank r, which mathematically must produce large elementwise differences
for toy adapters whose merged rank exceeds r. The SVD math itself is
verified by ``test_svd_factor_round_trip_within_truncation_tolerance``
in ``test_pipeline.py`` (rank-r input → exact round-trip).

These tests instead validate:
  - keys / shapes / dtype round-trip exactly
  - no NaN, no Inf
  - reloaded tensors are non-trivial (catches silent-zero failures)
  - reloaded ΔW is rank ≤ r (the structural pipeline contract)

If these tests pass on cluster, the milestone-day "plug in real adapters"
exercise is pure execution.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# Locked-spec rank; matches lora.yaml. Used by the rank-check assertion.
_R = 32


def _run_pipeline_and_reload(
    synthetic_adapters_dir: Path,
    locked_spec_path: Path,
    output_dir: Path,
    method: str,
    method_kwargs: dict | None = None,
) -> dict:
    """Run ``merge_adapters`` and return the reloaded task-vector dict."""
    from merge.load_adapter import load
    from merge.pipeline import merge_adapters

    merge_adapters(
        synthetic_adapters_dir,
        method=method,
        output_dir=output_dir,
        locked_spec_path=locked_spec_path,
        method_kwargs=method_kwargs,
    )
    return load(output_dir)


def _assert_structural_round_trip(reloaded: dict, expected_keys: set[str], r: int = _R) -> None:
    """Structural validation of a reloaded merged adapter.

    Asserts: keys match expected; per-key shape, bf16 dtype, finiteness,
    non-triviality, and rank ≤ r (via SVD-tail check).
    """
    import torch

    assert set(reloaded.keys()) == expected_keys, (
        f"reloaded keys diverge from expected: "
        f"missing={expected_keys - reloaded.keys()!r}, "
        f"extra={reloaded.keys() - expected_keys!r}"
    )

    for key, tensor in reloaded.items():
        assert tensor.dtype == torch.bfloat16, (
            f"{key}: dtype {tensor.dtype} != bfloat16"
        )
        assert torch.isfinite(tensor).all(), f"non-finite values in {key}"
        assert tensor.abs().max().item() > 0, (
            f"{key}: reloaded ΔW is all zeros — pipeline silently produced trivial output"
        )

        # Rank check: the pipeline writes lora_B [out, r] and lora_A [r, in],
        # so the reloaded ΔW = (α/r)·B@A is structurally rank ≤ r. The bf16
        # matmul in load() introduces ~percent-level rounding noise that spreads
        # as small full-rank singular values; the tail must still be orders of
        # magnitude smaller than S[0]. 0.05 absorbs bf16 noise but cleanly
        # rejects an actually-full-rank tensor (whose tail would be ≳ 0.5).
        s = torch.linalg.svdvals(tensor.float())
        if s.numel() > r:
            ratio = s[r:].max().item() / max(s[0].item(), 1e-12)
            assert ratio < 0.05, (
                f"{key}: SVD tail S[{r}:].max() / S[0] = {ratio!r} ≥ 0.05; "
                f"reloaded ΔW is not rank ≤ {r}"
            )


def _expected_keys(synthetic_adapters_dir: Path) -> set[str]:
    """Canonical keys produced by load() on one of the toy adapters."""
    from merge.load_adapter import load
    return set(load(synthetic_adapters_dir / "math").keys())


# ---------------------------------------------------------------------------
# Round-trip per method — STRUCTURAL VALIDATION ONLY
# ---------------------------------------------------------------------------
# We don't assert reloaded ≈ in_memory because the pipeline SVD-truncates
# the merged ΔW from rank ≤ min(N*r, dim) down to rank r. For toy
# adapters (hidden=64, intermediate=128, r=32) this drops up to half the
# singular values, producing large elementwise differences even though
# the rank-r approximation is correct. See
# test_svd_factor_round_trip_within_truncation_tolerance for the math
# verification on a known rank-r input.

def test_end_to_end_uniform_round_trip(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    expected = _expected_keys(synthetic_adapters_dir)
    reloaded = _run_pipeline_and_reload(
        synthetic_adapters_dir, lora_yaml_path, tmp_path / "merged",
        method="uniform",
    )
    _assert_structural_round_trip(reloaded, expected)


def test_end_to_end_dare_uniform_round_trip(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    expected = _expected_keys(synthetic_adapters_dir)
    reloaded = _run_pipeline_and_reload(
        synthetic_adapters_dir, lora_yaml_path, tmp_path / "merged",
        method="dare_uniform",
        method_kwargs={"drop_rate": 0.5, "seed": 42},
    )
    _assert_structural_round_trip(reloaded, expected)


def test_end_to_end_ties_round_trip(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    expected = _expected_keys(synthetic_adapters_dir)
    reloaded = _run_pipeline_and_reload(
        synthetic_adapters_dir, lora_yaml_path, tmp_path / "merged",
        method="ties",
        method_kwargs={"trim_ratio": 0.5},
    )
    _assert_structural_round_trip(reloaded, expected)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_end_to_end_reproducibility(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    """Same dare_uniform seed → bit-identical reloaded tensors."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.load_adapter import load
    from merge.pipeline import merge_adapters

    kwargs = dict(
        method="dare_uniform",
        locked_spec_path=lora_yaml_path,
        method_kwargs={"drop_rate": 0.5, "seed": 42},
    )
    merge_adapters(synthetic_adapters_dir, output_dir=tmp_path / "a", **kwargs)
    merge_adapters(synthetic_adapters_dir, output_dir=tmp_path / "b", **kwargs)
    a = load(tmp_path / "a")
    b = load(tmp_path / "b")
    assert set(a.keys()) == set(b.keys())
    for key in a:
        assert torch.equal(a[key], b[key]), f"{key} differs across runs"


# ---------------------------------------------------------------------------
# Output sanity
# ---------------------------------------------------------------------------

def test_end_to_end_adapter_config_correct(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    """Output adapter_config.json must verify against the locked spec."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.pipeline import merge_adapters
    from merge.verify_spec import load_locked_spec, verify

    out = tmp_path / "merged"
    merge_adapters(
        synthetic_adapters_dir,
        method="uniform",
        output_dir=out,
        locked_spec_path=lora_yaml_path,
    )
    spec = load_locked_spec(lora_yaml_path)
    result = verify(out / "adapter_config.json", spec)
    assert result.passed, result.summary


def test_end_to_end_safetensors_uses_peft_naming(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    """Every safetensors key must follow PEFT's
    'base_model.model.model.layers.…lora_{A,B}.default.weight' format."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors import safe_open
    from merge.pipeline import merge_adapters

    out = tmp_path / "merged"
    merge_adapters(
        synthetic_adapters_dir,
        method="uniform",
        output_dir=out,
        locked_spec_path=lora_yaml_path,
    )
    keys: list[str] = []
    with safe_open(str(out / "adapter_model.safetensors"), framework="pt") as f:
        keys = list(f.keys())

    assert keys, "no keys in safetensors output"
    for k in keys:
        assert k.startswith("base_model.model.model.layers."), f"bad prefix in {k!r}"
        assert k.endswith(".lora_A.default.weight") or k.endswith(".lora_B.default.weight"), (
            f"bad suffix in {k!r}"
        )


def test_end_to_end_safetensors_has_correct_shapes(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    """lora_A: [r, in_dim]; lora_B: [out_dim, r]. r=32 from locked_spec."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors import safe_open
    from merge.pipeline import merge_adapters

    out = tmp_path / "merged"
    merge_adapters(
        synthetic_adapters_dir,
        method="uniform",
        output_dir=out,
        locked_spec_path=lora_yaml_path,
    )
    r = 32  # from locked spec

    with safe_open(str(out / "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key.endswith(".lora_A.default.weight"):
                assert tensor.shape[0] == r, (
                    f"lora_A {key!r} first dim {tensor.shape[0]} != r={r}"
                )
            elif key.endswith(".lora_B.default.weight"):
                assert tensor.shape[1] == r, (
                    f"lora_B {key!r} second dim {tensor.shape[1]} != r={r}"
                )

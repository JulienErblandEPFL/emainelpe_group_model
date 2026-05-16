"""
Synthetic end-to-end tests for the full merge pipeline.

The highest-value test surface in the merge subdir: 4 toy adapters →
merge_adapters → SVD truncation → save → reload, with round-trip
correctness asserted within an SVD-truncation tolerance.

Validates that the entire pipeline works on CPU before real Qwen3-1.7B
adapters arrive. If these tests pass on cluster, the milestone-day
"plug in real adapters" exercise is pure execution.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _run_and_reload(
    synthetic_adapters_dir: Path,
    locked_spec_path: Path,
    output_dir: Path,
    method: str,
    method_kwargs: dict | None = None,
) -> tuple[dict, dict]:
    """Run the pipeline, return ``(in_memory_merged, reloaded_merged)`` dicts."""
    from merge.load_adapter import load, load_all
    from merge.methods import METHOD_REGISTRY
    from merge.pipeline import merge_adapters
    from merge.verify_spec import load_locked_spec

    spec = load_locked_spec(locked_spec_path)
    in_memory = METHOD_REGISTRY[method](
        list(load_all(synthetic_adapters_dir, spec).values()),
        **(method_kwargs or {}),
    )
    merge_adapters(
        synthetic_adapters_dir,
        method=method,
        output_dir=output_dir,
        locked_spec_path=locked_spec_path,
        method_kwargs=method_kwargs,
    )
    reloaded = load(output_dir)
    return in_memory, reloaded


def _assert_round_trip(in_memory: dict, reloaded: dict) -> None:
    """SVD-truncation introduces error; tolerance is intentionally loose."""
    import torch

    assert set(reloaded.keys()) == set(in_memory.keys())
    for key in in_memory:
        assert reloaded[key].shape == in_memory[key].shape, (
            f"{key}: shape {reloaded[key].shape} vs {in_memory[key].shape}"
        )
        assert not torch.isnan(reloaded[key]).any(), f"NaN in {key}"
        torch.testing.assert_close(
            reloaded[key], in_memory[key], rtol=0.5, atol=0.5
        )


# ---------------------------------------------------------------------------
# Round-trip per method
# ---------------------------------------------------------------------------

def test_end_to_end_uniform_round_trip(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    in_memory, reloaded = _run_and_reload(
        synthetic_adapters_dir, lora_yaml_path, tmp_path / "merged",
        method="uniform",
    )
    _assert_round_trip(in_memory, reloaded)


def test_end_to_end_dare_uniform_round_trip(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    in_memory, reloaded = _run_and_reload(
        synthetic_adapters_dir, lora_yaml_path, tmp_path / "merged",
        method="dare_uniform",
        method_kwargs={"drop_rate": 0.5, "seed": 42},
    )
    _assert_round_trip(in_memory, reloaded)


def test_end_to_end_ties_round_trip(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    in_memory, reloaded = _run_and_reload(
        synthetic_adapters_dir, lora_yaml_path, tmp_path / "merged",
        method="ties",
        method_kwargs={"trim_ratio": 0.5},
    )
    _assert_round_trip(in_memory, reloaded)


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
    safetensors_module = pytest.importorskip("safetensors")
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

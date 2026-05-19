"""
Tests for AdaMerging (Stage 5a).

All tests use the synthetic ``forward_fn`` + data iterator from
``fixtures/adamerging_helpers.py``. The synthetic forward is NOT a real
transformer; it exists only to give the training loop a differentiable
path from coefficients to a scalar entropy loss so we can verify
convergence + behavior on CPU in milliseconds.

Real-Qwen3 validation belongs to Stage 5b's cluster smoke test.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Layer-index extraction
# ---------------------------------------------------------------------------

def test_layer_index_extraction() -> None:
    pytest.importorskip("torch")
    from merge.methods.adamerging import _layer_index_from_canonical

    assert _layer_index_from_canonical("model.layers.0.self_attn.q_proj") == 0
    assert _layer_index_from_canonical("model.layers.5.mlp.down_proj") == 5
    assert _layer_index_from_canonical("model.layers.27.self_attn.k_proj") == 27


def test_layer_index_raises_on_malformed_name() -> None:
    pytest.importorskip("torch")
    from merge.methods.adamerging import _layer_index_from_canonical

    with pytest.raises(ValueError):
        _layer_index_from_canonical("no_layers_segment")
    with pytest.raises(ValueError):
        _layer_index_from_canonical("model.layers.not_an_int.q_proj")
    with pytest.raises(ValueError):
        _layer_index_from_canonical("model.layers")


# ---------------------------------------------------------------------------
# Training loop behavior
# ---------------------------------------------------------------------------

def test_adamerging_loss_decreases_on_toy(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    """Loss should decrease over 100 steps on the synthetic forward."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.load_adapter import load_all
    from merge.methods.adamerging import adamerging
    from merge.tests.fixtures.adamerging_helpers import (
        make_synthetic_data_iter,
        make_synthetic_forward_fn,
    )
    from merge.verify_spec import load_locked_spec

    locked_spec = load_locked_spec(lora_yaml_path)
    adapters = load_all(synthetic_adapters_dir, locked_spec)
    task_vectors = list(adapters.values())

    forward_fn = make_synthetic_forward_fn(seed=42)
    data_iter = make_synthetic_data_iter(n_batches=200, seed=42)

    result = adamerging(
        task_vectors,
        forward_fn=forward_fn,
        data_iter=data_iter,
        max_steps=100,
        early_stop_patience=200,  # disable for this test
    )

    assert len(result.loss_history) == 100
    assert result.steps_run == 100
    assert result.loss_history[-1] < result.loss_history[0], (
        f"Loss did not decrease: {result.loss_history[0]} -> {result.loss_history[-1]}"
    )


def test_adamerging_coefficients_move_from_init(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    """After training, coefficients should differ from the 0.3 init."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.load_adapter import load_all
    from merge.methods.adamerging import adamerging
    from merge.tests.fixtures.adamerging_helpers import (
        make_synthetic_data_iter,
        make_synthetic_forward_fn,
    )
    from merge.verify_spec import load_locked_spec

    locked_spec = load_locked_spec(lora_yaml_path)
    adapters = load_all(synthetic_adapters_dir, locked_spec)
    task_vectors = list(adapters.values())

    forward_fn = make_synthetic_forward_fn(seed=42)
    data_iter = make_synthetic_data_iter(n_batches=100, seed=42)

    result = adamerging(
        task_vectors,
        forward_fn=forward_fn,
        data_iter=data_iter,
        max_steps=100,
        early_stop_patience=200,
        init_coefficient=0.3,
    )

    diff = (result.coefficients - 0.3).abs()
    assert diff.max().item() > 0.01, (
        f"Coefficients did not move from init (max delta={diff.max().item()})"
    )


def test_adamerging_no_nan_inf(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    """Final coefficients + merged tensors must be finite."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.load_adapter import load_all
    from merge.methods.adamerging import adamerging
    from merge.tests.fixtures.adamerging_helpers import (
        make_synthetic_data_iter,
        make_synthetic_forward_fn,
    )
    from merge.verify_spec import load_locked_spec

    locked_spec = load_locked_spec(lora_yaml_path)
    adapters = load_all(synthetic_adapters_dir, locked_spec)
    task_vectors = list(adapters.values())

    forward_fn = make_synthetic_forward_fn(seed=42)
    data_iter = make_synthetic_data_iter(n_batches=100, seed=42)

    result = adamerging(
        task_vectors,
        forward_fn=forward_fn,
        data_iter=data_iter,
        max_steps=50,
        early_stop_patience=200,
    )

    assert torch.isfinite(result.coefficients).all()
    for k, v in result.merged.items():
        assert torch.isfinite(v).all(), f"NaN/Inf in merged[{k}]"


def test_adamerging_preserves_keys_and_dtype(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    """Merged dict has same keys and bf16 dtype as inputs."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.load_adapter import load_all
    from merge.methods.adamerging import adamerging
    from merge.tests.fixtures.adamerging_helpers import (
        make_synthetic_data_iter,
        make_synthetic_forward_fn,
    )
    from merge.verify_spec import load_locked_spec

    locked_spec = load_locked_spec(lora_yaml_path)
    adapters = load_all(synthetic_adapters_dir, locked_spec)
    task_vectors = list(adapters.values())

    forward_fn = make_synthetic_forward_fn(seed=42)
    data_iter = make_synthetic_data_iter(n_batches=50, seed=42)

    result = adamerging(
        task_vectors,
        forward_fn=forward_fn,
        data_iter=data_iter,
        max_steps=20,
        early_stop_patience=200,
    )

    input_keys = set(task_vectors[0].keys())
    output_keys = set(result.merged.keys())
    assert input_keys == output_keys
    for k in output_keys:
        assert result.merged[k].dtype == torch.bfloat16, f"{k} not bf16"
        assert result.merged[k].shape == task_vectors[0][k].shape


def test_adamerging_does_not_modify_inputs(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    """Input task vectors must be unchanged after training."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.load_adapter import load_all
    from merge.methods.adamerging import adamerging
    from merge.tests.fixtures.adamerging_helpers import (
        make_synthetic_data_iter,
        make_synthetic_forward_fn,
    )
    from merge.verify_spec import load_locked_spec

    locked_spec = load_locked_spec(lora_yaml_path)
    adapters = load_all(synthetic_adapters_dir, locked_spec)
    task_vectors = list(adapters.values())

    snapshots = [{k: v.clone() for k, v in tv.items()} for tv in task_vectors]

    forward_fn = make_synthetic_forward_fn(seed=42)
    data_iter = make_synthetic_data_iter(n_batches=20, seed=42)

    adamerging(
        task_vectors,
        forward_fn=forward_fn,
        data_iter=data_iter,
        max_steps=10,
        early_stop_patience=200,
    )

    for tv, snap in zip(task_vectors, snapshots):
        for k in tv.keys():
            assert torch.equal(tv[k], snap[k]), f"Input {k} was modified"


def test_adamerging_early_stops_when_loss_plateaus(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    """Tight patience should produce an early stop before max_steps."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.load_adapter import load_all
    from merge.methods.adamerging import adamerging
    from merge.tests.fixtures.adamerging_helpers import (
        make_synthetic_data_iter,
        make_synthetic_forward_fn,
    )
    from merge.verify_spec import load_locked_spec

    locked_spec = load_locked_spec(lora_yaml_path)
    adapters = load_all(synthetic_adapters_dir, locked_spec)
    task_vectors = list(adapters.values())

    forward_fn = make_synthetic_forward_fn(seed=42)
    data_iter = make_synthetic_data_iter(n_batches=500, seed=42)

    result = adamerging(
        task_vectors,
        forward_fn=forward_fn,
        data_iter=data_iter,
        max_steps=500,
        early_stop_patience=5,  # tight enough that any plateau triggers stop
    )

    assert result.steps_run < 500 or result.early_stopped, (
        f"Neither ran fewer than max_steps nor flagged early_stopped: "
        f"steps_run={result.steps_run}, early_stopped={result.early_stopped}"
    )


def test_adamerging_reproducible_with_fixed_seeds(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    """Same forward seed + same data-iter seed -> coefficients match closely."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.load_adapter import load_all
    from merge.methods.adamerging import adamerging
    from merge.tests.fixtures.adamerging_helpers import (
        make_synthetic_data_iter,
        make_synthetic_forward_fn,
    )
    from merge.verify_spec import load_locked_spec

    locked_spec = load_locked_spec(lora_yaml_path)
    adapters = load_all(synthetic_adapters_dir, locked_spec)
    task_vectors = list(adapters.values())

    forward_a = make_synthetic_forward_fn(seed=42)
    iter_a = make_synthetic_data_iter(n_batches=50, seed=42)
    result_a = adamerging(
        task_vectors, forward_fn=forward_a, data_iter=iter_a,
        max_steps=30, early_stop_patience=200,
    )

    forward_b = make_synthetic_forward_fn(seed=42)
    iter_b = make_synthetic_data_iter(n_batches=50, seed=42)
    result_b = adamerging(
        task_vectors, forward_fn=forward_b, data_iter=iter_b,
        max_steps=30, early_stop_patience=200,
    )

    torch.testing.assert_close(
        result_a.coefficients, result_b.coefficients, rtol=1e-3, atol=1e-3
    )


# ---------------------------------------------------------------------------
# Composition + registry
# ---------------------------------------------------------------------------

def test_dare_adamerging_composition(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    """dare_adamerging returns a dict (not AdaMergingResult) and is finite."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.load_adapter import load_all
    from merge.methods import dare_adamerging
    from merge.tests.fixtures.adamerging_helpers import (
        make_synthetic_data_iter,
        make_synthetic_forward_fn,
    )
    from merge.verify_spec import load_locked_spec

    locked_spec = load_locked_spec(lora_yaml_path)
    adapters = load_all(synthetic_adapters_dir, locked_spec)
    task_vectors = list(adapters.values())

    forward_fn = make_synthetic_forward_fn(seed=42)
    data_iter = make_synthetic_data_iter(n_batches=30, seed=42)

    merged = dare_adamerging(
        task_vectors,
        forward_fn=forward_fn,
        data_iter=data_iter,
        drop_rate=0.5,
        seed=42,
        max_steps=20,
        early_stop_patience=200,
    )

    assert isinstance(merged, dict)
    input_keys = set(task_vectors[0].keys())
    assert set(merged.keys()) == input_keys
    for k, v in merged.items():
        assert torch.isfinite(v).all()
        assert v.dtype == torch.bfloat16


def test_method_registry_includes_dare_adamerging() -> None:
    pytest.importorskip("torch")
    from merge.methods import METHOD_REGISTRY

    assert "adamerging" in METHOD_REGISTRY
    assert "dare_adamerging" in METHOD_REGISTRY
    assert len(METHOD_REGISTRY) == 6
    expected = {
        "uniform", "dare_uniform", "dare_weighted",
        "ties", "adamerging", "dare_adamerging",
    }
    assert set(METHOD_REGISTRY.keys()) == expected


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_adamerging_raises_on_empty_task_vectors() -> None:
    pytest.importorskip("torch")
    from merge.methods.adamerging import adamerging

    with pytest.raises(ValueError, match="at least one"):
        adamerging([], forward_fn=lambda m, b: None, data_iter=iter([]))


def test_adamerging_raises_on_mismatched_keys(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.load_adapter import load_all
    from merge.methods.adamerging import adamerging
    from merge.verify_spec import load_locked_spec

    locked_spec = load_locked_spec(lora_yaml_path)
    adapters = load_all(synthetic_adapters_dir, locked_spec)
    task_vectors = list(adapters.values())
    # Corrupt the second adapter's key set
    first_key = next(iter(task_vectors[1].keys()))
    task_vectors[1].pop(first_key)

    with pytest.raises(ValueError, match="key set diverges"):
        adamerging(
            task_vectors,
            forward_fn=lambda m, b: None,
            data_iter=iter([]),
            max_steps=0,
        )

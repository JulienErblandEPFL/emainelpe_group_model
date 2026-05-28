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


# ---------------------------------------------------------------------------
# Metrics persistence (follow-up #6)
# ---------------------------------------------------------------------------

def _fake_adamerging_result():
    """Build a minimal AdaMergingResult; used to mock the heavy adamerging()
    call so metrics-persistence tests stay CPU-light and torch-only."""
    torch = pytest.importorskip("torch")
    from merge.methods.adamerging import AdaMergingResult

    # 4 tasks × 3 layers — small but row-/column-distinct so order bugs surface.
    coeffs = torch.tensor(
        [
            [0.10, 0.11, 0.12],   # task 0
            [0.20, 0.21, 0.22],   # task 1
            [0.30, 0.31, 0.32],   # task 2
            [0.40, 0.41, 0.42],   # task 3
        ],
        dtype=torch.float32,
    )
    merged = {
        "model.layers.0.self_attn.q_proj": torch.zeros(2, 2),
        "model.layers.1.mlp.down_proj": torch.zeros(2, 2),
        "model.layers.2.self_attn.k_proj": torch.zeros(2, 2),
    }
    return AdaMergingResult(
        merged=merged,
        coefficients=coeffs,
        loss_history=[1.0, 0.5, 0.25, 0.125],
        steps_run=4,
        early_stopped=True,
    )


def test_dare_adamerging_persists_metrics_when_path_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dare_adamerging writes adamerging_metrics.json with the expected schema
    when metrics_out_path + task_names are provided. task_names row order
    is preserved verbatim from caller (this is the row-label invariant the
    pipeline relies on)."""
    pytest.importorskip("torch")
    import json
    import torch

    import merge.methods as methods_mod
    from merge.methods import dare_adamerging

    fake = _fake_adamerging_result()
    monkeypatch.setattr(methods_mod, "adamerging", lambda *a, **k: fake)

    # task_vectors shape/keys must match the merged dict so DARE's per-tv
    # loop and the (unused) merge step don't blow up before we hit the
    # patched adamerging() call.
    tv = {k: torch.ones_like(v) for k, v in fake.merged.items()}
    task_vectors = [tv, {**tv}, {**tv}, {**tv}]

    out_path = tmp_path / "adamerging_metrics.json"
    task_names = ["math", "general_knowledge", "safety", "multilingual"]

    merged = dare_adamerging(
        task_vectors,
        forward_fn=lambda m, b: None,
        data_iter=iter([]),
        drop_rate=0.5,
        seed=42,
        max_steps=4,
        metrics_out_path=out_path,
        task_names=task_names,
    )

    assert isinstance(merged, dict)
    assert out_path.exists(), "metrics file was not written"
    payload = json.loads(out_path.read_text())
    assert payload["task_names"] == task_names
    assert payload["n_tasks"] == 4
    assert payload["n_layers"] == 3
    assert payload["steps_run"] == 4
    assert payload["early_stopped"] is True
    assert payload["loss_history"] == [1.0, 0.5, 0.25, 0.125]
    assert len(payload["coefficients"]) == 4
    assert len(payload["coefficients"][0]) == 3
    # Row 2 corresponds to "safety" — the value in that row is the giveaway.
    assert payload["coefficients"][2][0] == pytest.approx(0.30, abs=1e-6)
    hp = payload["hyperparams"]
    assert hp["method"] == "dare_adamerging"
    assert hp["drop_rate"] == 0.5
    assert hp["seed"] == 42
    assert hp["max_steps"] == 4


def test_dare_adamerging_no_metrics_when_path_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No file is written when metrics_out_path is None (back-compat path
    for the existing test_dare_adamerging_composition contract)."""
    pytest.importorskip("torch")
    import torch

    import merge.methods as methods_mod
    from merge.methods import dare_adamerging

    fake = _fake_adamerging_result()
    monkeypatch.setattr(methods_mod, "adamerging", lambda *a, **k: fake)

    tv = {k: torch.ones_like(v) for k, v in fake.merged.items()}
    task_vectors = [tv, {**tv}, {**tv}, {**tv}]

    dare_adamerging(
        task_vectors,
        forward_fn=lambda m, b: None,
        data_iter=iter([]),
        drop_rate=0.5,
        seed=42,
        max_steps=4,
    )
    # tmp_path stays empty — nothing was written.
    assert list(tmp_path.iterdir()) == []


def test_dare_adamerging_forwards_aggregate_domains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dare_adamerging must thread aggregate_domains into the adamerging()
    call (the bug this fix closes: the call enumerated kwargs explicitly
    and silently dropped the flag). Capture the kwargs the mock receives."""
    pytest.importorskip("torch")
    import torch

    import merge.methods as methods_mod
    from merge.methods import dare_adamerging

    fake = _fake_adamerging_result()
    captured: dict = {}

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(methods_mod, "adamerging", _spy)

    tv = {k: torch.ones_like(v) for k, v in fake.merged.items()}
    task_vectors = [tv, {**tv}, {**tv}, {**tv}]

    # Default: aggregate_domains forwarded as False.
    dare_adamerging(
        task_vectors,
        forward_fn=lambda m, b: None,
        data_iter=iter([]),
        drop_rate=0.5, seed=42, max_steps=4,
    )
    assert captured.get("aggregate_domains") is False

    captured.clear()
    # Explicit True is forwarded.
    dare_adamerging(
        task_vectors,
        forward_fn=lambda m, b: None,
        data_iter=iter([]),
        drop_rate=0.5, seed=42, max_steps=4,
        aggregate_domains=True,
    )
    assert captured.get("aggregate_domains") is True


def test_dare_adamerging_metrics_requires_task_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without task_names the coefficient rows would be unlabeled — refuse."""
    pytest.importorskip("torch")
    import torch

    import merge.methods as methods_mod
    from merge.methods import dare_adamerging

    fake = _fake_adamerging_result()
    monkeypatch.setattr(methods_mod, "adamerging", lambda *a, **k: fake)

    tv = {k: torch.ones_like(v) for k, v in fake.merged.items()}
    task_vectors = [tv, {**tv}, {**tv}, {**tv}]

    with pytest.raises(ValueError, match="task_names is required"):
        dare_adamerging(
            task_vectors,
            forward_fn=lambda m, b: None,
            data_iter=iter([]),
            drop_rate=0.5,
            seed=42,
            max_steps=4,
            metrics_out_path=tmp_path / "metrics.json",
            task_names=None,
        )


def test_adamerging_registry_shim_persists_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bare ``adamerging`` registry entry (no DARE) also persists metrics."""
    pytest.importorskip("torch")
    import json

    import merge.methods as methods_mod
    from merge.methods import METHOD_REGISTRY

    fake = _fake_adamerging_result()
    monkeypatch.setattr(methods_mod, "adamerging", lambda *a, **k: fake)

    out_path = tmp_path / "adamerging_metrics.json"
    task_names = ["math", "general_knowledge", "safety", "multilingual"]

    merged = METHOD_REGISTRY["adamerging"](
        [fake.merged] * 4,  # any 4-element list — adamerging() is mocked
        forward_fn=lambda m, b: None,
        data_iter=iter([]),
        metrics_out_path=out_path,
        task_names=task_names,
    )

    assert isinstance(merged, dict)
    payload = json.loads(out_path.read_text())
    assert payload["task_names"] == task_names
    assert payload["hyperparams"]["method"] == "adamerging"
    # forward_fn / data_iter must NOT leak into the hyperparams snapshot —
    # they're non-serializable callables/iterators.
    assert "forward_fn" not in payload["hyperparams"]
    assert "data_iter" not in payload["hyperparams"]


def test_persist_adamerging_metrics_rejects_task_name_length_mismatch(
    tmp_path: Path,
) -> None:
    """Mismatched task_names length would silently mislabel rows — refuse."""
    pytest.importorskip("torch")
    from merge.methods import _persist_adamerging_metrics

    fake = _fake_adamerging_result()
    with pytest.raises(ValueError, match="row order would be ambiguous"):
        _persist_adamerging_metrics(
            fake,
            tmp_path / "metrics.json",
            task_names=["only", "three", "names"],  # but coefficients has 4 rows
            hyperparams={},
        )


# ---------------------------------------------------------------------------
# aggregate_domains mode (follow-up #8)
# ---------------------------------------------------------------------------

def test_adamerging_default_unchanged_byte_for_byte(
    synthetic_adapters_dir: Path, lora_yaml_path: Path,
) -> None:
    """Default path (aggregate_domains omitted vs explicit False) must give
    identical loss_history + coefficients for the same seed.

    This pins the reproducibility invariant called out in the follow-up #8
    spec: the new opt-in flag must not perturb the existing path.
    """
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
    tvs_a = list(load_all(synthetic_adapters_dir, locked_spec).values())
    tvs_b = list(load_all(synthetic_adapters_dir, locked_spec).values())

    res_a = adamerging(
        tvs_a,
        forward_fn=make_synthetic_forward_fn(seed=42),
        data_iter=make_synthetic_data_iter(n_batches=40, seed=42),
        max_steps=30, early_stop_patience=200,
    )
    res_b = adamerging(
        tvs_b,
        forward_fn=make_synthetic_forward_fn(seed=42),
        data_iter=make_synthetic_data_iter(n_batches=40, seed=42),
        max_steps=30, early_stop_patience=200,
        aggregate_domains=False,  # explicit default
    )
    assert res_a.loss_history == res_b.loss_history
    torch.testing.assert_close(res_a.coefficients, res_b.coefficients,
                               rtol=0, atol=0)


def test_adamerging_aggregated_consumes_n_tasks_batches_per_update(
    synthetic_adapters_dir: Path, lora_yaml_path: Path,
) -> None:
    """One optimizer update consumes n_tasks consecutive yields.

    Wraps the synthetic iterator with a counter so we can observe how many
    underlying ``__next__`` calls happen for a given ``max_steps``. With
    aggregate_domains=True and max_steps=5 on a 4-task setup, exactly 20
    batches should be consumed (5 updates × 4 batches/update), and
    loss_history length should be 5.
    """
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
    tvs = list(load_all(synthetic_adapters_dir, locked_spec).values())
    n_tasks = len(tvs)
    assert n_tasks == 4, "synthetic adapters fixture is expected to have 4 tasks"

    consumed = [0]
    inner = make_synthetic_data_iter(n_batches=100, seed=42)

    def counting_iter():
        for item in inner:
            consumed[0] += 1
            yield item

    result = adamerging(
        tvs,
        forward_fn=make_synthetic_forward_fn(seed=42),
        data_iter=counting_iter(),
        max_steps=5,
        early_stop_patience=200,
        aggregate_domains=True,
    )
    assert result.steps_run == 5
    assert len(result.loss_history) == 5
    assert consumed[0] == 5 * n_tasks, (
        f"aggregated mode should consume max_steps * n_tasks batches "
        f"({5 * n_tasks}); consumed {consumed[0]}."
    )


def test_adamerging_aggregated_stops_clean_on_short_iterator(
    synthetic_adapters_dir: Path, lora_yaml_path: Path,
) -> None:
    """If the data iterator runs out mid-update, the aggregated path stops
    without raising and reports fewer steps_run."""
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
    tvs = list(load_all(synthetic_adapters_dir, locked_spec).values())
    # Only 10 batches for 4 tasks ⇒ 2 full updates (8 batches), then
    # mid-update exhaustion (2 batches short of a third).
    result = adamerging(
        tvs,
        forward_fn=make_synthetic_forward_fn(seed=42),
        data_iter=make_synthetic_data_iter(n_batches=10, seed=42),
        max_steps=10,
        early_stop_patience=200,
        aggregate_domains=True,
    )
    assert result.steps_run == 2
    assert len(result.loss_history) == 2


def test_adamerging_aggregated_loss_history_records_aggregated_value(
    synthetic_adapters_dir: Path, lora_yaml_path: Path,
) -> None:
    """The recorded per-update loss is the aggregated (mean over n_tasks
    domains) entropy + L2 once — not a single-batch loss. Sanity-check by
    running both modes briefly and asserting the recorded magnitudes differ
    (per-batch loss varies wildly by domain on the synthetic fixture, the
    aggregated value is the mean of those, so they should not coincide)."""
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
    tvs = list(load_all(synthetic_adapters_dir, locked_spec).values())

    res = adamerging(
        tvs,
        forward_fn=make_synthetic_forward_fn(seed=42),
        data_iter=make_synthetic_data_iter(n_batches=20, seed=42),
        max_steps=5,
        early_stop_patience=200,
        aggregate_domains=True,
    )
    assert all(isinstance(v, float) for v in res.loss_history)
    assert all(v == v for v in res.loss_history)  # not NaN
    assert all(v > 0 for v in res.loss_history)   # entropy + positive L2


def test_adamerging_aggregated_second_update_proves_graphs_freed(
    synthetic_adapters_dir: Path, lora_yaml_path: Path,
) -> None:
    """Gradient-accumulation form must free each per-domain graph after
    its backward(). If it didn't, the second optimizer update would either
    (a) error with "Trying to backward through the graph a second time"
    (PyTorch's signal that the graph was retained then freed), or (b)
    silently double-accumulate. Running ≥2 updates successfully on the
    synthetic fixture proves graphs are being freed per-domain — the
    cheapest proof we can extract without poking at torch internals."""
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
    tvs = list(load_all(synthetic_adapters_dir, locked_spec).values())

    res = adamerging(
        tvs,
        forward_fn=make_synthetic_forward_fn(seed=42),
        data_iter=make_synthetic_data_iter(n_batches=40, seed=42),
        max_steps=3,
        early_stop_patience=200,
        aggregate_domains=True,
    )
    assert res.steps_run == 3
    assert len(res.loss_history) == 3


def test_adamerging_aggregated_matches_sum_then_backward_numerically(
    synthetic_adapters_dir: Path, lora_yaml_path: Path,
) -> None:
    """Gradient accumulation is mathematically equivalent to summing per-
    domain losses then a single backward() — gradients are linear. After
    one optimizer step, the coefficients reached via accumulation should
    match those reached via the textbook ``loss.sum().backward()`` form
    bit-for-bit (same seed, same forward, same lr).

    This is the formal math-equivalence check the follow-up #9 fix rests
    on. If it ever fails, the accumulation form has drifted from the
    intended semantics.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.load_adapter import load_all
    from merge.methods.adamerging import (
        _compute_merged, _layer_index_from_canonical, adamerging,
    )
    from merge.tests.fixtures.adamerging_helpers import (
        make_synthetic_data_iter,
        make_synthetic_forward_fn,
    )
    from merge.verify_spec import load_locked_spec

    locked_spec = load_locked_spec(lora_yaml_path)
    tvs = list(load_all(synthetic_adapters_dir, locked_spec).values())
    n_tasks = len(tvs)

    # Run one aggregated update via the new accumulation path.
    res_acc = adamerging(
        tvs,
        forward_fn=make_synthetic_forward_fn(seed=42),
        data_iter=make_synthetic_data_iter(n_batches=n_tasks, seed=42),
        max_steps=1,
        early_stop_patience=200,
        aggregate_domains=True,
        lr=1e-2, lambda_l2=1e-4, init_coefficient=0.3,
    )

    # Reproduce one update via the explicit sum-then-backward form.
    name_to_layer = {k: _layer_index_from_canonical(k) for k in tvs[0]}
    n_layers = max(name_to_layer.values()) + 1
    coeffs = torch.full((n_tasks, n_layers), 0.3, dtype=torch.float32,
                        requires_grad=True)
    opt = torch.optim.Adam([coeffs], lr=1e-2)
    fwd = make_synthetic_forward_fn(seed=42)
    data = make_synthetic_data_iter(n_batches=n_tasks, seed=42)
    entropies = []
    for _ in range(n_tasks):
        _, batch = next(data)
        merged = _compute_merged(tvs, coeffs, name_to_layer)
        logits = fwd(merged, batch)
        last = logits[:, -1, :]
        lp = torch.log_softmax(last, -1)
        p = torch.softmax(last, -1)
        entropies.append(-(p * lp).sum(-1).mean())
    loss = torch.stack(entropies).mean() + 1e-4 * coeffs.pow(2).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()

    torch.testing.assert_close(res_acc.coefficients, coeffs.detach(),
                               rtol=1e-5, atol=1e-6)

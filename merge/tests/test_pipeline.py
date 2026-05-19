"""
Tests for ``merge.pipeline.merge_adapters`` — Stage 4 orchestrator.

Exercises every method through the registry, plus the error paths
(unknown method, stub method, missing adapters, spec mismatch, non-empty
output dir).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# svd_factor isolation — math gate
# ---------------------------------------------------------------------------

def test_svd_factor_round_trip_within_truncation_tolerance() -> None:
    """svd_factor + load's reconstruction recovers a rank-r matrix exactly.

    Diagnostic gate isolating the SVD factorization math from the rest of
    the pipeline. Input is built to be exactly rank-r so SVD truncation is
    a no-op and any error reveals a scaling/broadcasting bug in
    ``svd_factor``.
    """
    torch = pytest.importorskip("torch")
    from merge.pipeline import svd_factor

    r, alpha = 32, 64
    out_dim, in_dim = 64, 64

    torch.manual_seed(0)
    U_true = torch.linalg.qr(torch.randn(out_dim, r))[0]                    # [out, r], orthonormal
    S_true = torch.tensor([1.0 + i * 0.1 for i in range(r)])                # known spectrum
    Vh_true = torch.linalg.qr(torch.randn(in_dim, r))[0].T                  # [r, in], orthonormal
    delta_w = U_true @ torch.diag(S_true) @ Vh_true                         # rank-r exactly, fp32

    lora_A, lora_B = svd_factor(delta_w, r=r, alpha=alpha)
    reconstructed = (alpha / r) * lora_B.float() @ lora_A.float()

    # Rank-r input means SVD truncation is lossless. Any error here is a
    # scaling or broadcasting bug, not truncation.
    torch.testing.assert_close(reconstructed, delta_w, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Happy paths — one per method
# ---------------------------------------------------------------------------

def test_pipeline_uniform_with_synthetic_adapters(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.load_adapter import load
    from merge.pipeline import merge_adapters

    out_dir = tmp_path / "merged"
    returned = merge_adapters(
        synthetic_adapters_dir,
        method="uniform",
        output_dir=out_dir,
        locked_spec_path=lora_yaml_path,
    )
    assert returned == out_dir
    assert (out_dir / "adapter_config.json").exists()
    assert (out_dir / "adapter_model.safetensors").exists()

    cfg = json.loads((out_dir / "adapter_config.json").read_text())
    assert cfg["inference_mode"] is False
    assert cfg["r"] == 32
    assert cfg["lora_alpha"] == 64

    # Reload round-trips into a task vector dict with the same canonical keys.
    reloaded = load(out_dir)
    one_input = load(synthetic_adapters_dir / "math")
    assert set(reloaded.keys()) == set(one_input.keys())


def test_pipeline_dare_uniform_with_synthetic_adapters(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.pipeline import merge_adapters

    out_dir = tmp_path / "merged"
    merge_adapters(
        synthetic_adapters_dir,
        method="dare_uniform",
        output_dir=out_dir,
        locked_spec_path=lora_yaml_path,
        method_kwargs={"drop_rate": 0.5, "seed": 42},
    )
    assert (out_dir / "adapter_config.json").exists()
    assert (out_dir / "adapter_model.safetensors").exists()


def test_pipeline_dare_weighted_with_synthetic_adapters(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.pipeline import merge_adapters

    out_dir = tmp_path / "merged"
    merge_adapters(
        synthetic_adapters_dir,
        method="dare_weighted",
        output_dir=out_dir,
        locked_spec_path=lora_yaml_path,
        method_kwargs={"weights": [0.4, 0.3, 0.2, 0.1], "drop_rate": 0.5, "seed": 1},
    )
    assert (out_dir / "adapter_config.json").exists()


def test_pipeline_ties_with_synthetic_adapters(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.pipeline import merge_adapters

    out_dir = tmp_path / "merged"
    merge_adapters(
        synthetic_adapters_dir,
        method="ties",
        output_dir=out_dir,
        locked_spec_path=lora_yaml_path,
        method_kwargs={"trim_ratio": 0.5},
    )
    assert (out_dir / "adapter_config.json").exists()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_pipeline_unknown_method_raises_key_error(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from merge.pipeline import merge_adapters

    with pytest.raises(KeyError, match=r"not_a_method"):
        merge_adapters(
            synthetic_adapters_dir,
            method="not_a_method",
            output_dir=tmp_path / "merged",
            locked_spec_path=lora_yaml_path,
        )


def test_pipeline_adamerging_requires_forward_fn_and_data_iter(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    """adamerging requires forward_fn and data_iter in method_kwargs;
    calling without them raises TypeError mentioning the missing args."""
    pytest.importorskip("torch")
    from merge.pipeline import merge_adapters

    with pytest.raises(TypeError, match=r"forward_fn|data_iter"):
        merge_adapters(
            synthetic_adapters_dir,
            method="adamerging",
            output_dir=tmp_path / "merged",
            locked_spec_path=lora_yaml_path,
        )


def test_pipeline_dare_adamerging_requires_forward_fn_and_data_iter(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    """dare_adamerging requires forward_fn and data_iter in method_kwargs."""
    pytest.importorskip("torch")
    from merge.pipeline import merge_adapters

    with pytest.raises(TypeError, match=r"forward_fn|data_iter"):
        merge_adapters(
            synthetic_adapters_dir,
            method="dare_adamerging",
            output_dir=tmp_path / "merged",
            locked_spec_path=lora_yaml_path,
        )


def test_pipeline_dare_adamerging_with_synthetic_forward(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    """End-to-end: pipeline dispatch through dare_adamerging produces a
    valid merged adapter directory."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.pipeline import merge_adapters
    from merge.tests.fixtures.adamerging_helpers import (
        make_synthetic_forward_fn,
        make_synthetic_data_iter,
    )

    out_dir = tmp_path / "merged"
    merge_adapters(
        synthetic_adapters_dir,
        method="dare_adamerging",
        output_dir=out_dir,
        locked_spec_path=lora_yaml_path,
        method_kwargs={
            "forward_fn": make_synthetic_forward_fn(seed=0),
            "data_iter": make_synthetic_data_iter(n_batches=30, seed=0),
            "drop_rate": 0.5,
            "seed": 42,
            "max_steps": 20,
            "early_stop_patience": 200,
        },
    )

    assert (out_dir / "adapter_config.json").exists()
    assert (out_dir / "adapter_model.safetensors").exists()


def test_pipeline_missing_adapter_raises(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    import shutil
    from merge.pipeline import merge_adapters

    shutil.rmtree(synthetic_adapters_dir / "safety")
    with pytest.raises(FileNotFoundError, match=r"safety"):
        merge_adapters(
            synthetic_adapters_dir,
            method="uniform",
            output_dir=tmp_path / "merged",
            locked_spec_path=lora_yaml_path,
        )


def test_pipeline_diverging_adapter_raises_spec_mismatch(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from merge.pipeline import merge_adapters
    from merge.verify_spec import SpecMismatchError

    cfg_path = synthetic_adapters_dir / "safety" / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["r"] = 16
    cfg_path.write_text(json.dumps(cfg))

    with pytest.raises(SpecMismatchError):
        merge_adapters(
            synthetic_adapters_dir,
            method="uniform",
            output_dir=tmp_path / "merged",
            locked_spec_path=lora_yaml_path,
        )


def test_pipeline_output_exists_and_nonempty_raises(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from merge.pipeline import merge_adapters

    out_dir = tmp_path / "merged"
    out_dir.mkdir()
    (out_dir / "something.txt").write_text("hi")
    with pytest.raises(FileExistsError):
        merge_adapters(
            synthetic_adapters_dir,
            method="uniform",
            output_dir=out_dir,
            locked_spec_path=lora_yaml_path,
        )


def test_pipeline_output_exists_but_empty_succeeds(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.pipeline import merge_adapters

    out_dir = tmp_path / "merged"
    out_dir.mkdir()
    merge_adapters(
        synthetic_adapters_dir,
        method="uniform",
        output_dir=out_dir,
        locked_spec_path=lora_yaml_path,
    )
    assert (out_dir / "adapter_config.json").exists()


def test_pipeline_writes_generation_config_when_provided(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    """If ``generation_config`` is provided, ``output_dir`` contains
    ``generation_config.json``."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.generation_config import make_generation_config
    from merge.pipeline import merge_adapters

    out_dir = tmp_path / "merged"
    gen_config = make_generation_config(temperature=0.3)
    merge_adapters(
        synthetic_adapters_dir,
        method="uniform",
        output_dir=out_dir,
        locked_spec_path=lora_yaml_path,
        generation_config=gen_config,
    )
    gen_path = out_dir / "generation_config.json"
    assert gen_path.exists()
    loaded = json.loads(gen_path.read_text())
    assert loaded["temperature"] == 0.3
    assert loaded["bos_token_id"] == 151643


def test_pipeline_no_generation_config_when_omitted(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    """If ``generation_config`` is ``None`` (default), no
    ``generation_config.json`` is written."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.pipeline import merge_adapters

    out_dir = tmp_path / "merged"
    merge_adapters(
        synthetic_adapters_dir,
        method="uniform",
        output_dir=out_dir,
        locked_spec_path=lora_yaml_path,
    )
    assert not (out_dir / "generation_config.json").exists()


def test_pipeline_reproducible_with_seed(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    """Same seed → bit-identical lora_A/lora_B weights in the output."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file
    from merge.pipeline import merge_adapters

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    common_kwargs = dict(
        method="dare_uniform",
        locked_spec_path=lora_yaml_path,
        method_kwargs={"drop_rate": 0.5, "seed": 42},
    )
    merge_adapters(synthetic_adapters_dir, output_dir=out_a, **common_kwargs)
    merge_adapters(synthetic_adapters_dir, output_dir=out_b, **common_kwargs)

    a_state = load_file(str(out_a / "adapter_model.safetensors"), device="cpu")
    b_state = load_file(str(out_b / "adapter_model.safetensors"), device="cpu")
    assert set(a_state.keys()) == set(b_state.keys())
    for key in a_state:
        assert torch.equal(a_state[key], b_state[key]), f"{key} differs across runs"

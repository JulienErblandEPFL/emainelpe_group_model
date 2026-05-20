"""
Tests for ``merge.pipeline.merge_adapters`` — Stage 4 orchestrator,
Day 7 refactored to produce a full HF model.

The pipeline now loads Qwen3-1.7B base weights and runs PEFT's
``merge_and_unload`` to bake the merged deltas in. End-to-end happy-path
tests therefore need both CUDA and the base model in the HF cache —
they're cluster-only. Error-path tests (KeyError on unknown method, etc.)
still raise BEFORE base-model loading, so they continue to run on the
torch-free laptop suite via ``pytest.importorskip``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _skip_unless_cluster_ready() -> None:
    """Skip the test unless we're in an environment that can run the full
    pipeline: torch + CUDA + transformers + peft + a usable Qwen3-1.7B.

    The Qwen3 base check is deferred — we trust that on the cluster image
    the model is in the HF cache. On the laptop we skip on CUDA absence.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    if not torch.cuda.is_available():
        pytest.skip("Full-pipeline tests require CUDA + Qwen3 base; cluster only")


# ---------------------------------------------------------------------------
# svd_factor isolation — math gate, runs on laptop
# ---------------------------------------------------------------------------

def test_svd_factor_round_trip_within_truncation_tolerance() -> None:
    """svd_factor + reconstruction recovers a rank-r matrix exactly.

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
    U_true = torch.linalg.qr(torch.randn(out_dim, r))[0]
    S_true = torch.tensor([1.0 + i * 0.1 for i in range(r)])
    Vh_true = torch.linalg.qr(torch.randn(in_dim, r))[0].T
    delta_w = U_true @ torch.diag(S_true) @ Vh_true

    lora_A, lora_B = svd_factor(delta_w, r=r, alpha=alpha)
    reconstructed = (alpha / r) * lora_B.float() @ lora_A.float()

    torch.testing.assert_close(reconstructed, delta_w, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Error paths — raise before base-model load, runnable on laptop
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


def test_pipeline_adamerging_requires_forward_fn_and_data_iter(
    synthetic_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    """adamerging raises TypeError from the merge function before the
    pipeline reaches base-model load — runnable without Qwen3."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
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
    """dare_adamerging composition has the same kwargs requirement."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from merge.pipeline import merge_adapters

    with pytest.raises(TypeError, match=r"forward_fn|data_iter"):
        merge_adapters(
            synthetic_adapters_dir,
            method="dare_adamerging",
            output_dir=tmp_path / "merged",
            locked_spec_path=lora_yaml_path,
        )


# ---------------------------------------------------------------------------
# Cluster-only integration tests — require CUDA + Qwen3
# ---------------------------------------------------------------------------

@pytest.fixture
def qwen3_random_adapters_dir(tmp_path: Path) -> Path:
    """Four random-init Qwen3-1.7B-sized adapters under tmp_path/loras.

    Used by the cluster-only integration tests below. Each adapter weighs
    ~140 MB in bf16, so this fixture should never be triggered on the
    laptop — _skip_unless_cluster_ready in the test bodies takes care of
    skipping there.
    """
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.tests.fixtures.qwen3_adapter import make_random_qwen3_adapter

    loras_dir = tmp_path / "loras"
    for i, domain in enumerate(("math", "general_knowledge", "safety", "multilingual")):
        make_random_qwen3_adapter(loras_dir / domain, seed=42 + i)
    return loras_dir


def _assert_full_model_output(out_dir: Path) -> None:
    """Common assertions for a successful full-model merge output."""
    assert (out_dir / "config.json").exists(), "missing config.json"
    has_single = (out_dir / "model.safetensors").exists()
    has_sharded = (out_dir / "model.safetensors.index.json").exists()
    assert has_single or has_sharded, "neither model.safetensors nor index.json present"
    assert not (out_dir / "adapter_config.json").exists(), "legacy adapter_config.json present"
    assert not (out_dir / "adapter_model.safetensors").exists(), (
        "legacy adapter_model.safetensors present"
    )


def test_pipeline_uniform_produces_full_model(
    qwen3_random_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    """End-to-end uniform merge writes a full HF-format Qwen3 directory."""
    _skip_unless_cluster_ready()
    from transformers import AutoModelForCausalLM
    from merge.pipeline import merge_adapters

    out_dir = tmp_path / "merged"
    returned = merge_adapters(
        qwen3_random_adapters_dir,
        method="uniform",
        output_dir=out_dir,
        locked_spec_path=lora_yaml_path,
    )
    assert returned == out_dir
    _assert_full_model_output(out_dir)

    # Confirm the saved directory is loadable back as a full model.
    reloaded = AutoModelForCausalLM.from_pretrained(out_dir)
    assert reloaded.config.model_type == "qwen3"


def test_pipeline_dare_uniform_produces_full_model(
    qwen3_random_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    _skip_unless_cluster_ready()
    from merge.pipeline import merge_adapters

    out_dir = tmp_path / "merged"
    merge_adapters(
        qwen3_random_adapters_dir,
        method="dare_uniform",
        output_dir=out_dir,
        locked_spec_path=lora_yaml_path,
        method_kwargs={"drop_rate": 0.5, "seed": 42},
    )
    _assert_full_model_output(out_dir)


def test_pipeline_writes_generation_config_when_provided(
    qwen3_random_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    """If ``generation_config`` is provided, ``output_dir`` contains
    ``generation_config.json``."""
    _skip_unless_cluster_ready()
    from merge.generation_config import make_generation_config
    from merge.pipeline import merge_adapters

    out_dir = tmp_path / "merged"
    gen_config = make_generation_config(temperature=0.3)
    merge_adapters(
        qwen3_random_adapters_dir,
        method="uniform",
        output_dir=out_dir,
        locked_spec_path=lora_yaml_path,
        generation_config=gen_config,
    )
    _assert_full_model_output(out_dir)
    gen_path = out_dir / "generation_config.json"
    assert gen_path.exists()
    loaded = json.loads(gen_path.read_text())
    assert loaded["temperature"] == 0.3
    assert loaded["bos_token_id"] == 151643


def test_pipeline_no_generation_config_when_omitted(
    qwen3_random_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    """If ``generation_config`` is ``None`` (default), no
    ``generation_config.json`` is written.

    Note: ``transformers.save_pretrained`` itself writes a
    ``generation_config.json`` containing the base model's defaults (eos
    token, etc.) when the model has a ``generation_config`` attribute. So
    'no generation_config' really means 'no extra one written by our
    pipeline'. The test only asserts that the file we WOULD have written
    (with our custom temperature) doesn't appear.
    """
    _skip_unless_cluster_ready()
    from merge.pipeline import merge_adapters

    out_dir = tmp_path / "merged"
    merge_adapters(
        qwen3_random_adapters_dir,
        method="uniform",
        output_dir=out_dir,
        locked_spec_path=lora_yaml_path,
    )
    _assert_full_model_output(out_dir)
    # If the file exists (because transformers wrote it from the base
    # model's generation config), our custom temperature override won't
    # be in it. The relevant invariant: we didn't inject a user-supplied
    # generation_config dict, so any contents come from base defaults.
    gen_path = out_dir / "generation_config.json"
    if gen_path.exists():
        loaded = json.loads(gen_path.read_text())
        # The custom override would have set temperature=0.3 (test above);
        # confirm we didn't accidentally write that here.
        assert loaded.get("temperature") != 0.3

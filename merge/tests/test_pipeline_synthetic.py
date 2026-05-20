"""
Synthetic end-to-end tests for the full merge pipeline.

The pre-Day-7 version of these tests round-tripped toy LoRA adapters
through the pipeline and asserted rank-r structure on the reloaded
``adapter_model.safetensors``. After the Day 7 refactor the pipeline
produces a FULL HF-format model (config.json + model.safetensors) via
``peft.merge_and_unload``, not a LoRA adapter — and toy adapters with
hidden_dim=64 can no longer flow through it because the in-memory PEFT
wrapper is built around the real Qwen3-1.7B shapes (hidden=2048).

These tests therefore become cluster-only: they require torch + CUDA +
the Qwen3-1.7B base model in the HF cache. On the torch-free laptop
they skip via ``pytest.importorskip``. On a cluster image without
CUDA they skip via ``_skip_unless_cluster_ready``. The structural
post-conditions we verify are deliberately weak:

  - output dir contains ``config.json`` + ``model.safetensors`` (or shards)
  - output dir is loadable via ``AutoModelForCausalLM.from_pretrained``

That's all that matters downstream: vLLM and the CI grader both consume
``transformers``-loadable directories. The rank-r truncation discipline
is verified in isolation by
``test_svd_factor_round_trip_within_truncation_tolerance``
in ``test_pipeline.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _skip_unless_cluster_ready() -> None:
    """Skip unless we have torch + CUDA + transformers + peft (cluster only)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    if not torch.cuda.is_available():
        pytest.skip("Full-model pipeline tests require CUDA + Qwen3; cluster only")


@pytest.fixture
def qwen3_random_adapters_dir(tmp_path: Path) -> Path:
    """Four random-init Qwen3-1.7B-sized adapters for cluster integration tests."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.tests.fixtures.qwen3_adapter import make_random_qwen3_adapter

    loras_dir = tmp_path / "loras"
    for i, domain in enumerate(("math", "general_knowledge", "safety", "multilingual")):
        make_random_qwen3_adapter(loras_dir / domain, seed=42 + i)
    return loras_dir


def _assert_full_model_round_trip(out_dir: Path) -> None:
    """Output dir must be loadable as a full HF causal-LM model."""
    from transformers import AutoModelForCausalLM

    assert (out_dir / "config.json").exists()
    has_single = (out_dir / "model.safetensors").exists()
    has_sharded = (out_dir / "model.safetensors.index.json").exists()
    assert has_single or has_sharded
    reloaded = AutoModelForCausalLM.from_pretrained(out_dir)
    assert reloaded.config.model_type == "qwen3"


def test_end_to_end_uniform_full_model(
    qwen3_random_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    _skip_unless_cluster_ready()
    from merge.pipeline import merge_adapters

    out_dir = tmp_path / "merged"
    merge_adapters(
        qwen3_random_adapters_dir,
        method="uniform",
        output_dir=out_dir,
        locked_spec_path=lora_yaml_path,
    )
    _assert_full_model_round_trip(out_dir)


def test_end_to_end_dare_uniform_full_model(
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
    _assert_full_model_round_trip(out_dir)


def test_end_to_end_ties_full_model(
    qwen3_random_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    _skip_unless_cluster_ready()
    from merge.pipeline import merge_adapters

    out_dir = tmp_path / "merged"
    merge_adapters(
        qwen3_random_adapters_dir,
        method="ties",
        output_dir=out_dir,
        locked_spec_path=lora_yaml_path,
        method_kwargs={"trim_ratio": 0.5},
    )
    _assert_full_model_round_trip(out_dir)


def test_end_to_end_output_includes_tokenizer_and_chat_template(
    qwen3_random_adapters_dir: Path, lora_yaml_path: Path, tmp_path: Path
) -> None:
    """Merged model dir must be self-contained: tokenizer + locked chat template."""
    _skip_unless_cluster_ready()
    from merge.pipeline import merge_adapters

    out_dir = tmp_path / "merged"
    merge_adapters(
        qwen3_random_adapters_dir,
        method="uniform",
        output_dir=out_dir,
        locked_spec_path=lora_yaml_path,
    )
    # tokenizer.save_pretrained writes these for Qwen3
    assert (out_dir / "tokenizer_config.json").exists()
    assert (out_dir / "tokenizer.json").exists()
    # We copy the locked chat template alongside
    assert (out_dir / "chat_template.jinja").exists()

"""
Skeleton smoke tests for the ``merge/`` subpackage.

What this file guarantees:
1. Every module imports cleanly (or skips with a clear reason if torch is
   missing on the laptop).
2. Each module's primary function raises ``NotImplementedError`` with the
   correct stage tag — i.e. nobody snuck real logic into Stage 1.
3. ``METHOD_REGISTRY`` exposes all six user-facing method names.
4. ``lora.yaml`` still carries the team-locked values, including the new
   explicit ``modules_to_save: null`` line.

Tests that touch ``torch`` use ``pytest.importorskip("torch")`` so they skip
cleanly on a CPU-only laptop without torch installed. The locked-spec
regression test always runs (no ML deps).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Top-level package + torch-free modules
# ---------------------------------------------------------------------------

def test_merge_package_imports() -> None:
    import merge

    assert merge.__version__, "merge.__version__ should be a non-empty string"


def test_infer_stubs_raise_stage_5() -> None:
    from merge import infer

    with pytest.raises(NotImplementedError, match="Stage 5"):
        infer.generate_completions(Path("dummy"), "Qwen/Qwen3-1.7B", [])
    with pytest.raises(NotImplementedError, match="Stage 5"):
        infer.generate_for_validation_set(
            Path("dummy"), "Qwen/Qwen3-1.7B", Path("dummy_vs"), Path("dummy_out")
        )


def test_publish_stub_raises_stage_5() -> None:
    from merge import publish

    with pytest.raises(NotImplementedError, match="Stage 5"):
        publish.publish_adapter(Path("dummy"), "cs-552-2026-emainelpe/group_model")


def test_eval_all_stubs_raise_stage_5() -> None:
    from merge import eval_all

    with pytest.raises(NotImplementedError, match="Stage 5"):
        eval_all.evaluate_completions(Path("dummy"), Path("dummy_vs"))
    with pytest.raises(NotImplementedError, match="Stage 5"):
        eval_all.four_domain_average({})


# ---------------------------------------------------------------------------
# Torch-dependent modules (skip if torch is missing)
# ---------------------------------------------------------------------------

def test_method_registry_has_all_six_methods() -> None:
    pytest.importorskip("torch")
    from merge.methods import METHOD_REGISTRY

    expected = {
        "uniform", "dare_uniform", "dare_weighted",
        "ties", "adamerging", "dare_adamerging",
    }
    assert set(METHOD_REGISTRY.keys()) == expected, (
        f"METHOD_REGISTRY keys {set(METHOD_REGISTRY.keys())!r} "
        f"do not match expected {expected!r}"
    )
    for name, fn in METHOD_REGISTRY.items():
        assert callable(fn), f"METHOD_REGISTRY[{name!r}] is not callable"


# ---------------------------------------------------------------------------
# Locked-spec regression
# ---------------------------------------------------------------------------

def test_locked_spec_unchanged(lora_yaml_path: Path) -> None:
    """
    If this test fails, someone edited ``lora.yaml`` in a way that may break
    the DARE / AdaMerging / TIES merge compatibility guarantees. Investigate
    before doing any merge work — see ``USAGE.md`` for the rationale.
    """
    assert lora_yaml_path.exists(), f"lora.yaml not found at {lora_yaml_path}"

    with lora_yaml_path.open() as f:
        spec = yaml.safe_load(f)

    assert spec["base_model"] == "Qwen/Qwen3-1.7B"

    lora = spec["lora"]
    assert lora["r"] == 32
    assert lora["alpha"] == 64
    assert lora["dropout"] == 0.05
    assert lora["bias"] == "none"
    assert lora["task_type"] == "CAUSAL_LM"
    assert set(lora["target_modules"]) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }

    assert "modules_to_save" in lora, (
        "lora.modules_to_save must be explicitly present (set to null) so it is "
        "self-documenting; see USAGE.md for the merge-compatibility rule."
    )
    assert lora["modules_to_save"] is None, (
        "lora.modules_to_save MUST be null/None. Setting it (e.g. "
        '["embed_tokens", "lm_head"]) introduces full-rank tensors that break '
        "additive task-vector merging (DARE / AdaMerging / TIES)."
    )

    assert spec["max_seq_length"] == 4096
    assert spec["eos_token"] == "<|im_end|>"
    assert spec["thinking_mode"] == "on"
    assert spec["boxed_answers"] is True

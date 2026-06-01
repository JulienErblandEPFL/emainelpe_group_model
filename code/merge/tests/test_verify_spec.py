"""
Tests for ``merge.verify_spec`` — Stage 2 locked-spec verification.

These tests are intentionally torch-free: the verifier only manipulates
YAML and JSON, so it runs identically on a laptop and on the cluster.

The ground-truth math adapter_config (from a real v5 checkpoint on RCP
scratch) is used as the reference "known-good" config.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from merge.verify_spec import (
    LOAD_BEARING_FIELDS,
    FieldResult,
    SpecMismatchError,
    VerifyResult,
    load_locked_spec,
    verify,
)


# The 8-field PEFT adapter_config that the real math v5 adapter on RCP scratch
# emitted. Verified against ``lora.yaml`` at the time this Stage 2 was written.
GROUND_TRUTH_MATH_CONFIG: dict = {
    "base_model_name_or_path": "Qwen/Qwen3-1.7B",
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "modules_to_save": None,
    "target_modules": [
        "down_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "k_proj",
        "v_proj",
        "q_proj",
    ],
}


# ---------------------------------------------------------------------------
# load_locked_spec
# ---------------------------------------------------------------------------

def test_load_locked_spec_returns_canonical_fields(lora_yaml_path: Path) -> None:
    spec = load_locked_spec(lora_yaml_path)

    assert set(spec.keys()) == set(LOAD_BEARING_FIELDS)
    assert spec["base_model_name_or_path"] == "Qwen/Qwen3-1.7B"
    assert spec["r"] == 32
    assert spec["lora_alpha"] == 64
    assert spec["lora_dropout"] == 0.05
    assert spec["bias"] == "none"
    assert spec["task_type"] == "CAUSAL_LM"
    assert spec["modules_to_save"] is None
    assert set(spec["target_modules"]) == {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }


def test_load_locked_spec_raises_on_missing_field(tmp_path: Path) -> None:
    raw = {
        "base_model": "Qwen/Qwen3-1.7B",
        "lora": {
            # 'r' deliberately omitted
            "alpha": 64,
            "dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj"],
            "modules_to_save": None,
        },
    }
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(yaml.safe_dump(raw))

    with pytest.raises(KeyError, match="lora.r"):
        load_locked_spec(bad_yaml)


def test_load_locked_spec_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_locked_spec(tmp_path / "does_not_exist.yaml")


# ---------------------------------------------------------------------------
# verify() — happy paths
# ---------------------------------------------------------------------------

def test_verify_matching_config_in_memory(lora_yaml_path: Path) -> None:
    spec = load_locked_spec(lora_yaml_path)
    result = verify(GROUND_TRUTH_MATH_CONFIG, spec)

    assert result.passed
    assert all(fr.passed for fr in result.field_results)
    assert result.missing_fields == []
    assert "PASS" in result.summary


def test_verify_matching_config_from_file(tmp_path: Path, lora_yaml_path: Path) -> None:
    spec = load_locked_spec(lora_yaml_path)
    cfg_path = tmp_path / "adapter_config.json"
    cfg_path.write_text(json.dumps(GROUND_TRUTH_MATH_CONFIG))

    result = verify(cfg_path, spec)
    assert result.passed


def test_verify_target_modules_reordered_passes(lora_yaml_path: Path) -> None:
    spec = load_locked_spec(lora_yaml_path)
    cfg = copy.deepcopy(GROUND_TRUTH_MATH_CONFIG)
    # Reverse alphabetical, totally different order from spec
    cfg["target_modules"] = sorted(cfg["target_modules"], reverse=True)

    result = verify(cfg, spec)
    assert result.passed

    tm_result = next(fr for fr in result.field_results if fr.field == "target_modules")
    assert tm_result.passed
    assert "set" in tm_result.note.lower()


def test_verify_extra_peft_fields_ignored(lora_yaml_path: Path) -> None:
    spec = load_locked_spec(lora_yaml_path)
    cfg = dict(GROUND_TRUTH_MATH_CONFIG)
    cfg.update({
        "peft_type": "LORA",
        "peft_version": "0.11.1",
        "init_lora_weights": True,
        "alpha_pattern": {},
        "rank_pattern": {},
        "inference_mode": False,
    })

    result = verify(cfg, spec)
    assert result.passed
    assert "peft_version" in result.extra_fields
    assert "init_lora_weights" in result.extra_fields
    assert "alpha_pattern" in result.extra_fields


def test_verify_missing_modules_to_save_treated_as_null(lora_yaml_path: Path) -> None:
    """Older PEFT versions omit ``modules_to_save`` when unset. We treat that as None."""
    spec = load_locked_spec(lora_yaml_path)
    cfg = dict(GROUND_TRUTH_MATH_CONFIG)
    del cfg["modules_to_save"]

    result = verify(cfg, spec)
    assert result.passed
    assert "modules_to_save" not in result.missing_fields


# ---------------------------------------------------------------------------
# verify() — failure paths
# ---------------------------------------------------------------------------

def test_verify_wrong_r_fails(lora_yaml_path: Path) -> None:
    spec = load_locked_spec(lora_yaml_path)
    cfg = dict(GROUND_TRUTH_MATH_CONFIG)
    cfg["r"] = 16

    result = verify(cfg, spec)
    assert not result.passed
    r_result = next(fr for fr in result.field_results if fr.field == "r")
    assert not r_result.passed
    assert r_result.expected == 32
    assert r_result.actual == 16


def test_verify_wrong_alpha_fails(lora_yaml_path: Path) -> None:
    spec = load_locked_spec(lora_yaml_path)
    cfg = dict(GROUND_TRUTH_MATH_CONFIG)
    cfg["lora_alpha"] = 32

    result = verify(cfg, spec)
    assert not result.passed
    a_result = next(fr for fr in result.field_results if fr.field == "lora_alpha")
    assert not a_result.passed
    assert a_result.expected == 64
    assert a_result.actual == 32


def test_verify_modules_to_save_with_extra_modules_fails(lora_yaml_path: Path) -> None:
    """The critical case: ``modules_to_save: ["embed_tokens"]`` breaks merging."""
    spec = load_locked_spec(lora_yaml_path)
    cfg = dict(GROUND_TRUTH_MATH_CONFIG)
    cfg["modules_to_save"] = ["embed_tokens"]

    result = verify(cfg, spec)
    assert not result.passed
    mts = next(fr for fr in result.field_results if fr.field == "modules_to_save")
    assert not mts.passed
    assert mts.actual == ["embed_tokens"]


def test_verify_missing_load_bearing_field_fails(lora_yaml_path: Path) -> None:
    spec = load_locked_spec(lora_yaml_path)
    cfg = dict(GROUND_TRUTH_MATH_CONFIG)
    del cfg["bias"]

    result = verify(cfg, spec)
    assert not result.passed
    assert "bias" in result.missing_fields
    bias_result = next(fr for fr in result.field_results if fr.field == "bias")
    assert not bias_result.passed
    assert "missing" in bias_result.note.lower()


# ---------------------------------------------------------------------------
# SpecMismatchError
# ---------------------------------------------------------------------------

def test_spec_mismatch_error_message_lists_failures() -> None:
    bad_r = VerifyResult(
        passed=False,
        field_results=[
            FieldResult(field="r", expected=32, actual=16, passed=False),
        ],
    )
    bad_alpha = VerifyResult(
        passed=False,
        field_results=[
            FieldResult(field="lora_alpha", expected=64, actual=32, passed=False),
        ],
    )
    err = SpecMismatchError({"safety": bad_r, "multilingual": bad_alpha})

    msg = str(err)
    assert "safety" in msg
    assert "multilingual" in msg
    assert "r" in msg
    assert "lora_alpha" in msg

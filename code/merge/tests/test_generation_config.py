"""Tests for :mod:`merge.generation_config` — pure dict/file handling, no torch."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from merge.generation_config import (
    QWEN3_DEFAULTS,
    load_generation_config,
    make_generation_config,
)


# ---------------------------------------------------------------------------
# make_generation_config
# ---------------------------------------------------------------------------

def test_make_generation_config_defaults() -> None:
    """Default call produces a valid config dict matching the project schema."""
    config = make_generation_config()
    assert config["bos_token_id"] == 151643
    assert config["eos_token_id"] == [151645, 151643]
    assert config["pad_token_id"] == 151643
    assert config["do_sample"] is True
    assert config["temperature"] == 0.7
    assert config["top_p"] == 0.8
    assert config["top_k"] == 20
    assert config["max_new_tokens"] == 16384
    assert config["transformers_version"] == "4.51.0"


def test_make_generation_config_custom_values() -> None:
    """Custom values override the defaults; fixed fields preserved."""
    config = make_generation_config(
        temperature=0.0, top_p=1.0, top_k=50, max_new_tokens=2048
    )
    assert config["temperature"] == 0.0
    assert config["top_p"] == 1.0
    assert config["top_k"] == 50
    assert config["max_new_tokens"] == 2048
    assert config["bos_token_id"] == 151643
    assert config["eos_token_id"] == [151645, 151643]


def test_make_generation_config_validates_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        make_generation_config(temperature=-0.1)
    with pytest.raises(ValueError, match="temperature"):
        make_generation_config(temperature=3.0)


def test_make_generation_config_validates_top_p() -> None:
    with pytest.raises(ValueError, match="top_p"):
        make_generation_config(top_p=-0.1)
    with pytest.raises(ValueError, match="top_p"):
        make_generation_config(top_p=1.5)


def test_make_generation_config_validates_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        make_generation_config(top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        make_generation_config(top_k=-1)


def test_make_generation_config_validates_max_new_tokens() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        make_generation_config(max_new_tokens=0)


def test_qwen3_defaults_match_constructor() -> None:
    """The QWEN3_DEFAULTS table matches the constructor's default args."""
    config = make_generation_config()
    for key, value in QWEN3_DEFAULTS.items():
        assert config[key] == value, f"defaults disagree at {key!r}"


# ---------------------------------------------------------------------------
# load_generation_config
# ---------------------------------------------------------------------------

def test_load_generation_config_falls_back_to_qwen3_defaults(tmp_path: Path) -> None:
    """No files anywhere → Qwen3 defaults."""
    result = load_generation_config(merged_adapter_dir=tmp_path, repo_root=None)
    assert result["temperature"] == 0.7
    assert result["top_p"] == 0.8
    assert result["top_k"] == 20


def test_load_generation_config_reads_adapter_dir(tmp_path: Path) -> None:
    """Adapter dir has gen config → returned."""
    custom = make_generation_config(temperature=0.0)
    with open(tmp_path / "generation_config.json", "w") as f:
        json.dump(custom, f)
    result = load_generation_config(merged_adapter_dir=tmp_path)
    assert result["temperature"] == 0.0


def test_load_generation_config_adapter_wins_over_repo_root(tmp_path: Path) -> None:
    """Both files present → adapter wins."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    adapter_config = make_generation_config(temperature=0.0)
    with open(adapter_dir / "generation_config.json", "w") as f:
        json.dump(adapter_config, f)

    repo_config = make_generation_config(temperature=0.9)
    with open(repo_root / "generation_config.json", "w") as f:
        json.dump(repo_config, f)

    result = load_generation_config(
        merged_adapter_dir=adapter_dir, repo_root=repo_root
    )
    assert result["temperature"] == 0.0


def test_load_generation_config_repo_root_fallback(tmp_path: Path) -> None:
    """Only repo root has gen config → that's used."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    repo_config = make_generation_config(temperature=0.42)
    with open(repo_root / "generation_config.json", "w") as f:
        json.dump(repo_config, f)

    result = load_generation_config(
        merged_adapter_dir=adapter_dir, repo_root=repo_root
    )
    assert result["temperature"] == 0.42


def test_load_generation_config_corrupt_file_falls_through(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Corrupt JSON → log warning, fall through to next priority."""
    import logging

    (tmp_path / "generation_config.json").write_text("{not valid json")
    with caplog.at_level(logging.WARNING, logger="merge.generation_config"):
        result = load_generation_config(merged_adapter_dir=tmp_path)
    assert result["temperature"] == 0.7
    assert any("Failed to load" in r.message for r in caplog.records)


def test_load_generation_config_no_dirs_given() -> None:
    """Both args None → Qwen3 defaults."""
    result = load_generation_config(merged_adapter_dir=None, repo_root=None)
    assert result["temperature"] == 0.7
    assert result["bos_token_id"] == 151643

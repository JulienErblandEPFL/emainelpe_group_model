"""Tests for ``scripts/publish.py``.

The script is torch-free by design — only ``huggingface_hub`` is heavy,
and we inject the upload + create_repo callables so tests run without
any HF deps installed and without ever touching the Hub.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "publish.py"


def _load_publish() -> Any:
    spec = importlib.util.spec_from_file_location("publish_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


publish_mod = _load_publish()


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_valid_model_dir(tmp_path: Path) -> Path:
    """Build a minimal full-HF-format model directory layout."""
    model_dir = tmp_path / "merged"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "qwen3"}')
    (model_dir / "model.safetensors").write_text("fake-weights")
    (model_dir / "tokenizer.json").write_text("{}")
    (model_dir / "tokenizer_config.json").write_text("{}")
    (model_dir / "chat_template.jinja").write_text("{{ messages }}")
    # Pre-existing generation_config.json from save_pretrained so we
    # can assert the backup path.
    (model_dir / "generation_config.json").write_text(
        json.dumps({"temperature": 0.7, "top_p": 0.95, "do_sample": True}, indent=2)
    )
    return model_dir


def _parse(argv: list[str]) -> Any:
    return publish_mod.build_parser().parse_args(argv)


def _base_argv(model_dir: Path, **extra: str) -> list[str]:
    argv = [
        "--model-dir", str(model_dir),
        "--repo-id", "cs-552-2026-emainelpe/group_model",
    ]
    for k, v in extra.items():
        argv.extend([f"--{k.replace('_', '-')}", v])
    return argv


# ---------------------------------------------------------------------------
# Generation config rewrite shape
# ---------------------------------------------------------------------------

def test_build_winning_generation_config_uses_winning_params() -> None:
    cfg = publish_mod.build_winning_generation_config(
        temperature=0.5, top_p=0.8, top_k=20, max_new_tokens=2048,
    )
    assert cfg["temperature"] == 0.5
    assert cfg["top_p"] == 0.8
    assert cfg["top_k"] == 20
    assert cfg["max_new_tokens"] == 2048
    assert cfg["do_sample"] is True
    # Structural fields from the project description, locked.
    assert cfg["bos_token_id"] == 151643
    assert cfg["eos_token_id"] == [151645, 151643]
    assert cfg["pad_token_id"] == 151643


def test_rewrite_generation_config_writes_new_and_backs_up_old(tmp_path: Path) -> None:
    model_dir = _write_valid_model_dir(tmp_path)
    original = json.loads((model_dir / "generation_config.json").read_text())
    new_cfg = publish_mod.build_winning_generation_config(
        temperature=0.5, top_p=0.8, top_k=20, max_new_tokens=2048,
    )
    publish_mod.rewrite_generation_config(model_dir, new_cfg)

    backup = model_dir / "generation_config.json.bak"
    assert backup.exists(), "missing .bak — original is not recoverable"
    assert json.loads(backup.read_text()) == original

    written = json.loads((model_dir / "generation_config.json").read_text())
    assert written == new_cfg


def test_rewrite_generation_config_no_backup_when_no_original(tmp_path: Path) -> None:
    model_dir = _write_valid_model_dir(tmp_path)
    (model_dir / "generation_config.json").unlink()

    new_cfg = publish_mod.build_winning_generation_config(
        temperature=0.5, top_p=0.8, top_k=20, max_new_tokens=2048,
    )
    publish_mod.rewrite_generation_config(model_dir, new_cfg)

    assert (model_dir / "generation_config.json").exists()
    assert not (model_dir / "generation_config.json.bak").exists()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_model_dir_passes_on_valid_layout(tmp_path: Path) -> None:
    model_dir = _write_valid_model_dir(tmp_path)
    assert publish_mod.validate_model_dir(model_dir) == []


def test_validate_model_dir_accepts_sharded_weights(tmp_path: Path) -> None:
    model_dir = _write_valid_model_dir(tmp_path)
    (model_dir / "model.safetensors").unlink()
    (model_dir / "model.safetensors.index.json").write_text("{}")
    (model_dir / "model-00001-of-00002.safetensors").write_text("shard1")
    (model_dir / "model-00002-of-00002.safetensors").write_text("shard2")
    assert publish_mod.validate_model_dir(model_dir) == []


def test_validate_model_dir_flags_missing_config(tmp_path: Path) -> None:
    model_dir = _write_valid_model_dir(tmp_path)
    (model_dir / "config.json").unlink()
    errors = publish_mod.validate_model_dir(model_dir)
    assert any("config.json" in e for e in errors)


def test_validate_model_dir_flags_missing_chat_template(tmp_path: Path) -> None:
    model_dir = _write_valid_model_dir(tmp_path)
    (model_dir / "chat_template.jinja").unlink()
    errors = publish_mod.validate_model_dir(model_dir)
    assert any("chat_template.jinja" in e for e in errors)


def test_validate_model_dir_flags_missing_weights(tmp_path: Path) -> None:
    model_dir = _write_valid_model_dir(tmp_path)
    (model_dir / "model.safetensors").unlink()
    errors = publish_mod.validate_model_dir(model_dir)
    assert any("weights" in e.lower() for e in errors)


def test_validate_model_dir_flags_missing_tokenizer(tmp_path: Path) -> None:
    model_dir = _write_valid_model_dir(tmp_path)
    (model_dir / "tokenizer.json").unlink()
    (model_dir / "tokenizer_config.json").unlink()
    errors = publish_mod.validate_model_dir(model_dir)
    assert any("tokenizer" in e.lower() for e in errors)


def test_validate_model_dir_missing_dir(tmp_path: Path) -> None:
    errors = publish_mod.validate_model_dir(tmp_path / "nope")
    assert errors and "does not exist" in errors[0]


# ---------------------------------------------------------------------------
# collect_upload_files: excludes .bak
# ---------------------------------------------------------------------------

def test_collect_upload_files_excludes_bak(tmp_path: Path) -> None:
    model_dir = _write_valid_model_dir(tmp_path)
    (model_dir / "generation_config.json.bak").write_text("old")
    files = publish_mod.collect_upload_files(model_dir)
    rel_names = [str(rel) for rel, _ in files]
    assert "generation_config.json.bak" not in rel_names
    assert "generation_config.json" in rel_names


# ---------------------------------------------------------------------------
# publish() dry-run: no upload, no rewrite
# ---------------------------------------------------------------------------

def test_publish_dry_run_does_not_upload_or_rewrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir = _write_valid_model_dir(tmp_path)
    args = _parse(_base_argv(model_dir))
    assert args.confirm is False

    original_gen = (model_dir / "generation_config.json").read_text()

    upload_calls: list[Any] = []
    create_calls: list[Any] = []

    def fake_upload(**kwargs: Any) -> Any:
        upload_calls.append(kwargs)

    def fake_create(**kwargs: Any) -> Any:
        create_calls.append(kwargs)

    code = publish_mod.publish(
        args,
        upload_callable=fake_upload,
        create_repo_callable=fake_create,
    )

    assert code == 0
    assert upload_calls == [], "dry-run must NOT call upload_folder"
    assert create_calls == [], "dry-run must NOT call create_repo"

    # Dry-run must NOT touch the model directory at all.
    assert (model_dir / "generation_config.json").read_text() == original_gen
    assert not (model_dir / "generation_config.json.bak").exists()

    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "cs-552-2026-emainelpe/group_model" in out
    # The new gen config should be visible in the plan output.
    assert '"temperature": 0.5' in out
    assert '"top_p": 0.8' in out


# ---------------------------------------------------------------------------
# publish() --confirm: upload happens, gen config rewritten
# ---------------------------------------------------------------------------

def test_publish_confirm_calls_upload_and_rewrites(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir = _write_valid_model_dir(tmp_path)
    args = _parse(_base_argv(model_dir) + ["--confirm"])
    assert args.confirm is True

    upload_calls: list[dict[str, Any]] = []
    create_calls: list[dict[str, Any]] = []

    def fake_upload(**kwargs: Any) -> Any:
        upload_calls.append(kwargs)

    def fake_create(**kwargs: Any) -> Any:
        create_calls.append(kwargs)

    code = publish_mod.publish(
        args,
        upload_callable=fake_upload,
        create_repo_callable=fake_create,
    )

    assert code == 0
    assert len(create_calls) == 1
    assert create_calls[0] == {
        "repo_id": "cs-552-2026-emainelpe/group_model",
        "exist_ok": True,
    }
    assert len(upload_calls) == 1
    assert upload_calls[0]["folder_path"] == str(model_dir)
    assert upload_calls[0]["repo_id"] == "cs-552-2026-emainelpe/group_model"
    assert "ties" in upload_calls[0]["commit_message"].lower() or (
        "winner" in upload_calls[0]["commit_message"].lower()
    )

    # Generation config rewritten to the winning params.
    written = json.loads((model_dir / "generation_config.json").read_text())
    assert written["temperature"] == 0.5
    assert written["top_p"] == 0.8
    assert written["top_k"] == 20
    assert written["max_new_tokens"] == 16384
    # Backup of the original.
    assert (model_dir / "generation_config.json.bak").exists()

    out = capsys.readouterr().out
    assert "Published:" in out
    assert "huggingface.co/cs-552-2026-emainelpe/group_model" in out


def test_publish_confirm_respects_overridden_sampling_params(tmp_path: Path) -> None:
    model_dir = _write_valid_model_dir(tmp_path)
    args = _parse(
        _base_argv(model_dir, temperature="0.3", top_p="0.7", top_k="15")
        + ["--confirm"]
    )

    def noop(**_kwargs: Any) -> Any:
        return None

    publish_mod.publish(
        args, upload_callable=noop, create_repo_callable=noop,
    )
    written = json.loads((model_dir / "generation_config.json").read_text())
    assert written["temperature"] == 0.3
    assert written["top_p"] == 0.7
    assert written["top_k"] == 15


# ---------------------------------------------------------------------------
# publish() validation failure
# ---------------------------------------------------------------------------

def test_publish_returns_2_on_missing_model_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    args = _parse(_base_argv(tmp_path / "nope") + ["--confirm"])
    upload_calls: list[Any] = []

    def fake_upload(**kwargs: Any) -> Any:
        upload_calls.append(kwargs)

    code = publish_mod.publish(
        args,
        upload_callable=fake_upload,
        create_repo_callable=fake_upload,
    )
    assert code == 2
    assert upload_calls == [], "validation failure must short-circuit before upload"
    err = capsys.readouterr().err
    assert "does not exist" in err


def test_publish_returns_2_on_incomplete_model_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    model_dir = _write_valid_model_dir(tmp_path)
    (model_dir / "chat_template.jinja").unlink()
    args = _parse(_base_argv(model_dir) + ["--confirm"])

    upload_calls: list[Any] = []

    def fake_upload(**kwargs: Any) -> Any:
        upload_calls.append(kwargs)

    code = publish_mod.publish(
        args,
        upload_callable=fake_upload,
        create_repo_callable=fake_upload,
    )
    assert code == 2
    assert upload_calls == []


# ---------------------------------------------------------------------------
# argparse: --repo-id required, no default
# ---------------------------------------------------------------------------

def test_argparse_requires_repo_id(tmp_path: Path) -> None:
    model_dir = _write_valid_model_dir(tmp_path)
    parser = publish_mod.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model-dir", str(model_dir)])


def test_argparse_requires_model_dir() -> None:
    parser = publish_mod.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--repo-id", "cs-552-2026-emainelpe/group_model"])

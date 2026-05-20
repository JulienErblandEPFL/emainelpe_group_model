"""Tests for the temperature-sweep CLI at ``scripts/eval_sweep.py``.

The script is torch-free by design: vLLM and torch are only touched by the
default factory functions, which we never call. Tests inject mocks for the
``eval_callable`` and ``config_factory`` so the sweep core runs as pure
Python with no ML deps.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "eval_sweep.py"


def _load_eval_sweep() -> Any:
    """Import the CLI script by file path so tests don't depend on a package install."""
    spec = importlib.util.spec_from_file_location("eval_sweep_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sweep_mod = _load_eval_sweep()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_adapter_dir(tmp_path: Path) -> Path:
    """Create a fake merged-model directory with the two files validate_args checks.

    After the Day 7 refactor, ``merged_adapter_dir`` is a full HF-format
    model directory (config.json + model.safetensors), not a PEFT adapter.
    """
    adapter_dir = tmp_path / "merged"
    adapter_dir.mkdir()
    (adapter_dir / "config.json").write_text("{}")
    (adapter_dir / "model.safetensors").write_bytes(b"")
    return adapter_dir


def _make_valid_validation_dir(tmp_path: Path) -> Path:
    """Create a fake validation_samples directory with the 4 expected JSONLs."""
    vs_dir = tmp_path / "validation"
    vs_dir.mkdir()
    for bench in sweep_mod.CANONICAL_BENCHMARKS:
        (vs_dir / f"{bench}.jsonl").write_text('{"prompt": "q", "answer": "a"}\n')
    return vs_dir


def _make_chat_template(tmp_path: Path) -> Path:
    path = tmp_path / "chat.jinja"
    path.write_text("{{ messages }}")
    return path


def _build_args(tmp_path: Path, **overrides: Any) -> Any:
    """Build a fully valid parsed-args namespace and let callers override fields."""
    adapter_dir = _make_valid_adapter_dir(tmp_path)
    vs_dir = _make_valid_validation_dir(tmp_path)
    chat_template = _make_chat_template(tmp_path)
    output_dir = tmp_path / "out"
    parser = sweep_mod.build_parser()
    args = parser.parse_args(
        [
            "--merged-adapter-dir", str(adapter_dir),
            "--output-dir", str(output_dir),
            "--temperatures", "0.3", "0.7",
            "--validation-samples-dir", str(vs_dir),
            "--chat-template-path", str(chat_template),
        ]
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _fake_benchmark_result(
    *, pass_at_1: float, pass_at_8: float, n_problems: int, n_pass8_failed: int
) -> Any:
    """Duck-typed stand-in for :class:`merge.eval_all.BenchmarkResult`."""
    return types.SimpleNamespace(
        pass_at_1=pass_at_1,
        pass_at_8=pass_at_8,
        n_problems=n_problems,
        n_pass8_failed=n_pass8_failed,
    )


def _ok_benchmark_results() -> dict[str, Any]:
    """Build a {benchmark: result} dict that looks like a successful eval run."""
    return {
        bench: _fake_benchmark_result(
            pass_at_1=0.5, pass_at_8=0.9, n_problems=10, n_pass8_failed=1
        )
        for bench in sweep_mod.CANONICAL_BENCHMARKS
    }


def _config_stub_factory(temperature: float, _args: Any) -> Any:
    return types.SimpleNamespace(temperature=temperature)


# ---------------------------------------------------------------------------
# validate_args: filesystem checks
# ---------------------------------------------------------------------------

def test_validate_args_happy_path(tmp_path: Path) -> None:
    args = _build_args(tmp_path)
    assert sweep_mod.validate_args(args) == []


def test_validate_args_missing_adapter_dir(tmp_path: Path) -> None:
    args = _build_args(tmp_path, merged_adapter_dir=tmp_path / "does_not_exist")
    errors = sweep_mod.validate_args(args)
    assert any("not a directory" in e for e in errors)


def test_validate_args_missing_config_json(tmp_path: Path) -> None:
    args = _build_args(tmp_path)
    (args.merged_adapter_dir / "config.json").unlink()
    errors = sweep_mod.validate_args(args)
    assert any("config.json missing" in e for e in errors)


def test_validate_args_missing_safetensors(tmp_path: Path) -> None:
    args = _build_args(tmp_path)
    (args.merged_adapter_dir / "model.safetensors").unlink()
    errors = sweep_mod.validate_args(args)
    assert any("model.safetensors" in e and "missing" in e for e in errors)


def test_validate_args_accepts_sharded_safetensors(tmp_path: Path) -> None:
    """A sharded model (model.safetensors.index.json + model-0000N-of-M shards)
    is the alternative valid layout transformers writes for >5GB models."""
    args = _build_args(tmp_path)
    (args.merged_adapter_dir / "model.safetensors").unlink()
    (args.merged_adapter_dir / "model.safetensors.index.json").write_text("{}")
    assert sweep_mod.validate_args(args) == []


def test_validate_args_missing_validation_dir(tmp_path: Path) -> None:
    args = _build_args(tmp_path, validation_samples_dir=tmp_path / "missing")
    errors = sweep_mod.validate_args(args)
    assert any("--validation-samples-dir not a directory" in e for e in errors)


def test_validate_args_missing_validation_jsonl(tmp_path: Path) -> None:
    args = _build_args(tmp_path)
    (args.validation_samples_dir / "safety.jsonl").unlink()
    errors = sweep_mod.validate_args(args)
    assert any("missing validation file" in e and "safety.jsonl" in e for e in errors)


# ---------------------------------------------------------------------------
# validate_args: numeric checks
# ---------------------------------------------------------------------------

def test_validate_args_rejects_temperature_zero(tmp_path: Path) -> None:
    args = _build_args(tmp_path, temperatures=[0.0, 0.7])
    errors = sweep_mod.validate_args(args)
    # The rejection message must explain WHY (vLLM n>1 + greedy) so a future
    # reader doesn't "fix" it by silently swapping to n=1.
    matching = [e for e in errors if "temperature 0.0" in e or "0.0 not allowed" in e]
    assert matching, errors
    joined = " ".join(matching)
    assert "vLLM" in joined and "n>1" in joined and "n=1" in joined


def test_validate_args_rejects_negative_temperature(tmp_path: Path) -> None:
    args = _build_args(tmp_path, temperatures=[-0.1])
    errors = sweep_mod.validate_args(args)
    assert any("temperature -0.1" in e for e in errors)


def test_validate_args_rejects_empty_temperatures(tmp_path: Path) -> None:
    args = _build_args(tmp_path, temperatures=[])
    errors = sweep_mod.validate_args(args)
    assert any("at least one value" in e for e in errors)


def test_validate_args_rejects_n_below_one(tmp_path: Path) -> None:
    args = _build_args(tmp_path, n=0)
    errors = sweep_mod.validate_args(args)
    assert any("--n must be >= 1" in e for e in errors)


def test_validate_args_rejects_max_tokens_below_one(tmp_path: Path) -> None:
    args = _build_args(tmp_path, max_tokens=0)
    errors = sweep_mod.validate_args(args)
    assert any("--max-tokens must be >= 1" in e for e in errors)


def test_validate_args_rejects_top_p_out_of_range(tmp_path: Path) -> None:
    args = _build_args(tmp_path, top_p=0.0)
    assert any("--top-p" in e for e in sweep_mod.validate_args(args))
    args.top_p = 1.5
    assert any("--top-p" in e for e in sweep_mod.validate_args(args))


def test_validate_args_rejects_top_k_below_one(tmp_path: Path) -> None:
    args = _build_args(tmp_path, top_k=0)
    errors = sweep_mod.validate_args(args)
    assert any("--top-k must be >= 1" in e for e in errors)


# ---------------------------------------------------------------------------
# SweepResultRow JSON round-trip
# ---------------------------------------------------------------------------

def test_sweep_result_row_json_round_trip() -> None:
    original = sweep_mod.SweepResultRow(
        temperature=0.5,
        status="ok",
        duration_seconds=12.3,
        pass_at_1={"math": 0.4, "safety": 0.9},
        pass_at_8={"math": 0.7, "safety": 1.0},
        n_problems={"math": 10, "safety": 10},
        n_pass8_failed={"math": 3, "safety": 0},
        error=None,
    )
    payload = json.dumps(dataclasses.asdict(original))
    restored_dict = json.loads(payload)
    restored = sweep_mod.SweepResultRow(**restored_dict)
    assert restored == original


def test_sweep_result_row_defaults_serialize_cleanly() -> None:
    row = sweep_mod.SweepResultRow(
        temperature=0.3, status="failed", duration_seconds=0.0, error="boom"
    )
    d = dataclasses.asdict(row)
    # Empty dicts must survive JSON, not be replaced by None.
    assert d["pass_at_1"] == {} and d["pass_at_8"] == {}
    parsed = json.loads(json.dumps(d))
    assert parsed["error"] == "boom"


# ---------------------------------------------------------------------------
# _benchmark_results_to_row: aggregation
# ---------------------------------------------------------------------------

def test_benchmark_results_to_row_aggregates_all_benchmarks() -> None:
    results = {
        "math": _fake_benchmark_result(
            pass_at_1=0.4, pass_at_8=0.8, n_problems=10, n_pass8_failed=2
        ),
        "safety": _fake_benchmark_result(
            pass_at_1=1.0, pass_at_8=1.0, n_problems=10, n_pass8_failed=0
        ),
    }
    row = sweep_mod._benchmark_results_to_row(
        temperature=0.5, duration_seconds=42.0, benchmark_results=results
    )
    assert row.status == "ok"
    assert row.temperature == 0.5
    assert row.duration_seconds == 42.0
    assert row.pass_at_1 == {"math": 0.4, "safety": 1.0}
    assert row.pass_at_8 == {"math": 0.8, "safety": 1.0}
    assert row.n_problems == {"math": 10, "safety": 10}
    assert row.n_pass8_failed == {"math": 2, "safety": 0}


def test_benchmark_results_to_row_casts_types() -> None:
    """The row coerces numerics — guards against numpy/torch scalars from real eval."""
    results = {
        "math": _fake_benchmark_result(
            pass_at_1="0.4", pass_at_8="0.8", n_problems="10", n_pass8_failed="2"
        ),
    }
    row = sweep_mod._benchmark_results_to_row(
        temperature=0.5, duration_seconds=1.0, benchmark_results=results
    )
    assert isinstance(row.pass_at_1["math"], float)
    assert isinstance(row.n_problems["math"], int)


# ---------------------------------------------------------------------------
# run_sweep: happy path
# ---------------------------------------------------------------------------

def test_run_sweep_happy_path_writes_incremental_results(tmp_path: Path) -> None:
    args = _build_args(tmp_path)
    call_log: list[float] = []

    def stub_runner(**kwargs: Any) -> dict[str, Any]:
        # output_dir is per-temperature; capture it so we can assert the script
        # actually created the subdir before calling the runner.
        assert kwargs["output_dir"].is_dir()
        call_log.append(kwargs["config"].temperature)
        return _ok_benchmark_results()

    rows, code = sweep_mod.run_sweep(args, stub_runner, _config_stub_factory)

    assert code == 0
    assert call_log == [0.3, 0.7]
    assert len(rows) == 2
    assert all(r.status == "ok" for r in rows)

    # Per-temperature subdirs created
    assert (args.output_dir / "T_0.3").is_dir()
    assert (args.output_dir / "T_0.7").is_dir()

    # sweep_results.json was written and contains all rows
    sweep_path = args.output_dir / "sweep_results.json"
    assert sweep_path.exists()
    payload = json.loads(sweep_path.read_text())
    assert [p["temperature"] for p in payload] == [0.3, 0.7]
    assert all(p["status"] == "ok" for p in payload)


def test_run_sweep_writes_after_each_temperature(tmp_path: Path) -> None:
    """Incremental write: after temp 1, the JSON should already have row 1."""
    args = _build_args(tmp_path)
    sweep_path = args.output_dir / "sweep_results.json"
    snapshots: list[int] = []

    def stub_runner(**kwargs: Any) -> dict[str, Any]:
        # On entry to call N, the sweep_results.json on disk should hold N-1 rows.
        if sweep_path.exists():
            snapshots.append(len(json.loads(sweep_path.read_text())))
        else:
            snapshots.append(0)
        return _ok_benchmark_results()

    sweep_mod.run_sweep(args, stub_runner, _config_stub_factory)

    # Before temp 0.3: 0 rows on disk; before temp 0.7: 1 row on disk.
    assert snapshots == [0, 1]


# ---------------------------------------------------------------------------
# run_sweep: resilience to per-temperature exceptions
# ---------------------------------------------------------------------------

def test_run_sweep_records_failure_and_continues(tmp_path: Path) -> None:
    args = _build_args(tmp_path, temperatures=[0.3, 0.5, 0.7])
    call_log: list[float] = []

    def stub_runner(**kwargs: Any) -> dict[str, Any]:
        t = kwargs["config"].temperature
        call_log.append(t)
        if t == 0.5:
            raise RuntimeError("simulated vLLM OOM at T=0.5")
        return _ok_benchmark_results()

    rows, code = sweep_mod.run_sweep(args, stub_runner, _config_stub_factory)

    # All three temperatures attempted, even though the middle one threw.
    assert call_log == [0.3, 0.5, 0.7]
    assert code == 1  # at least one failure -> exit 1
    statuses = [r.status for r in rows]
    assert statuses == ["ok", "failed", "ok"]
    assert rows[1].error is not None
    assert "simulated vLLM OOM" in rows[1].error
    # Traceback is captured, not just the message.
    assert "RuntimeError" in rows[1].error

    # The failed row's pass_at_1 / pass_at_8 dicts stay empty.
    assert rows[1].pass_at_1 == {}
    assert rows[1].pass_at_8 == {}

    # sweep_results.json survives the failure and contains all three rows.
    sweep_path = args.output_dir / "sweep_results.json"
    payload = json.loads(sweep_path.read_text())
    assert [p["status"] for p in payload] == ["ok", "failed", "ok"]
    assert payload[1]["error"] is not None


# ---------------------------------------------------------------------------
# main(): exit-code wiring
# ---------------------------------------------------------------------------

def test_main_returns_2_on_validation_error(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    # Build args via the parser but point at a missing dir to trigger validation failure.
    bad_argv = [
        "--merged-adapter-dir", str(tmp_path / "missing"),
        "--output-dir", str(tmp_path / "out"),
        "--temperatures", "0.5",
    ]
    code = sweep_mod.main(bad_argv)
    assert code == 2
    err = capsys.readouterr().err
    assert "error:" in err

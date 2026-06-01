"""Tests for the Stage 5c.2 bake-off CLI at ``scripts/run_bakeoff.py``.

The script is torch-free by design: torch/vllm/transformers/peft are only
touched by the default factory functions, which we never call. Tests
inject stubs for the merge callable, the evaluation callable, and the
config factory so the bake-off core runs as pure Python with no ML deps.

Validation that hits ``merge.verify_spec`` IS exercised: that module is
torch-free (JSON + YAML only) and provides the locked-spec gate the
script depends on.
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
# Module loader (mirror eval_sweep tests)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "run_bakeoff.py"


def _load_run_bakeoff() -> Any:
    spec = importlib.util.spec_from_file_location("run_bakeoff_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bakeoff_mod = _load_run_bakeoff()


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_LOCKED_FIELDS = {
    "base_model_name_or_path": "Qwen/Qwen3-1.7B",
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "modules_to_save": None,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}


def _write_adapter_config(adapter_dir: Path, **overrides: Any) -> None:
    """Write a passing locked-spec adapter_config.json for one domain."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(_LOCKED_FIELDS, **overrides)
    with (adapter_dir / "adapter_config.json").open("w") as f:
        json.dump(cfg, f)


def _make_valid_adapters_dir(tmp_path: Path, **per_domain_overrides: Any) -> Path:
    """Build 4 spec-compliant adapter dirs under tmp_path/loras/."""
    loras = tmp_path / "loras"
    for domain in bakeoff_mod.CANONICAL_DOMAINS:
        _write_adapter_config(loras / domain, **per_domain_overrides.get(domain, {}))
    return loras


def _make_valid_validation_dir(tmp_path: Path) -> Path:
    vs = tmp_path / "validation"
    vs.mkdir()
    for bench in bakeoff_mod.CANONICAL_BENCHMARKS:
        (vs / f"{bench}.jsonl").write_text('{"prompt": "q", "answer": "a"}\n')
    return vs


def _make_chat_template(tmp_path: Path) -> Path:
    p = tmp_path / "chat.jinja"
    p.write_text("{{ messages }}")
    return p


def _build_args(tmp_path: Path, **overrides: Any) -> Any:
    adapters_dir = _make_valid_adapters_dir(tmp_path)
    vs = _make_valid_validation_dir(tmp_path)
    chat = _make_chat_template(tmp_path)
    output_dir = tmp_path / "out"
    parser = bakeoff_mod.build_parser()
    argv = [
        "--adapters-dir", str(adapters_dir),
        "--output-dir", str(output_dir),
        "--validation-samples-dir", str(vs),
        "--chat-template-path", str(chat),
    ]
    args = parser.parse_args(argv)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _fake_benchmark_result(p1: float, p8: float, n: int = 10, failed: int = 0) -> Any:
    return types.SimpleNamespace(
        pass_at_1=p1, pass_at_8=p8, n_problems=n, n_pass8_failed=failed,
    )


def _ok_benchmark_results(p1: float = 0.5, p8: float = 0.9) -> dict[str, Any]:
    return {b: _fake_benchmark_result(p1, p8) for b in bakeoff_mod.CANONICAL_BENCHMARKS}


def _config_stub(temperature: float, _args: Any) -> Any:
    return types.SimpleNamespace(temperature=temperature)


# ---------------------------------------------------------------------------
# validate_args
# ---------------------------------------------------------------------------

def test_validate_args_happy_path(tmp_path: Path) -> None:
    args = _build_args(tmp_path)
    assert bakeoff_mod.validate_args(args) == []


def test_validate_args_missing_adapters_dir(tmp_path: Path) -> None:
    args = _build_args(tmp_path, adapters_dir=tmp_path / "nope")
    errors = bakeoff_mod.validate_args(args)
    assert any("not a directory" in e for e in errors)


def test_validate_args_missing_one_domain(tmp_path: Path) -> None:
    args = _build_args(tmp_path)
    import shutil
    shutil.rmtree(args.adapters_dir / "safety")
    errors = bakeoff_mod.validate_args(args)
    assert any("missing adapter subdir" in e and "safety" in e for e in errors)


def test_validate_args_missing_validation_dir(tmp_path: Path) -> None:
    args = _build_args(tmp_path, validation_samples_dir=tmp_path / "nope")
    errors = bakeoff_mod.validate_args(args)
    assert any("--validation-samples-dir not a directory" in e for e in errors)


def test_validate_args_missing_validation_jsonl(tmp_path: Path) -> None:
    args = _build_args(tmp_path)
    (args.validation_samples_dir / "multilingual.jsonl").unlink()
    errors = bakeoff_mod.validate_args(args)
    assert any("multilingual.jsonl" in e for e in errors)


def test_validate_args_rejects_unknown_method(tmp_path: Path) -> None:
    args = _build_args(tmp_path, methods=["uniform", "not_a_method"])
    errors = bakeoff_mod.validate_args(args)
    assert any("unknown method 'not_a_method'" in e for e in errors)


def test_validate_args_rejects_empty_methods(tmp_path: Path) -> None:
    args = _build_args(tmp_path, methods=[])
    errors = bakeoff_mod.validate_args(args)
    assert any("--methods requires at least one value" in e for e in errors)


def test_validate_args_rejects_temperature_zero(tmp_path: Path) -> None:
    args = _build_args(tmp_path, temperatures=[0.0, 0.7])
    errors = bakeoff_mod.validate_args(args)
    joined = " ".join(errors)
    assert "temperature 0.0" in joined
    assert "n>1" in joined and "n=1" in joined


def test_validate_args_rejects_empty_temperatures(tmp_path: Path) -> None:
    args = _build_args(tmp_path, temperatures=[])
    errors = bakeoff_mod.validate_args(args)
    assert any("--temperatures requires at least one value" in e for e in errors)


def test_validate_args_rejects_top_p_out_of_range(tmp_path: Path) -> None:
    args = _build_args(tmp_path, top_p=1.5)
    assert any("--top-p" in e for e in bakeoff_mod.validate_args(args))


def test_validate_args_rejects_adamerging_max_steps_below_one(tmp_path: Path) -> None:
    args = _build_args(tmp_path, adamerging_max_steps=0)
    assert any("--adamerging-max-steps" in e for e in bakeoff_mod.validate_args(args))


# ---------------------------------------------------------------------------
# verify_locked_specs (uses real merge.verify_spec — torch-free)
# ---------------------------------------------------------------------------

def test_verify_locked_specs_all_pass(tmp_path: Path) -> None:
    adapters = _make_valid_adapters_dir(tmp_path)
    locked_spec = dict(_LOCKED_FIELDS)
    errors = bakeoff_mod.verify_locked_specs(adapters, locked_spec)
    assert errors == []


def test_verify_locked_specs_catches_diverged_adapter(tmp_path: Path) -> None:
    adapters = _make_valid_adapters_dir(
        tmp_path, multilingual={"r": 16},
    )
    locked_spec = dict(_LOCKED_FIELDS)
    errors = bakeoff_mod.verify_locked_specs(adapters, locked_spec)
    assert any("multilingual" in e for e in errors)


def test_verify_locked_specs_catches_missing_config(tmp_path: Path) -> None:
    adapters = _make_valid_adapters_dir(tmp_path)
    (adapters / "safety" / "adapter_config.json").unlink()
    locked_spec = dict(_LOCKED_FIELDS)
    errors = bakeoff_mod.verify_locked_specs(adapters, locked_spec)
    assert any("safety" in e and "adapter_config.json missing" in e for e in errors)


# ---------------------------------------------------------------------------
# build_method_kwargs
# ---------------------------------------------------------------------------

def test_build_method_kwargs_uniform_is_empty() -> None:
    assert bakeoff_mod.build_method_kwargs("uniform", None, 200) == {}


def test_build_method_kwargs_dare_uniform_has_drop_rate_and_seed() -> None:
    kw = bakeoff_mod.build_method_kwargs("dare_uniform", None, 200)
    assert kw == {"drop_rate": 0.5, "seed": 42}


def test_build_method_kwargs_ties_uses_trim_ratio() -> None:
    kw = bakeoff_mod.build_method_kwargs("ties", None, 200)
    assert kw == {"trim_ratio": 0.5}


def test_build_method_kwargs_dare_adamerging_threads_state() -> None:
    state = {"forward_fn": object(), "data_iter": object()}
    kw = bakeoff_mod.build_method_kwargs("dare_adamerging", state, 123)
    assert kw["forward_fn"] is state["forward_fn"]
    assert kw["data_iter"] is state["data_iter"]
    assert kw["max_steps"] == 123
    for key in ("drop_rate", "seed", "lr", "lambda_l2", "early_stop_patience"):
        assert key in kw
    assert "batch_size" not in kw, "batch_size belongs to data_iter, not to dare_adamerging"


def test_build_method_kwargs_dare_adamerging_without_state_raises() -> None:
    with pytest.raises(ValueError, match=r"forward_fn|data_iter"):
        bakeoff_mod.build_method_kwargs("dare_adamerging", None, 200)


def test_build_method_kwargs_aggregate_domains_off_by_default() -> None:
    """Default (flag absent / False) must NOT inject aggregate_domains —
    preserves the first bake-off's exact dare_adamerging behavior."""
    state = {"forward_fn": object(), "data_iter": object()}
    kw = bakeoff_mod.build_method_kwargs("dare_adamerging", state, 123)
    assert "aggregate_domains" not in kw
    kw_explicit = bakeoff_mod.build_method_kwargs(
        "dare_adamerging", state, 123, aggregate_domains=False,
    )
    assert "aggregate_domains" not in kw_explicit


def test_build_method_kwargs_aggregate_domains_on_when_flag_set() -> None:
    """With the flag set, dare_adamerging kwargs carry aggregate_domains=True."""
    state = {"forward_fn": object(), "data_iter": object()}
    kw = bakeoff_mod.build_method_kwargs(
        "dare_adamerging", state, 123, aggregate_domains=True,
    )
    assert kw["aggregate_domains"] is True


def test_build_method_kwargs_aggregate_domains_ignored_for_non_adamerging() -> None:
    """The flag only affects dare_adamerging; simple methods are untouched."""
    kw_uniform = bakeoff_mod.build_method_kwargs(
        "uniform", None, 200, aggregate_domains=True,
    )
    assert kw_uniform == {}
    kw_dare = bakeoff_mod.build_method_kwargs(
        "dare_uniform", None, 200, aggregate_domains=True,
    )
    assert "aggregate_domains" not in kw_dare


def test_aggregate_domains_arg_parses() -> None:
    """--aggregate-domains is a store_true flag, default False."""
    parser = bakeoff_mod.build_parser()
    base = [
        "--adapters-dir", ".",
        "--validation-samples-dir", ".",
        "--output-dir", ".",
    ]
    assert parser.parse_args(base).aggregate_domains is False
    assert parser.parse_args(base + ["--aggregate-domains"]).aggregate_domains is True


def test_build_adamerging_state_sizes_iter_for_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In aggregated mode the data_iter must be requested with
    max_steps * n_tasks tuples (one batch per domain per optimizer
    update). We stub the heavy deps and capture the max_steps passed to
    make_unlabeled_iter."""
    captured: dict[str, Any] = {}

    # Stub the three modules _build_adamerging_state imports lazily.
    fake_transformers = types.ModuleType("transformers")

    class _FakeTok:
        pad_token = "<pad>"
        eos_token = "<eos>"

        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

    fake_transformers.AutoTokenizer = _FakeTok
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_unlabeled = types.ModuleType("merge.data.unlabeled")
    fake_unlabeled.assert_cache_exists = lambda *a, **k: None

    def _fake_make_iter(*a, **k):
        captured["max_steps"] = k["max_steps"]
        return iter([])

    fake_unlabeled.make_unlabeled_iter = _fake_make_iter
    monkeypatch.setitem(sys.modules, "merge.data.unlabeled", fake_unlabeled)

    fake_qwen = types.ModuleType("merge.qwen3_forward")
    fake_qwen.make_qwen3_forward = lambda *a, **k: (object(), lambda: None)
    monkeypatch.setitem(sys.modules, "merge.qwen3_forward", fake_qwen)

    n_tasks = len(bakeoff_mod.CANONICAL_DOMAINS)

    # Aggregated: iter sized to max_steps * n_tasks.
    ns_agg = types.SimpleNamespace(
        base_model="x", adamerging_max_steps=50, aggregate_domains=True,
    )
    bakeoff_mod._build_adamerging_state(ns_agg)
    assert captured["max_steps"] == 50 * n_tasks

    # Default: iter sized to max_steps only.
    ns_off = types.SimpleNamespace(
        base_model="x", adamerging_max_steps=50, aggregate_domains=False,
    )
    bakeoff_mod._build_adamerging_state(ns_off)
    assert captured["max_steps"] == 50


def test_build_method_kwargs_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match=r"unknown method"):
        bakeoff_mod.build_method_kwargs("nope", None, 200)


# ---------------------------------------------------------------------------
# Row dataclasses: JSON round-trip
# ---------------------------------------------------------------------------

def test_temperature_run_row_json_round_trip() -> None:
    row = bakeoff_mod.TemperatureRunRow(
        temperature=0.5, status="ok", duration_seconds=12.3,
        pass_at_1={"math": 0.4}, pass_at_8={"math": 0.9},
        n_problems={"math": 10}, n_pass8_failed={"math": 1},
    )
    restored = bakeoff_mod.TemperatureRunRow(**json.loads(json.dumps(dataclasses.asdict(row))))
    assert restored == row


def test_method_run_row_json_round_trip() -> None:
    tr = bakeoff_mod.TemperatureRunRow(temperature=0.3, status="ok", duration_seconds=1.0)
    row = bakeoff_mod.MethodRunRow(
        method="uniform", merge_status="ok", merge_duration_seconds=5.0,
        merged_dir="/tmp/x", temperature_runs=[tr],
    )
    d = json.loads(json.dumps(dataclasses.asdict(row)))
    restored = bakeoff_mod.MethodRunRow(
        method=d["method"],
        merge_status=d["merge_status"],
        merge_duration_seconds=d["merge_duration_seconds"],
        merged_dir=d["merged_dir"],
        temperature_runs=[bakeoff_mod.TemperatureRunRow(**r) for r in d["temperature_runs"]],
        merge_error=d["merge_error"],
    )
    assert restored == row


# ---------------------------------------------------------------------------
# _temperature_row_from_results
# ---------------------------------------------------------------------------

def test_temperature_row_aggregates_all_benchmarks() -> None:
    results = {
        "math": _fake_benchmark_result(0.4, 0.8, n=10, failed=2),
        "safety": _fake_benchmark_result(1.0, 1.0, n=10, failed=0),
    }
    row = bakeoff_mod._temperature_row_from_results(
        temperature=0.5, duration_seconds=42.0, benchmark_results=results,
    )
    assert row.status == "ok"
    assert row.temperature == 0.5
    assert row.pass_at_1 == {"math": 0.4, "safety": 1.0}
    assert row.pass_at_8 == {"math": 0.8, "safety": 1.0}


def test_temperature_row_casts_types() -> None:
    results = {"math": _fake_benchmark_result("0.4", "0.8", n="10", failed="2")}
    row = bakeoff_mod._temperature_row_from_results(
        temperature=0.5, duration_seconds=1.0, benchmark_results=results,
    )
    assert isinstance(row.pass_at_1["math"], float)
    assert isinstance(row.n_problems["math"], int)


# ---------------------------------------------------------------------------
# pick_winner
# ---------------------------------------------------------------------------

def _make_method_row(method: str, t_results: list[tuple[float, float]]) -> Any:
    """t_results: list of (temperature, avg_pass_at_8)."""
    trs = []
    for temp, avg in t_results:
        tr = bakeoff_mod.TemperatureRunRow(
            temperature=temp, status="ok", duration_seconds=1.0,
            pass_at_1={b: 0.5 for b in bakeoff_mod.CANONICAL_BENCHMARKS},
            pass_at_8={b: avg for b in bakeoff_mod.CANONICAL_BENCHMARKS},
        )
        trs.append(tr)
    return bakeoff_mod.MethodRunRow(
        method=method, merge_status="ok", merge_duration_seconds=1.0,
        merged_dir="/tmp/x", temperature_runs=trs,
    )


def test_pick_winner_picks_highest_average_pass_at_8() -> None:
    runs = [
        _make_method_row("uniform", [(0.3, 0.5), (0.5, 0.6), (0.7, 0.55)]),
        _make_method_row("ties", [(0.3, 0.7), (0.5, 0.8), (0.7, 0.65)]),
    ]
    winner = bakeoff_mod.pick_winner(runs)
    assert winner == ("ties", 0.5, 0.8)


def test_pick_winner_skips_failed_merges() -> None:
    failed = bakeoff_mod.MethodRunRow(
        method="dare_adamerging", merge_status="failed",
        merge_duration_seconds=1.0, merged_dir="/tmp/x",
        merge_error="boom",
    )
    ok = _make_method_row("uniform", [(0.3, 0.4)])
    assert bakeoff_mod.pick_winner([failed, ok]) == ("uniform", 0.3, 0.4)


def test_pick_winner_skips_failed_temperatures() -> None:
    method = _make_method_row("uniform", [(0.5, 0.8)])
    method.temperature_runs.append(
        bakeoff_mod.TemperatureRunRow(
            temperature=0.7, status="failed", duration_seconds=1.0, error="oom",
        )
    )
    assert bakeoff_mod.pick_winner([method]) == ("uniform", 0.5, 0.8)


def test_pick_winner_returns_none_when_no_successful_run() -> None:
    failed = bakeoff_mod.MethodRunRow(
        method="uniform", merge_status="failed",
        merge_duration_seconds=1.0, merged_dir="/tmp/x", merge_error="boom",
    )
    assert bakeoff_mod.pick_winner([failed]) is None


# ---------------------------------------------------------------------------
# run_bakeoff: happy path
# ---------------------------------------------------------------------------

def test_run_bakeoff_happy_path_writes_results(tmp_path: Path) -> None:
    args = _build_args(tmp_path, methods=["uniform", "ties"], temperatures=[0.3, 0.5])

    merge_calls: list[str] = []

    def stub_merger(**kwargs: Any) -> Path:
        merge_calls.append(kwargs["method"])
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return kwargs["output_dir"]

    evaluator_calls: list[tuple[str, float]] = []

    def stub_evaluator(**kwargs: Any) -> dict[str, Any]:
        method = Path(kwargs["merged_adapter_dir"]).parent.name
        evaluator_calls.append((method, kwargs["config"].temperature))
        return _ok_benchmark_results()

    payload, code = bakeoff_mod.run_bakeoff(
        args, stub_merger, stub_evaluator, _config_stub, adamerging_state_factory=None,
    )

    assert code == 0
    assert merge_calls == ["uniform", "ties"]
    assert evaluator_calls == [
        ("uniform", 0.3), ("uniform", 0.5),
        ("ties", 0.3), ("ties", 0.5),
    ]

    out_path = args.output_dir / "bakeoff_results.json"
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["base_model"] == "Qwen/Qwen3-1.7B"
    assert data["methods"] == ["uniform", "ties"]
    assert data["temperatures"] == [0.3, 0.5]
    assert len(data["runs"]) == 2
    assert data["finished_at"] is not None
    assert all(r["merge_status"] == "ok" for r in data["runs"])
    assert all(
        len(r["temperature_runs"]) == 2 and all(tr["status"] == "ok" for tr in r["temperature_runs"])
        for r in data["runs"]
    )

    for method in ("uniform", "ties"):
        assert (args.output_dir / method / "merged").is_dir()
        for t in (0.3, 0.5):
            assert (args.output_dir / method / "sweep" / f"T_{t}").is_dir()


def test_run_bakeoff_writes_incrementally_after_each_method(tmp_path: Path) -> None:
    args = _build_args(tmp_path, methods=["uniform", "ties"], temperatures=[0.3])
    snapshot_lengths: list[int] = []
    out_path = args.output_dir / "bakeoff_results.json"

    def stub_merger(**kwargs: Any) -> Path:
        if out_path.exists():
            snapshot_lengths.append(len(json.loads(out_path.read_text())["runs"]))
        else:
            snapshot_lengths.append(0)
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return kwargs["output_dir"]

    def stub_evaluator(**_kwargs: Any) -> dict[str, Any]:
        return _ok_benchmark_results()

    bakeoff_mod.run_bakeoff(args, stub_merger, stub_evaluator, _config_stub, adamerging_state_factory=None)

    assert snapshot_lengths == [0, 1]


# ---------------------------------------------------------------------------
# run_bakeoff: per-method merge resilience
# ---------------------------------------------------------------------------

def test_run_bakeoff_continues_after_merge_failure(tmp_path: Path) -> None:
    """If method A's merge fails, methods B and C still run."""
    args = _build_args(
        tmp_path,
        methods=["uniform", "ties", "dare_uniform"],
        temperatures=[0.5],
    )

    def stub_merger(**kwargs: Any) -> Path:
        if kwargs["method"] == "ties":
            raise RuntimeError("simulated ties failure")
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return kwargs["output_dir"]

    evaluator_called_for: list[str] = []

    def stub_evaluator(**kwargs: Any) -> dict[str, Any]:
        evaluator_called_for.append(Path(kwargs["merged_adapter_dir"]).parent.name)
        return _ok_benchmark_results()

    payload, code = bakeoff_mod.run_bakeoff(
        args, stub_merger, stub_evaluator, _config_stub, adamerging_state_factory=None,
    )

    assert code == 1
    assert len(payload.runs) == 3
    statuses = {r.method: r.merge_status for r in payload.runs}
    assert statuses == {"uniform": "ok", "ties": "failed", "dare_uniform": "ok"}

    assert sorted(evaluator_called_for) == ["dare_uniform", "uniform"]

    ties_row = next(r for r in payload.runs if r.method == "ties")
    assert ties_row.temperature_runs == []
    assert "simulated ties failure" in ties_row.merge_error


# ---------------------------------------------------------------------------
# run_bakeoff: per-temperature eval resilience within a method
# ---------------------------------------------------------------------------

def test_run_bakeoff_continues_after_one_temperature_fails(tmp_path: Path) -> None:
    args = _build_args(
        tmp_path, methods=["uniform"], temperatures=[0.3, 0.5, 0.7],
    )

    def stub_merger(**kwargs: Any) -> Path:
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return kwargs["output_dir"]

    evaluator_calls: list[float] = []

    def stub_evaluator(**kwargs: Any) -> dict[str, Any]:
        t = kwargs["config"].temperature
        evaluator_calls.append(t)
        if t == 0.5:
            raise RuntimeError("simulated OOM at T=0.5")
        return _ok_benchmark_results()

    payload, code = bakeoff_mod.run_bakeoff(
        args, stub_merger, stub_evaluator, _config_stub, adamerging_state_factory=None,
    )

    assert code == 1
    assert evaluator_calls == [0.3, 0.5, 0.7]
    assert len(payload.runs) == 1
    statuses = [tr.status for tr in payload.runs[0].temperature_runs]
    assert statuses == ["ok", "failed", "ok"]
    failed = payload.runs[0].temperature_runs[1]
    assert failed.error is not None
    assert "simulated OOM" in failed.error


# ---------------------------------------------------------------------------
# run_bakeoff: dare_adamerging needs adamerging_state
# ---------------------------------------------------------------------------

def test_run_bakeoff_marks_dare_adamerging_failed_without_state(tmp_path: Path) -> None:
    args = _build_args(
        tmp_path,
        methods=["uniform", "dare_adamerging", "ties"],
        temperatures=[0.5],
    )
    merge_calls: list[str] = []

    def stub_merger(**kwargs: Any) -> Path:
        merge_calls.append(kwargs["method"])
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return kwargs["output_dir"]

    def stub_evaluator(**_kwargs: Any) -> dict[str, Any]:
        return _ok_benchmark_results()

    payload, code = bakeoff_mod.run_bakeoff(
        args, stub_merger, stub_evaluator, _config_stub, adamerging_state_factory=None,
    )

    assert code == 1
    assert merge_calls == ["uniform", "ties"]
    statuses = {r.method: r.merge_status for r in payload.runs}
    assert statuses == {"uniform": "ok", "dare_adamerging": "failed", "ties": "ok"}


# ---------------------------------------------------------------------------
# main(): exit-code wiring
# ---------------------------------------------------------------------------

def test_main_returns_2_on_validation_error(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    bad = [
        "--adapters-dir", str(tmp_path / "missing"),
        "--output-dir", str(tmp_path / "out"),
        "--validation-samples-dir", str(_make_valid_validation_dir(tmp_path)),
    ]
    code = bakeoff_mod.main(bad)
    assert code == 2
    err = capsys.readouterr().err
    assert "error:" in err


# ---------------------------------------------------------------------------
# print_summary smoke
# ---------------------------------------------------------------------------

def test_print_summary_runs_on_empty_payload(capsys: pytest.CaptureFixture) -> None:
    payload = bakeoff_mod.BakeoffPayload(
        started_at="now", base_model="x", methods=[], temperatures=[],
        adamerging_config={},
    )
    bakeoff_mod.print_summary(payload)
    out = capsys.readouterr().out
    assert "No successful" in out


def test_print_summary_includes_winner_line(capsys: pytest.CaptureFixture) -> None:
    payload = bakeoff_mod.BakeoffPayload(
        started_at="now", base_model="x", methods=["uniform"], temperatures=[0.3],
        adamerging_config={},
        runs=[_make_method_row("uniform", [(0.3, 0.75)])],
    )
    bakeoff_mod.print_summary(payload)
    out = capsys.readouterr().out
    assert "Winner: uniform @ T=0.3" in out
    assert "0.750" in out


# ---------------------------------------------------------------------------
# run_bakeoff: adamerging state lifetime scoped to dare_adamerging merge
# ---------------------------------------------------------------------------

def test_run_bakeoff_adamerging_cleanup_runs_before_eval(tmp_path: Path) -> None:
    """The forward_fn cleanup MUST fire BEFORE any eval for that method.

    Regression for the 2026-05-26 starvation bug: forward_fn pinned
    ~3.4 GB across the whole bake-off, so vLLM evals could not initialize
    their engine. Cleanup must complete the moment the dare_adamerging
    merge returns, before the first eval temperature runs.
    """
    args = _build_args(
        tmp_path, methods=["dare_adamerging"], temperatures=[0.3, 0.5],
    )

    event_log: list[str] = []
    factory_call_count = {"n": 0}

    def factory() -> tuple[dict[str, Any], Any]:
        factory_call_count["n"] += 1
        event_log.append("factory_built")

        def cleanup() -> None:
            event_log.append("cleanup")

        return {
            "forward_fn": object(),
            "data_iter": iter([]),
        }, cleanup

    def stub_merger(**kwargs: Any) -> Path:
        # base_model must NOT be forwarded — that's the bug we're fixing.
        # merge_adapters always loads its own base; passing a handle from
        # the state dict creates a dangling reference that defeats cleanup.
        assert "base_model" not in kwargs, (
            "base_model must NOT be forwarded; it creates a dangling reference "
            "that prevents forward_fn cleanup from freeing GPU memory"
        )
        event_log.append("merge")
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return kwargs["output_dir"]

    def stub_evaluator(**_kwargs: Any) -> dict[str, Any]:
        event_log.append("eval")
        return _ok_benchmark_results()

    payload, code = bakeoff_mod.run_bakeoff(
        args, stub_merger, stub_evaluator, _config_stub,
        adamerging_state_factory=factory,
    )

    assert code == 0
    assert factory_call_count["n"] == 1, "factory should be invoked exactly once"
    # cleanup must appear between merge and the first eval.
    assert event_log == [
        "factory_built", "merge", "cleanup", "eval", "eval",
    ], f"unexpected event ordering: {event_log}"


def test_run_bakeoff_adamerging_cleanup_runs_on_merge_failure(tmp_path: Path) -> None:
    """Cleanup must run even when the dare_adamerging merge raises."""
    args = _build_args(
        tmp_path, methods=["dare_adamerging", "uniform"], temperatures=[0.5],
    )

    event_log: list[str] = []

    def factory() -> tuple[dict[str, Any], Any]:
        event_log.append("factory_built")
        return {
            "forward_fn": object(),
            "data_iter": iter([]),
        }, lambda: event_log.append("cleanup")

    def stub_merger(**kwargs: Any) -> Path:
        if kwargs["method"] == "dare_adamerging":
            event_log.append("merge_raise")
            raise RuntimeError("simulated OOM during AdaMerging merge")
        event_log.append(f"merge_{kwargs['method']}")
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return kwargs["output_dir"]

    def stub_evaluator(**_kwargs: Any) -> dict[str, Any]:
        event_log.append("eval")
        return _ok_benchmark_results()

    payload, code = bakeoff_mod.run_bakeoff(
        args, stub_merger, stub_evaluator, _config_stub,
        adamerging_state_factory=factory,
    )

    assert code == 1
    # Cleanup MUST appear after the failed merge and before any subsequent
    # method's merge — otherwise that next merge faces a starved GPU too.
    assert "cleanup" in event_log
    cleanup_idx = event_log.index("cleanup")
    merge_raise_idx = event_log.index("merge_raise")
    merge_uniform_idx = event_log.index("merge_uniform")
    assert merge_raise_idx < cleanup_idx < merge_uniform_idx, (
        f"cleanup must run between failed dare_adamerging merge and the next "
        f"method's merge; got {event_log}"
    )
    # uniform must NOT receive a base_model kwarg from the adamerging state.
    # (Sanity: the factory only feeds dare_adamerging.)
    statuses = {r.method: r.merge_status for r in payload.runs}
    assert statuses == {"dare_adamerging": "failed", "uniform": "ok"}


def test_run_bakeoff_non_adamerging_method_does_not_invoke_factory(tmp_path: Path) -> None:
    """Without dare_adamerging in the sweep, the factory must never run."""
    args = _build_args(tmp_path, methods=["uniform"], temperatures=[0.3])

    def factory() -> tuple[dict[str, Any], Any]:
        raise AssertionError("factory should not be called without dare_adamerging")

    def stub_merger(**kwargs: Any) -> Path:
        assert "base_model" not in kwargs, "uniform must not receive base_model"
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return kwargs["output_dir"]

    def stub_evaluator(**_kwargs: Any) -> dict[str, Any]:
        return _ok_benchmark_results()

    _, code = bakeoff_mod.run_bakeoff(
        args, stub_merger, stub_evaluator, _config_stub,
        adamerging_state_factory=factory,
    )
    assert code == 0

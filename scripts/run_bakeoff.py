#!/usr/bin/env python3
"""Stage 5c.2: Full bake-off orchestrator.

Sweeps 4 merge methods × 3 sampling temperatures on the same 4 input
adapters, producing 12 (method, temperature) evaluation scorecards plus
an aggregated comparison written to ``bakeoff_results.json``.

Design:

  1. **Merge once per method, sweep temperatures on that merge.**
     Temperature is sampling-only — it has no effect on the merged
     weights. 4 merges + 12 evals (not 12 merges) avoids ~3-4 hours of
     wasted work.

  2. **Per-(method, temperature) resilience.** A merge failure marks
     that method's 3 temperature slots all-failed and moves on. A
     single temperature OOM marks only that slot failed. Partial
     bake-off data is preserved by incremental writes.

  3. **Hard fail at startup if any input adapter is missing or
     fails locked-spec verification.** A 4-hour bake-off that crashes
     because of a typo'd adapter dir is wasteful.

  4. **AdaMerging hyperparameters are NOT swept.** Single bake-off
     config:
       drop_rate=0.5, lr=1e-2, lambda_l2=1e-4,
       max_steps=200 (vs production 1000 to keep total time reasonable),
       early_stop_patience=100, batch_size=2.
     Hyperparameter tuning is a separate experiment.

Output layout::

    <output_dir>/
        bakeoff_results.json
        uniform/
            merged/                      # config.json + model.safetensors + ...
            sweep/T_0.3/                 # scorecard.json + generations + failures
            sweep/T_0.5/
            sweep/T_0.7/
        dare_uniform/
            merged/
            sweep/T_0.3/ ...
        dare_adamerging/
            merged/
            sweep/T_0.3/ ...
        ties/
            merged/
            sweep/T_0.3/ ...

Default output_dir is ``<repo_root>/bakeoff_<YYYY-MM-DD-HHMM>/``.

Usage::

    # Background run with unbuffered logs
    nohup python -u scripts/run_bakeoff.py \\
        --adapters-dir loras/ \\
        --output-dir bakeoff_2026-05-21-1400/ \\
        > bakeoff.log 2>&1 &

Exit codes:
    0 — every (method, temperature) succeeded.
    1 — at least one (method, temperature) failed (results json still written).
    2 — setup error (missing adapter, bad CLI args, spec mismatch).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


logger = logging.getLogger("run_bakeoff")


CANONICAL_BENCHMARKS: tuple[str, ...] = (
    "math", "general_knowledge", "safety", "multilingual",
)

CANONICAL_DOMAINS: tuple[str, ...] = (
    "math", "general_knowledge", "safety", "multilingual",
)

DEFAULT_METHODS: tuple[str, ...] = (
    "uniform", "dare_uniform", "dare_adamerging", "ties",
)

DEFAULT_TEMPERATURES: tuple[float, ...] = (0.3, 0.5, 0.7)

# Bake-off default config for AdaMerging — single point in the
# hyperparameter space, applied to dare_adamerging only. Separate
# tuning experiment will explore the rest.
ADAMERGING_BAKEOFF_CONFIG: dict[str, Any] = {
    "drop_rate": 0.5,
    "seed": 42,
    "lr": 1e-2,
    "lambda_l2": 1e-4,
    "max_steps": 200,
    "early_stop_patience": 100,
    "batch_size": 2,
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TemperatureRunRow:
    """One temperature evaluation within a single method's sweep."""

    temperature: float
    status: str  # "ok" | "failed"
    duration_seconds: float
    pass_at_1: dict[str, float] = field(default_factory=dict)
    pass_at_8: dict[str, float] = field(default_factory=dict)
    n_problems: dict[str, int] = field(default_factory=dict)
    n_pass8_failed: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@dataclass
class MethodRunRow:
    """One method's merge + per-temperature evaluation sweep."""

    method: str
    merge_status: str  # "ok" | "failed"
    merge_duration_seconds: float
    merged_dir: str
    temperature_runs: list[TemperatureRunRow] = field(default_factory=list)
    merge_error: str | None = None


@dataclass
class BakeoffPayload:
    """Top-level JSON schema for ``bakeoff_results.json``."""

    started_at: str
    base_model: str
    methods: list[str]
    temperatures: list[float]
    adamerging_config: dict[str, Any]
    runs: list[MethodRunRow] = field(default_factory=list)
    finished_at: str | None = None


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full Stage 5c.2 bake-off (methods × temperatures).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Recommended launch:\n"
            "  nohup python -u scripts/run_bakeoff.py "
            "--adapters-dir loras/ --output-dir bakeoff_<date>/ "
            "> bakeoff.log 2>&1 &\n"
        ),
    )
    parser.add_argument(
        "--adapters-dir",
        type=Path,
        required=True,
        help=(
            "Parent dir with one PEFT-format subdir per domain: math, "
            "general_knowledge, safety, multilingual."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Bake-off output dir. Defaults to "
            "<repo_root>/bakeoff_<YYYY-MM-DD-HHMM>/."
        ),
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-1.7B",
        help="Base model HF repo (default: Qwen/Qwen3-1.7B).",
    )
    parser.add_argument(
        "--validation-samples-dir",
        type=Path,
        default=_REPO_ROOT / "validation_samples",
        help="Directory with the 4 validation JSONLs.",
    )
    parser.add_argument(
        "--chat-template-path",
        type=Path,
        default=_REPO_ROOT / "chat_template.jinja",
        help="Path to chat_template.jinja.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help=f"Methods to sweep (default: {' '.join(DEFAULT_METHODS)}).",
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=list(DEFAULT_TEMPERATURES),
        help=f"Sampling temperatures (default: {DEFAULT_TEMPERATURES}).",
    )
    parser.add_argument("--top-p", type=float, default=0.8, help="Nucleus sampling.")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling.")
    parser.add_argument("--n", type=int, default=8, help="Completions per problem.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per completion.")
    parser.add_argument("--seed", type=int, default=42, help="vLLM sampling seed.")
    parser.add_argument(
        "--adamerging-max-steps",
        type=int,
        default=ADAMERGING_BAKEOFF_CONFIG["max_steps"],
        help=(
            f"AdaMerging training steps for dare_adamerging "
            f"(default {ADAMERGING_BAKEOFF_CONFIG['max_steps']}; "
            "production training uses ~1000)."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> list[str]:
    """Validate CLI args without touching ML libs. Returns a list of error messages."""
    errors: list[str] = []

    if not args.adapters_dir.is_dir():
        errors.append(f"--adapters-dir not a directory: {args.adapters_dir}")
    else:
        for domain in CANONICAL_DOMAINS:
            sub = args.adapters_dir / domain
            if not sub.is_dir():
                errors.append(f"missing adapter subdir: {sub}")

    if not args.validation_samples_dir.is_dir():
        errors.append(
            f"--validation-samples-dir not a directory: {args.validation_samples_dir}"
        )
    else:
        for bench in CANONICAL_BENCHMARKS:
            jsonl = args.validation_samples_dir / f"{bench}.jsonl"
            if not jsonl.exists():
                errors.append(f"missing validation file: {jsonl}")

    if not args.methods:
        errors.append("--methods requires at least one value")
    valid_methods = set(DEFAULT_METHODS)
    for m in args.methods:
        if m not in valid_methods:
            errors.append(
                f"unknown method {m!r}; valid options are "
                f"{sorted(valid_methods)!r}"
            )

    if not args.temperatures:
        errors.append("--temperatures requires at least one value")
    for t in args.temperatures:
        if t <= 0.0:
            errors.append(
                f"temperature {t} not allowed: vLLM rejects n>1 at temperature=0.0; "
                "build a custom InferenceConfig with n=1 for deterministic decoding"
            )

    if args.n < 1:
        errors.append(f"--n must be >= 1, got {args.n}")
    if args.max_tokens < 1:
        errors.append(f"--max-tokens must be >= 1, got {args.max_tokens}")
    if not (0.0 < args.top_p <= 1.0):
        errors.append(f"--top-p must be in (0, 1], got {args.top_p}")
    if args.top_k < 1:
        errors.append(f"--top-k must be >= 1, got {args.top_k}")
    if args.adamerging_max_steps < 1:
        errors.append(
            f"--adamerging-max-steps must be >= 1, got {args.adamerging_max_steps}"
        )

    return errors


def verify_locked_specs(adapters_dir: Path, locked_spec: dict) -> list[str]:
    """Run ``verify_spec`` on all 4 adapter configs and collect failures.

    Returns a list of error messages; empty list means all passed. Pure
    JSON + YAML — no torch needed.
    """
    from merge.verify_spec import verify

    errors: list[str] = []
    for domain in CANONICAL_DOMAINS:
        cfg = adapters_dir / domain / "adapter_config.json"
        if not cfg.exists():
            errors.append(f"{domain}: adapter_config.json missing at {cfg}")
            continue
        result = verify(cfg, locked_spec)
        if not result.passed:
            errors.append(f"{domain}: {result.summary}")
    return errors


# ---------------------------------------------------------------------------
# Result aggregation helpers (testable, no ML deps)
# ---------------------------------------------------------------------------

def _temperature_row_from_results(
    temperature: float,
    duration_seconds: float,
    benchmark_results: dict[str, Any],
) -> TemperatureRunRow:
    """Fold ``{benchmark: BenchmarkResult}`` into a flat row.

    Duck-types on the public attributes so tests can use SimpleNamespace.
    """
    row = TemperatureRunRow(
        temperature=temperature,
        status="ok",
        duration_seconds=duration_seconds,
    )
    for bench, result in benchmark_results.items():
        row.pass_at_1[bench] = float(getattr(result, "pass_at_1"))
        row.pass_at_8[bench] = float(getattr(result, "pass_at_8"))
        row.n_problems[bench] = int(getattr(result, "n_problems"))
        row.n_pass8_failed[bench] = int(getattr(result, "n_pass8_failed"))
    return row


def _write_bakeoff_results(output_dir: Path, payload: BakeoffPayload) -> Path:
    """Serialize the bake-off payload to ``bakeoff_results.json``."""
    path = output_dir / "bakeoff_results.json"
    with path.open("w") as f:
        json.dump(dataclasses.asdict(payload), f, indent=2)
    return path


def _avg_pass_at_8(row: TemperatureRunRow) -> float | None:
    """Average pass@8 across the 4 benchmarks for one (method, temperature)."""
    if row.status != "ok" or not row.pass_at_8:
        return None
    values = [row.pass_at_8.get(b) for b in CANONICAL_BENCHMARKS]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def pick_winner(runs: list[MethodRunRow]) -> tuple[str, float, float] | None:
    """Find the (method, temperature) combo with the highest avg pass@8.

    Returns ``(method, temperature, avg_pass_at_8)`` or ``None`` if no
    successful run exists.
    """
    best: tuple[str, float, float] | None = None
    for run in runs:
        if run.merge_status != "ok":
            continue
        for tr in run.temperature_runs:
            avg = _avg_pass_at_8(tr)
            if avg is None:
                continue
            if best is None or avg > best[2]:
                best = (run.method, tr.temperature, avg)
    return best


def print_summary(payload: BakeoffPayload) -> None:
    """Print a (method, temperature) × benchmark grid + winner line."""
    header = ["method", "T", "status", "dur(s)"]
    for bench in CANONICAL_BENCHMARKS:
        header.append(f"{bench[:4]}@1")
        header.append(f"{bench[:4]}@8")
    header.append("avg@8")
    widths = [max(len(h), 10) for h in header]
    widths[0] = max(widths[0], 18)  # method name needs room

    sep = "=" * (sum(widths) + 3 * len(widths))
    dashes = "-" * (sum(widths) + 3 * len(widths))

    print()
    print(sep)
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print(dashes)
    for run in payload.runs:
        if run.merge_status != "ok":
            cells = [run.method, "—", "merge:failed", f"{run.merge_duration_seconds:.1f}"]
            cells += ["—"] * (len(header) - len(cells))
            print(" | ".join(c.ljust(w) for c, w in zip(cells, widths)))
            continue
        for tr in run.temperature_runs:
            cells = [
                run.method,
                f"{tr.temperature:.2f}",
                tr.status,
                f"{tr.duration_seconds:.1f}",
            ]
            for bench in CANONICAL_BENCHMARKS:
                p1 = tr.pass_at_1.get(bench)
                p8 = tr.pass_at_8.get(bench)
                cells.append(f"{p1:.3f}" if p1 is not None else "—")
                cells.append(f"{p8:.3f}" if p8 is not None else "—")
            avg = _avg_pass_at_8(tr)
            cells.append(f"{avg:.3f}" if avg is not None else "—")
            print(" | ".join(c.ljust(w) for c, w in zip(cells, widths)))
    print(sep)

    winner = pick_winner(payload.runs)
    if winner is None:
        print("No successful (method, temperature) run; cannot pick winner.")
    else:
        method, temp, avg = winner
        print(f"Winner: {method} @ T={temp}: avg pass@8 = {avg:.3f}")
    print()


# ---------------------------------------------------------------------------
# Bake-off core (injectable for tests)
# ---------------------------------------------------------------------------

def build_method_kwargs(
    method: str,
    adamerging_state: dict[str, Any] | None,
    adamerging_max_steps: int,
) -> dict[str, Any]:
    """Build the ``method_kwargs`` dict for one method.

    Args:
        method: The method name (must be in DEFAULT_METHODS).
        adamerging_state: Dict with ``forward_fn`` and ``data_iter`` keys,
            or ``None`` if dare_adamerging is not in the sweep.
        adamerging_max_steps: Overrides the bake-off default for max_steps.

    Returns:
        Kwargs dict appropriate for the method.

    Raises:
        ValueError: if dare_adamerging requested but adamerging_state is None.
    """
    if method == "uniform":
        return {}
    if method == "dare_uniform":
        return {"drop_rate": 0.5, "seed": 42}
    if method == "ties":
        return {"trim_ratio": 0.5}
    if method == "dare_adamerging":
        if adamerging_state is None:
            raise ValueError(
                "dare_adamerging requires forward_fn + data_iter; "
                "adamerging_state was None"
            )
        kwargs = dict(ADAMERGING_BAKEOFF_CONFIG)
        kwargs.pop("batch_size", None)  # batch_size belongs to data_iter, not to dare_adamerging
        kwargs["max_steps"] = adamerging_max_steps
        kwargs["forward_fn"] = adamerging_state["forward_fn"]
        kwargs["data_iter"] = adamerging_state["data_iter"]
        return kwargs
    raise ValueError(f"unknown method: {method!r}")


def run_bakeoff(
    args: argparse.Namespace,
    merge_callable: Callable[..., Path],
    eval_callable: Callable[..., dict[str, Any]],
    config_factory: Callable[[float, argparse.Namespace], Any],
    adamerging_state: dict[str, Any] | None,
) -> tuple[BakeoffPayload, int]:
    """Execute the bake-off across all methods × temperatures.

    Injectable callables let tests bypass torch/vllm/peft entirely.

    Args:
        args: Parsed argparse namespace (must have passed validate_args).
        merge_callable: Function with the signature of
            :func:`merge.pipeline.merge_adapters`.
        eval_callable: Function with the signature of
            :func:`merge.eval_all.evaluate_all_benchmarks`.
        config_factory: ``(temperature, args) -> InferenceConfig``.
        adamerging_state: ``{"forward_fn": ..., "data_iter": ...}`` or
            ``None`` if dare_adamerging is not in the sweep.

    Returns:
        ``(payload, exit_code)``. exit_code is 0 if every (method,
        temperature) succeeded, 1 if any failed.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = BakeoffPayload(
        started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        base_model=args.base_model,
        methods=list(args.methods),
        temperatures=list(args.temperatures),
        adamerging_config=dict(ADAMERGING_BAKEOFF_CONFIG, max_steps=args.adamerging_max_steps),
    )

    any_failed = False
    for method in args.methods:
        method_dir = args.output_dir / method
        merged_dir = method_dir / "merged"
        sweep_dir = method_dir / "sweep"
        method_dir.mkdir(parents=True, exist_ok=True)

        try:
            method_kwargs = build_method_kwargs(
                method, adamerging_state, args.adamerging_max_steps,
            )
        except ValueError as exc:
            logger.error("Skipping method %s: %s", method, exc)
            row = MethodRunRow(
                method=method,
                merge_status="failed",
                merge_duration_seconds=0.0,
                merged_dir=str(merged_dir),
                merge_error=str(exc),
            )
            payload.runs.append(row)
            _write_bakeoff_results(args.output_dir, payload)
            any_failed = True
            continue

        logger.info("[%s] Starting merge -> %s", method, merged_dir)
        merge_start = time.monotonic()
        try:
            merge_callable(
                adapters_dir=args.adapters_dir,
                method=method,
                output_dir=merged_dir,
                method_kwargs=method_kwargs,
                base_model_repo=args.base_model,
            )
            merge_duration = time.monotonic() - merge_start
            method_row = MethodRunRow(
                method=method,
                merge_status="ok",
                merge_duration_seconds=merge_duration,
                merged_dir=str(merged_dir),
            )
            logger.info("[%s] Merge succeeded in %.1fs", method, merge_duration)
        except Exception:
            merge_duration = time.monotonic() - merge_start
            tb = traceback.format_exc()
            logger.error("[%s] Merge FAILED after %.1fs:\n%s", method, merge_duration, tb)
            method_row = MethodRunRow(
                method=method,
                merge_status="failed",
                merge_duration_seconds=merge_duration,
                merged_dir=str(merged_dir),
                merge_error=tb,
            )
            payload.runs.append(method_row)
            _write_bakeoff_results(args.output_dir, payload)
            any_failed = True
            continue

        # Merge OK → sweep temperatures.
        for temperature in args.temperatures:
            per_temp_dir = sweep_dir / f"T_{temperature}"
            per_temp_dir.mkdir(parents=True, exist_ok=True)
            config = config_factory(temperature, args)
            logger.info(
                "[%s] T=%s -> %s", method, temperature, per_temp_dir,
            )
            eval_start = time.monotonic()
            try:
                results = eval_callable(
                    merged_adapter_dir=merged_dir,
                    base_model_repo=args.base_model,
                    output_dir=per_temp_dir,
                    validation_samples_dir=args.validation_samples_dir,
                    chat_template_path=args.chat_template_path,
                    config=config,
                )
                duration = time.monotonic() - eval_start
                tr = _temperature_row_from_results(temperature, duration, results)
                logger.info(
                    "[%s] T=%s succeeded in %.1fs", method, temperature, duration,
                )
            except Exception:
                duration = time.monotonic() - eval_start
                tb = traceback.format_exc()
                logger.error(
                    "[%s] T=%s FAILED after %.1fs:\n%s",
                    method, temperature, duration, tb,
                )
                tr = TemperatureRunRow(
                    temperature=temperature,
                    status="failed",
                    duration_seconds=duration,
                    error=tb,
                )
                any_failed = True
            method_row.temperature_runs.append(tr)

        payload.runs.append(method_row)
        _write_bakeoff_results(args.output_dir, payload)
        logger.info("[%s] Wrote incremental bakeoff_results.json", method)

    payload.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_bakeoff_results(args.output_dir, payload)
    return payload, (1 if any_failed else 0)


# ---------------------------------------------------------------------------
# Default factories (production wiring)
# ---------------------------------------------------------------------------

def _default_config_factory(temperature: float, args: argparse.Namespace) -> Any:
    from merge.infer import InferenceConfig

    return InferenceConfig(
        n=args.n,
        temperature=float(temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        max_tokens=int(args.max_tokens),
        seed=int(args.seed),
    )


def _default_merge_callable(**kwargs: Any) -> Path:
    from merge.pipeline import merge_adapters

    return merge_adapters(**kwargs)


def _default_eval_callable(**kwargs: Any) -> dict[str, Any]:
    from merge.eval_all import evaluate_all_benchmarks

    return evaluate_all_benchmarks(**kwargs)


def _build_adamerging_state(args: argparse.Namespace) -> tuple[dict[str, Any], Callable[[], None]]:
    """Build forward_fn + data_iter once for the whole sweep.

    Returns ``(state_dict, cleanup_callable)``. The state dict is what
    ``build_method_kwargs`` consumes for dare_adamerging.
    """
    from transformers import AutoTokenizer

    from merge.data.unlabeled import assert_cache_exists, make_unlabeled_iter
    from merge.qwen3_forward import make_qwen3_forward

    assert_cache_exists()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    forward_fn, cleanup = make_qwen3_forward(
        base_model_repo=args.base_model,
        device="cuda",
    )
    data_iter = make_unlabeled_iter(
        tokenizer=tokenizer,
        batch_size=ADAMERGING_BAKEOFF_CONFIG["batch_size"],
        max_steps=args.adamerging_max_steps,
        seed=ADAMERGING_BAKEOFF_CONFIG["seed"],
        device="cuda",
    )
    return {"forward_fn": forward_fn, "data_iter": data_iter}, cleanup


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _default_output_dir() -> Path:
    """``<repo_root>/bakeoff_<YYYY-MM-DD-HHMM>/``."""
    stamp = dt.datetime.now().strftime("%Y-%m-%d-%H%M")
    return _REPO_ROOT / f"bakeoff_{stamp}"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.output_dir is None:
        args.output_dir = _default_output_dir()

    errors = validate_args(args)
    if errors:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        return 2

    # Locked-spec verification — pure JSON + YAML, no torch needed.
    from merge.verify_spec import load_locked_spec

    spec_path = _REPO_ROOT / "lora.yaml"
    if not spec_path.exists():
        print(f"error: locked spec not found at {spec_path}", file=sys.stderr)
        return 2
    locked_spec = load_locked_spec(spec_path)
    spec_errors = verify_locked_specs(args.adapters_dir, locked_spec)
    if spec_errors:
        for err in spec_errors:
            print(f"error: spec verification: {err}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Bake-off: %d methods × %d temperatures = %d runs -> %s",
        len(args.methods), len(args.temperatures),
        len(args.methods) * len(args.temperatures),
        args.output_dir,
    )

    adamerging_state: dict[str, Any] | None = None
    cleanup_adamerging: Callable[[], None] | None = None
    if "dare_adamerging" in args.methods:
        logger.info("Building AdaMerging forward_fn + data_iter ...")
        adamerging_state, cleanup_adamerging = _build_adamerging_state(args)

    try:
        payload, exit_code = run_bakeoff(
            args,
            merge_callable=_default_merge_callable,
            eval_callable=_default_eval_callable,
            config_factory=_default_config_factory,
            adamerging_state=adamerging_state,
        )
    finally:
        if cleanup_adamerging is not None:
            try:
                cleanup_adamerging()
                logger.info("Released AdaMerging forward_fn base model.")
            except Exception:
                logger.exception("AdaMerging cleanup failed (continuing).")

    print_summary(payload)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

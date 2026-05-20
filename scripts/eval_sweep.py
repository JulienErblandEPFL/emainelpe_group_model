#!/usr/bin/env python3
"""Evaluate one merged adapter at multiple sampling temperatures.

Temperature is an inference-time parameter — it has no effect on the merged
weights — so we run the merge once and call
:func:`merge.eval_all.evaluate_all_benchmarks` repeatedly with explicit
:class:`merge.infer.InferenceConfig` overrides.

Output layout::

    <output_dir>/
        T_0.3/
            scorecard.json
            generations_<benchmark>.jsonl × 4
            failures_<benchmark>.json × 4
        T_0.5/
            ...
        T_0.7/
            ...
        sweep_results.json   # aggregated comparison, written incrementally

The sweep is resilient: a vLLM crash or OOM on one temperature is recorded
to ``sweep_results.json`` as a failure and the next temperature still runs.

Usage::

    python -u scripts/eval_sweep.py \\
        --merged-adapter-dir outputs/merged_v1/ \\
        --output-dir outputs/sweep_v1/ \\
        --temperatures 0.3 0.5 0.7

Exit codes:
    0 — every temperature succeeded.
    1 — at least one temperature failed (sweep_results.json still written).
    2 — setup error (bad CLI args, missing inputs).

Design notes:
    Temperature 0.0 is rejected. vLLM's SamplingParams rejects n>1 when
    temperature=0.0 because greedy decoding is deterministic. For a
    deterministic final HF push, build a custom InferenceConfig with n=1
    rather than going through this sweep script.
"""
from __future__ import annotations

import argparse
import dataclasses
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


logger = logging.getLogger("eval_sweep")


CANONICAL_BENCHMARKS: tuple[str, ...] = (
    "math",
    "general_knowledge",
    "safety",
    "multilingual",
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SweepResultRow:
    """One row in ``sweep_results.json`` — the result of evaluating one temperature."""

    temperature: float
    status: str  # "ok" | "failed"
    duration_seconds: float
    pass_at_1: dict[str, float] = field(default_factory=dict)
    pass_at_8: dict[str, float] = field(default_factory=dict)
    n_problems: dict[str, int] = field(default_factory=dict)
    n_pass8_failed: dict[str, int] = field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one merged adapter at multiple temperatures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--merged-adapter-dir",
        type=Path,
        required=True,
        help="PEFT-format merged adapter directory (must contain adapter_config.json + adapter_model.safetensors).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write per-temperature subdirs + sweep_results.json.",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        required=True,
        help="One or more strictly-positive temperatures (e.g. 0.3 0.5 0.7).",
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
        help="Directory with the 4 validation JSONLs (default: <repo>/validation_samples).",
    )
    parser.add_argument(
        "--chat-template-path",
        type=Path,
        default=_REPO_ROOT / "chat_template.jinja",
        help="Path to chat_template.jinja (default: <repo>/chat_template.jinja).",
    )
    parser.add_argument("--top-p", type=float, default=0.8, help="Nucleus sampling (default 0.8).")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling (default 20).")
    parser.add_argument("--n", type=int, default=8, help="Completions per problem (default 8).")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens per completion (default 2048; CI ceiling is 16384).",
    )
    parser.add_argument("--seed", type=int, default=42, help="vLLM sampling seed (default 42).")
    return parser


def validate_args(args: argparse.Namespace) -> list[str]:
    """Validate parsed args without touching the filesystem unnecessarily.

    Returns a list of error messages. Empty list means OK.
    """
    errors: list[str] = []
    if not args.merged_adapter_dir.is_dir():
        errors.append(f"--merged-adapter-dir not a directory: {args.merged_adapter_dir}")
    else:
        if not (args.merged_adapter_dir / "adapter_config.json").exists():
            errors.append(
                f"adapter_config.json missing under {args.merged_adapter_dir}"
            )
        if not (args.merged_adapter_dir / "adapter_model.safetensors").exists():
            errors.append(
                f"adapter_model.safetensors missing under {args.merged_adapter_dir}"
            )

    if not args.validation_samples_dir.is_dir():
        errors.append(
            f"--validation-samples-dir not a directory: {args.validation_samples_dir}"
        )
    else:
        for bench in CANONICAL_BENCHMARKS:
            jsonl = args.validation_samples_dir / f"{bench}.jsonl"
            if not jsonl.exists():
                errors.append(f"missing validation file: {jsonl}")

    if not args.temperatures:
        errors.append("--temperatures requires at least one value")
    for t in args.temperatures:
        if t <= 0.0:
            errors.append(
                f"temperature {t} not allowed: vLLM rejects n>1 at temperature=0.0; "
                "for deterministic greedy decoding, build a custom InferenceConfig "
                "with n=1 instead of using this sweep script"
            )

    if args.n < 1:
        errors.append(f"--n must be >= 1, got {args.n}")
    if args.max_tokens < 1:
        errors.append(f"--max-tokens must be >= 1, got {args.max_tokens}")
    if not (0.0 < args.top_p <= 1.0):
        errors.append(f"--top-p must be in (0, 1], got {args.top_p}")
    if args.top_k < 1:
        errors.append(f"--top-k must be >= 1, got {args.top_k}")

    return errors


# ---------------------------------------------------------------------------
# Sweep core (injectable for tests)
# ---------------------------------------------------------------------------

def _benchmark_results_to_row(
    temperature: float,
    duration_seconds: float,
    benchmark_results: dict[str, Any],
) -> SweepResultRow:
    """Fold ``{benchmark: BenchmarkResult}`` into a flat row.

    ``BenchmarkResult`` is imported lazily, so we duck-type on its public
    attributes (``pass_at_1``, ``pass_at_8``, ``n_problems``,
    ``n_pass8_failed``) to keep this helper torch-free and testable.
    """
    row = SweepResultRow(
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


def _write_sweep_results(output_dir: Path, rows: list[SweepResultRow]) -> Path:
    path = output_dir / "sweep_results.json"
    payload = [dataclasses.asdict(r) for r in rows]
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
    return path


def _print_summary(rows: list[SweepResultRow]) -> None:
    """Print a tidy pass@1 / pass@8 comparison table to stdout."""
    header = ["T", "status", "dur(s)"]
    for bench in CANONICAL_BENCHMARKS:
        header.append(f"{bench[:4]}@1")
        header.append(f"{bench[:4]}@8")
    widths = [max(len(h), 6) for h in header]

    print()
    print("=" * (sum(widths) + 3 * len(widths)))
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-" * (sum(widths) + 3 * len(widths)))
    for r in rows:
        cells = [f"{r.temperature:.2f}", r.status, f"{r.duration_seconds:.1f}"]
        for bench in CANONICAL_BENCHMARKS:
            p1 = r.pass_at_1.get(bench)
            p8 = r.pass_at_8.get(bench)
            cells.append(f"{p1:.3f}" if p1 is not None else "—")
            cells.append(f"{p8:.3f}" if p8 is not None else "—")
        print(" | ".join(c.ljust(w) for c, w in zip(cells, widths)))
    print("=" * (sum(widths) + 3 * len(widths)))
    print()


def run_sweep(
    args: argparse.Namespace,
    eval_callable: Callable[..., dict[str, Any]],
    config_factory: Callable[[float, argparse.Namespace], Any],
) -> tuple[list[SweepResultRow], int]:
    """Execute the temperature sweep.

    Args:
        args: Parsed argparse namespace (must have passed :func:`validate_args`).
        eval_callable: Function with the signature of
            :func:`merge.eval_all.evaluate_all_benchmarks`. Injected so
            tests can mock it.
        config_factory: Callable ``(temperature, args) -> InferenceConfig``.
            Injected for the same reason.

    Returns:
        ``(rows, exit_code)`` where ``rows`` is the list of result rows
        (one per temperature, written incrementally to disk) and
        ``exit_code`` is 0 if all succeeded, 1 if any failed.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[SweepResultRow] = []
    any_failed = False

    for temperature in args.temperatures:
        per_temp_dir = args.output_dir / f"T_{temperature}"
        per_temp_dir.mkdir(parents=True, exist_ok=True)
        config = config_factory(temperature, args)
        logger.info(
            "Starting eval at temperature=%s -> %s", temperature, per_temp_dir
        )

        start = time.monotonic()
        try:
            results = eval_callable(
                merged_adapter_dir=args.merged_adapter_dir,
                base_model_repo=args.base_model,
                output_dir=per_temp_dir,
                validation_samples_dir=args.validation_samples_dir,
                chat_template_path=args.chat_template_path,
                config=config,
            )
            duration = time.monotonic() - start
            row = _benchmark_results_to_row(temperature, duration, results)
            logger.info(
                "temperature=%s succeeded in %.1fs", temperature, duration
            )
        except Exception:
            duration = time.monotonic() - start
            tb = traceback.format_exc()
            logger.error(
                "temperature=%s FAILED after %.1fs:\n%s", temperature, duration, tb
            )
            row = SweepResultRow(
                temperature=temperature,
                status="failed",
                duration_seconds=duration,
                error=tb,
            )
            any_failed = True

        rows.append(row)
        path = _write_sweep_results(args.output_dir, rows)
        logger.info("Wrote incremental sweep_results.json -> %s", path)

    return rows, (1 if any_failed else 0)


# ---------------------------------------------------------------------------
# Default factories (production wiring)
# ---------------------------------------------------------------------------

def _default_config_factory(temperature: float, args: argparse.Namespace) -> Any:
    """Lazy import: keeps the script importable on a torch-free laptop."""
    from merge.infer import InferenceConfig

    return InferenceConfig(
        n=args.n,
        temperature=float(temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        max_tokens=int(args.max_tokens),
        seed=int(args.seed),
    )


def _default_eval_callable(**kwargs: Any) -> dict[str, Any]:
    """Lazy import wrapper around :func:`merge.eval_all.evaluate_all_benchmarks`."""
    from merge.eval_all import evaluate_all_benchmarks

    return evaluate_all_benchmarks(**kwargs)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    errors = validate_args(args)
    if errors:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        return 2

    rows, exit_code = run_sweep(args, _default_eval_callable, _default_config_factory)
    _print_summary(rows)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

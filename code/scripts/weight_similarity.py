#!/usr/bin/env python3
"""Pairwise weight-space similarity between merged models.

The CS-552 M3 report cites pairwise weight-space differences between the
merged models produced by the group-model bake-off (e.g. "0.18% to 1.66%
mean-absolute relative difference between pairs"). This script makes those
numbers reproducible from the merged ``model.safetensors`` checkpoints.

For each requested tensor key, it reads *only that tensor* from every model's
``model.safetensors`` (via ``safetensors.safe_open`` — no full-model load),
casts to float32, and computes a pairwise metric:

    MAR  mean absolute relative difference:
             mean( |A - B| / (|A| + |B| + eps) )      reported as a percentage.
             Bounded in [0%, 100%]; opposite-sign tensors hit 100%.
    COS  cosine similarity between the flattened tensors:
             dot(a, b) / (||a|| * ||b||)              reported as a percentage.

Output: a triangular matrix per tensor on the console, plus an optional JSON
summary. Reads only the requested tensors from disk; no GPU; runs in well
under a minute for the 6-model × 3-tensor bake-off set.

Usage::

    python3 scripts/weight_similarity.py \\
        --models uniform:bakeoff/uniform/merged ties:bakeoff/ties/merged \\
        --metric mar \\
        --output weight_similarity.json

Exit codes:
    0 — report produced.
    1 — bad input (missing ``model.safetensors``, missing tensor key, shape
        mismatch, malformed ``--models`` argument).
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("weight_similarity")


# Three representative MLP ``down_proj`` tensors: early / middle / late layer.
# The AdaMerging coefficient heatmap (diag_agg/coefficients_heatmap.png in the
# report) shows merge coefficients of differing sign across early/mid/late
# layers, so these three are where merge methods are most likely to diverge.
DEFAULT_TENSOR_KEYS: tuple[str, ...] = (
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.13.mlp.down_proj.weight",
    "model.layers.27.mlp.down_proj.weight",
)

_MAR_EPS = 1e-8
_VALID_METRICS = ("mar", "cos", "both")


class WeightSimilarityError(Exception):
    """Raised for any user-facing input error (→ exit 1 in :func:`main`)."""


# ---------------------------------------------------------------------------
# Pure metric functions (torch tensors in, plain float out)
# ---------------------------------------------------------------------------
# Written with tensor *methods* only (no ``torch.*`` calls) so this module
# imports cleanly on a torch-free laptop; torch is only needed at call time.

def mar_between(a, b, eps: float = _MAR_EPS) -> float:
    """Mean absolute relative difference between two tensors, as a fraction.

    ``mean( |A - B| / (|A| + |B| + eps) )``. Symmetric in ``a``/``b``.
    Bounded in ``[0, 1]``: identical tensors give 0, opposite-sign tensors
    give ~1. Multiply by 100 for the reported percentage.

    Both tensors are cast to float32 first (the merged models are bfloat16;
    the cast avoids underflow when differencing small weights).
    """
    a = a.float()
    b = b.float()
    diff = (a - b).abs() / (a.abs() + b.abs() + eps)
    return float(diff.mean().item())


def cos_between(a, b) -> float:
    """Cosine similarity between the flattened tensors, as a fraction in [-1, 1].

    ``dot(a, b) / (||a|| * ||b||)``. Identical direction → 1, opposite → -1.
    Multiply by 100 for the reported percentage. Cast to float32 first.
    """
    af = a.float().flatten()
    bf = b.float().flatten()
    denom = af.norm() * bf.norm()
    return float((af.dot(bf) / denom).item())


_METRIC_FUNCS = {"mar": mar_between, "cos": cos_between}


# ---------------------------------------------------------------------------
# Model / tensor loading
# ---------------------------------------------------------------------------

def parse_model_arg(spec: str) -> tuple[str, Path]:
    """Parse a ``name:path`` ``--models`` token into ``(name, Path)``.

    Splits on the first ``:`` so absolute paths work (the path part may
    legitimately contain no colon on POSIX). Both halves must be non-empty.
    """
    name, sep, path = spec.partition(":")
    if not sep or not name.strip() or not path.strip():
        raise WeightSimilarityError(
            f"--models entry {spec!r} is not of the form name:path "
            "(e.g. uniform:bakeoff/uniform/merged)"
        )
    return name.strip(), Path(path.strip())


def _safetensors_path(model_dir: Path) -> Path:
    """Return ``<model_dir>/model.safetensors`` or raise a clear error."""
    candidate = model_dir / "model.safetensors"
    if not candidate.is_file():
        sharded = model_dir / "model.safetensors.index.json"
        hint = (
            " (found a sharded index instead — this script expects a single "
            "model.safetensors)" if sharded.is_file() else ""
        )
        raise WeightSimilarityError(
            f"model.safetensors not found in {model_dir}{hint}"
        )
    return candidate


def load_tensor(model_dir: Path, key: str):
    """Load a single tensor by key from ``<model_dir>/model.safetensors``.

    Uses ``safe_open``/``get_tensor`` so only the requested tensor is read
    from disk, never the full model. Returns a CPU torch tensor.

    Raises:
        WeightSimilarityError: model.safetensors missing, or key absent.
    """
    # Lazy import: keep this module importable on a torch-free laptop.
    from safetensors import safe_open

    st_path = _safetensors_path(model_dir)
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        if key not in f.keys():
            raise WeightSimilarityError(
                f"tensor key {key!r} not present in {st_path}"
            )
        return f.get_tensor(key)


def _file_size_gb(model_dir: Path) -> float:
    return round(_safetensors_path(model_dir).stat().st_size / 1e9, 2)


# ---------------------------------------------------------------------------
# Pairwise computation
# ---------------------------------------------------------------------------

def compute_pairwise(
    models: list[tuple[str, Path]],
    tensor_keys: tuple[str, ...],
    metrics: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    """Compute pairwise metric values per tensor key.

    Returns ``{tensor_key: [ {"a", "b", "<metric>_pct"...}, ... ]}`` with one
    entry per unordered model pair, ordered by ``itertools.combinations``.

    Raises:
        WeightSimilarityError: shape mismatch between two models for a key
            (or any error surfaced by :func:`load_tensor`).
    """
    pairwise: dict[str, list[dict[str, Any]]] = {}
    for key in tensor_keys:
        # Load this tensor once per model, then compare all pairs.
        tensors = {name: load_tensor(path, key) for name, path in models}
        rows: list[dict[str, Any]] = []
        for (name_a, _), (name_b, _) in itertools.combinations(models, 2):
            ta, tb = tensors[name_a], tensors[name_b]
            if ta.shape != tb.shape:
                raise WeightSimilarityError(
                    f"shape mismatch for {key!r}: {name_a} has {tuple(ta.shape)}, "
                    f"{name_b} has {tuple(tb.shape)}"
                )
            row: dict[str, Any] = {"a": name_a, "b": name_b}
            for metric in metrics:
                row[f"{metric}_pct"] = round(_METRIC_FUNCS[metric](ta, tb) * 100.0, 4)
            rows.append(row)
        pairwise[key] = rows
    return pairwise


def summarize(
    pairwise: dict[str, list[dict[str, Any]]], metric: str
) -> dict[str, float]:
    """Min / max / mean of one metric across all tensors and all pairs."""
    values = [
        row[f"{metric}_pct"]
        for rows in pairwise.values()
        for row in rows
    ]
    if not values:
        return {"min_pct": 0.0, "max_pct": 0.0, "mean_pct": 0.0}
    return {
        "min_pct": round(min(values), 4),
        "max_pct": round(max(values), 4),
        "mean_pct": round(sum(values) / len(values), 4),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "mar": "MAR (mean absolute relative difference, %)",
    "cos": "COS (cosine similarity, %)",
}


def _render_matrix(model_names: list[str], rows: list[dict[str, Any]], metric: str) -> str:
    """Lower-triangular matrix for one tensor + one metric, indexed by model.

    Rows/columns are referenced by ``[i]`` indices (see the model legend in
    the header) so the layout stays narrow regardless of model-name length.
    """
    value_of = {(r["a"], r["b"]): r[f"{metric}_pct"] for r in rows}
    n = len(model_names)
    col_w = 9

    header = " " * 6 + "".join(f"[{j}]".rjust(col_w) for j in range(n - 1))
    lines = [header]
    for i in range(1, n):
        cells = []
        for j in range(i):
            val = value_of.get((model_names[i], model_names[j]))
            if val is None:  # combinations() emits (j, i) with j < i
                val = value_of.get((model_names[j], model_names[i]))
            cells.append(f"{val:.2f}".rjust(col_w))
        lines.append(f"[{i}]".ljust(6) + "".join(cells))
    return "\n".join(lines)


def render_report(
    models: list[tuple[str, Path]],
    sizes: dict[str, float],
    tensor_keys: tuple[str, ...],
    metrics: tuple[str, ...],
    pairwise: dict[str, list[dict[str, Any]]],
) -> str:
    """Build the full console report string."""
    model_names = [name for name, _ in models]
    out: list[str] = []
    out.append("=== Weight similarity report ===")
    out.append("Metric(s): " + ", ".join(_METRIC_LABELS[m] for m in metrics))
    out.append("Models:")
    for idx, (name, path) in enumerate(models):
        out.append(f"  [{idx}] {name:<24} {path}  ({sizes[name]:.2f} GB)")
    out.append("")

    for key in tensor_keys:
        out.append(f"--- {key} ---")
        for metric in metrics:
            if len(metrics) > 1:
                out.append(f"  {metric.upper()}:")
            out.append(_render_matrix(model_names, pairwise[key], metric))
        out.append("")

    for metric in metrics:
        s = summarize(pairwise, metric)
        out.append(
            f"=== Summary ({metric.upper()}) === "
            f"min pairwise: {s['min_pct']:.2f}%   "
            f"max: {s['max_pct']:.2f}%   mean: {s['mean_pct']:.2f}%"
        )
    return "\n".join(out)


def build_json(
    models: list[tuple[str, Path]],
    sizes: dict[str, float],
    tensor_keys: tuple[str, ...],
    metric_choice: str,
    metrics: tuple[str, ...],
    pairwise: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Assemble the JSON output object."""
    obj: dict[str, Any] = {
        "models": [
            {"name": name, "path": str(path), "size_gb": sizes[name]}
            for name, path in models
        ],
        "tensors_inspected": list(tensor_keys),
        "metric": metric_choice,
        "pairwise": pairwise,
    }
    if len(metrics) == 1:
        obj["summary"] = summarize(pairwise, metrics[0])
    else:
        obj["summary"] = {m: summarize(pairwise, m) for m in metrics}
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pairwise weight-space similarity (MAR / cosine) between merged "
            "models, reading only selected tensors from each model.safetensors."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="NAME:PATH",
        help="Two or more 'name:path' entries; path is a merged-model dir "
        "containing model.safetensors.",
    )
    parser.add_argument(
        "--tensor-keys",
        nargs="+",
        default=list(DEFAULT_TENSOR_KEYS),
        metavar="KEY",
        help="Tensor keys to inspect.",
    )
    parser.add_argument(
        "--metric",
        choices=_VALID_METRICS,
        default="mar",
        help="mar = mean abs relative diff, cos = cosine similarity, both.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional JSON output path (console report is always printed).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)

    try:
        models = [parse_model_arg(s) for s in args.models]
        if len(models) < 2:
            raise WeightSimilarityError("need at least 2 models to compare")
        names = [n for n, _ in models]
        if len(set(names)) != len(names):
            raise WeightSimilarityError(f"duplicate model names: {names}")

        metrics: tuple[str, ...] = (
            ("mar", "cos") if args.metric == "both" else (args.metric,)
        )
        tensor_keys = tuple(args.tensor_keys)

        sizes = {name: _file_size_gb(path) for name, path in models}
        pairwise = compute_pairwise(models, tensor_keys, metrics)
    except WeightSimilarityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = render_report(models, sizes, tensor_keys, metrics, pairwise)
    print(report)

    if args.output is not None:
        obj = build_json(models, sizes, tensor_keys, args.metric, metrics, pairwise)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(obj, indent=2) + "\n")
        print(f"\nWrote JSON to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

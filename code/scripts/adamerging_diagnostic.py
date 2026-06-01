#!/usr/bin/env python3
"""Standalone diagnostic re-run of AdaMerging for the final report.

The bake-off (2026-05-26) discarded AdaMergingResult's coefficients and
loss_history because the registry shim collapsed the result to a plain
dict for the pipeline's uniform contract. This script re-runs
``dare_adamerging`` end-to-end on the four real adapters with the same
hyperparameters as the bake-off, captures the full :class:`AdaMergingResult`,
and persists the artifacts the report needs:

  - ``metrics.json``               (loss curve + coefficients + hyperparams)
  - ``loss_curve.png``             (loss vs step)
  - ``coefficients_heatmap.png``   ([N_tasks × N_layers] heatmap)

Outputs are written under a PERSISTENT directory (default
``/scratch/Group/emainelpe_group_model/adamerging_diagnostic/``), NOT
``/tmp`` — the bake-off's loss logs lived in a /tmp stdout file that
disappeared on the next pod restart, which is the gap this script
closes.

Usage::

    python scripts/adamerging_diagnostic.py \
        --adapters-dir loras/ \
        --output-dir /scratch/Group/emainelpe_group_model/adamerging_diagnostic/

Prerequisites:
    - ``scripts/fetch_adamerging_data.py`` has been run (unlabeled dataset cache).
    - A100-40g (or equivalent) GPU.
    - ``matplotlib`` installed (``pip install matplotlib`` if not).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
# nohup'd processes don't inherit cwd; insert explicitly.
sys.path.insert(0, str(_REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapters-dir",
        type=Path,
        default=Path("loras"),
        help="Directory holding the 4 adapter subdirs (math/general_knowledge/safety/multilingual).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/scratch/Group/emainelpe_group_model/adamerging_diagnostic"),
        help="Persistent output directory. NOT /tmp.",
    )
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", default="cuda")
    # Bake-off defaults (must match scripts/run_bakeoff.py).
    parser.add_argument("--drop-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-coefficient", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda-l2", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=100)
    parser.add_argument(
        "--aggregate-domains",
        action="store_true",
        help=(
            "Use AdaMerging's aggregated-objective mode: each optimizer "
            "update sees one batch per domain (n_tasks=4 batches), the "
            "per-domain entropies are averaged, then one backward + step. "
            "Matches the original paper formulation; recommended for fresh "
            "runs. ``--max-steps`` then counts optimizer UPDATES, not "
            "batches, so the data iterator must yield "
            "``max_steps * n_tasks`` tuples (the script provisions this "
            "automatically). Outputs go to a separate dir suffixed "
            "``_aggregated`` so the round-robin baseline is preserved."
        ),
    )
    return parser.parse_args(argv)


def _refuse_tmp(output_dir: Path) -> None:
    """Bake-off lesson: /tmp evaporates on pod restart. Refuse to write there."""
    resolved = output_dir.resolve()
    parts = resolved.parts
    if parts[:2] == ("/", "tmp") or "/tmp/" in str(resolved) + "/":
        raise ValueError(
            f"--output-dir {resolved} is under /tmp; this script refuses "
            "to write report artifacts there. Use /scratch."
        )


def _write_loss_curve(loss_history: list[float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, linewidth=1.4)
    ax.set_xlabel("step")
    ax.set_ylabel("loss (entropy + L2)")
    ax.set_title(f"AdaMerging loss curve ({len(loss_history)} steps)")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_coefficients_heatmap(
    coefficients,  # torch.Tensor [N_tasks, N_layers]
    task_names: list[str],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.asarray(coefficients.detach().cpu().tolist(), dtype=float)
    n_tasks, n_layers = arr.shape
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * n_layers), 0.8 * n_tasks + 1.5))
    vmax = float(np.abs(arr).max())
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(task_names)
    ax.set_xticks(range(0, n_layers, max(1, n_layers // 14)))
    ax.set_xlabel("layer index")
    ax.set_title(f"AdaMerging learned coefficients ({n_tasks} tasks × {n_layers} layers)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("coefficient value")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Aggregated mode writes to a separate sibling dir so the round-robin
    # baseline outputs are preserved for before/after comparison.
    if args.aggregate_domains:
        args.output_dir = args.output_dir.with_name(args.output_dir.name + "_aggregated")
    _refuse_tmp(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("adamerging_diagnostic")
    logger.info("Output dir: %s", args.output_dir)

    # Imports here so --help works without torch installed.
    import torch
    from transformers import AutoTokenizer

    from merge.data.unlabeled import assert_cache_exists, make_unlabeled_iter
    from merge.load_adapter import CANONICAL_DOMAINS, load_all
    from merge.methods.adamerging import adamerging
    from merge.methods.dare import dare
    from merge.qwen3_forward import make_qwen3_forward
    from merge.verify_spec import load_locked_spec

    logger.info("Step 1/6: Verify unlabeled dataset cache present ...")
    assert_cache_exists()

    logger.info("Step 2/6: Load 4 real adapters from %s ...", args.adapters_dir)
    locked_spec = load_locked_spec(_REPO_ROOT / "lora.yaml")
    adapters_by_domain = load_all(args.adapters_dir, locked_spec, device=args.device)
    # Order MUST match dare_adamerging's expectations: list(values()) matches
    # list(keys()) in insertion order. load_all returns CANONICAL_DOMAINS order.
    task_names = list(adapters_by_domain.keys())
    task_vectors = list(adapters_by_domain.values())
    logger.info("  task order: %s", task_names)
    assert task_names == list(CANONICAL_DOMAINS), (
        f"unexpected task order {task_names!r}; expected {list(CANONICAL_DOMAINS)!r}"
    )

    logger.info("Step 3/6: Apply DARE (drop_rate=%g, seed=%d) ...",
                args.drop_rate, args.seed)
    # Match dare_adamerging composition exactly: per-tv seed = seed + i,
    # inplace=True, rescale=True (default). See merge/methods/__init__.py.
    dared = [
        dare(tv, args.drop_rate, seed=args.seed + i, rescale=True, inplace=True)
        for i, tv in enumerate(task_vectors)
    ]

    cleanup_qwen3 = lambda: None
    try:
        logger.info("Step 4/6: Build forward_fn + data_iter ...")
        forward_fn, cleanup_qwen3 = make_qwen3_forward(
            base_model_repo=args.base_model, device=args.device,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_tasks = len(task_vectors)
        # Aggregated mode: each optimizer update consumes n_tasks batches,
        # so the iterator must be sized accordingly. make_unlabeled_iter
        # caps internally at its own max_steps, so we provision exactly
        # what adamerging() will consume in the worst case.
        iter_steps = args.max_steps * n_tasks if args.aggregate_domains else args.max_steps
        data_iter = make_unlabeled_iter(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_steps=iter_steps,
            seed=args.seed,
            device=args.device,
        )

        mode = "aggregated" if args.aggregate_domains else "round-robin (baseline)"
        logger.info(
            "Step 5/6: Run adamerging mode=%s (max_steps=%d, lr=%g, lambda_l2=%g, init=%g) ...",
            mode, args.max_steps, args.lr, args.lambda_l2, args.init_coefficient,
        )
        result = adamerging(
            dared,
            forward_fn=forward_fn,
            data_iter=data_iter,
            init_coefficient=args.init_coefficient,
            lr=args.lr,
            lambda_l2=args.lambda_l2,
            max_steps=args.max_steps,
            early_stop_patience=args.early_stop_patience,
            aggregate_domains=args.aggregate_domains,
        )

        logger.info("Step 6/6: Persist metrics + figures ...")
        coeffs = result.coefficients.detach().cpu().tolist()
        n_tasks = len(coeffs)
        n_layers = len(coeffs[0]) if coeffs else 0
        metrics = {
            "task_names": task_names,
            "n_tasks": n_tasks,
            "n_layers": n_layers,
            "steps_run": result.steps_run,
            "early_stopped": result.early_stopped,
            "loss_history": list(result.loss_history),
            "coefficients": coeffs,
            "hyperparams": {
                "method": "dare_adamerging",
                "drop_rate": args.drop_rate,
                "seed": args.seed,
                "rescale": True,
                "init_coefficient": args.init_coefficient,
                "lr": args.lr,
                "lambda_l2": args.lambda_l2,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "early_stop_patience": args.early_stop_patience,
                "base_model": args.base_model,
                "aggregate_domains": args.aggregate_domains,
            },
        }
        # Metrics first: figures are nice-to-have, but losing the JSON
        # because matplotlib choked (the 2026-05-26 baseline run crashed
        # at plot time, losing the curve) is the failure mode this
        # ordering prevents.
        metrics_path = args.output_dir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("  wrote %s", metrics_path)

        loss_path = args.output_dir / "loss_curve.png"
        try:
            _write_loss_curve(result.loss_history, loss_path)
            logger.info("  wrote %s", loss_path)
        except Exception as exc:  # noqa: BLE001 — figures are best-effort
            logger.warning(
                "loss curve plot failed (%s: %s); metrics.json still written.",
                type(exc).__name__, exc,
            )

        heatmap_path = args.output_dir / "coefficients_heatmap.png"
        try:
            _write_coefficients_heatmap(result.coefficients, task_names, heatmap_path)
            logger.info("  wrote %s", heatmap_path)
        except Exception as exc:  # noqa: BLE001 — figures are best-effort
            logger.warning(
                "coefficients heatmap failed (%s: %s); metrics.json still written.",
                type(exc).__name__, exc,
            )

        # Console summary.
        loss = result.loss_history
        if loss:
            pct = 100.0 * (loss[0] - loss[-1]) / max(abs(loss[0]), 1e-9)
            print(f"\n=== AdaMerging diagnostic summary ===")
            print(f"steps_run:     {result.steps_run}")
            print(f"early_stopped: {result.early_stopped}")
            print(f"initial loss:  {loss[0]:.4f}")
            print(f"final loss:    {loss[-1]:.4f}  ({pct:+.2f}% change)")
            arr = result.coefficients.detach().cpu()
            print(f"coefficients shape: {tuple(arr.shape)} (tasks × layers)")
            for i, name in enumerate(task_names):
                row = arr[i]
                print(
                    f"  {name:<20s} min={row.min().item():+.4f} "
                    f"max={row.max().item():+.4f} mean={row.mean().item():+.4f}"
                )
            print(f"\nArtifacts: {args.output_dir}/")
        return 0
    except Exception:
        logger.exception("adamerging_diagnostic FAILED")
        return 1
    finally:
        cleanup_qwen3()
        if "torch" in sys.modules:
            import torch as _t  # noqa: PLC0415
            if _t.cuda.is_available():
                _t.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())

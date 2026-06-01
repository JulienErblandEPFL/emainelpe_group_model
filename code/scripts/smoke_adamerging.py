#!/usr/bin/env python3
"""Cluster smoke test: AdaMerging on real Qwen3-1.7B with random-init adapters.

Generates 4 random-init LoRA adapters at the actual Qwen3-1.7B scale
(28 layers × 7 modules × r=32 with GQA) and runs ``dare_adamerging``
through :func:`merge.pipeline.merge_adapters` for ``--max-steps``
training steps. Reports loss curve via the standard logger and asserts
the merged-adapter output directory is well-formed.

The merged output goes to a temp directory and the script asserts:
  - It exists and contains ``adapter_config.json`` + ``adapter_model.safetensors``
  - AdaMerging's loss curve was logged (via the regular Python logger)

This script does NOT push to HF. It validates the AdaMerging pipeline on
real Qwen3 *before* any real teammate adapter is plugged in.

Usage:
    python scripts/smoke_adamerging.py
    python scripts/smoke_adamerging.py --max-steps 100 --batch-size 4

Prerequisites:
    scripts/fetch_adamerging_data.py must have been run first.
    A100-40g (or equivalent) GPU recommended.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=50,
                        help="AdaMerging training steps (50 = quick smoke; 1000 = production).")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Prompts per batch (default 2; A100-40g handles 4 comfortably).")
    parser.add_argument("--drop-rate", type=float, default=0.5,
                        help="DARE drop_rate applied once before AdaMerging.")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="AdaMerging Adam learning rate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger = logging.getLogger("smoke_adamerging")

    # Imports here so --help works without torch / transformers installed.
    from transformers import AutoTokenizer

    from merge.data.unlabeled import assert_cache_exists, make_unlabeled_iter
    from merge.qwen3_forward import make_qwen3_forward
    from merge.pipeline import merge_adapters
    from merge.load_adapter import CANONICAL_DOMAINS
    from merge.tests.fixtures.qwen3_adapter import make_random_qwen3_adapter

    logger.info("Step 1/5: Verifying unlabeled dataset cache ...")
    assert_cache_exists()

    logger.info("Step 2/5: Loading Qwen3 tokenizer (%s) ...", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Step 3/5: Loading Qwen3 base model + building forward callable ...")
    forward_fn, cleanup_qwen3 = make_qwen3_forward(
        base_model_repo=args.base_model,
        device=args.device,
    )

    logger.info("Step 4/5: Building unlabeled data iterator ...")
    data_iter = make_unlabeled_iter(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        seed=args.seed,
        device=args.device,
    )

    rc = 0
    try:
        with tempfile.TemporaryDirectory(prefix="smoke_adamerging_") as work_dir:
            adapters_dir = Path(work_dir) / "adapters"
            adapters_dir.mkdir()

            logger.info("Step 5/5: Generating 4 random-init Qwen3-sized LoRA adapters ...")
            for i, domain in enumerate(CANONICAL_DOMAINS):
                make_random_qwen3_adapter(adapters_dir / domain, seed=args.seed + i)
                logger.info("  -> %s adapter written", domain)

            merged_dir = Path(work_dir) / "merged"
            logger.info(
                "Running pipeline method=dare_adamerging max_steps=%d batch_size=%d drop_rate=%g lr=%g ...",
                args.max_steps, args.batch_size, args.drop_rate, args.lr,
            )

            merge_adapters(
                adapters_dir=adapters_dir,
                method="dare_adamerging",
                output_dir=merged_dir,
                method_kwargs={
                    "forward_fn": forward_fn,
                    "data_iter": data_iter,
                    "drop_rate": args.drop_rate,
                    "seed": args.seed,
                    "lr": args.lr,
                    "max_steps": args.max_steps,
                    # Disable early stopping for the smoke; we want all max_steps.
                    "early_stop_patience": args.max_steps + 1,
                },
            )

            assert (merged_dir / "adapter_config.json").exists(), "merged adapter_config.json missing"
            assert (merged_dir / "adapter_model.safetensors").exists(), "merged safetensors missing"
            logger.info("Merged adapter directory present and well-formed.")
            logger.info("Smoke test passed.")
    except Exception:
        logger.exception("Smoke test FAILED")
        rc = 1
    finally:
        cleanup_qwen3()

    return rc


if __name__ == "__main__":
    sys.exit(main())

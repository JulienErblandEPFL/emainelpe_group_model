#!/usr/bin/env python3
"""Pre-download the 4 AdaMerging unlabeled datasets to HF_HOME cache.

Usage:
    python scripts/fetch_adamerging_data.py
    python scripts/fetch_adamerging_data.py --cache-dir /scratch/hf_cache

After this script succeeds, :func:`merge.data.unlabeled.make_unlabeled_iter`
can load the datasets from cache without further network access.

Cluster pods are preemptible. Running this script as a separate step (not
inside the training loop) means a failed download is a clean retry rather
than a corrupt cache that silently breaks a subsequent training run.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from merge.data.unlabeled import UNLABELED_DATASETS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "HF cache root. Datasets are stored under <cache-dir>/datasets. "
            "Defaults to HF_HOME env var or ~/.cache/huggingface."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("fetch_adamerging_data")

    if args.cache_dir is not None:
        cache_root = Path(args.cache_dir).expanduser()
    else:
        env = os.environ.get("HF_HOME")
        cache_root = Path(env).expanduser() if env else Path("~/.cache/huggingface").expanduser()
    datasets_cache = cache_root / "datasets"
    logger.info("HF cache root:       %s", cache_root)
    logger.info("Datasets subdir:     %s", datasets_cache)

    # Lazy import so --help works without datasets installed.
    from datasets import load_dataset

    failures: list[tuple[str, Exception]] = []
    for cfg in UNLABELED_DATASETS:
        logger.info(
            "Fetching domain=%-18s name=%s config=%s split=%s",
            cfg.domain_name, cfg.hf_name, cfg.hf_config, cfg.split,
        )
        try:
            if cfg.hf_config is not None:
                ds = load_dataset(
                    cfg.hf_name, cfg.hf_config,
                    split=cfg.split,
                    cache_dir=str(datasets_cache),
                )
            else:
                ds = load_dataset(
                    cfg.hf_name,
                    split=cfg.split,
                    cache_dir=str(datasets_cache),
                )
            n_total = len(ds)
            if cfg.type_filter_prefix is not None:
                prefix = cfg.type_filter_prefix
                ds = ds.filter(
                    lambda row: isinstance(row.get("type"), str) and row["type"].startswith(prefix)
                )
                logger.info(
                    "  -> loaded %d examples (filtered to %d via type prefix=%r)",
                    n_total, len(ds), prefix,
                )
            else:
                logger.info("  -> loaded %d examples", n_total)
        except Exception as e:  # network errors, schema changes, etc.
            logger.error("  -> FAILED: %s", e)
            failures.append((cfg.hf_name, e))

    if failures:
        logger.error("--- Summary: %d dataset(s) failed to fetch ---", len(failures))
        for name, e in failures:
            logger.error("  %s: %s", name, e)
        return 1

    logger.info(
        "All %d datasets cached. Run scripts/smoke_adamerging.py to validate the pipeline.",
        len(UNLABELED_DATASETS),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

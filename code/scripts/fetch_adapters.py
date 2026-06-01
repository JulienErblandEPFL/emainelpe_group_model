#!/usr/bin/env python3
"""Fetch the four specialist LoRA adapters from the Hugging Face Hub.

The merge pipeline consumes adapters from a local directory; it never
downloads from the Hub itself (see ``merge/load_adapter.py``). This script
is the missing first step: it pulls each teammate's adapter into the
canonical local layout the pipeline expects, then verifies every one
against the locked spec (``lora.yaml``) before anything GPU-bound runs.

Source repos (one per domain)::

    cs-552-2026-emainelpe/math_model
    cs-552-2026-emainelpe/general_knowledge_model
    cs-552-2026-emainelpe/safety_model
    cs-552-2026-emainelpe/multilingual_model

Output layout (``--target-dir``, default ``loras/``)::

    loras/
    ├── math/{adapter_config.json, adapter_model.safetensors}
    ├── general_knowledge/{...}
    ├── safety/{...}
    └── multilingual/{...}

The two adapter files are written *flat* in each domain subdir regardless
of where they live inside the source repo (some repos keep them under an
``adapter/`` subfolder, others at the root) — the pipeline only reads the
two files, flat, per ``adapters/README.md``.

Usage::

    python scripts/fetch_adapters.py --target-dir loras/

Auth: pass ``--token`` explicitly, or rely on the ambient Hugging Face
credentials (``HF_TOKEN`` env var, or a prior ``hf auth login``). The
source repos are public, so a token is only needed if the org later goes
private.

Exit codes: ``0`` = all four fetched and spec-verified; ``1`` = at least
one adapter missing its files or failing locked-spec verification.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
# nohup'd / cluster invocations don't inherit cwd on sys.path; insert
# the repo root (``code/``) explicitly so ``import merge`` resolves.
sys.path.insert(0, str(_REPO_ROOT))

from merge.load_adapter import CANONICAL_DOMAINS  # noqa: E402
from merge.verify_spec import load_locked_spec, verify  # noqa: E402


logger = logging.getLogger("fetch_adapters")

# HF org that owns the five model repos (see CLAUDE.md / README.md).
_HF_ORG = "cs-552-2026-emainelpe"

# The two files the merge pipeline reads. Everything else in a source
# repo (tokenizer, generation_config, training logs) is ignored.
_REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")


def repo_id_for(domain: str) -> str:
    """Map a canonical domain to its HF repo slug (``<org>/<domain>_model``)."""
    return f"{_HF_ORG}/{domain}_model"


def _flatten_into(snapshot_dir: Path, dest_dir: Path) -> None:
    """Copy the two adapter files from ``snapshot_dir`` (searched
    recursively) into ``dest_dir`` at the top level.

    Raises:
        FileNotFoundError: if either required file is absent from the
            downloaded snapshot.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for fname in _REQUIRED_FILES:
        matches = sorted(snapshot_dir.rglob(fname))
        if not matches:
            raise FileNotFoundError(
                f"{fname!r} not found anywhere in downloaded snapshot "
                f"{snapshot_dir} — the source repo layout is unexpected."
            )
        if len(matches) > 1:
            # Prefer a copy under an 'adapter/' subdir if present, else the
            # shallowest path. Ambiguity is logged so it is never silent.
            logger.warning(
                "%s found in %d locations under %s; picking %s",
                fname, len(matches), snapshot_dir, matches[0],
            )
        shutil.copy2(matches[0], dest_dir / fname)


def fetch_one(domain: str, target_dir: Path, token: str | None) -> Path:
    """Download one domain's adapter into ``target_dir/<domain>/``.

    Returns the per-domain destination directory. Lazy-imports
    ``huggingface_hub`` so ``--help`` works without the dep installed.
    """
    from huggingface_hub import snapshot_download

    repo_id = repo_id_for(domain)
    dest_dir = target_dir / domain
    logger.info("fetching %s -> %s", repo_id, dest_dir)

    # allow_patterns covers both layouts: files under an ``adapter/``
    # subfolder, or at the repo root. snapshot_download preserves repo
    # structure in its cache; _flatten_into then copies the two files flat.
    snapshot_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[
                "adapter/*",
                "adapter_config.json",
                "adapter_model.safetensors",
            ],
            token=token,
        )
    )
    _flatten_into(snapshot_dir, dest_dir)
    return dest_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("loras"),
        help="Parent dir to write the 4 adapter subdirs into (default: loras/).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "HF access token. If omitted, ambient credentials are used "
            "(HF_TOKEN env var or a prior 'hf auth login')."
        ),
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(CANONICAL_DOMAINS),
        help=(
            "Subset of domains to fetch (default: all 4). Must be a subset "
            f"of {list(CANONICAL_DOMAINS)}."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    unknown = set(args.domains) - set(CANONICAL_DOMAINS)
    if unknown:
        logger.error(
            "unknown domain(s) %s; valid domains are %s",
            sorted(unknown), list(CANONICAL_DOMAINS),
        )
        return 1

    locked_spec = load_locked_spec(_REPO_ROOT / "lora.yaml")

    failures: list[str] = []
    for domain in args.domains:
        try:
            dest_dir = fetch_one(domain, args.target_dir, args.token)
        except Exception as exc:  # network / missing-file / auth
            logger.error("FAILED to fetch %s: %s", domain, exc)
            failures.append(domain)
            continue

        # Both files present?
        missing = [f for f in _REQUIRED_FILES if not (dest_dir / f).is_file()]
        if missing:
            logger.error("%s: missing files after download: %s", domain, missing)
            failures.append(domain)
            continue

        # Locked-spec verification (same check the pipeline runs at startup).
        result = verify(dest_dir / "adapter_config.json", locked_spec)
        if not result.passed:
            logger.error("%s: FAILED locked-spec verification", domain)
            logger.error(result.summary)
            for fr in result.field_results:
                if not fr.passed:
                    logger.error(
                        "    %s: expected=%r, got=%r", fr.field, fr.expected, fr.actual
                    )
            failures.append(domain)
            continue

        logger.info("%s: OK (spec-verified) -> %s", domain, dest_dir)

    if failures:
        logger.error(
            "\n%d/%d adapter(s) failed: %s. Fix before running the bake-off.",
            len(failures), len(args.domains), failures,
        )
        return 1

    logger.info(
        "\nAll %d adapters fetched and spec-verified under %s/",
        len(args.domains), args.target_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

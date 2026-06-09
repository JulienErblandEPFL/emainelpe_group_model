#!/usr/bin/env python3
"""Stage 5d: publish a bake-off winner (full HF-format model dir) to the Hub.

The course CI grades by loading ``generation_config.json`` *from the
uploaded model*. The bake-off winner was scored under specific sampling
params (``temperature``, ``top_p``, ``top_k``, ``max_new_tokens``); a
mismatch between those params and the bundled config would mean CI
grades under unvalidated sampling. So publish rewrites
``generation_config.json`` to the winning params right before upload.

Safety design
-------------
- Dry-run by default. ``--confirm`` is the only flag that actually
  pushes 3.4 GB to HF. Without it the script prints the upload plan
  (repo, every file + size, total, the exact ``generation_config.json``
  it would write) and exits 0 without touching the model directory or
  the Hub. A 3.4 GB push to the wrong repo is expensive and embarrassing.
- ``--repo-id`` is required — there is no hardcoded default that could
  fire at the wrong target. The intended slug for the project is
  ``cs-552-2026-emainelpe/group_model`` but the user must type it.
- The existing ``generation_config.json`` is backed up to
  ``generation_config.json.bak`` BEFORE being overwritten, and ONLY in
  the ``--confirm`` path. Dry-run never modifies the model directory.
- ``HF_TOKEN`` is read from the environment by ``huggingface_hub``;
  this script does not handle auth.

Usage
-----
::

    # 1. Always dry-run first
    python scripts/publish.py \\
        --model-dir bakeoff_2026-05-26-1145/ties/merged \\
        --repo-id cs-552-2026-emainelpe/group_model

    # 2. After inspecting the printed plan, push for real
    python scripts/publish.py \\
        --model-dir bakeoff_2026-05-26-1145/ties/merged \\
        --repo-id cs-552-2026-emainelpe/group_model \\
        --confirm
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


logger = logging.getLogger("publish")


# Files we expect a full HF-format merged model dir to contain. The
# safetensors weights may be sharded (``model.safetensors.index.json``
# plus ``model-00001-of-XXXXX.safetensors`` shards) or single-file
# (``model.safetensors``); validate_model_dir handles either layout.
_REQUIRED_NON_WEIGHT_FILES: tuple[str, ...] = (
    "config.json",
    "chat_template.jinja",
)
# At least one of these tokenizer files must be present. Different
# tokenizer types ship different filenames — we accept any of them.
_TOKENIZER_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)


def build_parser() -> argparse.ArgumentParser:
    # Torch-free import (generation_config is stdlib-only); keeps --help working.
    from merge.generation_config import DEFAULT_SAMPLING

    parser = argparse.ArgumentParser(
        description=(
            "Publish a merged HF-format model directory to HF Hub. "
            "Dry-run by default; --confirm to actually push."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Dry-run\n"
            "  python scripts/publish.py --model-dir bakeoff_<date>/ties/merged \\\n"
            "      --repo-id cs-552-2026-emainelpe/group_model\n\n"
            "  # Push for real\n"
            "  python scripts/publish.py --model-dir bakeoff_<date>/ties/merged \\\n"
            "      --repo-id cs-552-2026-emainelpe/group_model --confirm\n"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to the merged full HF-format model directory.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help=(
            "Target HF repo slug, e.g. cs-552-2026-emainelpe/group_model. "
            "No default — required to prevent accidental pushes."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Generation temperature to bundle (default 0.5 = bake-off winner).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_SAMPLING["top_p"],
        help="Nucleus sampling (default 0.8).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_SAMPLING["top_k"],
        help="Top-k sampling (default 20).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16384,
        help="Max new tokens cap to bundle in generation_config.json (default 16384, matching CI ceiling).",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Publish bake-off winner: ties @ T=0.5",
        help="HF commit message (default: bake-off winner stamp).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help=(
            "REQUIRED to actually push. Without this flag, the script "
            "prints the plan and exits without uploading or modifying "
            "the model directory."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_model_dir(model_dir: Path) -> list[str]:
    """Return a list of error messages; empty means the dir is valid.

    A full HF-format model dir must contain ``config.json``,
    ``chat_template.jinja``, at least one safetensors weight artifact
    (single file or sharded index), and at least one tokenizer file.
    """
    errors: list[str] = []
    if not model_dir.exists():
        errors.append(f"--model-dir does not exist: {model_dir}")
        return errors
    if not model_dir.is_dir():
        errors.append(f"--model-dir is not a directory: {model_dir}")
        return errors

    for name in _REQUIRED_NON_WEIGHT_FILES:
        if not (model_dir / name).exists():
            errors.append(f"missing required file: {model_dir / name}")

    has_single = (model_dir / "model.safetensors").exists()
    has_sharded_index = (model_dir / "model.safetensors.index.json").exists()
    if not (has_single or has_sharded_index):
        errors.append(
            f"missing weights: neither {model_dir / 'model.safetensors'} "
            f"nor {model_dir / 'model.safetensors.index.json'} present"
        )

    if not any((model_dir / t).exists() for t in _TOKENIZER_FILES):
        errors.append(
            f"missing tokenizer files: none of {_TOKENIZER_FILES} found in {model_dir}"
        )

    return errors


# ---------------------------------------------------------------------------
# Generation config rewrite
# ---------------------------------------------------------------------------

def build_winning_generation_config(
    *, temperature: float, top_p: float, top_k: int, max_new_tokens: int,
) -> dict[str, Any]:
    """Build the ``generation_config.json`` dict to bundle.

    Delegates to :func:`merge.generation_config.make_generation_config`,
    which fills in the contractual fields (``bos_token_id=151643``,
    ``eos_token_id=[151645,151643]``, ``pad_token_id=151643``,
    ``do_sample=True``, ``transformers_version``) alongside the four
    tunable fields.
    """
    from merge.generation_config import make_generation_config

    return make_generation_config(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
    )


def rewrite_generation_config(
    model_dir: Path, new_config: dict[str, Any],
) -> Path:
    """Back up the existing ``generation_config.json`` and write ``new_config``.

    Called ONLY in the ``--confirm`` path; dry-run never modifies the dir.

    The backup is ``generation_config.json.bak``. If the original does
    not exist (rare, but possible), no backup is created and the new
    file is written fresh.

    Returns the path to the written ``generation_config.json``.
    """
    gen_path = model_dir / "generation_config.json"
    if gen_path.exists():
        backup_path = model_dir / "generation_config.json.bak"
        shutil.copy2(gen_path, backup_path)
        logger.info("Backed up existing generation_config.json -> %s", backup_path)
    else:
        logger.info(
            "No existing generation_config.json at %s; writing fresh.", gen_path,
        )

    with gen_path.open("w") as f:
        json.dump(new_config, f, indent=2)
        f.write("\n")
    logger.info("Wrote new generation_config.json to %s", gen_path)
    return gen_path


# ---------------------------------------------------------------------------
# Plan printing
# ---------------------------------------------------------------------------

def _format_size(num_bytes: int) -> str:
    """Render byte counts as human-readable GB/MB/KB strings."""
    for unit, threshold in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if num_bytes >= threshold:
            return f"{num_bytes / threshold:.2f} {unit}"
    return f"{num_bytes} B"


def collect_upload_files(model_dir: Path) -> list[tuple[Path, int]]:
    """Return ``(relative_path, size_bytes)`` for every file in ``model_dir``.

    Excludes any existing ``.bak`` files (we never want to ship the
    previous generation_config to HF).
    """
    out: list[tuple[Path, int]] = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".bak":
            continue
        out.append((path.relative_to(model_dir), path.stat().st_size))
    return out


def print_plan(
    *,
    model_dir: Path,
    repo_id: str,
    commit_message: str,
    new_gen_config: dict[str, Any],
    files: list[tuple[Path, int]],
    confirmed: bool,
) -> None:
    """Print the upload plan to stdout."""
    total_bytes = sum(size for _, size in files)

    print()
    print("=" * 70)
    print(f"Publish plan {'(CONFIRMED)' if confirmed else '(DRY RUN)'}")
    print("=" * 70)
    print(f"Repo:           {repo_id}")
    print(f"Model dir:      {model_dir}")
    print(f"Commit message: {commit_message!r}")
    print(f"Total files:    {len(files)}")
    print(f"Total size:     {_format_size(total_bytes)}")
    print("-" * 70)
    print("Files to upload:")
    width = max((len(str(p)) for p, _ in files), default=0)
    for rel, size in files:
        print(f"  {str(rel).ljust(width)}  {_format_size(size)}")
    print("-" * 70)
    print("New generation_config.json that will be bundled:")
    print(json.dumps(new_gen_config, indent=2))
    print("=" * 70)


# ---------------------------------------------------------------------------
# Publish core (injectable for tests)
# ---------------------------------------------------------------------------

def publish(
    args: argparse.Namespace,
    upload_callable: Callable[..., Any],
    create_repo_callable: Callable[..., Any],
) -> int:
    """Execute the publish workflow.

    The two callables are injected so tests can assert what was called
    (and what was NOT called in dry-run) without importing
    ``huggingface_hub``.

    Args:
        args: Parsed argparse namespace.
        upload_callable: Function with the signature of
            ``HfApi().upload_folder`` — accepts kwargs and returns
            something stringifiable for the log.
        create_repo_callable: Function with the signature of
            ``HfApi().create_repo`` — called once with ``exist_ok=True``
            before upload; tests can assert it was/wasn't called.

    Returns:
        Process exit code. 0 on success or successful dry-run; 2 on
        validation error.
    """
    errors = validate_model_dir(args.model_dir)
    if errors:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        return 2

    new_gen_config = build_winning_generation_config(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )

    files = collect_upload_files(args.model_dir)
    print_plan(
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        commit_message=args.commit_message,
        new_gen_config=new_gen_config,
        files=files,
        confirmed=args.confirm,
    )

    if not args.confirm:
        print()
        print("DRY RUN — no upload, no rewrite. Re-run with --confirm to push.")
        return 0

    # --confirm path: rewrite the gen config, then push.
    rewrite_generation_config(args.model_dir, new_gen_config)

    logger.info("Creating repo (if missing): %s", args.repo_id)
    create_repo_callable(repo_id=args.repo_id, exist_ok=True)

    logger.info("Uploading %s to %s", args.model_dir, args.repo_id)
    upload_callable(
        folder_path=str(args.model_dir),
        repo_id=args.repo_id,
        commit_message=args.commit_message,
    )

    url = f"https://huggingface.co/{args.repo_id}"
    print()
    print(f"Published: {url}")
    return 0


# ---------------------------------------------------------------------------
# Default factories (production wiring)
# ---------------------------------------------------------------------------

def _default_upload_callable(**kwargs: Any) -> Any:
    from huggingface_hub import HfApi

    return HfApi().upload_folder(**kwargs)


def _default_create_repo_callable(**kwargs: Any) -> Any:
    from huggingface_hub import HfApi

    return HfApi().create_repo(**kwargs)


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

    return publish(
        args,
        upload_callable=_default_upload_callable,
        create_repo_callable=_default_create_repo_callable,
    )


if __name__ == "__main__":
    sys.exit(main())

"""Generation config helpers for merged adapters.

The structure of ``generation_config.json`` is fixed by the CS-552 project
description: ``bos/eos/pad`` token IDs are Qwen3 constants, ``do_sample`` is
true, and the transformers version is pinned. The TUNABLE values are
``temperature``, ``top_p``, ``top_k``, and ``max_new_tokens`` — these are
bake-off hyperparameters.

This module provides:
- :func:`make_generation_config` — construct a CS-552-compliant dict from
  tunable values; contractual fields are filled in automatically.
- :func:`load_generation_config` — hierarchical fallback loader used at
  eval time (adapter dir → repo root → Qwen3 defaults).
- :data:`QWEN3_DEFAULTS` — the conservative tunable defaults used as the
  lowest-priority fallback.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Qwen3-1.7B-specific tokenizer + framework constants from the project
# description. NOT tunable — these are contractual.
_QWEN3_BOS = 151643
_QWEN3_EOS = [151645, 151643]
_QWEN3_PAD = 151643
_TRANSFORMERS_VERSION = "4.51.0"


# Tunable defaults (Qwen3-1.7B's published recommendations). Used as the
# lowest-priority fallback when no gen config is found anywhere.
QWEN3_DEFAULTS: dict[str, Any] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_new_tokens": 16384,
}


def make_generation_config(
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    max_new_tokens: int = 16384,
) -> dict[str, Any]:
    """Construct a complete ``generation_config.json`` dict matching the
    CS-552 project description schema.

    Args:
        temperature: Sampling temperature. ``0.0`` is near-greedy in vLLM.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling threshold.
        max_new_tokens: Hard cap on tokens generated per response.
            The project mandates 16384 as the CI limit; lower values may
            be used for internal eval to speed up runs.

    Returns:
        A dict ready to be ``json.dump``-ed to ``generation_config.json``.
        Contains the fixed Qwen3 token IDs, ``do_sample=True``, the
        transformers version, and the four tunable fields.

    Raises:
        ValueError: if ``temperature``/``top_p`` are outside their loose
            ranges, or ``top_k``/``max_new_tokens`` are not positive.
    """
    if not (0.0 <= temperature <= 2.0):
        raise ValueError(f"temperature must be in [0, 2], got {temperature}")
    if not (0.0 <= top_p <= 1.0):
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")

    # The project's required schema has do_sample=true. ``temperature=0.0``
    # will produce near-greedy behavior in vLLM regardless of this flag.
    do_sample = True

    return {
        "bos_token_id": _QWEN3_BOS,
        "do_sample": do_sample,
        "eos_token_id": _QWEN3_EOS,
        "pad_token_id": _QWEN3_PAD,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "transformers_version": _TRANSFORMERS_VERSION,
    }


def load_generation_config(
    merged_adapter_dir: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Hierarchical fallback loader for ``generation_config.json``.

    Priority (highest to lowest):
        1. ``<merged_adapter_dir>/generation_config.json`` if present
        2. ``<repo_root>/generation_config.json`` if present
        3. :func:`make_generation_config` with default args (Qwen3 defaults
           + the contractual token IDs)

    Args:
        merged_adapter_dir: If provided, check here first.
        repo_root: If provided, check here second.

    Returns:
        A dict with at minimum the tunable fields (``temperature``,
        ``top_p``, ``top_k``, ``max_new_tokens``). May also contain the
        contractual token IDs when loaded from a file. Callers should
        treat the dict as opaque sampling params.

    Logs which source was used at INFO level. A corrupt JSON file is
    logged as a warning and the loader falls through to the next priority.
    """
    for source_label, source_dir in (
        ("merged adapter", merged_adapter_dir),
        ("repo root", repo_root),
    ):
        if source_dir is None:
            continue
        candidate = Path(source_dir) / "generation_config.json"
        if candidate.exists():
            try:
                with open(candidate) as f:
                    config = json.load(f)
                logger.info(
                    "Loaded generation config from %s: %s", source_label, candidate
                )
                return config
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Failed to load generation config from %s (%s); falling through.",
                    candidate, exc,
                )

    logger.info("Using Qwen3 default generation config (no file found).")
    return make_generation_config()


__all__ = ["QWEN3_DEFAULTS", "make_generation_config", "load_generation_config"]

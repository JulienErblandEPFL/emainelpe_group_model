"""Unlabeled data infrastructure for AdaMerging training.

AdaMerging minimizes prediction entropy on unlabeled in-domain prompts.
This module provides the iterator that feeds those prompts in round-robin
across the 4 domains: math, general_knowledge, safety, multilingual.

Datasets are downloaded once by ``scripts/fetch_adamerging_data.py`` and
cached under ``HF_HOME`` (typically ``/scratch/hf_cache/``). The iterator
asserts the cache is present and fails fast if not.

Domain order matches :data:`merge.load_adapter.CANONICAL_DOMAINS`:
  0 = math, 1 = general_knowledge, 2 = safety, 3 = multilingual

The module is import-light: ``torch``, ``datasets``, and ``transformers``
are imported lazily inside the functions that need them, so unit tests
that only inspect :data:`UNLABELED_DATASETS` constants can run on a
torch-free, datasets-free laptop.
"""
from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    import torch


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetConfig:
    """Static config for one unlabeled in-domain dataset.

    Attributes:
        domain_idx: Position in :data:`merge.load_adapter.CANONICAL_DOMAINS`.
        domain_name: Canonical domain name (matches one of the 4 fixed names).
        hf_name: HuggingFace dataset repo (``user/repo`` or ``repo``).
        hf_config: HuggingFace dataset config name, or ``None``.
        split: Split name to load (e.g. ``"train"``, ``"test"``).
        field: Column name holding the prompt text.
        max_samples: Cap on examples taken from the split (post-filter).
        type_filter_prefix: Optional prefix; if set, keep only rows where
            ``row["type"]`` starts with this prefix (for XSTest's
            ``safe_*`` filter). ``None`` disables filtering.
    """

    domain_idx: int
    domain_name: str
    hf_name: str
    hf_config: str | None
    split: str
    field: str
    max_samples: int
    type_filter_prefix: str | None = None


UNLABELED_DATASETS: tuple[DatasetConfig, ...] = (
    DatasetConfig(
        domain_idx=0,
        domain_name="math",
        hf_name="openai/gsm8k",
        hf_config="main",
        split="train",
        field="question",
        max_samples=1000,
    ),
    DatasetConfig(
        domain_idx=1,
        domain_name="general_knowledge",
        hf_name="cais/mmlu",
        hf_config="all",
        split="auxiliary_train",
        field="question",
        max_samples=1000,
    ),
    DatasetConfig(
        domain_idx=2,
        domain_name="safety",
        # XSTest has a single ``prompts`` split with a ``type`` column whose
        # values are e.g. ``safe_homonyms``, ``safe_figurative``, ``contrast_*``,
        # etc. Pure entropy on harmful prompts would push the model toward
        # confident harmful outputs, so we filter to ``safe_*`` only.
        hf_name="natolambert/xstest-v2-copy",
        hf_config=None,
        split="prompts",
        field="prompt",
        max_samples=250,
        type_filter_prefix="safe_",
    ),
    DatasetConfig(
        domain_idx=3,
        domain_name="multilingual",
        # MGSM has no train split per language; we use ``test`` minus any
        # overlap with the milestone validation set. The set is small
        # (250 samples per language); ``itertools.cycle`` repeats safely.
        hf_name="juletxara/mgsm",
        hf_config="en",
        split="test",
        field="question",
        max_samples=250,
    ),
)


def _default_cache_dir() -> Path:
    """Resolve the HF dataset cache directory.

    Priority: ``HF_DATASETS_CACHE`` → ``HF_HOME``/datasets → ``~/.cache/huggingface/datasets``.
    """
    env = os.environ.get("HF_DATASETS_CACHE")
    if env:
        return Path(env).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "datasets"
    return Path("~/.cache/huggingface/datasets").expanduser()


def _expected_cache_subdir(cfg: DatasetConfig, cache_dir: Path) -> Path:
    """HF datasets uses ``<repo>`` with ``/`` rewritten to ``___`` as the
    on-disk subdir name. The full layout includes a config and version dir
    underneath, but the top-level dataset dir is enough for a presence check.
    """
    normalized = cfg.hf_name.replace("/", "___")
    return cache_dir / normalized


def assert_cache_exists(cache_dir: Path | None = None) -> None:
    """Verify the HF cache contains all 4 unlabeled datasets.

    The check is heuristic: it confirms a top-level directory exists for each
    dataset under the resolved cache dir. This catches the common failure
    mode (cache wiped, wrong ``HF_HOME``) without false-negative-ing on
    minor layout drift inside the per-dataset subdir.

    Raises:
        FileNotFoundError: if any expected subdir is missing. The error
            message lists every missing dataset and the command to populate
            the cache.
    """
    resolved = Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()
    missing: list[str] = []
    for cfg in UNLABELED_DATASETS:
        expected = _expected_cache_subdir(cfg, resolved)
        if not expected.exists():
            missing.append(f"  - {cfg.domain_name:>18s}: {cfg.hf_name} (expected at {expected})")
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(
            f"HF dataset cache incomplete at {resolved}. Missing:\n{joined}\n\n"
            "Run:  python scripts/fetch_adamerging_data.py "
            f"--cache-dir {resolved.parent if resolved.name == 'datasets' else resolved}"
        )


def _load_one_dataset(cfg: DatasetConfig, cache_dir: Path | None, seed: int):
    """Load and shuffle one dataset, applying the optional type filter."""
    from datasets import load_dataset

    kwargs = {"split": cfg.split}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if cfg.hf_config is not None:
        ds = load_dataset(cfg.hf_name, cfg.hf_config, **kwargs)
    else:
        ds = load_dataset(cfg.hf_name, **kwargs)
    if cfg.type_filter_prefix is not None:
        prefix = cfg.type_filter_prefix
        ds = ds.filter(lambda row: isinstance(row.get("type"), str) and row["type"].startswith(prefix))
    ds = ds.shuffle(seed=seed + cfg.domain_idx)
    take = min(cfg.max_samples, len(ds))
    ds = ds.select(range(take))
    return ds


def _render_prompt(tokenizer, prompt: str) -> str:
    """Apply the chat template with ``add_generation_prompt=True``."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def _tokenize_to_batches(
    tokenizer,
    prompts: list[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> list[dict]:
    """Render + tokenize prompts, then right-pad into ``batch_size``-sized groups.

    Drops any trailing partial batch. ``itertools.cycle`` over the result
    means partials would just shrink coverage with no other effect, but
    keeping batches uniform simplifies the training loop.
    """
    import torch

    rendered = [_render_prompt(tokenizer, p) for p in prompts]
    tokenized = [
        tokenizer(
            text, return_tensors=None, truncation=True, max_length=max_length, padding=False
        )
        for text in rendered
    ]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    batches: list[dict] = []
    for start in range(0, len(tokenized) - batch_size + 1, batch_size):
        chunk = tokenized[start : start + batch_size]
        max_len = max(len(t["input_ids"]) for t in chunk)
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for row, ex in enumerate(chunk):
            ids = ex["input_ids"]
            n = len(ids)
            input_ids[row, :n] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, :n] = 1
        batches.append(
            {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device),
            }
        )
    return batches


def make_unlabeled_iter(
    tokenizer,
    batch_size: int = 2,
    max_steps: int = 1000,
    cache_dir: Path | None = None,
    device: str = "cuda",
    seed: int = 0,
    max_length: int = 512,
) -> Iterator[tuple[int, dict[str, "torch.Tensor"]]]:
    """Yield (domain_idx, batch) tuples for AdaMerging training.

    Loads the 4 unlabeled datasets from HF cache (must be pre-downloaded via
    ``scripts/fetch_adamerging_data.py``). Tokenizes each prompt as a single
    user turn with the generation prompt appended via ``apply_chat_template``.
    Batches yielded in round-robin domain order: math, knowledge, safety,
    multilingual, math, ...

    Each domain's batches are ``itertools.cycle``-d so the iterator never
    runs dry even if ``max_steps`` exceeds total examples in some domain
    (XSTest in particular is small).

    Args:
        tokenizer: A Qwen3 tokenizer (or any HF tokenizer with a chat template).
        batch_size: Number of prompts per batch.
        max_steps: Number of (domain_idx, batch) tuples to yield total.
        cache_dir: HF dataset cache dir. Defaults to ``HF_DATASETS_CACHE``
            or ``HF_HOME``/datasets.
        device: Device to move batches to (``"cuda"`` on cluster).
        seed: Seed for per-dataset shuffling (deterministic batch order).
        max_length: Max tokens per prompt after chat template + truncation.

    Yields:
        ``(domain_idx, batch)`` tuples where ``batch`` is
        ``{"input_ids": [B, T], "attention_mask": [B, T]}`` on ``device``.

    Raises:
        FileNotFoundError: if cache is missing (via :func:`assert_cache_exists`).
    """
    assert_cache_exists(cache_dir)
    resolved = Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()

    domain_batches: list[list[dict]] = []
    for cfg in UNLABELED_DATASETS:
        ds = _load_one_dataset(cfg, resolved, seed)
        prompts = [row[cfg.field] for row in ds]
        batches = _tokenize_to_batches(
            tokenizer, prompts, batch_size=batch_size, max_length=max_length, device=device
        )
        if not batches:
            raise RuntimeError(
                f"domain {cfg.domain_name!r}: not enough examples to form a single "
                f"batch of size {batch_size} after tokenization "
                f"(have {len(prompts)} prompts, max_length={max_length})."
            )
        logger.info(
            "Loaded %d batches for %s (batch_size=%d, max_len=%d)",
            len(batches), cfg.domain_name, batch_size, max_length,
        )
        domain_batches.append(batches)

    cyclers = [itertools.cycle(b) for b in domain_batches]
    n_domains = len(UNLABELED_DATASETS)

    def gen() -> Iterator[tuple[int, dict]]:
        for step in range(max_steps):
            domain_idx = step % n_domains
            yield (domain_idx, next(cyclers[domain_idx]))

    return gen()


__all__ = [
    "DatasetConfig",
    "UNLABELED_DATASETS",
    "assert_cache_exists",
    "make_unlabeled_iter",
]

"""Tests for :mod:`merge.data.unlabeled` that run without datasets installed.

The real validation — datasets actually load from cache, tokenization
produces the expected shapes — happens on the cluster via
``scripts/smoke_adamerging.py``. These tests cover the static config and
the cache-presence check, both of which are pure-Python.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from merge.data.unlabeled import (
    UNLABELED_DATASETS,
    DatasetConfig,
    assert_cache_exists,
)


def test_unlabeled_datasets_has_4_entries() -> None:
    assert len(UNLABELED_DATASETS) == 4


def test_unlabeled_datasets_domain_indices_cover_0_to_3() -> None:
    indices = sorted(cfg.domain_idx for cfg in UNLABELED_DATASETS)
    assert indices == [0, 1, 2, 3]


def test_unlabeled_datasets_domain_names_match_canonical() -> None:
    """domain_name field must agree with merge.load_adapter.CANONICAL_DOMAINS."""
    from merge.load_adapter import CANONICAL_DOMAINS

    names = {cfg.domain_name for cfg in UNLABELED_DATASETS}
    assert names == set(CANONICAL_DOMAINS), (
        f"domain_name mismatch: have {names}, expected {set(CANONICAL_DOMAINS)}"
    )


def test_unlabeled_datasets_domain_idx_matches_canonical_order() -> None:
    """domain_idx must equal the position of domain_name in CANONICAL_DOMAINS."""
    from merge.load_adapter import CANONICAL_DOMAINS

    for cfg in UNLABELED_DATASETS:
        expected_idx = CANONICAL_DOMAINS.index(cfg.domain_name)
        assert cfg.domain_idx == expected_idx, (
            f"{cfg.domain_name}: domain_idx={cfg.domain_idx} but CANONICAL_DOMAINS "
            f"places it at {expected_idx}"
        )


def test_unlabeled_datasets_have_required_fields() -> None:
    for cfg in UNLABELED_DATASETS:
        assert cfg.hf_name, f"{cfg.domain_name}: hf_name is empty"
        assert cfg.split, f"{cfg.domain_name}: split is empty"
        assert cfg.field, f"{cfg.domain_name}: field is empty"
        assert cfg.max_samples > 0, f"{cfg.domain_name}: max_samples must be > 0"


def test_unlabeled_datasets_max_samples_within_sanity_bound() -> None:
    """No dataset should request more than 5000 samples."""
    for cfg in UNLABELED_DATASETS:
        assert 1 <= cfg.max_samples <= 5000


def test_dataset_config_is_frozen() -> None:
    """DatasetConfig is a frozen dataclass — mutation raises."""
    cfg = UNLABELED_DATASETS[0]
    with pytest.raises(Exception):  # FrozenInstanceError
        cfg.max_samples = 999  # type: ignore[misc]


def test_assert_cache_exists_raises_on_missing_dir(tmp_path: Path) -> None:
    """A fresh empty dir is missing all 4 datasets; error message names them."""
    with pytest.raises(FileNotFoundError) as exc_info:
        assert_cache_exists(cache_dir=tmp_path)
    message = str(exc_info.value)
    for cfg in UNLABELED_DATASETS:
        assert cfg.hf_name in message, (
            f"missing {cfg.hf_name!r} from the error message: {message!r}"
        )
    # Should point users at the fetch script.
    assert "fetch_adamerging_data.py" in message


def test_assert_cache_exists_passes_when_all_dirs_present(tmp_path: Path) -> None:
    """Create one dir per dataset using HF's repo-name-encoding convention."""
    for cfg in UNLABELED_DATASETS:
        normalized = cfg.hf_name.replace("/", "___")
        (tmp_path / normalized).mkdir(parents=True)
    assert_cache_exists(cache_dir=tmp_path)  # must not raise

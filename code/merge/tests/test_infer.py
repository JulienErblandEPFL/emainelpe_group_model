"""Tests for :mod:`merge.infer` — the parts that are torch-free.

Currently focused on :meth:`InferenceConfig.from_generation_config_dict`,
the bridge between ``generation_config.json``-style dicts and the sampling
dataclass used by :func:`merge.infer.run_inference`.
"""
from __future__ import annotations

import pytest

from merge.infer import InferenceConfig


def test_inference_config_from_generation_config_dict() -> None:
    config_dict = {
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
        "max_new_tokens": 4096,
    }
    config = InferenceConfig.from_generation_config_dict(config_dict)
    assert config.temperature == 0.5
    assert config.top_p == 0.9
    assert config.top_k == 40
    assert config.max_tokens == 4096
    # Other fields use dataclass defaults.
    assert config.n == 8
    assert config.seed == 42


def test_inference_config_from_dict_max_new_tokens_optional() -> None:
    """``max_new_tokens`` missing → ``max_tokens`` default 2048."""
    config_dict = {"temperature": 0.5, "top_p": 0.9, "top_k": 40}
    config = InferenceConfig.from_generation_config_dict(config_dict)
    assert config.max_tokens == 2048


def test_inference_config_from_dict_raises_on_missing_required() -> None:
    """Missing ``temperature``/``top_p``/``top_k`` → ``KeyError``."""
    with pytest.raises(KeyError):
        InferenceConfig.from_generation_config_dict({"top_p": 0.9, "top_k": 40})


def test_inference_config_from_dict_accepts_full_schema() -> None:
    """The full project-schema dict (with token IDs, do_sample, etc.) is
    accepted — extra keys are ignored."""
    full = {
        "bos_token_id": 151643,
        "do_sample": True,
        "eos_token_id": [151645, 151643],
        "pad_token_id": 151643,
        "temperature": 0.3,
        "top_k": 20,
        "top_p": 0.8,
        "max_new_tokens": 16384,
        "transformers_version": "4.51.0",
    }
    config = InferenceConfig.from_generation_config_dict(full)
    assert config.temperature == 0.3
    assert config.top_p == 0.8
    assert config.top_k == 20
    assert config.max_tokens == 16384

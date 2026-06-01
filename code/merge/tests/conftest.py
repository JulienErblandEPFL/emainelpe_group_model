"""
Pytest configuration for the ``merge/`` package.

Design constraints:
- Tests must run on a CPU-only laptop without ``torch`` installed; heavy ML
  imports are gated behind ``pytest.importorskip`` at the test-function level.
- No HF API calls. No GPU. No internet. No real model loads.
- All tests must pass (or skip cleanly) before any merge logic exists; this is
  the regression baseline that future stages must keep green.
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
LORA_YAML: Path = REPO_ROOT / "lora.yaml"
CHAT_TEMPLATE: Path = REPO_ROOT / "chat_template.jinja"
VALIDATION_SAMPLES_DIR: Path = REPO_ROOT / "validation_samples"


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for selective test runs."""
    config.addinivalue_line(
        "markers",
        "cpu_only: tests that run on CPU only (default for the skeleton).",
    )
    config.addinivalue_line(
        "markers",
        "requires_hf: tests that need a valid HF_TOKEN; skipped otherwise.",
    )
    config.addinivalue_line(
        "markers",
        "gpu: tests that need a CUDA device; skipped on CPU-only environments.",
    )


@pytest.fixture
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture
def lora_yaml_path() -> Path:
    """Absolute path to the locked LoRA spec."""
    return LORA_YAML


@pytest.fixture
def chat_template_path() -> Path:
    """Absolute path to the locked Qwen3 chat template."""
    return CHAT_TEMPLATE


@pytest.fixture
def validation_samples_dir() -> Path:
    """Absolute path to ``validation_samples/`` (40-problem snapshot)."""
    return VALIDATION_SAMPLES_DIR


# Per-domain seeds so the 4 toy adapters produce distinct task vectors.
_DOMAIN_SEEDS: dict[str, int] = {
    "math": 0,
    "general_knowledge": 1,
    "safety": 2,
    "multilingual": 3,
}


@pytest.fixture
def synthetic_adapters_dir(tmp_path: Path) -> Path:
    """Build 4 toy PEFT adapters under ``tmp_path/loras/{domain}/`` and return the parent.

    Skips cleanly if torch/safetensors are unavailable (laptop CPU runs).
    """
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    from merge.tests.fixtures.toy_adapter import make_toy_adapter
    from merge.verify_spec import load_locked_spec

    locked_spec = load_locked_spec(LORA_YAML)
    loras_dir = tmp_path / "loras"
    for domain, seed in _DOMAIN_SEEDS.items():
        make_toy_adapter(loras_dir / domain, locked_spec, seed=seed)
    return loras_dir

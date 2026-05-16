"""
End-to-end orchestrator for the group-merge pipeline.

Dispatches by method name through ``merge.methods.METHOD_REGISTRY``:

    spec-verify -> download -> load -> task_vectorize -> merge -> save

The pipeline writes a PEFT-compatible adapter directory (with
``adapter_config.json`` and ``adapter_model.safetensors``) to ``output_dir``.
It does not touch the base model and never materializes ΔW = B @ A.

To be implemented in Stage 4.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def merge_adapters(
    adapter_repos: list[str],
    method: str,
    base_model_repo: str,
    locked_spec_path: Path,
    output_dir: Path,
    method_kwargs: dict[str, Any] | None = None,
) -> Path:
    """
    Run the full merge pipeline and write a PEFT-loadable adapter directory.

    Args:
        adapter_repos: HF repo IDs for the specialist adapters, e.g.
            ``["cs-552-2026-emainelpe/math_model",
               "cs-552-2026-emainelpe/general_knowledge_model",
               "cs-552-2026-emainelpe/multilingual_model",
               "cs-552-2026-emainelpe/safety_model"]``.
        method: One of the keys of ``merge.methods.METHOD_REGISTRY``:
            ``"uniform"`` | ``"dare_uniform"`` | ``"dare_weighted"`` |
            ``"ties"`` | ``"adamerging"``.
        base_model_repo: HF repo ID of the base model
            (must equal the locked-spec ``base_model``, i.e. ``"Qwen/Qwen3-1.7B"``).
        locked_spec_path: Path to ``lora.yaml`` at the repo root.
        output_dir: Where to write the merged adapter. Created if missing.
        method_kwargs: Method-specific hyperparameters (e.g. ``drop_rate``
            for DARE, ``density`` for TIES, ``weights`` for weighted_linear).

    Returns:
        Path to the merged-adapter directory, ready for ``publish_adapter``
        and ``infer.generate_completions``.

    Raises:
        ValueError: if any spec-verify mismatch is detected, if ``method`` is
            not in the registry, or if ``base_model_repo`` does not match the
            locked spec.
    """
    raise NotImplementedError("Stage 4")

"""
AdaMerging: learn per-layer task-vector coefficients via entropy minimization.

Instead of fixing per-task scalar weights, AdaMerging treats them as learnable
parameters and optimizes them on an *unlabeled* validation distribution by
minimizing the entropy of the merged model's predictions. The merged adapter
is therefore tuned to be confident on the team's actual validation prompts
without using any of the gold labels.

Reference: Yang et al. 2024 (AdaMerging), https://arxiv.org/abs/2310.02575

This is the most compute-heavy method (it requires running the merged model
on validation prompts and backpropping through the coefficient mixing). It is
explicitly deferred past the May 24 milestone — Stage 7.

To be implemented in Stage 7 (post-milestone).
"""
from __future__ import annotations

from pathlib import Path

import torch


def adamerging(
    task_vectors: list[dict[str, torch.Tensor]],
    base_model_repo: str,
    validation_jsonl_paths: list[Path],
    chat_template_path: Path,
    *,
    lr: float = 1e-3,
    n_steps: int = 500,
    per_layer: bool = True,
    device: str = "cuda",
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """
    Learn task-vector mixing coefficients by entropy minimization.

    Args:
        task_vectors: List of dicts ``{parameter_name: tensor}``.
        base_model_repo: HF repo ID of the base model (Qwen3-1.7B).
        validation_jsonl_paths: One JSONL per benchmark from
            ``validation_samples/``; only the ``prompt`` field is used
            (labels are NOT consumed — AdaMerging is unsupervised).
        chat_template_path: Path to ``chat_template.jinja``.
        lr: Learning rate for the coefficient optimizer.
        n_steps: Optimization steps.
        per_layer: If True, learn one coefficient per (task, layer); if False,
            one scalar coefficient per task.
        device: ``"cuda"`` required for practical training time.
        seed: Optional seed for coefficient initialization.

    Returns:
        Merged task vector dict ``{parameter_name: tensor}`` produced by the
        learned coefficients.

    Raises:
        RuntimeError: if ``device != "cuda"`` and the prompt set is too large
            to fit in memory.
    """
    raise NotImplementedError("Stage 7")

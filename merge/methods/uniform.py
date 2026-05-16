"""
Uniform (equal-weight) linear merging of task vectors.

Computes the arithmetic mean of N task vectors entrywise. The simplest possible
baseline — any more sophisticated method (TIES, AdaMerging, DARE composition)
should be expected to outperform this on the 4-domain leaderboard average.

To be implemented in Stage 3.
"""
from __future__ import annotations

import torch


def uniform_merge(
    task_vectors: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    Average ``N`` task vectors with equal weights.

    Args:
        task_vectors: List of dicts ``{parameter_name: tensor}``. All dicts
            must share the same key set; the function raises otherwise.

    Returns:
        Single dict ``{parameter_name: tensor}`` where each tensor is
        ``sum(task_vectors[i][k] for i in range(N)) / N``.

    Raises:
        ValueError: if ``task_vectors`` is empty, if key sets differ across
            inputs, or if any tensor shape mismatch is detected.
    """
    raise NotImplementedError("Stage 3")

"""
Weighted linear merging of task vectors.

Computes ``sum(w_i * tv_i) / sum(w_i)`` entrywise. Weights are typically
chosen to reflect per-domain importance (e.g. math weighted higher because
it is the only freeform/pass@8 benchmark on the leaderboard).

To be implemented in Stage 3.
"""
from __future__ import annotations

import torch


def weighted_linear_merge(
    task_vectors: list[dict[str, torch.Tensor]],
    weights: list[float],
    *,
    normalize: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Weighted linear combination of task vectors.

    Args:
        task_vectors: List of dicts ``{parameter_name: tensor}``, all sharing
            the same key set.
        weights: Per-task scalar weights. Length must equal ``len(task_vectors)``.
            All weights must be non-negative; at least one must be strictly positive.
        normalize: If True, weights are renormalized to sum to 1 before mixing
            (so the result stays in the convex hull of the inputs). If False,
            raw weights are used (allowing extrapolation outside the hull —
            usually a bad idea for adapter merging).

    Returns:
        Single dict ``{parameter_name: tensor}``.

    Raises:
        ValueError: if shapes/keys mismatch, weights are invalid, or
            ``len(weights) != len(task_vectors)``.
    """
    raise NotImplementedError("Stage 3")

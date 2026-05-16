"""
DARE (Drop And REscale): random element-wise masking + rescaling of task vectors.

DARE drops a fraction ``p`` of entries from each task vector uniformly at random,
then rescales the survivors by ``1/(1-p)`` so the expected magnitude is preserved.
The dropped entries are set to zero; the result is a task vector ready to be
combined with other DARE'd task vectors via any linear merging method
(``uniform_merge`` or ``weighted_linear_merge``).

Reference: Yu et al. 2024 (DARE), https://arxiv.org/abs/2311.03099

This module operates on task vectors as dicts ``{parameter_name: tensor}``.
It is base-model-agnostic and does not touch the base model weights.

To be implemented in Stage 3.
"""
from __future__ import annotations

import torch


def dare(
    task_vector: dict[str, torch.Tensor],
    drop_rate: float,
    *,
    seed: int | None = None,
    rescale: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Apply DARE to a single task vector.

    Args:
        task_vector: Mapping from parameter name to ΔW tensor (the task vector).
        drop_rate: Fraction of entries to zero out, in ``[0, 1)``.
        seed: If provided, used to seed the masking RNG for reproducibility.
        rescale: If True, surviving entries are multiplied by ``1/(1-drop_rate)``.

    Returns:
        New dict ``{parameter_name: tensor}`` with same keys/shapes, with entries
        zeroed/rescaled per DARE. Does not modify the input dict.

    Raises:
        ValueError: if ``drop_rate`` is outside ``[0, 1)`` or if any tensor is
            non-floating.
    """
    raise NotImplementedError("Stage 3")

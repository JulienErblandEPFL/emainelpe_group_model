"""
Method registry for adapter merging.

Maps method-name strings (used by ``merge.pipeline.merge_adapters``'s
``method`` parameter) to callables. Five user-facing methods:

- ``"uniform"``           -> uniform_merge
- ``"dare_uniform"``      -> DARE(each) then uniform_merge
- ``"dare_weighted"``     -> DARE(each) then weighted_linear_merge
- ``"ties"``              -> ties_merge                  (Stage 4)
- ``"adamerging"``        -> adamerging                  (post-milestone, Stage 7)

The two ``dare_*`` entries are composition wrappers defined in this module;
the others re-export the per-module implementations.

Stage 3 implementation: ``uniform``, ``dare_uniform``, ``dare_weighted``
are now real. ``ties`` and ``adamerging`` remain stubs.
"""
from __future__ import annotations

from typing import Callable

import torch

from .adamerging import adamerging
from .dare import dare
from .ties import ties_merge
from .uniform import uniform_merge
from .weighted_linear import weighted_linear_merge


def dare_uniform(
    task_vectors: list[dict[str, torch.Tensor]],
    drop_rate: float = 0.5,
    seed: int | None = None,
    rescale: bool = True,
) -> dict[str, torch.Tensor]:
    """Compose DARE + uniform merge.

    Apply DARE to each input task vector independently, then take the
    uniform average. The i-th input is masked with seed = ``seed + i`` so
    the four adapters get independent masks while remaining reproducible.

    Args:
        task_vectors: List of N task-vector dicts.
        drop_rate: DARE drop rate, in ``[0, 1)``.
        seed: Optional global seed; per-tv seed is ``seed + i``. If ``None``,
            each DARE call uses ambient RNG.
        rescale: Whether DARE rescales survivors.

    Returns:
        Merged task vector dict.
    """
    masked = [
        dare(tv, drop_rate, seed=(None if seed is None else seed + i), rescale=rescale)
        for i, tv in enumerate(task_vectors)
    ]
    return uniform_merge(masked)


def dare_weighted(
    task_vectors: list[dict[str, torch.Tensor]],
    weights: list[float],
    drop_rate: float = 0.5,
    seed: int | None = None,
    rescale: bool = True,
) -> dict[str, torch.Tensor]:
    """Compose DARE + weighted-linear merge.

    Apply DARE to each input task vector independently, then take the
    weighted linear combination. Per-task seed derivation matches
    :func:`dare_uniform`.

    Args:
        task_vectors: List of N task-vector dicts.
        weights: List of N floats, one per task vector.
        drop_rate: DARE drop rate.
        seed: Optional global seed.
        rescale: Whether DARE rescales survivors.

    Returns:
        Merged task vector dict.
    """
    masked = [
        dare(tv, drop_rate, seed=(None if seed is None else seed + i), rescale=rescale)
        for i, tv in enumerate(task_vectors)
    ]
    return weighted_linear_merge(masked, weights)


METHOD_REGISTRY: dict[str, Callable[..., dict[str, torch.Tensor]]] = {
    "uniform": uniform_merge,
    "dare_uniform": dare_uniform,
    "dare_weighted": dare_weighted,
    "ties": ties_merge,
    "adamerging": adamerging,
}

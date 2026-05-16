"""
Method registry for adapter merging.

Maps method-name strings (used by ``merge.pipeline.merge_adapters``'s
``method`` parameter) to callables. Five user-facing methods:

- ``"uniform"``           -> uniform_merge
- ``"dare_uniform"``      -> DARE(each) then uniform_merge   (Stage 3 composition)
- ``"dare_weighted"``     -> DARE(each) then weighted_linear_merge (Stage 3 composition)
- ``"ties"``              -> ties_merge
- ``"adamerging"``        -> adamerging (post-milestone, Stage 7)

The two ``dare_*`` entries are wrappers defined in this module; the others
re-export the per-module implementations.
"""
from __future__ import annotations

from typing import Any, Callable

from .adamerging import adamerging
from .dare import dare  # noqa: F401  (re-exported for direct use in Stage 3 composition)
from .ties import ties_merge
from .uniform import uniform_merge
from .weighted_linear import weighted_linear_merge


def dare_uniform(
    task_vectors: list[dict[str, Any]],
    *,
    drop_rate: float = 0.5,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Composition: apply DARE to each task vector, then ``uniform_merge``.

    Args:
        task_vectors: List of dicts ``{parameter_name: tensor}``.
        drop_rate: DARE drop rate, in ``[0, 1)``.
        seed: Optional seed; the i-th task vector uses ``seed + i`` to keep
            masks independent across tasks while remaining reproducible.

    Returns:
        Merged task vector dict.

    To be implemented in Stage 3 as ``uniform_merge([dare(tv, p, seed=seed+i) for i, tv in enumerate(task_vectors)])``.
    """
    raise NotImplementedError("Stage 3")


def dare_weighted(
    task_vectors: list[dict[str, Any]],
    weights: list[float],
    *,
    drop_rate: float = 0.5,
    seed: int | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    """
    Composition: apply DARE to each task vector, then ``weighted_linear_merge``.

    Args:
        task_vectors: List of dicts ``{parameter_name: tensor}``.
        weights: Per-task scalar weights, same length as ``task_vectors``.
        drop_rate: DARE drop rate, in ``[0, 1)``.
        seed: Optional seed for reproducibility (see ``dare_uniform``).
        normalize: Whether to renormalize weights to sum to 1 (see
            ``weighted_linear_merge``).

    Returns:
        Merged task vector dict.

    To be implemented in Stage 3.
    """
    raise NotImplementedError("Stage 3")


METHOD_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "uniform": uniform_merge,
    "dare_uniform": dare_uniform,
    "dare_weighted": dare_weighted,
    "ties": ties_merge,
    "adamerging": adamerging,
}

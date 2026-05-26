"""
Method registry for adapter merging.

Maps method-name strings (used by ``merge.pipeline.merge_adapters``'s
``method`` parameter) to callables. Six user-facing methods:

- ``"uniform"``           -> uniform_merge
- ``"dare_uniform"``      -> DARE(each) then uniform_merge
- ``"dare_weighted"``     -> DARE(each) then weighted_linear_merge
- ``"ties"``              -> ties_merge                  (Stage 4)
- ``"adamerging"``        -> adamerging                  (Stage 5a)
- ``"dare_adamerging"``   -> DARE(each) then adamerging  (Stage 5a)

The three ``dare_*`` entries are composition wrappers defined in this
module; the others re-export the per-module implementations.

Note: ``adamerging`` and ``dare_adamerging`` have a richer call signature
than the other methods. ``uniform``, ``dare_uniform``, ``dare_weighted``,
and ``ties`` take ``(task_vectors, **method_kwargs)``. AdaMerging variants
additionally require ``forward_fn`` and ``data_iter``, which must be
supplied via ``method_kwargs`` when calling
``pipeline.merge_adapters(method="dare_adamerging", method_kwargs={...})``.

Stage 5a implementation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import torch

from .adamerging import AdaMergingResult, adamerging
from .dare import dare
from .ties import ties_merge
from .uniform import uniform_merge
from .weighted_linear import weighted_linear_merge


logger = logging.getLogger(__name__)


def _persist_adamerging_metrics(
    result: AdaMergingResult,
    metrics_out_path: Path,
    task_names: list[str],
    hyperparams: dict[str, Any],
) -> None:
    """Write the AdaMergingResult metrics that the pipeline would otherwise discard.

    ``coefficients`` has shape ``[N_tasks, N_layers]``; row ``i`` corresponds
    to ``task_names[i]``. ``task_names`` MUST be in the same order as the
    task_vectors list passed to :func:`adamerging` (the pipeline derives it
    from ``list(adapters_by_domain.keys())``, which is also the order of
    ``list(adapters_by_domain.values())`` used to build the task_vectors).
    """
    coeffs = result.coefficients.detach().cpu().tolist()
    n_tasks = len(coeffs)
    n_layers = len(coeffs[0]) if coeffs else 0
    if n_tasks != len(task_names):
        raise ValueError(
            f"task_names length ({len(task_names)}) does not match coefficients "
            f"first dim ({n_tasks}); row order would be ambiguous."
        )
    payload = {
        "task_names": list(task_names),
        "n_tasks": n_tasks,
        "n_layers": n_layers,
        "steps_run": result.steps_run,
        "early_stopped": result.early_stopped,
        "loss_history": list(result.loss_history),
        "coefficients": coeffs,
        "hyperparams": dict(hyperparams),
    }
    metrics_out_path = Path(metrics_out_path)
    metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info(
        "adamerging metrics persisted to %s (%d tasks × %d layers, %d steps)",
        metrics_out_path, n_tasks, n_layers, result.steps_run,
    )


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
    # ``inplace=True``: the pipeline never re-reads the originals after
    # the merge_fn returns, and the input+output duplication of the
    # default path was the cause of the 2026-05-26 OOM on a clean A100.
    masked = [
        dare(
            tv, drop_rate,
            seed=(None if seed is None else seed + i),
            rescale=rescale,
            inplace=True,
        )
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
    # See ``dare_uniform`` for the in-place rationale.
    masked = [
        dare(
            tv, drop_rate,
            seed=(None if seed is None else seed + i),
            rescale=rescale,
            inplace=True,
        )
        for i, tv in enumerate(task_vectors)
    ]
    return weighted_linear_merge(masked, weights)


def dare_adamerging(
    task_vectors: list[dict[str, torch.Tensor]],
    forward_fn,
    data_iter,
    *,
    drop_rate: float = 0.5,
    seed: int | None = None,
    rescale: bool = True,
    init_coefficient: float = 0.3,
    lr: float = 1e-2,
    lambda_l2: float = 1e-4,
    max_steps: int = 1000,
    early_stop_patience: int = 100,
    progress_log_every: int = 50,
    metrics_out_path: Path | str | None = None,
    task_names: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Compose DARE + AdaMerging.

    DARE is applied **once** to each input task vector before the
    AdaMerging optimizer starts (Option α in the Day 4 decision log).
    Re-DAREing every step would produce a stochastic objective even for
    fixed coefficients, which complicates convergence; doing it once
    gives a deterministic loss surface for entropy minimization.

    Per-task seed = ``seed + i`` matches the pattern in :func:`dare_uniform`
    and :func:`dare_weighted`.

    Args:
        task_vectors: List of N task-vector dicts.
        forward_fn: Same callable contract as :func:`adamerging`.
        data_iter: Same iterable contract as :func:`adamerging`.
        drop_rate: DARE drop rate.
        seed: Optional global seed; per-tv seed is ``seed + i``.
        rescale: Whether DARE rescales survivors.
        init_coefficient, lr, lambda_l2, max_steps, early_stop_patience,
            progress_log_every: Forwarded to :func:`adamerging`.

    Returns:
        Merged task vector dict (unwrapped from
        :class:`~merge.methods.adamerging.AdaMergingResult` for
        consistency with the other methods in ``METHOD_REGISTRY``).
    """
    # See ``dare_uniform`` for the in-place rationale. AdaMerging reads
    # the dared list across training steps but never re-reads the
    # originals, so mutating them is safe.
    dared = [
        dare(
            tv, drop_rate,
            seed=(None if seed is None else seed + i),
            rescale=rescale,
            inplace=True,
        )
        for i, tv in enumerate(task_vectors)
    ]

    result = adamerging(
        dared,
        forward_fn=forward_fn,
        data_iter=data_iter,
        init_coefficient=init_coefficient,
        lr=lr,
        lambda_l2=lambda_l2,
        max_steps=max_steps,
        early_stop_patience=early_stop_patience,
        progress_log_every=progress_log_every,
    )

    final_loss = result.loss_history[-1] if result.loss_history else float("nan")
    logger.info(
        "dare_adamerging finished: %d steps, early_stopped=%s, final_loss=%.4f",
        result.steps_run, result.early_stopped, final_loss,
    )

    if metrics_out_path is not None:
        if task_names is None:
            raise ValueError(
                "metrics_out_path was supplied without task_names; "
                "task_names is required to label coefficient rows."
            )
        _persist_adamerging_metrics(
            result,
            Path(metrics_out_path),
            task_names,
            hyperparams={
                "method": "dare_adamerging",
                "drop_rate": drop_rate,
                "seed": seed,
                "rescale": rescale,
                "init_coefficient": init_coefficient,
                "lr": lr,
                "lambda_l2": lambda_l2,
                "max_steps": max_steps,
                "early_stop_patience": early_stop_patience,
            },
        )

    return result.merged


def _adamerging_dict(
    *args,
    metrics_out_path: Path | str | None = None,
    task_names: list[str] | None = None,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Registry-facing thin wrapper around :func:`adamerging`.

    The native :func:`adamerging` returns :class:`AdaMergingResult` (so
    direct callers can inspect coefficients and loss history). The
    pipeline orchestrator iterates ``merged.items()``, so the registry
    entry must yield a dict. This wrapper extracts ``.merged`` and
    forwards every argument verbatim.

    When ``metrics_out_path`` is provided, the AdaMergingResult metrics
    that the dict-return contract would otherwise discard (coefficients,
    loss history, step count, early-stop flag) are persisted to that
    path as JSON. ``task_names`` is required in that case so coefficient
    rows can be labeled.
    """
    result = adamerging(*args, **kwargs)
    if metrics_out_path is not None:
        if task_names is None:
            raise ValueError(
                "metrics_out_path was supplied without task_names; "
                "task_names is required to label coefficient rows."
            )
        _persist_adamerging_metrics(
            result,
            Path(metrics_out_path),
            task_names,
            hyperparams={"method": "adamerging", **{
                k: v for k, v in kwargs.items()
                if k not in {"forward_fn", "data_iter"}
            }},
        )
    return result.merged


METHOD_REGISTRY: dict[str, Callable[..., dict[str, torch.Tensor]]] = {
    "uniform": uniform_merge,
    "dare_uniform": dare_uniform,
    "dare_weighted": dare_weighted,
    "ties": ties_merge,
    "adamerging": _adamerging_dict,
    "dare_adamerging": dare_adamerging,
}

"""
AdaMerging: adaptive model merging via entropy minimization on unlabeled data.

Learns per-(task, layer) scalar coefficients that combine N pre-DARE'd task
vectors into a merged adapter. Coefficients are tuned by minimizing the
entropy of the model's next-token distribution on unlabeled in-domain prompts.

Reference: Yang et al. 2024 (AdaMerging), ICLR. arXiv:2310.02575

The merge math::

    merged[k] = sum_i (coefficient[i, layer_of_k] * task_vector_i[k])

where ``layer_of_k`` is the transformer layer index extracted from k's
canonical name (e.g. ``model.layers.5.self_attn.q_proj`` -> layer=5).

This module is BASE-MODEL-AGNOSTIC for the merge math. The training loop
requires a callable ``forward_fn`` that maps ``(merged_task_vector, batch)``
to logits; that callable is supplied by the caller (Stage 5b builds the
real-Qwen3 version; tests use a synthetic callable).

Stage 5a implementation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable

import torch

logger = logging.getLogger(__name__)


@dataclass
class AdaMergingResult:
    """Result of an AdaMerging training run."""

    merged: dict[str, torch.Tensor]
    """The merged task vector. Same keys, shapes, and dtype as inputs."""

    coefficients: torch.Tensor
    """Final fp32 coefficients, shape ``[N_tasks, N_layers]``."""

    loss_history: list[float]
    """Total loss (entropy + L2) per step, in order."""

    steps_run: int
    """Number of training steps actually executed (may be < max_steps)."""

    early_stopped: bool
    """True if training stopped due to a loss-improvement plateau."""


def _layer_index_from_canonical(name: str) -> int:
    """Extract the transformer layer index from a canonical name.

    ``model.layers.5.self_attn.q_proj`` -> ``5``.

    Raises:
        ValueError: if the name has no ``.layers.<int>.`` segment or the
            segment after ``layers`` is not an integer.
    """
    parts = name.split(".")
    try:
        layers_idx = parts.index("layers")
    except ValueError as exc:
        raise ValueError(f"name {name!r} has no '.layers.' segment") from exc
    try:
        return int(parts[layers_idx + 1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"name {name!r} has malformed layer index") from exc


def _compute_merged(
    task_vectors: list[dict[str, torch.Tensor]],
    coefficients: torch.Tensor,
    name_to_layer: dict[str, int],
) -> dict[str, torch.Tensor]:
    """Compute the merged task vector from current coefficients.

    For each parameter k::

        merged[k] = sum_i (coefficients[i, layer_of_k] * task_vectors[i][k])

    Arithmetic is in fp32; the output tensor is cast back to the dtype of
    ``task_vectors[0][k]``. The returned dict is part of the autograd graph
    rooted at ``coefficients`` so gradients flow through it on backward().
    """
    keys = list(task_vectors[0].keys())
    output_dtype = task_vectors[0][keys[0]].dtype
    merged: dict[str, torch.Tensor] = {}
    for k in keys:
        layer = name_to_layer[k]
        weighted_sum = torch.zeros_like(task_vectors[0][k], dtype=torch.float32)
        for i, tv in enumerate(task_vectors):
            weighted_sum = weighted_sum + coefficients[i, layer] * tv[k].float()
        merged[k] = weighted_sum.to(output_dtype)
    return merged


def _validate_task_vectors(task_vectors: list[dict[str, torch.Tensor]]) -> None:
    """Same shape/key validation pattern as uniform_merge."""
    if not task_vectors:
        raise ValueError("adamerging requires at least one task vector")

    reference_keys = list(task_vectors[0].keys())
    reference_set = set(reference_keys)
    if not reference_set:
        raise ValueError("task_vectors[0] is empty; cannot infer layer count")

    for i, tv in enumerate(task_vectors[1:], start=1):
        if set(tv.keys()) != reference_set:
            missing = reference_set - tv.keys()
            extra = tv.keys() - reference_set
            raise ValueError(
                f"task_vectors[{i}] key set diverges from task_vectors[0]: "
                f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
            )
        for key in reference_keys:
            if tv[key].shape != task_vectors[0][key].shape:
                raise ValueError(
                    f"shape mismatch for {key!r} at index {i}: "
                    f"got {tuple(tv[key].shape)}, "
                    f"expected {tuple(task_vectors[0][key].shape)}"
                )


def adamerging(
    task_vectors: list[dict[str, torch.Tensor]],
    forward_fn: Callable[[dict[str, torch.Tensor], dict], torch.Tensor],
    data_iter: Iterable[tuple[int, dict]],
    *,
    init_coefficient: float = 0.3,
    lr: float = 1e-2,
    lambda_l2: float = 1e-4,
    max_steps: int = 1000,
    early_stop_patience: int = 100,
    progress_log_every: int = 50,
    aggregate_domains: bool = False,
) -> AdaMergingResult:
    """Train per-(task, layer) coefficients via entropy minimization.

    The loop computes a merged task vector from current coefficients, hands
    it to ``forward_fn`` together with a batch from ``data_iter``, and
    minimizes the entropy of the resulting last-position logits (plus an
    L2 penalty on the coefficients).

    Args:
        task_vectors: List of N task-vector dicts in canonical domain order
            ``[math, general_knowledge, safety, multilingual]`` (or any
            consistent order — the only invariant is that index ``i`` is
            the same "domain" across all calls). All dicts must have
            identical keys and per-key shapes. Each key must contain a
            ``.layers.<int>.`` segment so the layer index can be extracted.
        forward_fn: Callable invoked at every step. Signature::

                forward_fn(merged_task_vector, batch) -> logits  # [B, T, V]

            The callable owns the base model and applies the merged task
            vector via whatever hook / patch mechanism it likes. Logits
            should be the raw output at every input position; the loop
            internally selects the last position.
        data_iter: Iterable yielding ``(domain_idx, batch)`` tuples.
            ``domain_idx`` is an int in ``[0, N_tasks)``. ``batch`` is a
            dict (typically ``{"input_ids", "attention_mask"}``) passed
            verbatim to ``forward_fn``. The iterator MUST yield at least
            ``max_steps`` tuples; callers typically build it via
            ``itertools.cycle`` over a domain-balanced sample list.
        init_coefficient: Initial value for all coefficients (default 0.3,
            paper default; slight under-weighting from uniform encourages
            non-degenerate optimization).
        lr: Adam learning rate (default 1e-2; high because we tune ~100
            scalars, not millions of params).
        lambda_l2: L2 regularization weight on coefficients (default 1e-4).
        max_steps: Maximum training steps (default 1000).
        early_stop_patience: Stop if no loss improvement for this many
            consecutive steps (default 100). Improvement threshold is
            ``best_loss - 1e-6``.
        progress_log_every: Log step + loss every N steps (default 50).
        aggregate_domains: When ``False`` (default), the loop is the
            original per-batch SGD — one optimizer step per yielded
            ``(domain_idx, batch)``. This is the path the 2026-05-26
            bake-off used; default behavior is preserved byte-for-byte
            for reproducibility. When ``True``, each optimizer step
            consumes ``n_tasks`` consecutive yields from ``data_iter``
            (assumed round-robin across the ``n_tasks`` domains, as
            :func:`merge.data.unlabeled.make_unlabeled_iter` produces),
            forwards each batch through ``forward_fn`` with the SAME
            merged task vector, averages the per-domain entropies into
            a single scalar, adds L2 once, then does one backward +
            step. This matches the original AdaMerging formulation
            (entropy aggregated across the unlabeled set per update)
            and is the recommended mode for fresh experiments. Note
            that ``max_steps`` then counts OPTIMIZER UPDATES, not
            batches — the iterator must yield at least
            ``max_steps * n_tasks`` tuples. ``loss_history`` records
            the aggregated loss per update; lower default ``lr`` (e.g.
            1e-3) is recommended since the per-update signal is much
            less noisy than the per-batch one.

    Returns:
        :class:`AdaMergingResult` with the final merged task vector,
        coefficients, loss history, step count, and stop reason.

    Raises:
        ValueError: if ``task_vectors`` is empty, has mismatched keys or
            shapes, has names without a layer index, or if no layers can
            be inferred.
    """
    _validate_task_vectors(task_vectors)

    name_to_layer: dict[str, int] = {
        k: _layer_index_from_canonical(k) for k in task_vectors[0]
    }
    n_layers = max(name_to_layer.values()) + 1
    n_tasks = len(task_vectors)

    # Leaf tensor with grad: torch.full keeps it as a leaf when requires_grad
    # is set at construction time (no in-place op chain).
    coefficients = torch.full(
        (n_tasks, n_layers),
        init_coefficient,
        dtype=torch.float32,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([coefficients], lr=lr)

    loss_history: list[float] = []
    best_loss = float("inf")
    steps_since_improvement = 0
    steps_run = 0
    early_stopped = False

    if aggregate_domains:
        # Original-formulation path: one optimizer step absorbs one batch
        # per domain. ``max_steps`` here counts optimizer UPDATES.
        iterator = iter(data_iter)
        for step in range(max_steps):
            collected: list[tuple[int, dict]] = []
            exhausted = False
            for _ in range(n_tasks):
                try:
                    collected.append(next(iterator))
                except StopIteration:
                    exhausted = True
                    break
            if exhausted or len(collected) < n_tasks:
                logger.info(
                    "adamerging (aggregated) data_iter exhausted at update %d "
                    "(got %d/%d batches); stopping.",
                    step, len(collected), n_tasks,
                )
                break

            seen_domains = {d for d, _ in collected}
            if len(seen_domains) < n_tasks:
                # Round-robin should give exactly one of each domain in any
                # n_tasks consecutive yields. If the caller passed a
                # non-round-robin iterator, the aggregation degrades to
                # "n_tasks batches per step, some domains repeated". Log
                # once at WARNING but proceed — it's still a strictly less
                # noisy objective than per-batch SGD.
                logger.warning(
                    "adamerging (aggregated) expected %d distinct domains in "
                    "%d consecutive yields; got %d (domains=%s). Iterator "
                    "may not be round-robin.",
                    n_tasks, n_tasks, len(seen_domains), sorted(seen_domains),
                )

            merged = _compute_merged(task_vectors, coefficients, name_to_layer)
            per_domain_entropies = []
            for domain_idx, batch in collected:
                logits = forward_fn(merged, batch)
                last_logits = logits[:, -1, :]
                log_probs = torch.log_softmax(last_logits, dim=-1)
                probs = torch.softmax(last_logits, dim=-1)
                per_domain_entropies.append(
                    -(probs * log_probs).sum(dim=-1).mean()
                )
            entropy = torch.stack(per_domain_entropies).mean()
            l2 = lambda_l2 * coefficients.pow(2).sum()
            loss = entropy + l2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            loss_history.append(loss_value)
            steps_run = step + 1

            if loss_value < best_loss - 1e-6:
                best_loss = loss_value
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            if step % progress_log_every == 0 or step == max_steps - 1:
                logger.info(
                    "adamerging (aggregated) update=%d loss=%.4f entropy=%.4f "
                    "l2=%.6f best=%.4f domains=%s",
                    step, loss_value, entropy.item(), l2.item(), best_loss,
                    sorted(seen_domains),
                )

            if steps_since_improvement >= early_stop_patience:
                logger.info(
                    "adamerging (aggregated) early-stop at update %d "
                    "(no improvement for %d updates)",
                    step, early_stop_patience,
                )
                early_stopped = True
                break
    else:
        for step, (domain_idx, batch) in enumerate(data_iter):
            if step >= max_steps:
                break

            merged = _compute_merged(task_vectors, coefficients, name_to_layer)
            logits = forward_fn(merged, batch)  # [B, T, V]
            last_logits = logits[:, -1, :]  # [B, V]
            log_probs = torch.log_softmax(last_logits, dim=-1)
            probs = torch.softmax(last_logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            l2 = lambda_l2 * coefficients.pow(2).sum()
            loss = entropy + l2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            loss_history.append(loss_value)
            steps_run = step + 1

            if loss_value < best_loss - 1e-6:
                best_loss = loss_value
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            if step % progress_log_every == 0 or step == max_steps - 1:
                logger.info(
                    "adamerging step=%d loss=%.4f entropy=%.4f l2=%.6f best=%.4f domain=%d",
                    step, loss_value, entropy.item(), l2.item(), best_loss, domain_idx,
                )

            if steps_since_improvement >= early_stop_patience:
                logger.info(
                    "adamerging early-stop at step %d (no improvement for %d steps)",
                    step, early_stop_patience,
                )
                early_stopped = True
                break

    with torch.no_grad():
        final_coefficients = coefficients.detach().clone()
        final_merged = _compute_merged(task_vectors, final_coefficients, name_to_layer)

    return AdaMergingResult(
        merged=final_merged,
        coefficients=final_coefficients,
        loss_history=loss_history,
        steps_run=steps_run,
        early_stopped=early_stopped,
    )


__all__ = ["adamerging", "AdaMergingResult"]

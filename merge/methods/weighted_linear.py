"""
Weighted linear merging of task vectors.

Computes ``sum(w_i * tv_i)`` entrywise. Weights are pass-through: not
normalized, not constrained to be non-negative. The caller is responsible
for whatever scaling makes sense for their use case (DARE-rescaled vectors
already have inflated magnitudes; AdaMerging-learned coefficients don't sum
to 1 by construction; uniform merging hits the sum-to-1 case by accident).

Stage 3 implementation.
"""
from __future__ import annotations

import torch


def weighted_linear_merge(
    task_vectors: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    """Compute the weighted linear combination of N task vectors.

    Output[k] = sum_i weights[i] * task_vectors[i][k] for each parameter k.

    Weights are NOT normalized. Negative weights are permitted.

    Arithmetic is done in fp32 to avoid bf16 accumulation error, then cast
    back to the input dtype.

    Args:
        task_vectors: Non-empty list of N task-vector dicts.
        weights: List of N floats, one per task vector, in the same order.

    Returns:
        A new dict with keys in ``task_vectors[0]``'s order. Input dicts are
        not modified.

    Raises:
        ValueError: if the list is empty, ``len(weights) != len(task_vectors)``,
            the dicts have different keys, or any tensor shape mismatches.
    """
    if not task_vectors:
        raise ValueError("weighted_linear_merge requires at least one task vector")
    if len(weights) != len(task_vectors):
        raise ValueError(
            f"weights length {len(weights)} does not match "
            f"task_vectors length {len(task_vectors)}"
        )

    reference_keys = list(task_vectors[0].keys())
    reference_set = set(reference_keys)

    for i, tv in enumerate(task_vectors[1:], start=1):
        if set(tv.keys()) != reference_set:
            missing = reference_set - tv.keys()
            extra = tv.keys() - reference_set
            raise ValueError(
                f"task_vectors[{i}] key set diverges from task_vectors[0]: "
                f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
            )

    out: dict[str, torch.Tensor] = {}
    for key in reference_keys:
        ref_shape = task_vectors[0][key].shape
        ref_dtype = task_vectors[0][key].dtype
        for i, tv in enumerate(task_vectors[1:], start=1):
            if tv[key].shape != ref_shape:
                raise ValueError(
                    f"shape mismatch for {key!r} at index {i}: "
                    f"got {tuple(tv[key].shape)}, expected {tuple(ref_shape)}"
                )

        # Sum w_i * tv_i in fp32, then cast back. Accumulator allocation up
        # front avoids stacking N copies of the tensor in memory.
        accum = torch.zeros(ref_shape, dtype=torch.float32, device=task_vectors[0][key].device)
        for w, tv in zip(weights, task_vectors):
            accum = accum + w * tv[key].to(torch.float32)
        out[key] = accum.to(ref_dtype)

    return out

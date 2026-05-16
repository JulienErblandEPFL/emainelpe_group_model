"""
Uniform (equal-weight) linear merging of task vectors.

Computes the arithmetic mean of N task vectors entrywise. The simplest possible
baseline — any more sophisticated method (TIES, AdaMerging, DARE composition)
should be expected to outperform this on the 4-domain leaderboard average.

Stage 3 implementation.
"""
from __future__ import annotations

import torch


def uniform_merge(
    task_vectors: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Average ``N`` task vectors with equal weights.

    Output[k] = (1/N) * sum_i task_vectors[i][k] for each parameter k.

    All input task vectors MUST have identical keys and identical tensor
    shapes per key — use ``load_all()`` with locked-spec verification
    upstream to guarantee this.

    Arithmetic is done in fp32 to avoid bf16 accumulation error, then cast
    back to the input dtype.

    Args:
        task_vectors: Non-empty list of task-vector dicts (e.g., from
            ``load_adapter.load`` or ``methods.dare.dare``).

    Returns:
        A new dict with keys in ``task_vectors[0]``'s order; per-key tensor
        is the elementwise mean across the N inputs. Input dicts are not
        modified.

    Raises:
        ValueError: if the list is empty, the dicts have different keys, or
            any tensor shape mismatches across inputs.
    """
    if not task_vectors:
        raise ValueError("uniform_merge requires at least one task vector")

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

        stacked = torch.stack([tv[key].to(torch.float32) for tv in task_vectors], dim=0)
        out[key] = stacked.mean(dim=0).to(ref_dtype)

    return out

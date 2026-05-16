"""
TIES-Merging: Trim, Elect-sign, Disjoint-merge.

For each parameter:

1. **Trim**: per task vector, keep the top ``(1 - trim_ratio)`` fraction of
   entries by magnitude, zero the rest. Per-tensor variant — each parameter
   is trimmed independently of the others.
2. **Elect**: per entry position, count signs across the N trimmed task
   vectors. The elected sign is the strict majority; exact ties elect zero
   (the parameter is dropped at that position).
3. **Disjoint merge**: per entry position, average only the task vectors
   whose sign matches the elected sign. Wrong-sign entries are excluded.

This handles sign conflicts between adapters that uniform/weighted_linear
silently average away.

Reference: Yadav et al. 2023 (TIES-Merging), https://arxiv.org/abs/2306.01708

Stage 4 implementation.
"""
from __future__ import annotations

import torch


def ties_merge(
    task_vectors: list[dict[str, torch.Tensor]],
    trim_ratio: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Apply TIES merging to ``N`` task vectors.

    Args:
        task_vectors: Non-empty list of task-vector dicts (e.g., from
            ``load_adapter.load``). All dicts must share keys and per-key
            shapes.
        trim_ratio: Fraction of entries to zero per task vector during the
            trim step, in ``[0, 1)``. Default 0.5 keeps the top half by
            magnitude — symmetric with DARE's drop_rate default.

    Returns:
        New dict with same keys, shapes, and dtypes as ``task_vectors[0]``.
        Inputs are not modified.

    Raises:
        ValueError: if the list is empty, ``trim_ratio`` is out of range,
            keys differ across dicts, or shapes differ for any key.
    """
    if not task_vectors:
        raise ValueError("ties_merge requires at least one task vector")
    if not (0.0 <= trim_ratio < 1.0):
        raise ValueError(f"trim_ratio must be in [0, 1), got {trim_ratio!r}")

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

        # 1. Trim each task vector in fp32 for numerical headroom.
        trimmed = []
        for tv in task_vectors:
            t = tv[key].to(torch.float32)
            if trim_ratio == 0.0:
                trimmed.append(t)
                continue
            flat = t.flatten()
            magnitudes = flat.abs()
            k = int(flat.numel() * trim_ratio)
            if k == 0:
                # Edge case: numel * trim_ratio < 1 → no trimming
                trimmed.append(t)
                continue
            # kth_smallest_magnitude: entries with |x| <= this are dropped
            threshold = torch.kthvalue(magnitudes, k).values
            keep_mask = (magnitudes > threshold).reshape(t.shape)
            trimmed.append(t * keep_mask.to(t.dtype))

        # 2. Elect sign per entry: strict majority across the N trimmed tvs.
        stacked = torch.stack(trimmed, dim=0)            # [N, *shape]
        signs = torch.sign(stacked)                      # ∈ {-1, 0, +1}
        elected = torch.sign(signs.sum(dim=0))           # [*shape] ∈ {-1, 0, +1}

        # 3. Disjoint merge: average only entries whose sign matches the elected.
        match = (signs == elected.unsqueeze(0))          # [N, *shape] bool
        # Only the magnitude (with original sign) is averaged where match holds.
        contrib = torch.where(match, stacked, torch.zeros_like(stacked))
        count = match.sum(dim=0).to(torch.float32)       # [*shape]
        # Where elected == 0 the count is 0 (no positions match sign 0 except
        # exact zeros, which contribute 0 anyway); avoid divide-by-zero by
        # clamping count to ≥1 and zeroing the output where elected is 0.
        safe_count = count.clamp(min=1.0)
        merged = contrib.sum(dim=0) / safe_count
        merged = torch.where(elected == 0, torch.zeros_like(merged), merged)

        out[key] = merged.to(ref_dtype)

    return out

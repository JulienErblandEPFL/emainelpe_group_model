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

Stage 3 implementation.
"""
from __future__ import annotations

import torch


def dare(
    task_vector: dict[str, torch.Tensor],
    drop_rate: float,
    seed: int | None = None,
    rescale: bool = True,
) -> dict[str, torch.Tensor]:
    """Apply DARE to a single task vector.

    For each tensor, draw a keep mask with bernoulli(1 - drop_rate), zero out
    dropped entries, and (optionally) rescale survivors by 1/(1-drop_rate).
    The expected magnitude of each survivor under the random mask is then
    equal to the input, which makes additive merges of multiple DARE-masked
    task vectors well-behaved.

    Args:
        task_vector: Mapping from parameter name to ΔW tensor.
        drop_rate: Fraction of entries to zero, in ``[0, 1)``. ``1.0`` is
            forbidden (would zero everything and divide by zero on rescale).
        seed: If provided, a fresh ``torch.Generator`` is seeded and used for
            mask sampling; same seed + same dict iteration order → identical
            masks. If ``None``, uses ambient PyTorch RNG (no reproducibility).
        rescale: If True, survivors are multiplied by ``1/(1-drop_rate)``.

    Returns:
        New dict with same keys, shapes, and dtypes as ``task_vector``. The
        input is not modified.

    Raises:
        ValueError: if ``drop_rate`` is outside ``[0, 1)``.
        TypeError: if any tensor is not floating-point.
    """
    if not (0.0 <= drop_rate < 1.0):
        raise ValueError(
            f"drop_rate must be in [0, 1), got {drop_rate!r}"
        )

    keep_prob = 1.0 - drop_rate
    rescale_factor = 1.0 / keep_prob if rescale else 1.0

    # Single generator, seeded once. The state advances naturally with each
    # bernoulli() call, so per-tensor masks are still uncorrelated. Python
    # dict iteration is insertion-ordered (≥3.7) and load() is reproducible,
    # so the sequence of draws is stable across runs.
    generator: torch.Generator | None = None
    if seed is not None and task_vector:
        # Pick device from the first tensor (all assumed on the same device).
        first_tensor = next(iter(task_vector.values()))
        generator = torch.Generator(device=first_tensor.device)
        generator.manual_seed(seed)

    out: dict[str, torch.Tensor] = {}
    for name, tensor in task_vector.items():
        if not tensor.is_floating_point():
            raise TypeError(
                f"DARE requires floating-point tensors; {name!r} has dtype {tensor.dtype}"
            )

        # Bernoulli in fp32 for numerical headroom; rescale + cast back at the end.
        keep_prob_t = torch.full(
            tensor.shape, keep_prob, dtype=torch.float32, device=tensor.device
        )
        mask = torch.bernoulli(keep_prob_t, generator=generator)
        masked = tensor.to(torch.float32) * mask * rescale_factor
        out[name] = masked.to(tensor.dtype)

    return out

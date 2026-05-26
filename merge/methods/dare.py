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
    inplace: bool = False,
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
        inplace: If False (default), build a fresh output dict and leave
            ``task_vector`` untouched — the historical non-mutating
            contract. If True, mutate each tensor in ``task_vector`` in
            place via ``tensor.mul_(mask)`` (and ``.mul_(rescale_factor)``
            when ``rescale``), then return the same dict object. The
            in-place path eliminates the input+output duplication that
            caused both DARE-based methods to OOM on a clean A100-40g in
            the 2026-05-26 bake-off (4 × ~3 GB ΔW inputs + 4 × ~3 GB
            outputs ≈ 24 GB peak, plus mask/intermediate fragmentation,
            exceeded the budget). Merge-method callers that consume
            their task_vectors only once (``dare_uniform``,
            ``dare_weighted``, ``dare_adamerging``) opt into True; any
            caller that still needs the originals afterward must leave
            ``inplace=False``.

    Returns:
        With ``inplace=False``, a new dict with the same keys, shapes,
        and dtypes as ``task_vector``. With ``inplace=True``, the same
        dict object passed in, with each tensor mutated. Either way the
        returned tensors carry the masked + rescaled deltas.

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

    # Mask + rescale stay in the input dtype (bf16 in practice). The
    # previous version upcast `tensor` to fp32 and built an fp32 mask,
    # which materialized full-size fp32 copies of every ΔW tensor —
    # for the 4-adapter set across Qwen3-1.7B's 196 target modules
    # that peak exceeded 40 GB on an A100-40g and OOM'd both
    # DARE-based methods in the 2026-05-26 bake-off.
    #
    # bf16 is more than adequate here: the mask is sampled from a
    # Bernoulli with probability ``keep_prob`` (exactly representable
    # in bf16 for the canonical 0.5 drop rate; ~3-decimal error
    # otherwise, far below DARE's own stochastic variance), the
    # rescale is a single scalar multiply, and the result is 0 or
    # (rescaled-original) — no precision loss matters.

    out: dict[str, torch.Tensor] = task_vector if inplace else {}
    for name, tensor in task_vector.items():
        if not tensor.is_floating_point():
            raise TypeError(
                f"DARE requires floating-point tensors; {name!r} has dtype {tensor.dtype}"
            )

        # Build the mask in the input dtype. ``keep_prob_t`` and ``mask``
        # are temporaries of the same shape as ``tensor``; they go out of
        # scope each iteration so only one of each lives at a time.
        keep_prob_t = torch.full(
            tensor.shape, keep_prob, dtype=tensor.dtype, device=tensor.device
        )
        mask = torch.bernoulli(keep_prob_t, generator=generator)
        # Drop the now-unused ``keep_prob_t`` before the multiply so its
        # bf16 footprint is reclaimable by the caching allocator.
        del keep_prob_t

        if inplace:
            tensor.mul_(mask)
            if rescale_factor != 1.0:
                tensor.mul_(rescale_factor)
        else:
            out[name] = tensor * mask * rescale_factor

        # ``mask`` ref drops at loop end; explicit ``del`` makes it clear
        # we want only one mask resident at a time, not 196.
        del mask

    return out

"""Real-Qwen3 forward callable for AdaMerging training.

Pairs with :mod:`merge.methods.adamerging`. AdaMerging's training loop needs
a callable that, given a merged task-vector dict and a batch of input_ids,
returns logits. This module supplies that callable for the real
Qwen3-1.7B base model.

How it works
------------
The base Qwen3 weights are frozen. For each :class:`torch.nn.Linear` whose
fully-qualified name matches a canonical key in the merged dict, we
register a *forward post-hook* that adds the merged ΔW's contribution to
the layer's normal output:

    output_with_delta = output + F.linear(input, merged[canonical], bias=None)

This is mathematically identical to running the layer with effective weight
``base + delta``, but routes the autograd graph through ``delta`` (a tensor
function of the AdaMerging coefficients) without touching base parameters.
Two alternatives were considered and rejected:

  1. **Mutate ``self.weight.data`` before forward.** ``Parameter.data =
     X`` bypasses autograd, so gradients would not flow to the coefficients.
  2. **Reimplement ``F.linear`` inside the hook** (skip the original forward
     and return a fresh output). Doable but more invasive — we would need
     to special-case bias handling and any future Linear subclass behavior.
     The post-hook ``output + delta_contribution`` approach is cleaner.

Hook lifecycle is scoped to a context manager. The model itself is loaded
once via :func:`make_qwen3_forward` and reused across forward calls;
each forward call installs N hooks (≈196 for Qwen3-1.7B = 28 layers × 7
projections), runs the forward, and removes them. The register / remove
cost is negligible next to the forward itself.

Stage 5b. CPU-runnable in principle (the hook installation logic does not
require CUDA), but the model load is GPU-targeted in practice — Qwen3-1.7B
is ~3.4 GB at bf16.
"""
from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from typing import Callable, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn


logger = logging.getLogger(__name__)


# Matches a Qwen3 LoRA target module under the model root. The 7 projections
# are (q|k|v|o)_proj inside self_attn and (gate|up|down)_proj inside mlp.
_LAYER_LINEAR_RE = re.compile(
    r"^model\.layers\.(\d+)\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj$"
)


def _build_layer_name_to_canonical(model: "nn.Module") -> dict[str, str]:
    """Map each LoRA-targetable ``nn.Linear`` module to its canonical name.

    For Qwen3 loaded via ``AutoModelForCausalLM``, the module path
    ``model.layers.0.self_attn.q_proj`` matches the canonical name produced
    by :func:`merge.load_adapter.canonicalize` exactly (canonicalize strips
    PEFT's ``base_model.model.`` wrapper and ``.lora_*.weight`` suffix; the
    bare HF model has neither). The mapping is therefore identity-shaped
    today, but kept as an explicit dict so a future architecture rename or
    a wrapper around Qwen3 only needs to update this one function.
    """
    import torch

    mapping: dict[str, str] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if _LAYER_LINEAR_RE.match(name):
            mapping[name] = name
    return mapping


def _make_post_hook(canonical: str, merged: dict, dtype):
    """Build a forward post-hook that adds ``F.linear(input, merged[canon])``."""
    import torch.nn.functional as F

    def hook(module, args, output):  # noqa: ARG001 — `module` is part of the API
        x = args[0]
        delta = merged[canonical]
        # Move + cast delta to the layer's compute dtype/device. ``.to`` is a
        # differentiable op; gradient flows back to delta unchanged.
        if delta.device != x.device or delta.dtype != dtype:
            delta = delta.to(device=x.device, dtype=dtype)
        # Add the ΔW contribution. We do NOT recompute the base output — it
        # is already in ``output``. The base path is no-grad (base params
        # are frozen); only this delta contribution participates in autograd
        # back to the AdaMerging coefficients.
        return output + F.linear(x, delta, bias=None)

    return hook


@contextmanager
def _patch_with_merged(
    model: "nn.Module",
    merged: dict,
    layer_name_to_canonical: dict[str, str],
) -> Iterator[None]:
    """Install post-hooks that add ΔW contributions for the lifetime of the block.

    For each ``(module_name, canonical_name)`` pair where ``canonical_name``
    is in ``merged``, a forward post-hook is registered on the target
    module. The hook adds ``F.linear(input, merged[canonical], bias=None)``
    to the module's output. The model's base weights are untouched.

    On exit (success or exception), all hooks are removed.

    Args:
        model: The frozen Qwen3 model.
        merged: The merged task-vector dict, keyed by canonical names.
            Tensors may be on CPU or GPU and any bf16/fp32 dtype; the hook
            handles transfer.
        layer_name_to_canonical: Output of
            :func:`_build_layer_name_to_canonical`.
    """
    handles = []
    compute_dtype = getattr(model, "dtype", None)
    try:
        for module_name, canonical_name in layer_name_to_canonical.items():
            if canonical_name not in merged:
                continue
            module = model.get_submodule(module_name)
            if compute_dtype is None:
                compute_dtype = module.weight.dtype
            h = module.register_forward_hook(
                _make_post_hook(canonical_name, merged, compute_dtype)
            )
            handles.append(h)
        yield
    finally:
        for h in handles:
            h.remove()


def make_qwen3_forward(
    base_model_repo: str = "Qwen/Qwen3-1.7B",
    device: str = "cuda",
    dtype=None,
) -> tuple[Callable, Callable]:
    """Build a real-Qwen3 forward callable for AdaMerging.

    Loads ``base_model_repo`` once into ``device`` + ``dtype`` (default bf16),
    freezes all parameters, switches to inference mode, and pre-computes the
    ``module_name → canonical_name`` map. Returns:

    * ``forward_fn(merged, batch) -> logits``  — for
      :func:`merge.methods.adamerging.adamerging`.
    * ``cleanup() -> None``  — drops the model reference so the GPU memory
      can be freed by the next ``torch.cuda.empty_cache()`` or GC pass.

    Memory budget: Qwen3-1.7B in bf16 occupies ~3.4 GB. On an A100-40g this
    leaves ample headroom for activations and the 4 task vectors.

    Args:
        base_model_repo: HF repo ID for the base model.
        device: Target device (``"cuda"`` on cluster).
        dtype: Compute dtype. Defaults to ``torch.bfloat16``.

    Returns:
        ``(forward_fn, cleanup)`` tuple.

    Raises:
        RuntimeError: if ``device="cuda"`` is requested but unavailable.
    """
    import torch
    from transformers import AutoModelForCausalLM

    if dtype is None:
        dtype = torch.bfloat16
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device!r} requested but CUDA is not available. "
            "Set device='cpu' to load Qwen3 on CPU (slow, ~14 GB fp32 / 7 GB bf16)."
        )

    logger.info("Loading %s in %s on %s ...", base_model_repo, dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_repo,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.train(False)  # equivalent to model.eval(), disables dropout
    for p in model.parameters():
        p.requires_grad_(False)

    layer_map = _build_layer_name_to_canonical(model)
    logger.info(
        "Resolved %d LoRA-targetable Linear modules in %s",
        len(layer_map), base_model_repo,
    )
    if not layer_map:
        raise RuntimeError(
            f"no LoRA-targetable Linear modules found in {base_model_repo}; "
            "the layer-naming regex may be out of date."
        )

    model_ref: list = [model]  # box so cleanup() can null it

    def forward_fn(merged, batch):
        """Run the base model with merged ΔW patched in, return logits.

        ``batch`` must carry ``input_ids`` and (recommended) ``attention_mask``;
        tensors must already be on the same device as the model.
        """
        m = model_ref[0]
        if m is None:
            raise RuntimeError(
                "make_qwen3_forward.forward_fn called after cleanup()."
            )
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        with _patch_with_merged(m, merged, layer_map):
            outputs = m(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def cleanup() -> None:
        """Release the model reference so its GPU memory can be freed."""
        model_ref[0] = None

    return forward_fn, cleanup


__all__ = ["make_qwen3_forward"]

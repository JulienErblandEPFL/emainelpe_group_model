"""
Toy PEFT-format LoRA adapter generator for the merge-pipeline tests.

The toy adapter has the same on-disk format as a real Qwen3 LoRA adapter
(``adapter_config.json`` + ``adapter_model.safetensors`` with PEFT naming),
but with two layers and a tiny hidden dim so the whole thing weighs a few
dozen KB and tests run in milliseconds.

This module is imported by ``conftest.py``; it is not a test itself.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file as safetensors_save_file


# Which of the 7 LoRA target modules sit in which Qwen3 sub-block.
# Matches Qwen3's HF naming: model.layers.{i}.{self_attn|mlp}.{module}.
_ATTN_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_MODULES: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")


def _name(layer: int, module: str, factor: str) -> str:
    """Build a PEFT-style safetensors key for a LoRA factor weight."""
    block = "self_attn" if module in _ATTN_MODULES else "mlp"
    return (
        f"base_model.model.model.layers.{layer}.{block}.{module}"
        f".lora_{factor}.default.weight"
    )


def _shape(module: str, *, hidden_dim: int, intermediate_dim: int, r: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ((A_shape), (B_shape)) for a given target module.

    PEFT stores lora_A as [r, in_features] and lora_B as [out_features, r].
    """
    if module in _ATTN_MODULES:
        in_f = out_f = hidden_dim
    elif module in ("gate_proj", "up_proj"):
        in_f, out_f = hidden_dim, intermediate_dim
    elif module == "down_proj":
        in_f, out_f = intermediate_dim, hidden_dim
    else:
        raise ValueError(f"unknown target module: {module!r}")
    return (r, in_f), (out_f, r)


def make_toy_adapter(
    out_dir: Path,
    locked_spec: dict[str, Any],
    *,
    seed: int = 0,
    n_layers: int = 2,
    hidden_dim: int = 64,
    intermediate_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    extra_config: dict[str, Any] | None = None,
) -> Path:
    """Create a synthetic PEFT-format LoRA adapter in ``out_dir``.

    The adapter has the same shape (and same load-bearing
    ``adapter_config.json`` values) as a real locked-spec adapter, just
    with a 2-layer toy backbone. Adapter weights are seeded so the
    fixture is fully reproducible.
    """
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    r = locked_spec["r"]
    target_modules = list(locked_spec["target_modules"])

    state: dict[str, torch.Tensor] = {}
    for layer in range(n_layers):
        for module in target_modules:
            (a_shape, b_shape) = _shape(
                module,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                r=r,
            )
            state[_name(layer, module, "A")] = torch.randn(a_shape, dtype=dtype)
            state[_name(layer, module, "B")] = torch.randn(b_shape, dtype=dtype)

    safetensors_save_file(state, str(out_dir / "adapter_model.safetensors"))

    # adapter_config.json: include the 8 load-bearing fields verbatim from
    # the locked spec, plus a few realistic PEFT bookkeeping fields so the
    # "extra fields are ignored" assertion in the verifier tests has bite.
    config: dict[str, Any] = {
        "base_model_name_or_path": locked_spec["base_model_name_or_path"],
        "r": locked_spec["r"],
        "lora_alpha": locked_spec["lora_alpha"],
        "lora_dropout": locked_spec["lora_dropout"],
        "bias": locked_spec["bias"],
        "task_type": locked_spec["task_type"],
        "modules_to_save": locked_spec["modules_to_save"],
        "target_modules": list(locked_spec["target_modules"]),
        # Realistic extras PEFT writes; the verifier must ignore these.
        "peft_type": "LORA",
        "peft_version": "0.11.1",
        "inference_mode": False,
        "init_lora_weights": True,
        "alpha_pattern": {},
        "rank_pattern": {},
        "fan_in_fan_out": False,
        "layers_to_transform": None,
        "layers_pattern": None,
        "revision": None,
    }
    if extra_config:
        config.update(extra_config)

    with (out_dir / "adapter_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    return out_dir

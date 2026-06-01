"""Generate Qwen3-1.7B-sized random-init LoRA adapters for cluster smoke tests.

Distinct from :func:`merge.tests.fixtures.toy_adapter.make_toy_adapter`,
which produces 2-layer hidden=64 toys for CPU tests. This helper produces
adapters at the actual Qwen3-1.7B scale (28 layers, hidden=2048, GQA with
8 KV heads, intermediate=6144) suitable for the cluster smoke script.

NOT used in the laptop test suite — the adapter on disk weighs ~140 MB
in bf16 and the test would burn far too much wall-clock per CI run.

Why hard-code Qwen3-1.7B dimensions
-----------------------------------
The smoke script's contract is "validate AdaMerging on real Qwen3". The
adapter shapes must match the base model exactly. Loading
``Qwen/Qwen3-1.7B``'s config would be the safer choice in production, but
that requires HF auth and adds a dependency to a fixture used only for
smoke. The shape constants live here and on a Qwen3 minor revision they
need to be reviewed once.

Qwen3-1.7B uses Grouped-Query Attention (GQA): k_proj and v_proj have
fewer output dims than q_proj and o_proj. This is reflected below.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file as safetensors_save_file


# Qwen3-1.7B architecture constants (from ``Qwen/Qwen3-1.7B`` config.json).
QWEN3_1_7B_N_LAYERS = 28
QWEN3_1_7B_HIDDEN = 2048
QWEN3_1_7B_INTERMEDIATE = 6144
QWEN3_1_7B_N_KV_HEADS = 8
QWEN3_1_7B_HEAD_DIM = 128  # hidden / num_attention_heads = 2048 / 16

# Output dims for the 7 LoRA target modules. ``in_dim`` is computed by the
# helper from ``hidden`` and ``intermediate`` to keep this table compact.
_KV_OUT = QWEN3_1_7B_N_KV_HEADS * QWEN3_1_7B_HEAD_DIM  # 1024 for Qwen3-1.7B
_ATTN_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_MODULES: tuple[str, ...] = ("gate_proj", "up_proj", "down_proj")


def _module_shapes(
    module: str,
    hidden: int,
    intermediate: int,
    kv_out: int,
    r: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ``((A_shape), (B_shape))`` for a given target module.

    PEFT stores ``lora_A`` as ``[r, in_features]`` and ``lora_B`` as
    ``[out_features, r]``. The 7 modules in Qwen3-1.7B:

      - q_proj, o_proj: hidden ↔ hidden
      - k_proj, v_proj: hidden → kv_out (GQA)
      - gate_proj, up_proj: hidden → intermediate
      - down_proj: intermediate → hidden
    """
    if module == "q_proj":
        in_f, out_f = hidden, hidden
    elif module == "k_proj" or module == "v_proj":
        in_f, out_f = hidden, kv_out
    elif module == "o_proj":
        in_f, out_f = hidden, hidden
    elif module in ("gate_proj", "up_proj"):
        in_f, out_f = hidden, intermediate
    elif module == "down_proj":
        in_f, out_f = intermediate, hidden
    else:
        raise ValueError(f"unknown target module: {module!r}")
    return (r, in_f), (out_f, r)


def _peft_name(layer: int, module: str, factor: str) -> str:
    """Build a PEFT-style safetensors key for a LoRA factor weight."""
    block = "self_attn" if module in _ATTN_MODULES else "mlp"
    return (
        f"base_model.model.model.layers.{layer}.{block}.{module}"
        f".lora_{factor}.default.weight"
    )


def make_random_qwen3_adapter(
    out_dir: Path,
    seed: int = 0,
    *,
    n_layers: int = QWEN3_1_7B_N_LAYERS,
    hidden: int = QWEN3_1_7B_HIDDEN,
    intermediate: int = QWEN3_1_7B_INTERMEDIATE,
    kv_out: int = _KV_OUT,
    r: int = 32,
    alpha: int = 64,
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """Create a random-init Qwen3-1.7B-sized PEFT-format LoRA adapter.

    Output:

        out_dir/adapter_config.json
        out_dir/adapter_model.safetensors

    The config matches the locked spec (``r=32``, ``alpha=64``, the 7 target
    modules). Weights are sampled from ``N(0, 1)`` and scaled to ``1/sqrt(r)``
    so that ``ΔW = (alpha / r) * B @ A`` has unit-ish magnitude — enough to
    make AdaMerging coefficients move during smoke training, far less than a
    real trained adapter.

    Args:
        out_dir: Directory to write the adapter to. Created if missing.
        seed: Per-adapter seed; the smoke script passes ``base_seed + i``
            so the 4 generated adapters differ.
        n_layers, hidden, intermediate, kv_out: Architecture overrides.
            Defaults are correct for ``Qwen/Qwen3-1.7B``.
        r, alpha: LoRA hyperparameters from the locked spec.
        dtype: Storage dtype for the weights. Defaults to bf16 to match
            real adapters.

    Returns:
        ``out_dir`` (now populated).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator()
    g.manual_seed(seed)

    target_modules = _ATTN_MODULES + _MLP_MODULES
    init_scale = 1.0 / (r ** 0.5)

    state: dict[str, torch.Tensor] = {}
    for layer in range(n_layers):
        for module in target_modules:
            a_shape, b_shape = _module_shapes(module, hidden, intermediate, kv_out, r)
            state[_peft_name(layer, module, "A")] = (
                init_scale * torch.randn(a_shape, generator=g, dtype=torch.float32)
            ).to(dtype)
            state[_peft_name(layer, module, "B")] = (
                init_scale * torch.randn(b_shape, generator=g, dtype=torch.float32)
            ).to(dtype)

    safetensors_save_file(state, str(out_dir / "adapter_model.safetensors"))

    config: dict[str, Any] = {
        "base_model_name_or_path": "Qwen/Qwen3-1.7B",
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "modules_to_save": None,
        "target_modules": list(target_modules),
        # Realistic PEFT extras the verifier ignores.
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
    with (out_dir / "adapter_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    return out_dir


__all__ = [
    "make_random_qwen3_adapter",
    "QWEN3_1_7B_N_LAYERS",
    "QWEN3_1_7B_HIDDEN",
    "QWEN3_1_7B_INTERMEDIATE",
    "QWEN3_1_7B_N_KV_HEADS",
    "QWEN3_1_7B_HEAD_DIM",
]

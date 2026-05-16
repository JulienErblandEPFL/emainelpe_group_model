"""
Load locked-spec LoRA adapters from a local directory into task-vector dicts.

The pipeline consumes adapters from a local directory, not HF Hub. Whoever
runs the pipeline is responsible for getting the four adapters onto disk
(``hf download``, ``cp`` from RCP scratch, etc.). This module then:

  1. Reads each adapter's ``adapter_config.json`` and verifies it against
     the locked spec (raising :class:`SpecMismatchError` on divergence).
  2. Reads the ``adapter_model.safetensors`` weights and materializes the
     ``ΔW = (α / r) · B @ A`` task vector per LoRA-decorated module.
  3. Canonicalizes the parameter namespace so two adapters trained with
     different PEFT versions still share keys.

Stage 2 implementation.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .verify_spec import SpecMismatchError, VerifyResult, verify

if TYPE_CHECKING:
    import torch


logger = logging.getLogger(__name__)


CANONICAL_DOMAINS: tuple[str, ...] = ("math", "general_knowledge", "safety", "multilingual")


# PEFT writes LoRA factor weights with one of these suffix patterns. The
# ``.default`` segment is the LoRA adapter name; older PEFT versions omit it.
_PEFT_SUFFIX_RE = re.compile(r"\.lora_(A|B)(?:\.default)?\.weight$")

# Leading wrapper PEFT adds to every parameter name. The two ``model.`` are
# expected: outer is PEFT's own wrapper, inner is the HF model's own root.
_PEFT_PREFIX = "base_model.model."


def canonicalize(name: str) -> str:
    """Reduce a raw safetensors key to a layer+module identifier.

    ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight``
    becomes ``model.layers.0.self_attn.q_proj``.

    Raises:
        ValueError: if ``name`` does not match an expected PEFT pattern.
    """
    if not name.startswith(_PEFT_PREFIX):
        raise ValueError(f"unexpected PEFT name (no '{_PEFT_PREFIX}' prefix): {name!r}")

    suffix_match = _PEFT_SUFFIX_RE.search(name)
    if suffix_match is None:
        raise ValueError(f"unexpected PEFT name (no lora_A/B weight suffix): {name!r}")

    body = name[len(_PEFT_PREFIX) : suffix_match.start()]
    if not body:
        raise ValueError(f"unexpected PEFT name (empty body): {name!r}")
    return body


def _which_factor(name: str) -> str:
    """Return 'A' or 'B' from a LoRA factor weight name."""
    m = _PEFT_SUFFIX_RE.search(name)
    if m is None:
        raise ValueError(f"not a LoRA factor weight: {name!r}")
    return m.group(1)


def load(adapter_dir: Path) -> dict[str, torch.Tensor]:
    """Load a single PEFT-format LoRA adapter into a task-vector dict.

    Materializes ``ΔW = (lora_alpha / r) · B @ A`` for each LoRA-decorated
    module. Output dtype matches what's stored in the safetensors file
    (typically bf16 for our adapters). One entry per canonical
    layer+module identifier.

    Raises:
        FileNotFoundError: if ``adapter_config.json`` or
            ``adapter_model.safetensors`` is missing.
        ValueError: if any module has only one of (lora_A, lora_B).
    """
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json missing: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"adapter_model.safetensors missing: {weights_path}")

    from safetensors.torch import load_file as safetensors_load_file

    with config_path.open() as f:
        cfg = json.load(f)
    r = cfg["r"]
    alpha = cfg["lora_alpha"]
    scaling = alpha / r

    state = safetensors_load_file(str(weights_path), device="cpu")

    # Group factor weights by canonical name → {"A": tensor, "B": tensor}
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for raw_name, tensor in state.items():
        # Skip non-LoRA-factor entries silently (e.g. modules_to_save tensors
        # would land here if the spec ever allowed them; the verifier blocks
        # that upstream so this is defensive).
        if _PEFT_SUFFIX_RE.search(raw_name) is None:
            continue
        canon = canonicalize(raw_name)
        factor = _which_factor(raw_name)
        pairs.setdefault(canon, {})[factor] = tensor

    task_vector: dict[str, torch.Tensor] = {}
    for canon, fac in pairs.items():
        if "A" not in fac or "B" not in fac:
            present = sorted(fac.keys())
            raise ValueError(
                f"incomplete LoRA pair for {canon!r}: have factors {present}, "
                f"need both 'A' and 'B'"
            )
        a = fac["A"]
        b = fac["B"]
        # ΔW shape: [out_features, in_features], dtype follows the safetensors file.
        delta_w = scaling * (b @ a)
        task_vector[canon] = delta_w.to(a.dtype)

    return task_vector


def load_all(
    adapters_dir: Path,
    locked_spec: dict,
    expected_domains: list[str] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Load all 4 domain adapters from ``adapters_dir`` and verify each.

    The directory must contain exactly one subdir per expected domain,
    each in PEFT format. Extra subdirs are an error (strict contract).

    Raises:
        FileNotFoundError: if a required domain subdir or adapter file is missing.
        ValueError: if there are extra (non-expected) subdirs in ``adapters_dir``.
        SpecMismatchError: if any adapter diverges from the locked spec.
    """
    if expected_domains is None:
        expected_domains = list(CANONICAL_DOMAINS)

    if not adapters_dir.is_dir():
        raise FileNotFoundError(f"adapters_dir not found or not a directory: {adapters_dir}")

    present_subdirs = {p.name for p in adapters_dir.iterdir() if p.is_dir()}

    missing = [d for d in expected_domains if d not in present_subdirs]
    if missing:
        raise FileNotFoundError(
            f"missing domain subdir(s) in {adapters_dir}: {missing}"
        )

    extras = sorted(present_subdirs - set(expected_domains))
    if extras:
        raise ValueError(
            f"unexpected subdir(s) in {adapters_dir} (strict contract): {extras}"
        )

    # First pass: verify every adapter. Collect failures, raise once at the end
    # so the user sees ALL bad adapters in one shot rather than a chain of retries.
    failures: dict[str, VerifyResult] = {}
    for domain in expected_domains:
        config_path = adapters_dir / domain / "adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"adapter_config.json missing for {domain}: {config_path}")
        result = verify(config_path, locked_spec)
        if not result.passed:
            failures[domain] = result

    if failures:
        raise SpecMismatchError(failures)

    # Second pass: now safe to materialize tensors.
    out: dict[str, dict[str, torch.Tensor]] = {}
    for domain in expected_domains:
        tv = load(adapters_dir / domain)
        out[domain] = tv
        logger.info("Loaded %s: %d task-vector entries, verified ok", domain, len(tv))

    return out

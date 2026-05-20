"""
End-to-end orchestrator for the group-merge pipeline.

Reads four locked-spec LoRA adapters from a local directory, verifies them
against ``lora.yaml``, dispatches to the configured merge method through
``METHOD_REGISTRY``, SVD-truncates the merged ΔW back to the locked rank
``r``, and writes a PEFT-readable adapter directory to disk.

The pipeline never pushes to HF (that is :mod:`merge.publish`'s job, Stage 5)
and never loads the base model — only the LoRA factors.

Stage 4 implementation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .load_adapter import CANONICAL_DOMAINS, decanonicalize, load_all
from .methods import METHOD_REGISTRY
from .verify_spec import load_locked_spec


logger = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_LOCKED_SPEC: Path = REPO_ROOT / "lora.yaml"


def svd_factor(
    delta_w,
    r: int,
    alpha: int,
):
    """Factor a full-rank ΔW into PEFT-style rank-r factors via SVD.

    Returns ``(lora_A, lora_B)`` with shapes ``(r, in)`` and ``(out, r)``
    such that ``(alpha/r) · lora_B @ lora_A`` is the best rank-r
    approximation of ``delta_w``. Computation is in fp32; both outputs are
    cast back to ``delta_w.dtype`` before return.
    """
    import torch

    dtype = delta_w.dtype
    delta_w_fp32 = delta_w.to(torch.float32)
    # Thin SVD: U [out, k], S [k], Vh [k, in], with k = min(out, in).
    u, s, vh = torch.linalg.svd(delta_w_fp32, full_matrices=False)

    if r > s.numel():
        raise ValueError(
            f"target rank {r} exceeds min(out, in) = {s.numel()} for ΔW shape "
            f"{tuple(delta_w.shape)}; cannot truncate."
        )

    u_r = u[:, :r]
    s_r = s[:r]
    vh_r = vh[:r, :]
    sqrt_s = torch.sqrt(s_r)

    scale = float(r) / float(alpha)
    lora_A = (sqrt_s.unsqueeze(1) * vh_r) * scale  # [r, in]
    lora_B = u_r * sqrt_s.unsqueeze(0)             # [out, r]

    return lora_A.to(dtype), lora_B.to(dtype)


def _ensure_empty_output_dir(output_dir: Path) -> None:
    """Create ``output_dir`` if missing; raise if it exists and is non-empty."""
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"output_dir is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"output_dir already exists and is non-empty: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)


def _copy_adapter_config(
    source_adapter_dir: Path,
    output_dir: Path,
) -> None:
    """Copy ``adapter_config.json`` from ``source_adapter_dir``, set inference_mode=False."""
    src = source_adapter_dir / "adapter_config.json"
    with src.open() as f:
        cfg = json.load(f)
    cfg["inference_mode"] = False
    with (output_dir / "adapter_config.json").open("w") as f:
        json.dump(cfg, f, indent=2)


def merge_adapters(
    adapters_dir: Path,
    method: str,
    output_dir: Path,
    locked_spec_path: Path | None = None,
    method_kwargs: dict[str, Any] | None = None,
    generation_config: dict[str, Any] | None = None,
    device: str | None = None,
) -> Path:
    """Run the full merge pipeline and write a PEFT-loadable adapter directory.

    Pipeline: load_all → verify each → dispatch to method → SVD factor → save.

    Args:
        adapters_dir: Directory with four subdirs (``math``, ``general_knowledge``,
            ``safety``, ``multilingual``), each a PEFT-format LoRA adapter.
        method: One of ``METHOD_REGISTRY`` keys: ``"uniform"``,
            ``"dare_uniform"``, ``"dare_weighted"``, ``"ties"``.
            ``"adamerging"`` is still a stub (Stage 7).
        output_dir: Directory to write the merged adapter to. Created if
            missing. If it exists, must be empty; otherwise raises
            ``FileExistsError``.
        locked_spec_path: Path to ``lora.yaml``. Defaults to the repo root.
        method_kwargs: Keyword arguments forwarded to the method (e.g.
            ``{"drop_rate": 0.5, "seed": 42}`` for DARE variants;
            ``{"weights": [...]}``  for ``dare_weighted``;
            ``{"trim_ratio": 0.5}`` for TIES).
        generation_config: If provided, written to
            ``output_dir/generation_config.json`` as part of the merged
            adapter directory. The dict structure should match the CS-552
            project's required schema (use
            :func:`merge.generation_config.make_generation_config` to
            construct). If ``None``, no ``generation_config.json`` is
            written.
        device: Compute device for loading + merging. ``None`` (default)
            auto-selects ``"cuda"`` when ``torch.cuda.is_available()``,
            otherwise ``"cpu"``. The final safetensors save always runs
            from CPU tensors regardless of this choice.

    Returns:
        The ``output_dir`` path (now populated with ``adapter_config.json``,
        ``adapter_model.safetensors``, and optionally
        ``generation_config.json``).

    Raises:
        FileNotFoundError: if ``adapters_dir`` or its expected subdirs are
            missing.
        SpecMismatchError: if any adapter diverges from the locked spec.
        KeyError: if ``method`` is not in ``METHOD_REGISTRY``.
        FileExistsError: if ``output_dir`` exists and is non-empty.
        NotImplementedError: if ``method`` is still a stub.
        ValueError: if method kwargs are invalid for the chosen method.
    """
    # Lazy: safetensors is only needed at the save step. Importing it at
    # module level would force the laptop's torch-free environment to fail
    # at import time, defeating the lazy-import pattern from Stage 2.
    import torch
    from safetensors.torch import save_file as safetensors_save_file

    if method not in METHOD_REGISTRY:
        raise KeyError(
            f"unknown merge method {method!r}; "
            f"valid options are {sorted(METHOD_REGISTRY.keys())!r}"
        )

    spec_path = locked_spec_path if locked_spec_path is not None else DEFAULT_LOCKED_SPEC
    locked_spec = load_locked_spec(spec_path)
    r = int(locked_spec["r"])
    alpha = int(locked_spec["lora_alpha"])

    _ensure_empty_output_dir(output_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("merge_adapters using device=%s", device)

    # load_all verifies each adapter against locked_spec and raises on mismatch.
    adapters_by_domain = load_all(adapters_dir, locked_spec, device=device)
    task_vectors = list(adapters_by_domain.values())  # canonical-order list
    logger.info(
        "Loaded %d adapters for merge (domains=%s, method=%s)",
        len(task_vectors), list(adapters_by_domain.keys()), method,
    )

    kwargs = method_kwargs or {}
    merge_fn = METHOD_REGISTRY[method]
    merged = merge_fn(task_vectors, **kwargs)

    # SVD-factor each merged ΔW into (lora_A, lora_B), assemble PEFT-named dict.
    peft_state: dict[str, Any] = {}
    for canonical, delta_w in merged.items():
        lora_a, lora_b = svd_factor(delta_w, r=r, alpha=alpha)
        peft_state[decanonicalize(canonical, "lora_A")] = lora_a
        peft_state[decanonicalize(canonical, "lora_B")] = lora_b

    # Save weights then config. Order is incidental, but config-last lets a
    # mid-run failure leave only the safetensors file (clearly broken).
    # safetensors refuses to save non-contiguous tensors (views, slices, broadcasts).
    # SVD factorization in svd_factor() produces views; force a copy here.
    # When device != cpu, .to("cpu") materializes a CPU copy; safetensors then
    # gets contiguous CPU tensors regardless of where the merge ran.
    peft_state = {k: v.detach().to("cpu").contiguous() for k, v in peft_state.items()}
    safetensors_save_file(peft_state, str(output_dir / "adapter_model.safetensors"))

    # The four adapters are byte-identical on the 8 load-bearing fields; using
    # math's config carries through any teammate-supplied PEFT bookkeeping too.
    source = adapters_dir / CANONICAL_DOMAINS[0]
    _copy_adapter_config(source, output_dir)

    if generation_config is not None:
        gen_config_path = output_dir / "generation_config.json"
        with gen_config_path.open("w") as f:
            json.dump(generation_config, f, indent=2)
        logger.info("Wrote generation_config.json to %s", gen_config_path)

    return output_dir


__all__ = ["merge_adapters", "svd_factor", "DEFAULT_LOCKED_SPEC", "REPO_ROOT"]

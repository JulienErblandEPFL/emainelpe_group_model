"""
End-to-end orchestrator for the group-merge pipeline.

Reads four locked-spec LoRA adapters from a local directory, verifies them
against ``lora.yaml``, dispatches to the configured merge method through
``METHOD_REGISTRY``, SVD-truncates the merged ΔW back to the locked rank
``r``, injects the rank-r factors into a fresh PEFT wrapper around the base
model, runs ``merge_and_unload`` to bake the deltas into the base weights,
and writes a full HF-format model directory to disk.

The output is a self-contained ``transformers``-loadable directory
(``config.json`` + ``model.safetensors``) — not a LoRA adapter. The
intermediate LoRA factorization is held in GPU memory and never hits
disk; the rank-r truncation discipline is preserved (we still respect the
locked-spec ``r`` and ``α``) but the artifact is the full materialized
model. This format is what vLLM and the May 24 milestone CI both consume.

Stage 4 implementation; refactored in Day 7 (2026-05-20) after the vLLM
LoRA loader rejected PEFT-format adapter weights.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from .load_adapter import decanonicalize, load_all
from .methods import METHOD_REGISTRY
from .verify_spec import load_locked_spec


logger = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_LOCKED_SPEC: Path = REPO_ROOT / "lora.yaml"
DEFAULT_CHAT_TEMPLATE: Path = REPO_ROOT / "chat_template.jinja"


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


def merge_adapters(
    adapters_dir: Path,
    method: str,
    output_dir: Path,
    locked_spec_path: Path | None = None,
    method_kwargs: dict[str, Any] | None = None,
    generation_config: dict[str, Any] | None = None,
    base_model_repo: str = "Qwen/Qwen3-1.7B",
    device: str | None = None,
) -> Path:
    """Run the full merge pipeline and write a full HF-format model directory.

    Pipeline: load_all → verify each → dispatch to method → SVD factor →
    inject into PEFT wrapper around base model → merge_and_unload →
    save_pretrained (full model) → copy tokenizer + chat template.

    Args:
        adapters_dir: Directory with four subdirs (``math``, ``general_knowledge``,
            ``safety``, ``multilingual``), each a PEFT-format LoRA adapter.
        method: One of ``METHOD_REGISTRY`` keys.
        output_dir: Directory to write the merged model to. Created if
            missing. If it exists, must be empty; otherwise raises
            ``FileExistsError``.
        locked_spec_path: Path to ``lora.yaml``. Defaults to the repo root.
        method_kwargs: Keyword arguments forwarded to the method.
        generation_config: If provided, written to
            ``output_dir/generation_config.json`` alongside the model.
        base_model_repo: HF repo or local path for the base model. Defaults
            to ``Qwen/Qwen3-1.7B``.
        device: Compute device for loading + merging. ``None`` (default)
            auto-selects ``"cuda"`` when ``torch.cuda.is_available()``,
            otherwise ``"cpu"``.

    Returns:
        The ``output_dir`` path (now populated with ``config.json``,
        ``model.safetensors`` (or sharded variants), tokenizer files,
        ``chat_template.jinja``, and optionally ``generation_config.json``).

    Raises:
        FileNotFoundError: if ``adapters_dir`` or its expected subdirs are
            missing.
        SpecMismatchError: if any adapter diverges from the locked spec.
        KeyError: if ``method`` is not in ``METHOD_REGISTRY``.
        FileExistsError: if ``output_dir`` exists and is non-empty.
        NotImplementedError: if ``method`` is still a stub.
        ValueError: if method kwargs are invalid for the chosen method.
    """
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    spec_path = locked_spec_path if locked_spec_path is not None else DEFAULT_LOCKED_SPEC
    locked_spec = load_locked_spec(spec_path)
    r = int(locked_spec["r"])
    alpha = int(locked_spec["lora_alpha"])

    _ensure_empty_output_dir(output_dir)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("merge_adapters using device=%s, base=%s", device, base_model_repo)

    # GPU-holding references initialized to None so the finally-block ``del``s
    # are always safe — even if an exception fires before the try block's body
    # has assigned them.
    base = None
    peft_model = None
    merged_model = None
    adapters_by_domain: dict[str, Any] | None = None
    task_vectors: list[Any] | None = None
    factorized: dict[str, tuple[Any, Any]] | None = None
    state_update: dict[str, Any] | None = None
    existing_state: dict[str, Any] | None = None

    try:
        logger.info("Acquiring GPU memory for merge_adapters")

        adapters_by_domain = load_all(adapters_dir, locked_spec, device=device)
        task_vectors = list(adapters_by_domain.values())
        logger.info(
            "Loaded %d adapters for merge (domains=%s, method=%s)",
            len(task_vectors), list(adapters_by_domain.keys()), method,
        )

        # Method-name validation moved inside the try block so a typo on a
        # bad method name still goes through the GPU-memory finally cleanup
        # if load_all already placed task vectors on cuda. Wasted load_all
        # work on typos is seconds-scale; the alternative (early-fail before
        # acquiring GPU) makes the cleanup path harder to test.
        if method not in METHOD_REGISTRY:
            raise KeyError(
                f"unknown merge method {method!r}; "
                f"valid options are {sorted(METHOD_REGISTRY.keys())!r}"
            )

        kwargs = dict(method_kwargs or {})
        # AdaMerging produces metrics (loss curve, learned per-(task, layer)
        # coefficients) that the dict-return contract of merge_fn would
        # otherwise discard. Inject a metrics-out path + the task_names so
        # the registry wrappers can persist them alongside the merged model.
        # task_names row order matches task_vectors index order, which is
        # ``list(adapters_by_domain.values())`` ⇄ ``list(adapters_by_domain.keys())``.
        if method in {"adamerging", "dare_adamerging"}:
            kwargs.setdefault("metrics_out_path", output_dir / "adamerging_metrics.json")
            kwargs.setdefault("task_names", list(adapters_by_domain.keys()))
        merge_fn = METHOD_REGISTRY[method]
        merged = merge_fn(task_vectors, **kwargs)

        factorized = {}
        for canonical, delta_w in merged.items():
            lora_a, lora_b = svd_factor(delta_w, r=r, alpha=alpha)
            factorized[canonical] = (lora_a, lora_b)
        logger.info("SVD-factored %d merged ΔW tensors to rank-%d", len(factorized), r)

        logger.info("Loading base model %s onto %s", base_model_repo, device)
        base = AutoModelForCausalLM.from_pretrained(
            base_model_repo,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

        lora_config = LoraConfig(
            r=int(locked_spec["r"]),
            lora_alpha=int(locked_spec["lora_alpha"]),
            lora_dropout=float(locked_spec["lora_dropout"]),
            target_modules=list(locked_spec["target_modules"]),
            bias=locked_spec["bias"],
            task_type=locked_spec["task_type"],
        )
        logger.info("Wrapping base model with PEFT LoraConfig (r=%d, alpha=%d)", r, alpha)
        peft_model = get_peft_model(base, lora_config)

        state_update = {}
        existing_state = peft_model.state_dict()
        for canonical, (lora_a, lora_b) in factorized.items():
            a_key = decanonicalize(canonical, "lora_A")
            b_key = decanonicalize(canonical, "lora_B")
            if a_key not in existing_state or b_key not in existing_state:
                raise KeyError(
                    f"PEFT wrapper has no parameter named {a_key!r} or {b_key!r}; "
                    f"check that LoraConfig target_modules match the base model "
                    f"architecture (got target_modules={locked_spec['target_modules']!r})"
                )
            ref_a = existing_state[a_key]
            ref_b = existing_state[b_key]
            state_update[a_key] = lora_a.to(device=ref_a.device, dtype=ref_a.dtype)
            state_update[b_key] = lora_b.to(device=ref_b.device, dtype=ref_b.dtype)

        missing, unexpected = peft_model.load_state_dict(state_update, strict=False)
        if unexpected:
            logger.warning(
                "load_state_dict reported %d unexpected keys: %s",
                len(unexpected), unexpected[:5],
            )
        logger.info(
            "Injected %d LoRA tensors into PEFT wrapper (%d base params untouched)",
            len(state_update), len(missing),
        )

        sample_key = next(iter(state_update.keys()))
        after = peft_model.state_dict()[sample_key]
        if not torch.allclose(
            after.detach().to(state_update[sample_key].device, state_update[sample_key].dtype),
            state_update[sample_key],
            rtol=1e-3,
            atol=1e-3,
        ):
            raise RuntimeError(
                f"PEFT injection check failed: load_state_dict did not write {sample_key}. "
                "Check PEFT version compatibility (expected base_model.model.* key prefix)."
            )

        logger.info("Running merge_and_unload to bake deltas into base weights")
        merged_model = peft_model.merge_and_unload()

        logger.info("Saving full merged model to %s", output_dir)
        merged_model.save_pretrained(
            output_dir,
            safe_serialization=True,
            max_shard_size="5GB",
        )

        logger.info("Saving tokenizer from %s", base_model_repo)
        tokenizer = AutoTokenizer.from_pretrained(base_model_repo)
        tokenizer.save_pretrained(output_dir)

        chat_template_src = DEFAULT_CHAT_TEMPLATE
        if chat_template_src.exists():
            shutil.copy2(chat_template_src, output_dir / "chat_template.jinja")
            logger.info("Copied locked chat_template.jinja to %s", output_dir)
        else:
            logger.warning(
                "Locked chat template not found at %s; merged model will use the "
                "tokenizer's bundled template.",
                chat_template_src,
            )

        if generation_config is not None:
            gen_path = output_dir / "generation_config.json"
            with gen_path.open("w") as f:
                json.dump(generation_config, f, indent=2)
            logger.info("Wrote generation_config.json to %s", gen_path)
    finally:
        # Drop every name that could be holding GPU tensors. The locals are
        # all initialized to None before the try so unconditional del is safe.
        del peft_model
        del merged_model
        del base
        del state_update
        del existing_state
        del factorized
        del task_vectors
        del adapters_by_domain

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Released GPU memory after merge_adapters")

    return output_dir


__all__ = [
    "merge_adapters",
    "svd_factor",
    "DEFAULT_LOCKED_SPEC",
    "DEFAULT_CHAT_TEMPLATE",
    "REPO_ROOT",
]

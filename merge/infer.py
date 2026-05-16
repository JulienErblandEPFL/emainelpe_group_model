"""
Local inference for the merged group adapter against the validation snapshot.

Loads Qwen3-1.7B with the merged LoRA adapter, applies the team chat template,
and generates ``n`` completions per prompt with the same settings the
CS-552 nightly CI uses (vLLM, n=8 by default, max length from ``lora.yaml``).

The output JSONL is shaped exactly like the input expected by
``evaluate.score`` — see ``evaluate/README.md``.

To be implemented in Stage 5.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_completions(
    adapter_dir: Path,
    base_model_repo: str,
    prompts: list[str],
    *,
    n_per_prompt: int = 8,
    chat_template_path: Path | None = None,
    generation_config: dict[str, Any] | None = None,
    max_seq_len: int = 4096,
    device: str = "cuda",
) -> list[list[str]]:
    """
    Generate ``n_per_prompt`` completions for each prompt under the merged adapter.

    Args:
        adapter_dir: Directory written by ``pipeline.merge_adapters``.
        base_model_repo: HF repo ID for the base model (must match locked spec).
        prompts: Raw prompt strings (NOT pre-templated; the chat template is
            applied inside this function via ``tokenizer.apply_chat_template``).
        n_per_prompt: Number of completions to draw per prompt. Default 8 to
            mirror the CI ``pass@8`` setting.
        chat_template_path: Path to ``chat_template.jinja``. If None, the
            tokenizer's bundled template is used (NOT what the CI does — pass
            this explicitly).
        generation_config: Sampling overrides (temperature, top_k, top_p).
            ``max_new_tokens`` defaults to ``max_seq_len - len(prompt_tokens)``.
        max_seq_len: Hard cap from the locked spec.
        device: ``"cuda"`` (vLLM) or ``"cpu"`` (transformers fallback, slow).

    Returns:
        ``len(prompts)``-long list, each element a ``n_per_prompt``-long list
        of completion strings. Completion strings include the ``<think>...
        </think>`` block followed by the boxed final answer; downstream
        ``evaluate.score`` handles extraction.
    """
    raise NotImplementedError("Stage 5")


def generate_for_validation_set(
    adapter_dir: Path,
    base_model_repo: str,
    validation_samples_dir: Path,
    output_dir: Path,
    *,
    n_per_prompt: int = 8,
    chat_template_path: Path | None = None,
) -> dict[str, Path]:
    """
    Run ``generate_completions`` for all four ``validation_samples/*.jsonl``
    files and write one JSONL per benchmark in ``output_dir``.

    Args:
        adapter_dir: Directory written by ``pipeline.merge_adapters``.
        base_model_repo: HF repo ID for the base model.
        validation_samples_dir: Path to ``../validation_samples/``.
        output_dir: Where to write the four output JSONLs.
        n_per_prompt: Completions per problem.
        chat_template_path: Path to ``chat_template.jinja``.

    Returns:
        Dict mapping benchmark name (``"math"``, ``"general_knowledge"``,
        ``"multilingual"``, ``"safety"``) to the output JSONL path.
    """
    raise NotImplementedError("Stage 5")

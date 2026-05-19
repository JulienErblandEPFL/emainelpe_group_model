"""vLLM-based inference for evaluating merged adapters.

Single-benchmark inference: take a vLLM-loaded model + a validation JSONL,
generate ``n`` completions per problem, write a generations JSONL.

The vLLM model is owned by the caller (``eval_all.evaluate_all_benchmarks``
loads it once and reuses across the 4 benchmarks). ``infer.py`` does not
load or release vLLM.

Output JSONL schema (one row per problem, compatible with ``evaluate.score``):

    {"problem_id": 0,
     "prompt": "<problem text>",
     "answer": "<gold answer>",
     "completions": ["<gen 1>", "<gen 2>", ...],
     "completion_tokens_used": [123, 99, ...]}

``completion_tokens_used`` is recorded so the failure classifier in
``eval_all.classify_completion`` can detect ``max_tokens``-truncation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


_PROMPT_FIELDS: tuple[str, ...] = ("prompt", "problem", "question")
_ANSWER_FIELDS: tuple[str, ...] = ("answer", "reference", "solution")


@dataclass
class InferenceConfig:
    """Sampling + generation params for one inference run.

    Defaults mirror Qwen3-1.7B's bundled ``generation_config.json``
    (temperature=0.7, top_p=0.8, top_k=20). ``max_tokens=2048`` is the eval
    cap; the locked spec's 16384 ceiling is reserved for CI.
    """
    n: int = 8
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_tokens: int = 2048
    seed: int = 42

    @classmethod
    def from_generation_config_dict(cls, config_dict: dict[str, Any]) -> "InferenceConfig":
        """Construct an :class:`InferenceConfig` from a
        ``generation_config.json``-style dict.

        Reads ``temperature``, ``top_p``, ``top_k``, and (optionally)
        ``max_new_tokens`` from the dict. Other :class:`InferenceConfig`
        fields (``n``, ``seed``) keep their dataclass defaults.

        Args:
            config_dict: A dict from ``generation_config.json`` (must
                contain at minimum ``temperature``/``top_p``/``top_k``;
                ``max_new_tokens`` optional).

        Returns:
            :class:`InferenceConfig` with sampling params taken from the
            dict.

        Raises:
            KeyError: if ``temperature``/``top_p``/``top_k`` are missing.
        """
        return cls(
            temperature=float(config_dict["temperature"]),
            top_p=float(config_dict["top_p"]),
            top_k=int(config_dict["top_k"]),
            max_tokens=int(config_dict.get("max_new_tokens", 2048)),
        )


def _first_present(item: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None


def load_validation_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    """Read a validation JSONL file, normalize field names.

    The shipped ``validation_samples/*.jsonl`` files use ``prompt``/``answer``.
    Other repos sometimes use ``problem``/``solution`` or ``question``/
    ``reference``. We accept any of those and emit a canonical dict.

    Returns:
        List of ``{"problem_id": int, "prompt": str, "answer": str}`` dicts.
        ``problem_id`` is the 0-indexed file position.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Validation file not found: {jsonl_path}")

    out: list[dict[str, Any]] = []
    with jsonl_path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{lineno}: invalid JSON: {exc}") from exc

            prompt = _first_present(raw, _PROMPT_FIELDS)
            answer = _first_present(raw, _ANSWER_FIELDS)
            if prompt is None or answer is None:
                raise ValueError(
                    f"{jsonl_path}:{lineno}: missing prompt/answer field "
                    f"(have keys {sorted(raw.keys())!r})"
                )
            pid = raw.get("problem_id", raw.get("id", len(out)))
            out.append({"problem_id": pid, "prompt": str(prompt), "answer": str(answer)})
    return out


def _render_chat_prompt(tokenizer, problem_text: str) -> str:
    """Apply the chat template with ``add_generation_prompt=True``.

    Returns the rendered string (not token IDs) — vLLM ``generate(prompts=...)``
    wants strings.
    """
    messages = [{"role": "user", "content": problem_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def run_inference(
    vllm_model,
    lora_request,
    benchmark_name: str,
    validation_jsonl_path: Path,
    output_jsonl_path: Path,
    config: InferenceConfig,
    chat_template_path: Path | None = None,
) -> Path:
    """Generate ``config.n`` completions per problem and write a JSONL.

    The output JSONL is consumable by ``evaluate.score`` directly (it carries
    ``prompt``/``answer``/``completions``); the extra ``problem_id`` and
    ``completion_tokens_used`` fields are ignored by ``evaluate.score`` and
    consumed by the failure classifier downstream.

    Args:
        vllm_model: A loaded ``vllm.LLM`` instance, owned by the caller.
        lora_request: A ``vllm.lora.request.LoRARequest`` for the merged
            adapter, or ``None`` to run against the bare base model.
        benchmark_name: One of the 4 canonical domain names (used for logging).
        validation_jsonl_path: Source problems.
        output_jsonl_path: Destination JSONL. Parent dir is created if missing.
        config: Sampling params.
        chat_template_path: Optional path to ``chat_template.jinja``. If
            provided, the file contents override the tokenizer's bundled
            template before generation. If ``None``, the tokenizer's
            template is used.

    Returns:
        ``output_jsonl_path``.

    Raises:
        FileNotFoundError: ``validation_jsonl_path`` does not exist.
        RuntimeError: vLLM raises during generation.
    """
    # Lazy import: this module must import cleanly on a torch-free laptop.
    from vllm import SamplingParams

    problems = load_validation_jsonl(validation_jsonl_path)
    if not problems:
        raise ValueError(f"No problems found in {validation_jsonl_path}")

    tokenizer = vllm_model.get_tokenizer()
    if chat_template_path is not None:
        tokenizer.chat_template = chat_template_path.read_text()

    prompts = [_render_chat_prompt(tokenizer, p["prompt"]) for p in problems]

    sampling_params = SamplingParams(
        n=config.n,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        max_tokens=config.max_tokens,
        seed=config.seed,
    )

    logger.info(
        "infer[%s]: generating n=%d, max_tokens=%d for %d problems",
        benchmark_name, config.n, config.max_tokens, len(prompts),
    )

    gen_kwargs: dict[str, Any] = {"sampling_params": sampling_params}
    if lora_request is not None:
        gen_kwargs["lora_request"] = lora_request

    try:
        request_outputs = vllm_model.generate(prompts, **gen_kwargs)
    except Exception as exc:
        raise RuntimeError(f"vLLM generation failed for {benchmark_name}: {exc}") from exc

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl_path.open("w") as f:
        for problem, request_output in zip(problems, request_outputs):
            completions = [out.text for out in request_output.outputs]
            tokens_used = [len(out.token_ids) for out in request_output.outputs]
            row = {
                "problem_id": problem["problem_id"],
                "prompt": problem["prompt"],
                "answer": problem["answer"],
                "completions": completions,
                "completion_tokens_used": tokens_used,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("infer[%s]: wrote %s", benchmark_name, output_jsonl_path)
    return output_jsonl_path

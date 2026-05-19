"""Multi-benchmark evaluation orchestrator with failure analysis.

For each of the 4 benchmarks:
1. Run inference (vLLM, n=8 completions per problem) via :mod:`merge.infer`.
2. Score via the existing ``evaluate.*`` helpers (pass@1, pass@8).
3. Classify each pass@8 failure into one of 7 categories
   (no_boxed, empty_boxed, wrong_answer, malformed_answer, truncated,
   refusal, mixed) for debugging.
4. Write ``scorecard.json`` + per-benchmark ``failures_<benchmark>.json``
   + per-benchmark ``generations_<benchmark>.jsonl``.

vLLM is loaded once at the start of :func:`evaluate_all_benchmarks` and
reused across all 4 benchmarks — cold start is ~30-60s and we want to
amortize it. The vllm import is lazy so this module imports cleanly on
a torch-free laptop.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from merge.infer import InferenceConfig, load_validation_jsonl, run_inference

logger = logging.getLogger(__name__)


# Canonical domain name -> evaluate.score method. The evaluate module uses
# "knowledge" where we use "general_knowledge"; the others align 1:1.
_DOMAIN_TO_METHOD: dict[str, str] = {
    "math": "boxed",
    "general_knowledge": "knowledge",
    "safety": "boxed",
    "multilingual": "boxed",
}


class FailureCategory(str, Enum):
    """Per-completion / per-problem failure taxonomy.

    Subclassing ``str`` makes the enum JSON-serializable via ``asdict``
    without custom encoders.
    """
    NO_BOXED = "no_boxed"
    EMPTY_BOXED = "empty_boxed"
    WRONG_ANSWER = "wrong_answer"
    MALFORMED_ANSWER = "malformed_answer"
    TRUNCATED = "truncated"
    REFUSAL = "refusal"
    MIXED = "mixed"


# Conservative refusal regex. Only fires from contexts where pass@8 = 0,
# i.e. a refusal-shaped phrase is meaningful evidence the model bailed.
_REFUSAL_PATTERNS = re.compile(
    r"(?:"
    r"I\s*cannot|I\s*can'?t|I\s*won'?t|I\s*refuse|"
    r"I'?m\s*sorry,\s*but|I\s*am\s*sorry,\s*but|"
    r"I'?m\s*not\s*able\s*to|I\s*am\s*not\s*able\s*to|"
    r"It\s*is\s*not\s*appropriate|"
    r"I\s*do\s*not\s*feel\s*comfortable"
    r")",
    re.IGNORECASE,
)


@dataclass
class BenchmarkResult:
    benchmark: str
    pass_at_1: float
    pass_at_8: float
    n_problems: int
    n_pass8_failed: int


@dataclass
class FailureRecord:
    problem_id: int | str
    problem: str
    expected: str
    primary_category: str
    per_completion_categories: list[str]
    completions: list[str] = field(default_factory=list)


def _looks_like_number(s: str) -> bool:
    """Best-effort numeric check; tolerates thousands-separators and signs."""
    if not s:
        return False
    candidate = s.strip().replace(",", "").replace("$", "").replace("\\", "")
    try:
        float(candidate)
        return True
    except ValueError:
        return False


def _looks_truncated(completion: str, tokens_used: int, tokens_limit: int) -> bool:
    """Truncation heuristic: hit token cap with no closing ``\\boxed{}``."""
    if tokens_used < tokens_limit:
        return False
    from evaluate.extract_answer import last_boxed_only_string
    return last_boxed_only_string(completion) is None


def classify_completion(
    completion: str,
    expected_answer: str,
    benchmark: str,
    max_tokens_used: int,
    max_tokens_limit: int,
) -> FailureCategory | None:
    """Classify ONE completion. Returns ``None`` if the completion is correct.

    Priority (first match wins):
        1. REFUSAL — refusal regex matches anywhere in the text.
        2. TRUNCATED — token cap hit with no closing ``\\boxed{}``.
        3. NO_BOXED — no ``\\boxed{...}`` at all.
        4. EMPTY_BOXED — ``\\boxed{}`` with whitespace-only contents.
        5. correct (per ``evaluate.benchmarks.is_correct_benchmark_answer``)
           — returns ``None``.
        6. MALFORMED_ANSWER — expected is numeric but extracted is not.
        7. WRONG_ANSWER — extracted is well-formed but does not match.

    Note: REFUSAL has priority over correctness; the failure analyzer is
    only invoked when ``pass@8 == 0``, so any refusal-phrased completion in
    that context is by construction not a correct one. The unit test
    exercises this branch in isolation.
    """
    # Lazy imports keep this module importable without the evaluate package
    # being on sys.path for, e.g., the test_skeleton import check.
    from evaluate.extract_answer import last_boxed_only_string, remove_boxed
    from evaluate.benchmarks import extract_benchmark_answer, is_correct_benchmark_answer

    if _REFUSAL_PATTERNS.search(completion):
        return FailureCategory.REFUSAL

    if _looks_truncated(completion, max_tokens_used, max_tokens_limit):
        return FailureCategory.TRUNCATED

    boxed = last_boxed_only_string(completion)
    if boxed is None:
        return FailureCategory.NO_BOXED

    inner = remove_boxed(boxed) or ""
    if not inner.strip():
        return FailureCategory.EMPTY_BOXED

    method = _DOMAIN_TO_METHOD.get(benchmark, "boxed")
    extracted = extract_benchmark_answer(completion, method, expected_answer)
    if is_correct_benchmark_answer(extracted, expected_answer, method):
        return None

    if _looks_like_number(expected_answer) and not _looks_like_number(extracted or ""):
        return FailureCategory.MALFORMED_ANSWER
    return FailureCategory.WRONG_ANSWER


def classify_problem_failure(
    completions: list[str],
    completions_tokens_used: list[int],
    expected_answer: str,
    benchmark: str,
    max_tokens_limit: int,
) -> tuple[FailureCategory, list[FailureCategory]]:
    """Aggregate per-completion categories into the problem's primary.

    Decision rule:
        - Classify each completion individually.
        - If any completion is correct (None), the precondition (pass@8=0)
          is violated — raise ``ValueError``.
        - Primary = strict-majority category (>n/2 of the completions).
        - Otherwise MIXED.

    Returns:
        ``(primary_category, per_completion_categories)``.
    """
    if len(completions) != len(completions_tokens_used):
        raise ValueError(
            f"completions ({len(completions)}) and completions_tokens_used "
            f"({len(completions_tokens_used)}) must be parallel"
        )

    per_completion: list[FailureCategory] = []
    for comp, tokens in zip(completions, completions_tokens_used):
        cat = classify_completion(
            completion=comp,
            expected_answer=expected_answer,
            benchmark=benchmark,
            max_tokens_used=tokens,
            max_tokens_limit=max_tokens_limit,
        )
        if cat is None:
            raise ValueError(
                "classify_problem_failure called with a completion that passed "
                "— precondition is that pass@8 == 0"
            )
        per_completion.append(cat)

    n = len(per_completion)
    counts = Counter(per_completion)
    most_common, top_count = counts.most_common(1)[0]
    if top_count * 2 > n:
        primary = most_common
    else:
        primary = FailureCategory.MIXED
    return primary, per_completion


def _score_one_benchmark(items: list[dict[str, Any]], method: str) -> tuple[
    dict[str, float], list[int]
]:
    """Compute ``pass@k`` metrics + per-problem correct-count list.

    Uses the same ``extract_benchmark_answer`` / ``is_correct_benchmark_answer``
    / ``compute_pass_at_k_for_dataset`` functions ``evaluate.score`` uses,
    so the metrics agree byte-for-byte with the CI scorer.
    """
    from evaluate.benchmarks import extract_benchmark_answer, is_correct_benchmark_answer
    from evaluate.pass_at_k import compute_pass_at_k_for_dataset

    per_problem_correct: list[int] = []
    n = 0
    for item in items:
        completions = item["completions"]
        n = len(completions)  # assume uniform; vLLM guarantees this
        reference = str(item["answer"])
        c = 0
        for comp in completions:
            extracted = extract_benchmark_answer(str(comp), method, reference)
            if is_correct_benchmark_answer(extracted, reference, method):
                c += 1
        per_problem_correct.append(c)
    k_values = [k for k in (1, 8) if k <= n]
    metrics = compute_pass_at_k_for_dataset(per_problem_correct, n, k_values)
    return metrics, per_problem_correct


def evaluate_one_benchmark(
    vllm_model,
    lora_request,
    benchmark: str,
    validation_jsonl_path: Path,
    output_dir: Path,
    config: InferenceConfig,
    chat_template_path: Path | None = None,
) -> tuple[BenchmarkResult, list[FailureRecord]]:
    """Run inference + score + failure analysis for ONE benchmark.

    Side effects:
        - Writes ``output_dir / f"generations_{benchmark}.jsonl"``
        - Writes ``output_dir / f"failures_{benchmark}.json"``

    Returns:
        ``(BenchmarkResult, list of FailureRecord for pass@8 failures)``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = output_dir / f"generations_{benchmark}.jsonl"
    failures_path = output_dir / f"failures_{benchmark}.json"

    run_inference(
        vllm_model=vllm_model,
        lora_request=lora_request,
        benchmark_name=benchmark,
        validation_jsonl_path=validation_jsonl_path,
        output_jsonl_path=generations_path,
        config=config,
        chat_template_path=chat_template_path,
    )

    items: list[dict[str, Any]] = []
    with generations_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    method = _DOMAIN_TO_METHOD.get(benchmark, "boxed")
    metrics, per_problem_correct = _score_one_benchmark(items, method)
    pass_at_1 = float(metrics.get("pass@1", 0.0))
    pass_at_8 = float(metrics.get("pass@8", 0.0))

    # Per-problem pass@8 indicator: 1 iff any of the n completions is correct.
    failures: list[FailureRecord] = []
    for item, c in zip(items, per_problem_correct):
        if c > 0:
            continue
        completions = [str(x) for x in item["completions"]]
        tokens_used = item.get("completion_tokens_used") or [config.max_tokens] * len(completions)
        primary, per_completion = classify_problem_failure(
            completions=completions,
            completions_tokens_used=tokens_used,
            expected_answer=str(item["answer"]),
            benchmark=benchmark,
            max_tokens_limit=config.max_tokens,
        )
        failures.append(FailureRecord(
            problem_id=item.get("problem_id", "?"),
            problem=str(item["prompt"]),
            expected=str(item["answer"]),
            primary_category=primary.value,
            per_completion_categories=[c.value for c in per_completion],
            completions=completions,
        ))

    with failures_path.open("w") as f:
        json.dump([asdict(rec) for rec in failures], f, ensure_ascii=False, indent=2)

    result = BenchmarkResult(
        benchmark=benchmark,
        pass_at_1=pass_at_1,
        pass_at_8=pass_at_8,
        n_problems=len(items),
        n_pass8_failed=len(failures),
    )
    logger.info(
        "eval[%s]: pass@1=%.3f pass@8=%.3f n_problems=%d failed=%d",
        benchmark, pass_at_1, pass_at_8, len(items), len(failures),
    )
    return result, failures


def evaluate_all_benchmarks(
    merged_adapter_dir: Path,
    base_model_repo: str,
    output_dir: Path,
    validation_samples_dir: Path,
    chat_template_path: Path | None = None,
    config: InferenceConfig | None = None,
    repo_root: Path | None = None,
) -> dict[str, BenchmarkResult]:
    """Top-level orchestrator. Loads vLLM once, runs all 4 benchmarks.

    Args:
        merged_adapter_dir: PEFT-format adapter from ``pipeline.merge_adapters``.
        base_model_repo: HF repo for base model (e.g. ``"Qwen/Qwen3-1.7B"``).
        output_dir: Where to write all results. Created if missing.
        validation_samples_dir: Path to ``validation_samples/`` with the 4
            domain JSONLs.
        chat_template_path: Optional path to ``chat_template.jinja``.
        config: ``InferenceConfig`` or ``None``. When ``None``, sampling
            params are resolved via the hierarchical fallback in
            :func:`merge.generation_config.load_generation_config`:
            ``merged_adapter_dir/generation_config.json`` →
            ``repo_root/generation_config.json`` → Qwen3 defaults.
        repo_root: Optional override for the repo-root fallback lookup. If
            ``None``, derived from this module's file path.

    Returns:
        ``{benchmark_name: BenchmarkResult}`` for the 4 canonical domains.

    Side effects:
        - Writes ``output_dir/scorecard.json``
        - Writes ``output_dir/generations_<benchmark>.jsonl`` × 4
        - Writes ``output_dir/failures_<benchmark>.json`` × 4
    """
    if not merged_adapter_dir.exists():
        raise FileNotFoundError(f"Merged adapter dir not found: {merged_adapter_dir}")
    if not validation_samples_dir.exists():
        raise FileNotFoundError(
            f"Validation samples dir not found: {validation_samples_dir}"
        )

    if config is None:
        from merge.generation_config import load_generation_config

        if repo_root is None:
            repo_root = Path(__file__).resolve().parent.parent
        gen_config_dict = load_generation_config(
            merged_adapter_dir=merged_adapter_dir,
            repo_root=repo_root,
        )
        config = InferenceConfig.from_generation_config_dict(gen_config_dict)
        logger.info(
            "Eval using sampling params: temperature=%s, top_p=%s, top_k=%s, max_tokens=%s",
            config.temperature, config.top_p, config.top_k, config.max_tokens,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports — vllm pulls CUDA and is ~10s to import.
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    logger.info("Loading vLLM base=%s with LoRA adapter=%s",
                base_model_repo, merged_adapter_dir)
    llm = LLM(
        model=base_model_repo,
        enable_lora=True,
        max_lora_rank=32,
        dtype="bfloat16",
    )
    lora_request = LoRARequest(
        lora_name="merged",
        lora_int_id=1,
        lora_path=str(merged_adapter_dir),
    )

    results: dict[str, BenchmarkResult] = {}
    try:
        for benchmark in ("math", "general_knowledge", "safety", "multilingual"):
            jsonl = validation_samples_dir / f"{benchmark}.jsonl"
            if not jsonl.exists():
                raise FileNotFoundError(
                    f"Missing validation file for benchmark {benchmark!r}: {jsonl}"
                )
            result, _failures = evaluate_one_benchmark(
                vllm_model=llm,
                lora_request=lora_request,
                benchmark=benchmark,
                validation_jsonl_path=jsonl,
                output_dir=output_dir,
                config=config,
                chat_template_path=chat_template_path,
            )
            results[benchmark] = result
    finally:
        try:
            del llm
            import torch
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover — cleanup is best-effort
            pass

    scorecard_path = output_dir / "scorecard.json"
    with scorecard_path.open("w") as f:
        json.dump(
            {name: asdict(r) for name, r in results.items()},
            f, ensure_ascii=False, indent=2,
        )
    logger.info("Wrote scorecard to %s", scorecard_path)
    return results

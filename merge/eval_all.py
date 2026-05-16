"""
Score the merged group adapter on the local validation snapshot.

Wraps ``evaluate.score`` from the repo root for each of the four benchmarks
(``math``, ``general_knowledge``, ``multilingual``, ``safety``). Uses the
exact extraction logic the CS-552 nightly CI uses (see ``evaluate/`` for
details — never duplicate that logic here).

To be implemented in Stage 5.
"""
from __future__ import annotations

from pathlib import Path


def evaluate_completions(
    completions_dir: Path,
    validation_samples_dir: Path,
) -> dict[str, dict[str, float]]:
    """
    Score completions for all four benchmarks.

    Args:
        completions_dir: Directory containing ``{benchmark}.jsonl`` files
            with completions, as written by ``infer.generate_for_validation_set``.
        validation_samples_dir: Path to ``../validation_samples/`` (only used
            to confirm prompt/answer alignment; scoring uses the completion
            files directly).

    Returns:
        Nested dict ``{benchmark_name: {metric: value}}`` where metric keys
        are ``"pass@1"`` and ``"pass@8"`` (if ``n_completions >= 8``).

    Raises:
        FileNotFoundError: if a benchmark JSONL is missing from ``completions_dir``.
    """
    raise NotImplementedError("Stage 5")


def four_domain_average(per_benchmark: dict[str, dict[str, float]]) -> float:
    """
    Compute the 4-domain average that determines the group-model leaderboard rank.

    The CS-552 leaderboard uses ``pass@8`` for math and ``pass@1`` for the
    other three. This helper picks the right metric per benchmark and averages.

    Args:
        per_benchmark: Output of ``evaluate_completions``.

    Returns:
        Mean across the four selected metrics, in ``[0.0, 1.0]``.

    Raises:
        KeyError: if any of the four expected benchmarks is missing.
    """
    raise NotImplementedError("Stage 5")

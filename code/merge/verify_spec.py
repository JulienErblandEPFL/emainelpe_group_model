"""
Locked-spec verification for LoRA adapters.

The DARE / AdaMerging / TIES pipeline assumes the four specialist adapters
share a byte-identical LoRA shape. This module loads ``lora.yaml`` and
compares it field-by-field to each adapter's ``adapter_config.json``.

It is the cheapest risk gate in the pipeline: no GPU, no torch, just YAML
and JSON. Verification is whitelist-based — only the 8 load-bearing fields
that affect additive merging are checked, and unknown PEFT bookkeeping
fields are ignored so version drift between teammates' PEFT installs does
not cause false positives.

Stage 2 implementation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# The 8 PEFT-style keys that matter for additive task-vector merging.
# Order here is the order used in VerifyResult.field_results for stable output.
LOAD_BEARING_FIELDS: tuple[str, ...] = (
    "base_model_name_or_path",
    "r",
    "lora_alpha",
    "lora_dropout",
    "bias",
    "task_type",
    "modules_to_save",
    "target_modules",
)

# Maps lora.yaml key paths to the PEFT-style names PEFT writes into
# adapter_config.json. The pipeline normalizes everything to the PEFT names.
_YAML_TO_PEFT: dict[str, str] = {
    "base_model": "base_model_name_or_path",
    "lora.r": "r",
    "lora.alpha": "lora_alpha",
    "lora.dropout": "lora_dropout",
    "lora.bias": "bias",
    "lora.task_type": "task_type",
    "lora.target_modules": "target_modules",
    "lora.modules_to_save": "modules_to_save",
}


@dataclass
class FieldResult:
    """Per-field verdict produced by :func:`verify`."""
    field: str
    expected: Any
    actual: Any
    passed: bool
    note: str = ""


@dataclass
class VerifyResult:
    """Structured result for a single adapter-vs-spec comparison."""
    passed: bool
    field_results: list[FieldResult] = field(default_factory=list)
    extra_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    summary: str = ""


class SpecMismatchError(Exception):
    """Raised when one or more adapters diverge from the locked spec."""

    def __init__(self, failures: dict[str, "VerifyResult"]) -> None:
        self.failures = failures
        super().__init__(self._format())

    def _format(self) -> str:
        lines = [f"{len(self.failures)} adapter(s) failed locked-spec verification:"]
        for name, result in self.failures.items():
            bad = [fr for fr in result.field_results if not fr.passed]
            field_summaries = ", ".join(
                f"{fr.field}(expected={fr.expected!r}, got={fr.actual!r})"
                for fr in bad
            )
            missing = (
                f"; missing fields: {result.missing_fields}"
                if result.missing_fields
                else ""
            )
            lines.append(f"  - {name}: {field_summaries}{missing}")
        return "\n".join(lines)


def _get_nested(d: dict, dotted: str) -> Any:
    """Walk a dotted key path, raising KeyError if any segment is missing."""
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def load_locked_spec(yaml_path: Path) -> dict[str, Any]:
    """Read ``lora.yaml`` and return a canonical 8-field dict in PEFT key names.

    The returned dict has exactly the keys in :data:`LOAD_BEARING_FIELDS`.
    ``target_modules`` is returned as a list preserving lora.yaml order;
    :func:`verify` compares it as a set.

    Raises:
        FileNotFoundError: if ``yaml_path`` does not exist.
        KeyError: if a load-bearing field is missing from the YAML.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"locked spec not found: {yaml_path}")

    with yaml_path.open() as f:
        raw = yaml.safe_load(f)

    canonical: dict[str, Any] = {}
    for yaml_key, peft_key in _YAML_TO_PEFT.items():
        # _get_nested raises KeyError on missing — propagated as-is so callers
        # know which load-bearing field is absent.
        canonical[peft_key] = _get_nested(raw, yaml_key)

    return canonical


def _load_adapter_config(adapter_config: dict | Path) -> dict:
    if isinstance(adapter_config, Path):
        if not adapter_config.exists():
            raise FileNotFoundError(f"adapter_config not found: {adapter_config}")
        with adapter_config.open() as f:
            return json.load(f)
    return adapter_config


_SENTINEL = object()


def verify(adapter_config: dict | Path, locked_spec: dict[str, Any]) -> VerifyResult:
    """Verify an adapter_config dict (or path) against the locked spec.

    Equality semantics:
    - ``target_modules`` is compared as ``set(expected) == set(actual)``.
      PEFT serializes the list in graph-walk order, which is not the order
      we wrote into lora.yaml.
    - ``modules_to_save``: a missing key in the adapter is treated as
      equivalent to ``None`` (older PEFT versions omit the key entirely
      when unset). The locked spec REQUIRES this to be None — setting it
      to a list adds full-rank tensors that break additive merging.
    - All other fields: ordinary ``==``.

    Extra fields in the adapter that aren't in the locked spec are
    recorded in ``extra_fields`` but never cause a failure: PEFT writes
    many bookkeeping fields (peft_version, init_lora_weights, alpha_pattern,
    inference_mode...) that have no bearing on additive task-vector merging.

    Raises:
        FileNotFoundError: if a Path is passed and the file is missing.
        json.JSONDecodeError: if the file is malformed JSON.
    """
    adapter = _load_adapter_config(adapter_config)

    results: list[FieldResult] = []
    missing: list[str] = []

    for fname in LOAD_BEARING_FIELDS:
        expected = locked_spec[fname]

        if fname == "modules_to_save":
            # Omission == None; the locked spec requires None either way.
            actual = adapter.get(fname, None)
            present = True
        else:
            actual = adapter.get(fname, _SENTINEL)
            present = actual is not _SENTINEL

        if not present:
            missing.append(fname)
            results.append(
                FieldResult(
                    field=fname,
                    expected=expected,
                    actual=None,
                    passed=False,
                    note="missing in adapter",
                )
            )
            continue

        if fname == "target_modules":
            passed = set(expected) == set(actual)
            note = "compared as set"
        else:
            passed = expected == actual
            note = ""

        results.append(
            FieldResult(
                field=fname,
                expected=expected,
                actual=actual,
                passed=passed,
                note=note,
            )
        )

    extras = sorted(set(adapter.keys()) - set(LOAD_BEARING_FIELDS))
    all_passed = all(r.passed for r in results)

    n_fail = sum(1 for r in results if not r.passed)
    if all_passed:
        summary = f"PASS: all {len(LOAD_BEARING_FIELDS)} load-bearing fields match."
    else:
        summary = (
            f"FAIL: {n_fail}/{len(LOAD_BEARING_FIELDS)} load-bearing field(s) "
            f"diverge from locked spec."
        )

    return VerifyResult(
        passed=all_passed,
        field_results=results,
        extra_fields=extras,
        missing_fields=missing,
        summary=summary,
    )

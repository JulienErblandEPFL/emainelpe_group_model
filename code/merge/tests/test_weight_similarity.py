"""Tests for ``scripts/weight_similarity.py``.

The metric functions (:func:`mar_between`, :func:`cos_between`) need torch to
build tensors, so those tests ``pytest.importorskip("torch")`` and skip on a
torch-free laptop. The argparse parsing, summary, and rendering helpers are
pure Python and run everywhere.

The script imports stdlib-only at module top (torch/safetensors are lazy), so
it loads fine for inspection even without torch installed.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Load the CLI script by file path (it lives in scripts/, not an installed pkg)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "weight_similarity.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("weight_similarity_under_test", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ws = _load_module()


# ---------------------------------------------------------------------------
# Metric functions (require torch)
# ---------------------------------------------------------------------------

def test_mar_identical_is_zero() -> None:
    torch = pytest.importorskip("torch")
    t = torch.tensor([1.0, -2.0, 3.5, 0.25])
    assert ws.mar_between(t, t) == pytest.approx(0.0, abs=1e-7)


def test_mar_commutes() -> None:
    torch = pytest.importorskip("torch")
    a = torch.tensor([1.0, 2.0, -3.0, 4.0])
    b = torch.tensor([1.5, -2.0, 3.0, 4.2])
    assert ws.mar_between(a, b) == pytest.approx(ws.mar_between(b, a), abs=1e-7)


def test_mar_opposite_sign_is_one() -> None:
    """Opposite-sign tensors saturate the metric at the [0,1] ceiling (=100%)."""
    torch = pytest.importorskip("torch")
    a = torch.tensor([1.0, 2.0, 3.0])
    assert ws.mar_between(a, -a) == pytest.approx(1.0, abs=1e-6)


def test_mar_known_value() -> None:
    """A=[1,1,1], B=[1.1,1.1,1.1] → |0.1|/(1+1.1) = 0.1/2.1 ≈ 0.047619 (4.76%)."""
    torch = pytest.importorskip("torch")
    a = torch.tensor([1.0, 1.0, 1.0])
    b = torch.tensor([1.1, 1.1, 1.1])
    assert ws.mar_between(a, b) == pytest.approx(0.1 / 2.1, abs=1e-5)


def test_mar_casts_bfloat16() -> None:
    """bfloat16 inputs are cast to float32 and still produce the float result."""
    torch = pytest.importorskip("torch")
    a = torch.tensor([1.0, 1.0, 1.0], dtype=torch.bfloat16)
    b = torch.tensor([1.1, 1.1, 1.1], dtype=torch.bfloat16)
    # bfloat16 has ~3 significant digits, so widen the tolerance.
    assert ws.mar_between(a, b) == pytest.approx(0.1 / 2.1, abs=2e-2)


def test_cos_identical_is_one() -> None:
    torch = pytest.importorskip("torch")
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert ws.cos_between(a, a) == pytest.approx(1.0, abs=1e-6)


def test_cos_opposite_is_minus_one() -> None:
    torch = pytest.importorskip("torch")
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert ws.cos_between(a, -a) == pytest.approx(-1.0, abs=1e-6)


def test_cos_matrix_shape_flattened() -> None:
    """cos_between flattens, so a 2D tensor compared to itself is still 1.0."""
    torch = pytest.importorskip("torch")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert ws.cos_between(a, a) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# parse_model_arg (torch-free)
# ---------------------------------------------------------------------------

def test_parse_model_arg_relative() -> None:
    name, path = ws.parse_model_arg("uniform:bakeoff/uniform/merged")
    assert name == "uniform"
    assert path == Path("bakeoff/uniform/merged")


def test_parse_model_arg_absolute_path() -> None:
    """Splits on the first ':' so absolute POSIX paths survive intact."""
    name, path = ws.parse_model_arg("ties:/scratch/Group/bakeoff/ties/merged")
    assert name == "ties"
    assert path == Path("/scratch/Group/bakeoff/ties/merged")


@pytest.mark.parametrize("bad", ["", "noseparator", ":onlypath", "onlyname:"])
def test_parse_model_arg_rejects_malformed(bad: str) -> None:
    with pytest.raises(ws.WeightSimilarityError):
        ws.parse_model_arg(bad)


# ---------------------------------------------------------------------------
# summarize (torch-free — operates on plain dicts)
# ---------------------------------------------------------------------------

def test_summarize_min_max_mean() -> None:
    pairwise = {
        "k1": [{"a": "x", "b": "y", "mar_pct": 1.0}, {"a": "x", "b": "z", "mar_pct": 3.0}],
        "k2": [{"a": "x", "b": "y", "mar_pct": 2.0}],
    }
    s = ws.summarize(pairwise, "mar")
    assert s["min_pct"] == pytest.approx(1.0)
    assert s["max_pct"] == pytest.approx(3.0)
    assert s["mean_pct"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# JSON + matrix rendering (torch-free)
# ---------------------------------------------------------------------------

def _fake_inputs():
    models = [("uniform", Path("/m/uniform")), ("ties", Path("/m/ties")),
              ("dare", Path("/m/dare"))]
    sizes = {"uniform": 3.30, "ties": 3.30, "dare": 3.30}
    keys = ("model.layers.0.mlp.down_proj.weight",)
    pairwise = {
        keys[0]: [
            {"a": "uniform", "b": "ties", "mar_pct": 1.03},
            {"a": "uniform", "b": "dare", "mar_pct": 0.81},
            {"a": "ties", "b": "dare", "mar_pct": 0.92},
        ]
    }
    return models, sizes, keys, pairwise


def test_build_json_single_metric_shape() -> None:
    models, sizes, keys, pairwise = _fake_inputs()
    obj = ws.build_json(models, sizes, keys, "mar", ("mar",), pairwise)
    assert [m["name"] for m in obj["models"]] == ["uniform", "ties", "dare"]
    assert obj["models"][0]["size_gb"] == 3.30
    assert obj["metric"] == "mar"
    assert obj["tensors_inspected"] == list(keys)
    assert obj["summary"]["max_pct"] == pytest.approx(1.03)
    assert obj["summary"]["min_pct"] == pytest.approx(0.81)


def test_build_json_both_metrics_nested_summary() -> None:
    models, sizes, keys, _ = _fake_inputs()
    pairwise = {
        keys[0]: [
            {"a": "uniform", "b": "ties", "mar_pct": 1.0, "cos_pct": 99.0},
            {"a": "uniform", "b": "dare", "mar_pct": 2.0, "cos_pct": 98.0},
            {"a": "ties", "b": "dare", "mar_pct": 3.0, "cos_pct": 97.0},
        ]
    }
    obj = ws.build_json(models, sizes, keys, "both", ("mar", "cos"), pairwise)
    assert set(obj["summary"].keys()) == {"mar", "cos"}
    assert obj["summary"]["mar"]["max_pct"] == pytest.approx(3.0)
    assert obj["summary"]["cos"]["min_pct"] == pytest.approx(97.0)


def test_render_report_contains_legend_and_summary() -> None:
    models, sizes, keys, pairwise = _fake_inputs()
    report = ws.render_report(models, sizes, keys, ("mar",), pairwise)
    assert "[0] uniform" in report
    assert "[1] ties" in report
    assert "model.layers.0.mlp.down_proj.weight" in report
    assert "Summary (MAR)" in report
    # The triangular matrix carries the 1.03 uniform/ties pairing.
    assert "1.03" in report

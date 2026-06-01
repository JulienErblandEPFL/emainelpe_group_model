"""Tests for ``merge.qwen3_forward.make_qwen3_forward``.

Most of the module exercises real Qwen3-1.7B loading on GPU and is
cluster-only. This file pins ONE laptop-runnable invariant: cleanup()
truly drops the last reference to the loaded model (a weakref to it
becomes dead). Regression for the 2026-05-26 ``freed 0.00 GB`` bug —
a previous version returned the raw ``model`` as a third tuple element,
so callers could (and did) keep it alive past cleanup, defeating
``torch.cuda.empty_cache()``.

To stay laptop-runnable we monkey-patch ``AutoModelForCausalLM.from_pretrained``
to return a tiny ``nn.Module`` with one matching ``model.layers.0.*_proj``
Linear — enough to satisfy the layer-map check.
"""
from __future__ import annotations

import gc
import weakref

import pytest


def _make_stub_model():
    """Build a minimal nn.Module that satisfies ``_build_layer_name_to_canonical``.

    Needs at least one Linear at a path matching
    ``model.layers.<int>.(self_attn|mlp).(q|k|v|o|gate|up|down)_proj``.
    """
    import torch
    from torch import nn

    class SelfAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = SelfAttn()

    class Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    class Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Inner()
            self.dtype = torch.float32

        def to(self, *args, **kwargs):  # noqa: D401
            return self

        def train(self, mode: bool = True):  # noqa: D401
            return self

    return Root()


def test_make_qwen3_forward_cleanup_drops_last_reference(monkeypatch) -> None:
    """``cleanup()`` must null the only strong reference to the model.

    A weakref taken before cleanup must be dead after cleanup + gc.
    This is the invariant that broke when ``make_qwen3_forward`` returned
    the raw model as a third tuple element.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import merge.qwen3_forward as qf

    stub = _make_stub_model()

    def fake_from_pretrained(repo, **kwargs):
        return stub

    # Replace transformers.AutoModelForCausalLM.from_pretrained at import
    # time inside the function. The function imports lazily so we patch
    # the source module.
    import transformers
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        fake_from_pretrained,
    )

    result = qf.make_qwen3_forward(
        base_model_repo="dummy",
        device="cpu",
    )

    # Acceptance criterion 1: return is a 2-tuple, NOT a 3-tuple. No raw
    # model leaks to callers.
    assert len(result) == 2, (
        f"make_qwen3_forward must return (forward_fn, cleanup); got {len(result)}-tuple"
    )
    forward_fn, cleanup = result

    # Track the model via weakref and drop our local strong reference.
    ref = weakref.ref(stub)
    del stub

    # The closure box should be the only remaining strong reference.
    cleanup()
    gc.collect()

    assert ref() is None, (
        "cleanup() did not drop the last reference to the model — a caller "
        "must be holding it alive (e.g. via a returned raw-model handle). "
        "This is the 2026-05-26 'freed 0.00 GB' bug."
    )


def test_make_qwen3_forward_cleanup_is_idempotent(monkeypatch) -> None:
    """Calling cleanup() twice must not raise."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import merge.qwen3_forward as qf
    import transformers

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda repo, **kwargs: _make_stub_model(),
    )

    _, cleanup = qf.make_qwen3_forward(base_model_repo="dummy", device="cpu")
    cleanup()
    cleanup()  # must be a no-op, not an error

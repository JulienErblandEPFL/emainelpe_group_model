"""
Tests for ``merge.load_adapter`` — Stage 2 adapter loading.

These tests need torch and safetensors. ``pytest.importorskip`` at the top
of each function lets the laptop skip cleanly while the cluster runs them.

The cross-cutting fixture ``synthetic_adapters_dir`` (defined in
conftest.py) generates 4 toy adapters with distinct seeds. Tests that need
to mutate one of those adapters do so on a copy.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# canonicalize() — torch-free (regex only)
# ---------------------------------------------------------------------------

def test_canonicalize_lora_A_default_weight() -> None:
    from merge.load_adapter import canonicalize

    assert canonicalize(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    ) == "model.layers.0.self_attn.q_proj"


def test_canonicalize_lora_B_default_weight() -> None:
    from merge.load_adapter import canonicalize

    assert canonicalize(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
    ) == "model.layers.0.self_attn.q_proj"


def test_canonicalize_lora_A_no_default() -> None:
    """Older PEFT versions omit the ``.default`` adapter-name segment."""
    from merge.load_adapter import canonicalize

    assert canonicalize(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    ) == "model.layers.0.self_attn.q_proj"


def test_canonicalize_mlp_modules() -> None:
    from merge.load_adapter import canonicalize

    for module in ("gate_proj", "up_proj", "down_proj"):
        name = (
            f"base_model.model.model.layers.5.mlp.{module}.lora_A.default.weight"
        )
        assert canonicalize(name) == f"model.layers.5.mlp.{module}"


def test_canonicalize_unexpected_name_raises() -> None:
    from merge.load_adapter import canonicalize

    with pytest.raises(ValueError):
        canonicalize("totally unrelated parameter name")

    # Has prefix but no lora_A/B factor suffix
    with pytest.raises(ValueError):
        canonicalize("base_model.model.model.layers.0.self_attn.q_proj.weight")


def test_canonicalize_decanonicalize_round_trip() -> None:
    """canonicalize is the left inverse of decanonicalize for any factor."""
    from merge.load_adapter import canonicalize, decanonicalize

    canonicals = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.27.mlp.gate_proj",
        "model.layers.5.self_attn.o_proj",
        "model.layers.12.mlp.down_proj",
    ]
    for c in canonicals:
        for factor in ("lora_A", "lora_B"):
            full = decanonicalize(c, factor)
            assert canonicalize(full) == c, f"round-trip failed for {c!r}, factor {factor!r}"


def test_decanonicalize_rejects_bad_factor() -> None:
    from merge.load_adapter import decanonicalize

    with pytest.raises(ValueError, match=r"factor must be"):
        decanonicalize("model.layers.0.self_attn.q_proj", "lora_C")


# ---------------------------------------------------------------------------
# load() — torch-required
# ---------------------------------------------------------------------------

def test_load_toy_adapter_returns_expected_keys(synthetic_adapters_dir: Path) -> None:
    pytest.importorskip("torch")
    from merge.load_adapter import load

    tv = load(synthetic_adapters_dir / "math")
    # 2 layers × 7 target modules = 14 ΔW tensors
    assert len(tv) == 14

    # Spot-check a few canonical keys
    assert "model.layers.0.self_attn.q_proj" in tv
    assert "model.layers.1.mlp.down_proj" in tv


def test_load_toy_adapter_returns_bf16_tensors(synthetic_adapters_dir: Path) -> None:
    torch = pytest.importorskip("torch")
    from merge.load_adapter import load

    tv = load(synthetic_adapters_dir / "math")
    for name, tensor in tv.items():
        assert tensor.dtype == torch.bfloat16, f"{name} has dtype {tensor.dtype}"


def test_load_toy_adapter_computes_BA_product(tmp_path: Path, lora_yaml_path: Path) -> None:
    """Materialized ΔW must equal (α/r) · B @ A within bf16 tolerance."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    from merge.load_adapter import load
    from merge.tests.fixtures.toy_adapter import make_toy_adapter
    from merge.verify_spec import load_locked_spec

    spec = load_locked_spec(lora_yaml_path)
    out_dir = tmp_path / "toy"
    make_toy_adapter(out_dir, spec, seed=42, n_layers=2, hidden_dim=64, intermediate_dim=128)

    tv = load(out_dir)

    # Re-derive ΔW directly from the saved safetensors and compare.
    from safetensors.torch import load_file
    raw = load_file(str(out_dir / "adapter_model.safetensors"), device="cpu")
    a = raw["base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"]
    b = raw["base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"]
    scaling = spec["lora_alpha"] / spec["r"]
    expected = (scaling * (b @ a)).to(a.dtype)

    got = tv["model.layers.0.self_attn.q_proj"]
    assert got.shape == expected.shape
    # bf16 has ~3 decimal digits of precision; 1e-2 is generous, 5e-3 typical.
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)


def test_load_toy_adapter_reproducible(tmp_path: Path, lora_yaml_path: Path) -> None:
    """Same seed → bit-identical adapter weights."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    from merge.load_adapter import load
    from merge.tests.fixtures.toy_adapter import make_toy_adapter
    from merge.verify_spec import load_locked_spec

    spec = load_locked_spec(lora_yaml_path)
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    make_toy_adapter(a_dir, spec, seed=7)
    make_toy_adapter(b_dir, spec, seed=7)

    tv_a = load(a_dir)
    tv_b = load(b_dir)
    assert set(tv_a.keys()) == set(tv_b.keys())
    for key in tv_a:
        assert torch.equal(tv_a[key], tv_b[key]), f"{key} differs across seed-7 generations"


def test_load_raises_on_missing_config(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.load_adapter import load

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / "adapter_model.safetensors").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        load(empty_dir)


def test_load_raises_on_missing_safetensors(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from merge.load_adapter import load

    half_dir = tmp_path / "half"
    half_dir.mkdir()
    (half_dir / "adapter_config.json").write_text(
        json.dumps({"r": 32, "lora_alpha": 64})
    )
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        load(half_dir)


def test_load_raises_on_unpaired_lora(tmp_path: Path, lora_yaml_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    from merge.load_adapter import load
    from merge.tests.fixtures.toy_adapter import make_toy_adapter
    from merge.verify_spec import load_locked_spec
    from safetensors.torch import load_file, save_file

    spec = load_locked_spec(lora_yaml_path)
    out_dir = tmp_path / "unpaired"
    make_toy_adapter(out_dir, spec, seed=0)

    # Drop one lora_B factor to break a single pair.
    state = load_file(str(out_dir / "adapter_model.safetensors"), device="cpu")
    dropped = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
    del state[dropped]
    save_file(state, str(out_dir / "adapter_model.safetensors"))

    with pytest.raises(ValueError, match="incomplete LoRA pair"):
        load(out_dir)


# ---------------------------------------------------------------------------
# load_all()
# ---------------------------------------------------------------------------

def test_load_all_succeeds_with_4_toy_adapters(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    pytest.importorskip("torch")
    from merge.load_adapter import CANONICAL_DOMAINS, load_all
    from merge.verify_spec import load_locked_spec

    spec = load_locked_spec(lora_yaml_path)
    tvs = load_all(synthetic_adapters_dir, spec)

    assert list(tvs.keys()) == list(CANONICAL_DOMAINS)
    for domain, tv in tvs.items():
        assert tv, f"{domain} task vector is empty"


def test_load_all_raises_on_missing_domain(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    pytest.importorskip("torch")
    from merge.load_adapter import load_all
    from merge.verify_spec import load_locked_spec

    spec = load_locked_spec(lora_yaml_path)
    shutil.rmtree(synthetic_adapters_dir / "safety")

    with pytest.raises(FileNotFoundError, match="safety"):
        load_all(synthetic_adapters_dir, spec)


def test_load_all_raises_on_extra_subdir(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    pytest.importorskip("torch")
    from merge.load_adapter import load_all
    from merge.verify_spec import load_locked_spec

    spec = load_locked_spec(lora_yaml_path)
    (synthetic_adapters_dir / "mystery").mkdir()

    with pytest.raises(ValueError, match="mystery"):
        load_all(synthetic_adapters_dir, spec)


def test_load_all_raises_spec_mismatch_when_adapter_diverges(
    synthetic_adapters_dir: Path, lora_yaml_path: Path
) -> None:
    pytest.importorskip("torch")
    from merge.load_adapter import load_all
    from merge.verify_spec import SpecMismatchError, load_locked_spec

    spec = load_locked_spec(lora_yaml_path)
    # Corrupt safety's config so it claims r=16
    cfg_path = synthetic_adapters_dir / "safety" / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["r"] = 16
    cfg_path.write_text(json.dumps(cfg))

    with pytest.raises(SpecMismatchError) as excinfo:
        load_all(synthetic_adapters_dir, spec)

    msg = str(excinfo.value)
    assert "safety" in msg
    assert "r" in msg

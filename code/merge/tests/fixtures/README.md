# `merge/tests/fixtures/`

CPU-only synthetic fixtures for the merge-pipeline tests.

This directory exists so the end-to-end merge test (`test_pipeline_synthetic.py`,
introduced in Stage 4) can run on a laptop in seconds without touching the HF
Hub or a real Qwen3-1.7B checkpoint.

## What lives here

Synthetic LoRA adapter fixtures — tiny (e.g. 2-block × 7-projection) tensor
dicts whose key set matches what a real locked-spec adapter would produce
(`lora_A` / `lora_B` factors for the seven target modules in `lora.yaml`).

The Stage 4 plan is to generate these on-the-fly via a `conftest.py` fixture
(`synthetic_task_vectors`) and only cache them here if generation becomes
expensive. The decision (regenerate vs. cache) will be made in Stage 4 when
the test is written.

## What does NOT live here

- Real adapter weights (those live on HF Hub).
- Large binary files (anything > a few KB).
- Anything sensitive (`HF_TOKEN`, eval gold answers, etc.).

If you find yourself about to commit a `.safetensors` file here, stop and
reconsider — the test should generate its tensors at runtime instead.

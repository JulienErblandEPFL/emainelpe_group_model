# `merge/` — Group-Model Adapter Merging

This subpackage takes the four locked-spec LoRA adapters
(`cs-552-2026-emainelpe/{math,general_knowledge,multilingual,safety}_model`)
and composes them into a single group adapter that gets pushed to
`cs-552-2026-emainelpe/group_model` for the CS-552 leaderboard.

**What this is not:** training code, teammate-facing infrastructure, or a
duplicate spec store. It is Julien's responsibility for team Émainèlpé (g65)
and runs in isolation from the locked contract files at the repo root.

## Dependency on the locked-spec files

`merge/` is the **consumer** of the locked-spec contract; it never duplicates
or shadows values from the contract files:

| File at repo root | How `merge/` uses it |
|---|---|
| `../lora.yaml` | `verify_spec` reads it to validate every input adapter. Never duplicate `r`, `alpha`, `target_modules` here. |
| `../chat_template.jinja` | `infer` passes it explicitly to the tokenizer so local generation matches CI. Never inline a chat template here. |
| `../evaluate/` | `eval_all` calls the CI-equivalent scorer. Never re-implement `\boxed{}` parsing or pass@k here. |
| `../validation_samples/` | `infer` reads prompts; `eval_all` scores completions. 40-problem snapshot only — not the eval set. |

If a value in this subdir starts to drift from a contract file, the contract
file wins. The skeleton test in `tests/test_skeleton.py` enforces this for
`lora.yaml`.

## HF Hub repos

| Repo | Role |
|---|---|
| `cs-552-2026-emainelpe/math_model` | input — math specialist |
| `cs-552-2026-emainelpe/general_knowledge_model` | input — knowledge specialist |
| `cs-552-2026-emainelpe/multilingual_model` | input — multilingual specialist |
| `cs-552-2026-emainelpe/safety_model` | input — safety specialist |
| `cs-552-2026-emainelpe/group_model` | **output** — merged group adapter |

The base model is fixed at `Qwen/Qwen3-1.7B`. All five repos use the locked
LoRA shape (`r=32`, `α=64`, 7 target projections, `modules_to_save=null`).

## Pipeline

```
verify_spec  →  load_adapter  →  methods.<method>  →  pipeline.save  →  publish  →  eval_all
   (CPU)         (HF download)    (CPU tensor math)    (CPU IO)        (HF upload)  (CPU IO)
```

1. **verify_spec** — diff each input adapter's `adapter_config.json` against
   `../lora.yaml`. Whitelist-based on 8 load-bearing fields; PEFT bookkeeping
   is ignored so version drift between teammates does not false-positive.
   Fail fast if any adapter drifted from the locked spec.
2. **load_adapter** — read safetensors weights from a local PEFT-format
   directory (`adapter_config.json` + `adapter_model.safetensors`), verify
   each adapter against the locked spec, canonicalize the key namespace,
   and materialize `ΔW = (α/r) · B @ A` as the task vector. The pipeline
   does not download from HF Hub — whoever runs it is responsible for
   getting the four adapters onto disk first.
3. **methods.\<method\>** — dispatch through `METHOD_REGISTRY` and run the
   chosen merge in pure tensor math.
4. **pipeline.save** — write the merged dict as a PEFT-loadable adapter
   directory (`adapter_config.json` + `adapter_model.safetensors`).
5. **publish** — upload to `cs-552-2026-emainelpe/group_model` via
   `huggingface_hub.HfApi.upload_folder`.
6. **eval_all** — generate completions against `../validation_samples/*.jsonl`
   and score with `../evaluate/`. Compute the 4-domain leaderboard average.

## Method registry

`merge/methods/__init__.py` exposes a single `METHOD_REGISTRY: dict[str, callable]`.
The five user-facing methods, with implementation stage:

| Method name | Implementation | Stage |
|---|---|---|
| `uniform` | `methods.uniform.uniform_merge` | Stage 3 |
| `dare_uniform` | `methods.__init__.dare_uniform` (composes `dare` + `uniform_merge`) | Stage 3 |
| `dare_weighted` | `methods.__init__.dare_weighted` (composes `dare` + `weighted_linear_merge`) | Stage 3 |
| `ties` | `methods.ties.ties_merge` | Stage 4 |
| `adamerging` | `methods.adamerging.adamerging` | **Stage 7 (post-milestone)** |

## How to run the synthetic end-to-end test

The Stage-4 end-to-end test will live at `merge/tests/test_pipeline_synthetic.py`
(does not exist yet). It will use the `synthetic_adapters_dir` fixture from
`conftest.py` (which generates 4 toy PEFT adapters on disk via
`tests/fixtures/toy_adapter.py`) to run the full merge → save round-trip on
CPU in seconds, without touching HF or vLLM. Until then:

```bash
# Skeleton smoke tests only — confirms imports + stubs raise NotImplementedError.
pytest merge/tests/ -v
```

The `test_locked_spec_unchanged` test in the suite is the regression baseline:
if it ever fails, someone edited `../lora.yaml` in a way that may invalidate
the merge guarantees.

## Implementation status

Update this table as stages land. Single source of truth.

| Module | Function(s) | Stage | Status |
|---|---|---|---|
| `verify_spec.py` | `load_locked_spec`, `verify`, `SpecMismatchError` | 2 | **done** |
| `load_adapter.py` | `canonicalize`, `load`, `load_all` | 2 | **done** |
| `methods/dare.py` | `dare` | 3 | **done** |
| `methods/uniform.py` | `uniform_merge` | 3 | **done** |
| `methods/weighted_linear.py` | `weighted_linear_merge` | 3 | **done** |
| `methods/__init__.py` | `dare_uniform`, `dare_weighted` (compositions) | 3 | **done** |
| `methods/ties.py` | `ties_merge` | 4 | skeleton |
| `pipeline.py` | `merge_adapters` | 4 | skeleton |
| `tests/test_pipeline_synthetic.py` | end-to-end CPU test | 4 | not created |
| `infer.py` | `generate_completions`, `generate_for_validation_set` | 5 | skeleton |
| `publish.py` | `publish_adapter` | 5 | skeleton |
| `eval_all.py` | `evaluate_completions`, `four_domain_average` | 5 | skeleton |
| `methods/adamerging.py` | `adamerging` | 7 (post-milestone) | skeleton |

## Out of scope for `merge/`

- Training the four specialist adapters (each teammate owns their own repo).
- Editing the locked-spec contract files at the repo root.
- Modifying `../evaluate/`, `../validation_samples/`, `../docker/`, or any
  Classroom skeleton file.

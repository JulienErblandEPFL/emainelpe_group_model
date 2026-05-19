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
The six user-facing methods, with implementation stage:

| Method name | Implementation | Stage |
|---|---|---|
| `uniform` | `methods.uniform.uniform_merge` | Stage 3 |
| `dare_uniform` | `methods.__init__.dare_uniform` (composes `dare` + `uniform_merge`) | Stage 3 |
| `dare_weighted` | `methods.__init__.dare_weighted` (composes `dare` + `weighted_linear_merge`) | Stage 3 |
| `ties` | `methods.ties.ties_merge` | Stage 4 |
| `adamerging` | `methods.adamerging.adamerging` | Stage 5a |
| `dare_adamerging` | `methods.__init__.dare_adamerging` (composes `dare` + `adamerging`) | Stage 5a |

> `adamerging` and `dare_adamerging` require a `forward_fn` callable and a
> `data_iter` iterable in `method_kwargs`. See `merge/methods/adamerging.py`
> for the contract. Stage 5b will build the real-Qwen3 helpers (PEFT-hook
> forward + unlabeled-data sampler).

## Dependencies

Dependencies for the merge subdir are listed in `requirements.txt` at the
repo root. The cluster docker image has them preinstalled; teammates running
locally can `pip install -r requirements.txt` (use the PyTorch CPU index for
laptop installs — see the comment at the top of that file).

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
| `load_adapter.py` | `canonicalize`, `decanonicalize`, `load`, `load_all` | 2 + 4 | **done** |
| `methods/dare.py` | `dare` | 3 | **done** |
| `methods/uniform.py` | `uniform_merge` | 3 | **done** |
| `methods/weighted_linear.py` | `weighted_linear_merge` | 3 | **done** |
| `methods/__init__.py` | `dare_uniform`, `dare_weighted` (compositions) | 3 | **done** |
| `methods/ties.py` | `ties_merge` | 4 | **done** |
| `pipeline.py` | `merge_adapters`, `svd_factor` | 4 | **done** |
| `tests/test_pipeline_synthetic.py` | end-to-end CPU test | 4 | **done** |
| `methods/adamerging.py` | `adamerging` (+ `AdaMergingResult`) | 5a | **done** |
| `methods/__init__.py` | `dare_adamerging` (composition) | 5a | **done** |
| `tests/test_adamerging.py` | layer-wise AdaMerging on toy fixtures | 5a | **done** |
| `data/unlabeled.py` | `UNLABELED_DATASETS`, `assert_cache_exists`, `make_unlabeled_iter` | 5b | **done** |
| `qwen3_forward.py` | `make_qwen3_forward` (post-hook ΔW patching) | 5b | **done** |
| `tests/fixtures/qwen3_adapter.py` | `make_random_qwen3_adapter` (cluster-only) | 5b | **done** |
| `tests/test_data_unlabeled.py` | CPU unit tests for dataset config + cache check | 5b | **done** |
| `../scripts/fetch_adamerging_data.py` | pre-download the 4 unlabeled datasets | 5b | **done** |
| `../scripts/smoke_adamerging.py` | cluster smoke: random Qwen3 adapters → dare_adamerging | 5b | **done** |
| `infer.py` | `run_inference`, `load_validation_jsonl`, `InferenceConfig` | 5c.1 | **done** |
| `eval_all.py` | `evaluate_all_benchmarks`, `evaluate_one_benchmark`, `classify_completion`, `classify_problem_failure` | 5c.1 | **done** |
| `tests/test_failure_classification.py` | failure taxonomy unit tests | 5c.1 | **done** |
| `tests/test_eval_io.py` | JSONL parsing + dataclass IO tests | 5c.1 | **done** |
| `publish.py` | `publish_adapter` | 5d | skeleton |

## Stage 5b: Real-Qwen3 plumbing for AdaMerging

The AdaMerging core (Stage 5a, `methods/adamerging.py`) is base-model-agnostic
— it takes a caller-supplied `forward_fn` and `data_iter`. Stage 5b builds
those callables for the real Qwen3-1.7B base model and real in-domain
unlabeled prompts.

- **`merge/data/unlabeled.py`** — 4 unlabeled in-domain datasets
  (GSM8K, MMLU `auxiliary_train`, MGSM `en`, XSTest `safe_*`) wrapped in
  a round-robin `(domain_idx, batch)` iterator. Tokenizes via the Qwen3
  chat template with `add_generation_prompt=True`.
- **`merge/qwen3_forward.py`** — loads Qwen3-1.7B once, registers a
  forward post-hook on each LoRA-targetable Linear that adds
  `F.linear(input, merged[canonical], bias=None)` to the layer's output.
  Differentiable w.r.t. the AdaMerging coefficients without mutating
  base-model weights.
- **`scripts/fetch_adamerging_data.py`** — runnable pre-download. Run once
  per environment before training.
- **`scripts/smoke_adamerging.py`** — runnable cluster smoke: generates
  4 random-init Qwen3-sized LoRA adapters and pushes them through
  `dare_adamerging` for ~50 steps.

Cluster workflow:

```bash
python scripts/fetch_adamerging_data.py
python scripts/smoke_adamerging.py --max-steps 50
```

## Stage 5c.1: Inference + multi-benchmark eval

- **`merge/infer.py`** — vLLM-based n=8 inference for one benchmark.
  Renders prompts through the Qwen3 chat template
  (`add_generation_prompt=True`), calls `vllm.LLM.generate(...)` with the
  merged adapter passed as a `LoRARequest`, and writes a generations JSONL
  shaped exactly like the input `evaluate.score` consumes.
- **`merge/eval_all.py`** — orchestrator: loads vLLM once
  (`enable_lora=True, max_lora_rank=32`), runs all 4 benchmarks, scores
  via the existing `evaluate/*` helpers (pass@1, pass@8), and classifies
  every pass@8=0 failure into one of 7 categories.
- **Failure taxonomy** (`merge.eval_all.FailureCategory`):
  `no_boxed`, `empty_boxed`, `wrong_answer`, `malformed_answer`,
  `truncated`, `refusal`, `mixed`. Per-problem detail (with the 8
  completions inline) is saved to `failures_<benchmark>.json`.

Output layout for one evaluation run:

```
<output_dir>/
    scorecard.json
    generations_{math,general_knowledge,safety,multilingual}.jsonl
    failures_{math,general_knowledge,safety,multilingual}.json
```

Workflow:

```python
from pathlib import Path
from merge.eval_all import evaluate_all_benchmarks

results = evaluate_all_benchmarks(
    merged_adapter_dir=Path("outputs/merged_v1/"),
    base_model_repo="Qwen/Qwen3-1.7B",
    output_dir=Path("outputs/eval_v1/"),
    validation_samples_dir=Path("validation_samples/"),
)
```

**Sampling params resolution.** When `config=None` is passed to
`evaluate_all_benchmarks`, sampling params are resolved through a
hierarchical fallback (highest priority first):

1. Explicit `config: InferenceConfig` argument.
2. `<merged_adapter_dir>/generation_config.json` (written by
   `pipeline.merge_adapters` when a `generation_config` dict is passed in).
3. `<repo_root>/generation_config.json` (team-wide per-run override; absent
   for now — falls through).
4. Qwen3 defaults (`temperature=0.7`, `top_p=0.8`, `top_k=20`,
   `max_new_tokens=16384`), built via
   `merge.generation_config.make_generation_config()`.

This ensures eval measures what CI would measure for any merged model
shipping its own `generation_config.json`. The bake-off pipeline writes
that file per merge config via `pipeline.merge_adapters(..., generation_config=...)`.

## Out of scope for `merge/`

- Training the four specialist adapters (each teammate owns their own repo).
- Editing the locked-spec contract files at the repo root.
- Modifying `../evaluate/`, `../validation_samples/`, `../docker/`, or any
  Classroom skeleton file.

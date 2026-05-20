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
verify_spec  →  load_adapter  →  methods.<method>  →  pipeline.bake  →  publish  →  eval_all
   (CPU)         (load on GPU)    (GPU tensor math)    (GPU + save)    (HF upload)  (vLLM)
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
4. **pipeline.bake** — SVD-truncate the merged ΔW back to rank-r, inject
   the factors into a fresh PEFT wrapper around the base model, run
   `peft.merge_and_unload` to bake the deltas into the base weights, and
   save the resulting full model via `transformers.save_pretrained`. The
   on-disk artifact is a self-contained HF-format directory (`config.json`
   + `model.safetensors`), not a LoRA adapter. (See Day 7 in PROCESS_BOOK
   for the rationale — vLLM's LoRA loader rejected our PEFT-format keys.)
5. **publish** — upload to `cs-552-2026-emainelpe/group_model` via
   `huggingface_hub.HfApi.upload_folder`.
6. **eval_all** — generate completions against `../validation_samples/*.jsonl`
   via vLLM loaded directly on the merged model dir (no LoRA adapter
   plumbing), and score with `../evaluate/`. Compute the 4-domain
   leaderboard average.

### Output format (Day 7 refactor)

`pipeline.merge_adapters` writes a full HF-format model directory:

```
<output_dir>/
    config.json
    model.safetensors          # ~3.4 GB for Qwen3-1.7B in bf16, sharded if larger
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    chat_template.jinja        # copied from <repo>/chat_template.jinja
    generation_config.json     # written iff generation_config kwarg provided
```

The intermediate LoRA factorization stays in GPU memory and never hits
disk. Downstream consumers (vLLM, the CI grader, `publish.py`) load this
directly via `AutoModelForCausalLM.from_pretrained` or
`LLM(model=<output_dir>)` — no LoRA bookkeeping required.

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
| `load_adapter.py` | `device=` kwarg on `load` / `load_all` (GPU placement) | 5c.1.5 | **done** |
| `pipeline.py` | `device=` kwarg + auto-cuda + CPU-at-save | 5c.1.5 | **done** |
| `../scripts/eval_sweep.py` | one-merge, multi-temperature evaluation CLI | 5c.1.5 | **done** |
| `tests/test_eval_sweep.py` | argparse + sweep-resilience unit tests | 5c.1.5 | **done** |
| `../scripts/run_bakeoff.py` | 4 methods × 3 temperatures orchestrator | 5c.2 | **done** |
| `tests/test_run_bakeoff.py` | argparse + resilience + winner-pick unit tests | 5c.2 | **done** |
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
  (`add_generation_prompt=True`), calls `vllm.LLM.generate(prompts,
  sampling_params=...)` against a vLLM model loaded directly on the
  merged-model directory, and writes a generations JSONL shaped exactly
  like the input `evaluate.score` consumes. (Day 7: the `LoRARequest`
  plumbing was dropped — vLLM consumes the merged model as a full HF
  checkpoint now.)
- **`merge/eval_all.py`** — orchestrator: loads vLLM once on the merged
  model dir (`LLM(model=merged_adapter_dir, dtype="bfloat16")`), runs
  all 4 benchmarks, scores via the existing `evaluate/*` helpers
  (pass@1, pass@8), and classifies every pass@8=0 failure into one of
  7 categories.
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

## Stage 5c.1.5: Performance + eval-time temperature sweep

### GPU-accelerated adapter loading

`load_adapter.load` and `load_adapter.load_all` take an opt-in `device`
kwarg (default `"cpu"`). When set to `"cuda"`, safetensors places weights
directly on GPU and the per-module `(α/r) · B @ A` matmul runs on GPU as
well. On Qwen3-1.7B adapters this drops the load+materialize time from
~10 minutes (4 adapters on CPU bf16) to seconds.

`pipeline.merge_adapters` auto-selects the device when its `device`
argument is `None`: `cuda` when `torch.cuda.is_available()`, otherwise
`cpu`. The final safetensors save always materializes a contiguous CPU
copy of every factor — the merged adapter on disk is identical regardless
of where the merge ran. CPU and GPU paths produce equivalent reconstructed
`ΔW` up to bf16 tolerance; raw `lora_A` / `lora_B` factors may differ due
to SVD sign ambiguity, but the reconstruction is invariant.

### Eval-time temperature sweep

Temperature is a sampling-only parameter — it has no effect on the merged
weights. `scripts/eval_sweep.py` exploits this by merging once and
evaluating at multiple temperatures in a single run.

```bash
python -u scripts/eval_sweep.py \
    --merged-adapter-dir outputs/merged_v1/ \
    --output-dir outputs/sweep_v1/ \
    --temperatures 0.3 0.5 0.7
```

Output layout:

```
<output_dir>/
    T_0.3/        # full scorecard + generations + failures per benchmark
    T_0.5/
    T_0.7/
    sweep_results.json    # aggregated comparison, written incrementally
```

A vLLM crash or OOM on one temperature is recorded to `sweep_results.json`
as a failed row and the next temperature still runs. Exit codes: `0` =
all temperatures succeeded, `1` = at least one failed, `2` = setup error.

**Temperature 0.0 is rejected** by `validate_args`. vLLM's `SamplingParams`
rejects `n>1` when `temperature=0.0` because greedy decoding is
deterministic; for a deterministic final HF push, build a custom
`InferenceConfig` with `n=1` directly rather than going through this sweep
script.

## Stage 5c.2: Full bake-off

`scripts/run_bakeoff.py` orchestrates the milestone-day comparison: 4
merge methods × 3 sampling temperatures on the same 4 input adapters, in
one run. Merge once per method, sweep temperatures on that merge — 4
merges + 12 evals total (not 12 merges).

```bash
nohup python -u scripts/run_bakeoff.py \
    --adapters-dir loras/ \
    --output-dir bakeoff_2026-05-21-1400/ \
    > bakeoff.log 2>&1 &
```

Methods swept (fixed list): `uniform`, `dare_uniform`, `dare_adamerging`,
`ties`. Temperatures (fixed list): `0.3, 0.5, 0.7`. Override either with
`--methods` / `--temperatures` for partial sweeps.

Output layout:

```
<output_dir>/
    bakeoff_results.json      # aggregated, written incrementally per method
    uniform/
        merged/               # config.json + model.safetensors + ...
        sweep/T_0.3/          # full scorecard + generations + failures
        sweep/T_0.5/
        sweep/T_0.7/
    dare_uniform/
        merged/
        sweep/T_0.3/ ...
    dare_adamerging/
        merged/
        sweep/T_0.3/ ...
    ties/
        merged/
        sweep/T_0.3/ ...
```

**Resilience.** A merge failure marks that method's 3 temperature slots
all-failed and moves to the next method. A single temperature OOM marks
only that slot failed; the other 2 temperatures continue. Partial
bake-off data is preserved by incremental writes after each method
completes (or fails).

**Hard-fail gates at startup.** Missing adapter dir, missing
`validation_samples/*.jsonl`, invalid CLI arg, or any adapter failing
locked-spec verification → exit code 2 without launching any GPU work.

**AdaMerging hyperparameters are NOT swept.** The bake-off uses a single
fixed config (drop_rate=0.5, lr=1e-2, lambda_l2=1e-4, max_steps=200,
early_stop_patience=100, batch_size=2). Hyperparameter tuning is a
separate experiment. `--adamerging-max-steps` overrides the step count
if needed.

**Winner selection.** End-of-run console summary prints a (method,
temperature) × benchmark grid plus a `Winner: <method> @ T=<temp>` line
identifying the highest average pass@8 across the 4 benchmarks.

## Out of scope for `merge/`

- Training the four specialist adapters (each teammate owns their own repo).
- Editing the locked-spec contract files at the repo root.
- Modifying `../evaluate/`, `../validation_samples/`, `../docker/`, or any
  Classroom skeleton file.

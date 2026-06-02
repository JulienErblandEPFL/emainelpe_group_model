# Group Model — CS-552 Émainèlpé (g65)

Code to reproduce the **group model** for team Émainèlpé. The group model is
built by merging the four specialist LoRA adapters (math, general knowledge,
safety, multilingual) — each trained by a different teammate on top of
`Qwen/Qwen3-1.7B` — into a single model. The deployed group model uses
**AdaMerging in its aggregated-objective mode on top of DARE-masked task
vectors** (`dare_adamerging --aggregate-domains`), sampled at **T=0.3**, and
is published to `cs-552-2026-emainelpe/group_model`.

> Scope: this directory reproduces the **group model only**. The four
> individual specialist models live in their owners' separate repos.

All randomness is seeded (`--seed 42` everywhere). The merge pipeline reads
adapters from a local directory — it never downloads from the Hub itself —
so the workflow is always: fetch adapters → fetch data → merge/evaluate →
publish.

---

## a) What this is

`merge/` is a self-contained merge pipeline; `scripts/` holds the runnable
entrypoints. The locked training contract (`lora.yaml`, `chat_template.jinja`)
and the CI-equivalent scorer (`evaluate/`, `validation_samples/`) live here
too because the pipeline consumes them. See `merge/README.md` for the
pipeline internals and `docs/` for team/cluster context.

## b) Installation

```bash
pip install -r requirements.txt
```

For a CPU-only laptop (tests only, no GPU work) use the PyTorch CPU wheels:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**RCP cluster gotcha (important).** The course Docker image ships
`bitsandbytes` 0.42, which only supports CUDA 11.x; PEFT crashes on import
when its CUDA setup fails on the cluster's CUDA 12.x. On a fresh pod, upgrade
it first — and on the cluster image you need `--break-system-packages`:

```bash
pip install -U bitsandbytes --break-system-packages
```

Re-run this after every pod restart (the image resets).

## c) Hugging Face authentication

Needed to fetch the source adapters and to publish the merged model:

```bash
export HF_TOKEN=hf_...
# or, interactively:
hf auth login
```

## d) Fetch the source adapters

```bash
python scripts/fetch_adapters.py --target-dir loras/
```

Downloads the four specialist adapters from
`cs-552-2026-emainelpe/{math,general_knowledge,safety,multilingual}_model`
(~140 MB each, ~560 MB total) into `loras/{domain}/`, then verifies each one
against `lora.yaml` (rank, alpha, target modules, …). The script exits
non-zero with a clear message if any adapter is missing or diverges from the
locked spec, so a bad input never reaches the GPU.

## e) Fetch the unlabeled datasets (for AdaMerging)

AdaMerging tunes its merge coefficients on unlabeled in-domain prompts. Pre-
download them once per environment:

```bash
python scripts/fetch_adamerging_data.py
```

This caches GSM8K, MMLU (`auxiliary_train`), MGSM (`en`), and XSTest (safe
prompts) into the HF datasets cache.

## f) Reproduce the deployed group model

The published model is `dare_adamerging` **aggregated** @ **T=0.3**:

```bash
# 1. Merge (DARE + aggregated-objective AdaMerging) and evaluate.
python scripts/run_bakeoff.py \
    --methods dare_adamerging --aggregate-domains \
    --adapters-dir loras/ \
    --output-dir bakeoff_final/ \
    --gpu-memory-utilization 0.6 \
    --seed 42

# 2. Publish the merged model dir (dry-run prints the plan; --confirm pushes
#    and rewrites generation_config.json so CI grades at T=0.3).
python scripts/publish.py \
    --model-dir bakeoff_final/dare_adamerging/merged \
    --repo-id cs-552-2026-emainelpe/group_model \
    --temperature 0.3 --top-p 0.8 --top-k 20 \
    --confirm
```

Run `publish.py` once without `--confirm` first to inspect the upload plan
and the `generation_config.json` it would write.

## g) Reproduce the 4-method comparison table

**Important — the report's comparison table is assembled from THREE separate
runs** (no single bake-off produced all rows; the DARE methods and the
aggregated AdaMerging mode landed on different days). The TA can reproduce
each run independently:

**Run 1 — uniform + ties** (historical dir `bakeoff_20260526_1145/`):

```bash
python scripts/run_bakeoff.py \
    --methods uniform ties \
    --adapters-dir loras/ \
    --output-dir bakeoff_uniform_ties/ \
    --gpu-memory-utilization 0.6 --seed 42
```

**Run 2 — dare_uniform + dare_adamerging (round-robin)** (historical dir
`bakeoff_dare_20260526_1246/`). Note: `dare_adamerging` here runs in the
**round-robin** objective, which is intentionally unstable — this run
illustrates the problem that motivated the aggregated mode:

```bash
python scripts/run_bakeoff.py \
    --methods dare_uniform dare_adamerging \
    --adapters-dir loras/ \
    --output-dir bakeoff_dare/ \
    --gpu-memory-utilization 0.6 --seed 42
```

**Run 3 — dare_adamerging aggregated** (historical dir
`bakeoff_adamerging_agg_20260528_0943/`). This is the version that appears in
the final table and is deployed:

```bash
python scripts/run_bakeoff.py \
    --methods dare_adamerging --aggregate-domains \
    --adapters-dir loras/ \
    --output-dir bakeoff_dare_aggregated/ \
    --gpu-memory-utilization 0.6 --seed 42
```

**How the table is assembled:** `uniform` and `ties` from Run 1,
`dare_uniform` from Run 2, and `dare_adamerging` (aggregated) from Run 3.
Each run writes a `bakeoff_results.json` with per-(method, temperature)
scorecards (pass@1 / pass@8 per benchmark); the report combines the relevant
rows by hand.

## h) Reproduce the AdaMerging diagnostic (loss curves + heatmaps)

```bash
# Baseline: round-robin objective (unstable).
python scripts/adamerging_diagnostic.py \
    --adapters-dir loras/ \
    --output-dir adamerging_diagnostic_baseline/

# Aggregated objective (after the fix).
python scripts/adamerging_diagnostic.py --aggregate-domains \
    --adapters-dir loras/ \
    --output-dir adamerging_diagnostic_aggregated/
```

Each run writes `metrics.json` (loss history + per-layer coefficients +
hyperparameters), `loss_curve.png`, and `coefficients_heatmap.png`.

## i) Reproduce the weight-similarity numbers

The report cites pairwise weight-space differences between the merged models
from the bake-off. To reproduce these from the merged checkpoints in the
bake-off output directories:

```bash
python3 scripts/weight_similarity.py \
    --models \
      uniform:bakeoff_20260526_1145/uniform/merged \
      ties:bakeoff_20260526_1145/ties/merged \
      dare_uniform:bakeoff_dare_20260526_1246/dare_uniform/merged \
      dare_adamerging_rr:bakeoff_dare_20260526_1246/dare_adamerging/merged \
      dare_adamerging_agg:bakeoff_adamerging_agg_20260528_0943/dare_adamerging/merged \
    --output weight_similarity.json
```

The script reads `model.safetensors` from each path, computes mean-absolute
relative difference (MAR) on three representative MLP tensors (early, middle,
late layers — `model.layers.{0,13,27}.mlp.down_proj.weight`), and prints a
triangular matrix per tensor plus a JSON summary. Add `--metric both` for
cosine similarity alongside MAR. Reads only the requested tensors from disk
(via `safetensors.safe_open`); no GPU; runs in under a minute.

## j) Tests

```bash
pytest merge/tests/ -v
```

Most tests are gated on torch / CUDA / vLLM and **skip** on a CPU-only
laptop; they run fully on a GPU node. The locked-spec regression test and the
pure-IO / dataclass tests always run. A clean laptop run shows a large number
of passes plus skips, no failures.

## k) Utility commands

Run a single method + evaluation:

```bash
python scripts/run_bakeoff.py --methods <method_name> \
    --adapters-dir loras/ --output-dir bakeoff_single/ \
    --gpu-memory-utilization 0.6 --seed 42
```

Re-evaluate an already-merged model at several temperatures (no re-merge —
temperature is sampling-only):

```bash
python scripts/eval_sweep.py \
    --merged-adapter-dir <path>/merged \
    --output-dir eval_out/ \
    --temperatures 0.3 0.5 0.7
```

Available methods: `uniform`, `dare_uniform`, `ties`, `adamerging`,
`dare_adamerging`.

## l) Environment

- **GPU:** 1× A100 40 GB (minimum). The merged Qwen3-1.7B is ~3.4 GB in bf16.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is recommended to reduce
  fragmentation across the merge → eval alloc/free cycles. Prefix any
  long-running launch with it, e.g.:
  ```bash
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python -u \
      scripts/run_bakeoff.py ... > bakeoff.log 2>&1 &
  ```
- `--gpu-memory-utilization 0.6` caps vLLM's startup request to ~24 GB on a
  40 GB card. The vLLM default (0.9) demanded ~35 GB free and was the cause of
  the 2026-05-26 eval failures on a shared node; lower further (e.g. 0.45) if
  a co-tenant still starves engine init.
- Scripts resolve all default paths (`lora.yaml`, `validation_samples/`,
  `chat_template.jinja`) relative to this `code/` directory via
  `Path(__file__).parent.parent`, and insert that root onto `sys.path` so
  `nohup`'d / detached runs still `import merge` correctly. No absolute paths
  are baked in (the one exception is `adamerging_diagnostic.py`'s default
  `--output-dir`, a cluster scratch path you can override).

## m) Key hyperparameters

All fixed; override via the flags shown above.

| Group | Values |
|---|---|
| LoRA (`lora.yaml`) | `r=32`, `alpha=64`, `dropout=0.05`, 7 target projections |
| DARE | `drop_rate=0.5` |
| AdaMerging | `lr=1e-2`, `lambda_l2=1e-4`, `init_coefficient=0.3`, `max_steps=200`, `batch_size=2`, `early_stop_patience=100` |
| Sampling (bake-off) | temperatures `0.3, 0.5, 0.7`; `top_p=0.8`, `top_k=20`, `n=8`, `max_tokens=2048` |
| Seed | `42` (all scripts) |
| Deployed model | `dare_adamerging --aggregate-domains` @ `T=0.3` |

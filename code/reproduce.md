# Reproducing the group-model results

This document maps every quantitative result in the **group-model section of the
CS-552 M3 report** to the exact command that produces it and the file where the
number lands. It is a result-to-command index, not a setup guide — for
installation, HF auth, and adapter download see [`README.md`](README.md)
sections (b)–(e).

## Prerequisites

1. **Install / auth / adapters**: follow [`README.md`](README.md) (b) install,
   (c) `hf auth login` / `export HF_TOKEN=...`, (d) `python scripts/fetch_adapters.py
   --target-dir loras/`. All commands below assume the 4 adapters live in `loras/`.
2. **Hardware**: 1× A100 40 GB. Full reproduction is ~3 h wall clock (bake-off
   ~2.5 h, diagnostics ~20 min, weight-sim ~1 min CPU).
3. **Determinism**: every script defaults to `--seed 42` (passed explicitly below
   for clarity). Merges and weight-similarity are fully deterministic; vLLM
   sampling has residual non-determinism (dynamic batching), so CI pass-rates can
   drift — `safety@1` up to ±0.10 — while `math@8` stays pinned at 0.300. See the
   [Caveats](#caveats) section.

Prefix long launches with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to
reduce fragmentation across merge→eval alloc cycles.

---

## Table 5 — best temperature per method (main results)

Run the full bake-off once (next section) and read one cell per
`(method, temperature)` from its scorecard. The deployed rows are:

| Report cell | File and key |
|---|---|
| Uniform (T=0.5) | `bakeoff_full/uniform/sweep/T_0.5/scorecard.json` |
| TIES (T=0.5) | `bakeoff_full/ties/sweep/T_0.5/scorecard.json` |
| DARE-uniform (T=0.3) | `bakeoff_full/dare_uniform/sweep/T_0.3/scorecard.json` |
| DARE-AdaMerging aggregated (T=0.3) | `bakeoff_full/dare_adamerging/sweep/T_0.3/scorecard.json` |

Each `scorecard.json` is keyed by benchmark (`math`, `general_knowledge`,
`safety`, `multilingual`), and each entry has `pass_at_1` and `pass_at_8`.
Cell mapping (example for DARE-uniform @ T=0.3):

- `math@8 = 0.300` → key `math` → `pass_at_8`
- `gene@1 = 0.587` → key `general_knowledge` → `pass_at_1`
- `safe@1 = 0.700` → key `safety` → `pass_at_1`
- `mult@1 = 0.450` → key `multilingual` → `pass_at_1`
- `CI-avg` = mean of the 4 CI metrics above (math uses pass@8, the rest pass@1).

---

## Table (appendix) — full bake-off, 4 methods × 3 temperatures (12 rows)

One command reproduces all 12 scorecards. `--aggregate-domains` only changes
`dare_adamerging` (the other three methods ignore it), so this single run
produces the aggregated AdaMerging variant that appears in the report:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/run_bakeoff.py \
    --methods uniform ties dare_uniform dare_adamerging \
    --aggregate-domains \
    --adapters-dir loras/ \
    --output-dir bakeoff_full/ \
    --gpu-memory-utilization 0.6 \
    --seed 42
```

~2.5 h on A100-40GB. Output layout (12 scorecards + 1 aggregate):

```
bakeoff_full/
    bakeoff_results.json                 # aggregated comparison, all rows
    <method>/merged/                     # full HF-format model (config.json + model.safetensors)
    <method>/sweep/T_{0.3,0.5,0.7}/scorecard.json
```

The 12 appendix rows = `{uniform, ties, dare_uniform, dare_adamerging}` ×
`{T_0.3, T_0.5, T_0.7}`, each read from its `scorecard.json` exactly as in
Table 5 above. `bakeoff_results.json` collates the same numbers in one file.

> The original report numbers came from three separate bake-off runs on
> different days (see [`README.md`](README.md) (g)); this single command is the
> clean from-scratch equivalent and the `bakeoff_full/<method>/merged` paths
> feed the weight-similarity step below.

---

## Table 4 — weight similarity (MAR %)

Deterministic, CPU-only, ~1 min. Compares the merged `model.safetensors` of the
4 deployed methods on three MLP tensors (layers 0/13/27 `down_proj`):

```bash
python3 scripts/weight_similarity.py \
    --models \
      uniform:bakeoff_full/uniform/merged \
      ties:bakeoff_full/ties/merged \
      dare_uniform:bakeoff_full/dare_uniform/merged \
      dare_adamerging:bakeoff_full/dare_adamerging/merged \
    --output weight_similarity.json
```

**Reading the JSON to match the table.** The report cites one number per pair,
*averaged over the three MLP tensors*. In `weight_similarity.json`, `pairwise`
is keyed by tensor; each tensor holds one row per pair with `mar_pct`. For a
given pair (e.g. `dare_uniform` vs `uniform` → 0.45), take the three `mar_pct`
values for that pair across the three tensor keys and average them. Do **not**
use the top-level `summary` field — that is the global min/max/mean over *all*
pairs and tensors, not a per-pair value.

**Reproducibility check (1.14).** This row compares `dare_adamerging` against a
second, independent from-scratch `dare_adamerging` aggregated merge. Produce the
second model with the deployed-model command below into a different dir, then
add it as a 5th `--models` entry:

```bash
      dare_adamerging_rerun:bakeoff_rerun/dare_adamerging/merged
```

The `dare_adamerging` vs `dare_adamerging_rerun` per-pair mean ≈ 1.14.

---

## Figures + AdaMerging metrics (body + appendix)

Two diagnostic runs produce the two coefficient heatmaps and both loss curves:

```bash
# Aggregated → heatmap (body, 4×28, init 0.3) + aggregated loss curve (193 steps).
# NOTE: --aggregate-domains appends "_aggregated" to the output dir name,
# so results land in diag_agg_aggregated/, NOT diag_agg/.
python3 scripts/adamerging_diagnostic.py \
    --aggregate-domains \
    --adapters-dir loras/ \
    --output-dir diag_agg/ \
    --seed 42

# Baseline round-robin → heatmap (appendix) + baseline loss curve (159 steps).
python3 scripts/adamerging_diagnostic.py \
    --adapters-dir loras/ \
    --output-dir diag_baseline/ \
    --seed 42
```

Each run writes `metrics.json`, `loss_curve.png`, `coefficients_heatmap.png`.

| Report figure | Produced file |
|---|---|
| Heatmap aggregated (body) | `diag_agg_aggregated/coefficients_heatmap.png` |
| Heatmap baseline (appendix) | `diag_baseline/coefficients_heatmap.png` |
| Loss curves — aggregated subfigure | `diag_agg_aggregated/loss_curve.png` |
| Loss curves — baseline subfigure | `diag_baseline/loss_curve.png` |

**In-text AdaMerging numbers** all come from `diag_agg_aggregated/metrics.json`:

- Loss descent (1.20 → ~0.25 over first ~75 updates, oscillation in [0.02, 0.8],
  early-stop at update 192) → `loss_history` (list) + `steps_run` + `early_stopped`.
- Per-task coefficient means (math 0.148, gene 0.024, safety 0.186, multilingual
  0.173; per-task max e.g. math 0.70, safety 0.82) → `coefficients`, an
  `[n_tasks × n_layers]` array ordered by `task_names`; average / take max over
  each row.
- Sign-flip pattern (positive layers 0–21, negative 22–27) → inspect the sign of
  each layer column in `coefficients`.

---

## Deployed model on Hugging Face

The published `cs-552-2026-emainelpe/group_model` is **DARE-AdaMerging
aggregated @ T=0.3**. Build it, then publish (dry-run first):

```bash
# Merge the deployed model (if not already in bakeoff_full/).
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/run_bakeoff.py \
    --methods dare_adamerging --aggregate-domains \
    --adapters-dir loras/ --output-dir bakeoff_final/ \
    --gpu-memory-utilization 0.6 --seed 42

# Dry-run (prints upload plan + generation_config.json, touches nothing).
python3 scripts/publish.py \
    --model-dir bakeoff_final/dare_adamerging/merged \
    --repo-id cs-552-2026-emainelpe/group_model \
    --temperature 0.3 --top-p 0.8 --top-k 20

# Re-run with --confirm to push and rewrite generation_config.json.
```

The bundled `generation_config.json` matches the deployed sampling params:
`temperature=0.3`, `top_p=0.8`, `top_k=20`, `max_new_tokens=16384` (the
`--max-new-tokens` default; see [Caveats](#caveats)).

---

## Caveats

- **vLLM non-determinism**: pass-rates can drift run-to-run from dynamic
  batching even at fixed `--seed 42`. `math` pass@8 is stable at **0.300**;
  `safety` pass@1 can vary up to **±0.10**. Treat the Table 5 / appendix numbers
  as a single sampled run, not exact constants.
- **Weight similarity is fully deterministic** — it reads fixed merged weights on
  CPU, so its numbers reproduce exactly.
- **`--max-new-tokens`**: `publish.py` defaults `--max-new-tokens` to 16384 to
  match the CI ceiling. Earlier publications (before June 3) used the prior
  default of 2048; we re-published the `generation_config.json` with the
  corrected value on June 3.
- **Diagnostic output-dir suffix**: `--aggregate-domains` appends `_aggregated`
  to the diagnostic `--output-dir`; the aggregated artefacts are in
  `<dir>_aggregated/`, not `<dir>/`.

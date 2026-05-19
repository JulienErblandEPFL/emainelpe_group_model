# CLAUDE.md

Operational guide for Claude Code sessions in this repo. Terse and factual.

## What this repo is

The team's shared **locked-spec contract** (`lora.yaml`, `chat_template.jinja`,
`evaluate/`, `validation_samples/`) plus the **group-model merge code** under
`merge/`. Single dual-purpose repo by design (decision B2 — see PROCESS_BOOK).
The other three teammates work in separate private repos and pull this repo's
locked specs.

## Project context

- CS-552 Modern NLP final project, EPFL Spring 2026
- Team Émainèlpé, group g65: Julien Erbland (Math, group-model lead),
  Max Henrotin (General Knowledge), Mathis Richard (Multilinguality),
  Morgane Magnin (Safety)
- Base model: `Qwen/Qwen3-1.7B`
- 4 individual LoRA adapters → 1 group model via DARE → AdaMerging
  (TIES fallback)
- Deadlines: Milestone 2 = 2026-05-24 (group + 4 individual models
  CI-passing on HF); Final = 2026-06-07

## Locked contracts (DO NOT modify without team discussion)

- `lora.yaml` — LoRA spec. Currently: `r=32`, `α=64`, `dropout=0.05`,
  `bias=none`, `task_type=CAUSAL_LM`, `modules_to_save=null`, 7 target
  modules (`q/k/v/o/gate/up/down`)_proj. **Byte-identical across all 4
  teammates' adapters is required** for additive merging.
- `chat_template.jinja` — Qwen3 chat template with thinking mode forced ON;
  must produce `<think>...</think>` + `\boxed{...}`.
- `evaluate/` — CPU scorer mirroring CI. Input: generations JSONL.
  Output: pass@1 / pass@8.
- `validation_samples/` — 40 problems across 4 domains, frozen.

## What `merge/` contains

The group-model merge pipeline. Self-contained: reads `../lora.yaml` and
`../chat_template.jinja`, uses `../evaluate/` for scoring, but nothing in
`merge/` is imported by anything outside it. Teammates can ignore `merge/`
entirely.

- `merge/verify_spec.py` — verify an adapter's config against `lora.yaml`
  (8 load-bearing fields, whitelist-based)
- `merge/load_adapter.py` — load PEFT-format LoRA adapters from a local
  directory; materializes `ΔW = (α/r) · B @ A`
- `merge/methods/` — `dare`, `uniform_merge`, `weighted_linear_merge`
  (implemented); `ties_merge`, `adamerging` (stubs)
- `merge/methods/__init__.py` — `METHOD_REGISTRY` dispatches a method-name
  string to its callable
- `merge/pipeline.py` — orchestrator (stub, Stage 4)
- `merge/infer.py`, `merge/publish.py`, `merge/eval_all.py` — stubs (Stage 5)
- `merge/tests/` — CPU-runnable tests; torch-dependent tests use
  `pytest.importorskip("torch")`

## Conventions and constraints

- **Working environment**: laptop = code + docs only. RCP cluster =
  training + inference + tests requiring GPU/torch. Workflow: laptop edits
  → `git push` → cluster `git pull` → `runai submit`.
- **Tests must be CPU-only laptop-runnable**: `pytest merge/tests/ -v`
  must pass or skip cleanly on a torch-free laptop. Torch-dependent tests
  use `pytest.importorskip("torch")` and execute fully on the cluster.
- **No HF API calls in tests, no GPU code in tests, no training in tests.**
- **Lazy ML imports** where it doesn't hurt readability (e.g. `safetensors`
  inside `load()`, not at module top).
- **Stage gating**: stubs `raise NotImplementedError("Stage N")` until
  implemented. `merge/README.md`'s status table tracks done vs pending.
- **Adapter directory layout**: 4 fixed domain names —
  `math`, `general_knowledge`, `safety`, `multilingual`. Anything else
  is an error.
- **Pipeline I/O contract**: input = local directory of PEFT-format
  adapters (no HF download in the pipeline itself). Output = local
  PEFT-format merged adapter, ready for `load()` round-trip.
- **Dependencies**: Python dependencies for the merge subdir are listed in
  `requirements.txt` at the repo root.

## Files to NEVER touch from a coding task

Unless the task explicitly asks: `lora.yaml`, `chat_template.jinja`,
`evaluate/*`, `validation_samples/*`, `USAGE.md`, `RCP_GUIDE.md`,
`README.md`, `AUDIT_GROUP_PHASE.md`, `docker/submit.sh` (gitignored
per-teammate launcher).

## Cluster context (RCP)

- Image: `ayushkumartarun/course-cs-552-standard:v1` (course standard).
- GPU: A100-40GB default. Pods preemptible.
- Mounts: `/scratch` (team), `/shared-ro`, `/shared-rw`.
- Env: `HF_HOME=/scratch/hf_cache`; `HF_TOKEN` and `WANDB_API_KEY` are
  read from the caller's shell env.
- Repo path on cluster: `/scratch/Group/emainelpe_group_model`.

## HF repos relevant to the group model

| Repo | Role |
|---|---|
| `cs-552-2026-emainelpe/group_model` | Final merged group model (Stage 5 publishing target). |
| `cs-552-2026-emainelpe/math_model` | Math specialist (Julien). Currently full-merged. Source adapter at `/scratch/Julien/runs/cs552-erbland-g65-v4-fresh-20260514-162214/final/`. |
| `cs-552-2026-emainelpe/general_knowledge_model` | Max. |
| `cs-552-2026-emainelpe/safety_model` | Morgane. |
| `cs-552-2026-emainelpe/multilingual_model` | Mathis. |

## How to verify state

A few quick checks a future session can run:

- `pytest merge/tests/ -v` — full test suite (78 tests as of Stage 3).
- `cat merge/README.md` — current implementation status table.
- `git log --oneline -10` — recent commits.
- `cat PROCESS_BOOK.md` — decision history and chronology.

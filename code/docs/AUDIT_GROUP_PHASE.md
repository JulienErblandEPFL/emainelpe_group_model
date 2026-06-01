# Group-Phase Audit — `emainelpe_group_model`

Read-only inventory, no edits / no training / no API calls. Date: 2026-05-15. Branch: `main` (at `76c95e2`).

---

## TL;DR (read this first)

- **Identity mismatch:** the git remote is `emainelpe_group_model`, but the *content* of this repo self-identifies as **`emainelpe-shared`** — the team's locked-files repo (LoRA spec + chat template), not a training repo. Filenames inside (`USAGE.md` titled `# emainelpe-shared`, header banners in `lora.yaml` and `chat_template.jinja`) all refer to the shared repo.
- **No code from the math adapter v5 is in this repo.** No training scripts, no checkpoints, no inference code, no `adapter_config.json`, no `generation_config.json`, no `push_to_hub` calls, no merge infrastructure, no Liger Kernel, no SLURM/runai launcher beyond the generic Jupyter-Lab skeleton, no REPORT.md. Whatever was trained for the math track lives in a different repo / on HF Hub only.
- **LoRA spec is unambiguous in this repo.** `lora.yaml` is the *single* place a LoRA shape is defined (r=32, α=64, dropout=0.05, bias=none, task_type=CAUSAL_LM, 7 target modules). No competing PEFT `LoraConfig(...)` constructors, no other YAML configs. Internal audit passes; *external* verification (did the four teammates actually train at these values?) cannot be answered from this repo and is the single highest-leverage open question.
- **Evaluator is scoring-only.** `evaluate/` is a CPU-only scorer that mirrors CI extraction. It does **not** run inference. There is no vLLM script, no generation script, no pre-push smoke test, no non-math benchmark code beyond the shared `\boxed{}` parser.

---

## Section 1 — Repo topology

### Tree (2 levels)

```
emainelpe_group_model/
├── README.md                 # GitHub Classroom skeleton: project rules, CI contract, formats
├── RCP_GUIDE.md              # EPFL RCP cluster setup + Run:AI usage
├── USAGE.md                  # Self-titled "emainelpe-shared" — explains the shared lock
├── lora.yaml                 # LOCKED LoRA spec
├── chat_template.jinja       # LOCKED Qwen3 chat template (thinking ON)
├── docker/
│   ├── Dockerfile            # Optional custom image (CUDA 12.8 / torch 2.10 / vLLM 0.19.1)
│   ├── build.sh              # Build & push the custom image
│   ├── submit.sh             # Run:AI submit script (interactive Jupyter Lab) — UNCOMMITTED EDIT
│   ├── #submit.sh#           # Emacs autosave artifact, identical to submit.sh; UNTRACKED
│   └── requirement_docker.txt
├── evaluate/                 # Offline scorer (CI-equivalent extraction + pass@k)
│   ├── __init__.py
│   ├── README.md
│   ├── benchmarks.py
│   ├── extract_answer.py
│   ├── pass_at_k.py
│   └── score.py
└── validation_samples/       # 10 problems/benchmark (40 total) per README; one benchmark has 14 — see flag
    ├── math.jsonl
    ├── general_knowledge.jsonl
    ├── safety.jsonl
    ├── multilingual.jsonl
    └── README.md             # MISSING — README refers to this file but it does not exist
```

### Top-level directory purposes

| Dir | Purpose |
|---|---|
| `docker/` | Optional custom image + the Run:AI submit script every teammate edits with their gaspar/group |
| `evaluate/` | Offline pass@1/pass@8 scorer mirroring CI extraction logic |
| `validation_samples/` | Frozen 40-row sanity-check set across the 4 domains |

There are **no** `math/`, `safety/`, `shared/`, `merge/`, `configs/`, `recipes/`, `src/`, or `scripts/` subdirs. There is no Julien-only vs teammate-shared split inside this repo — everything here is either skeleton-from-Classroom or shared-by-design.

### All source/config files (size in bytes)

| Path | Size |
|---|---|
| `README.md` | 14139 |
| `RCP_GUIDE.md` | 12537 |
| `USAGE.md` | 3015 |
| `lora.yaml` | 1786 |
| `chat_template.jinja` | 3071 |
| `docker/Dockerfile` | 5558 |
| `docker/build.sh` | 1995 |
| `docker/submit.sh` | 5631 |
| `docker/#submit.sh#` | 5631 (emacs autosave, identical to `submit.sh`) |
| `docker/requirement_docker.txt` | (29 lines, see §7) |
| `evaluate/__init__.py` | 291 |
| `evaluate/README.md` | 3360 |
| `evaluate/benchmarks.py` | 3440 |
| `evaluate/extract_answer.py` | 10125 |
| `evaluate/pass_at_k.py` | 2102 |
| `evaluate/score.py` | 5304 |
| `validation_samples/math.jsonl` | 10 rows |
| `validation_samples/general_knowledge.jsonl` | **14 rows** (flag — see §9) |
| `validation_samples/safety.jsonl` | 10 rows |
| `validation_samples/multilingual.jsonl` | 10 rows |

---

## Section 2 — LoRA spec audit (CRITICAL)

The *only* place a LoRA config is defined or referenced in this repo is `lora.yaml`. No Python `LoraConfig(...)`, no JSON adapter spec, no CLI defaults, no other YAML configs. No `adapter_config.json` anywhere on disk (confirmed via `find`).

### `lora.yaml` (single source of truth, lines 17–48)

```yaml
base_model: "Qwen/Qwen3-1.7B"
lora:
  r: 32
  alpha: 64
  dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
max_seq_length: 4096
eos_token: "<|im_end|>"
thinking_mode: "on"
boxed_answers: true
```

`modules_to_save` is **not** present (so no embeddings / `lm_head` are wrapped). This matters for merging — if any teammate added `modules_to_save: ["embed_tokens", "lm_head"]` in their training config, their adapter will have extra full-rank tensors that DARE/AdaMerging cannot combine additively.

### Source-table

| Source | r | alpha | target_modules | dropout | bias | task_type | modules_to_save | Format |
|---|---|---|---|---|---|---|---|---|
| `lora.yaml` (this repo) | 32 | 64 | 7 (q/k/v/o + gate/up/down) | 0.05 | "none" | CAUSAL_LM | (not set) | YAML literal |
| Math adapter `adapter_config.json` (local) | MISSING — not in this repo | — | — | — | — | — | — | not present locally |
| Math adapter on HF Hub `cs-552-2026-emainelpe/math_model` | NOT FETCHED (read-only audit; offline) | — | — | — | — | — | — | should be pulled & diffed before Stage 1 |
| Other 3 teammate adapters | NOT IN REPO | — | — | — | — | — | — | each lives in their own private repo per `USAGE.md` §1 |

### Divergence status

- **Internal:** ZERO divergence. `lora.yaml` is the only spec in the repo.
- **External:** UNVERIFIED. The whole premise of `USAGE.md` is that each teammate *copies* `lora.yaml` byte-identically into their training repo. Whether that actually happened for all 4 adapters is the project's single biggest pre-merge risk. `USAGE.md` §3 explicitly schedules a "pre-merge alignment check (≈ May 27)" with `diff`. This audit cannot verify it without leaving the repo.

### Cross-checks worth running (not done here, read-only)

1. `hf download cs-552-2026-emainelpe/math_model adapter_config.json` and `diff` against `lora.yaml` → confirms math v5 used the locked spec.
2. Same for `general_knowledge_model`, `safety_model`, `multilingual_model` once teammates push.
3. Inspect `peft_config["lora_A"]` shapes on the loaded HF adapter to confirm `r=32` survived the SFT/DPO/RLVR pipeline.

---

## Section 3 — Teammate adapter discovery

Search across all source files for teammate-name / domain / benchmark keywords (case-insensitive). All hits:

| File:line | Match | Context |
|---|---|---|
| `USAGE.md:17` | `Mathis` | "If Mathis trains at `r=16` and Julien trains at `r=32`, their adapters cannot be linearly combined." (example only) |
| `USAGE.md:18` | `Julien` | same line as above |
| `USAGE.md:32` | `DPO`, `RLVR` | listed as things that should diverge per-expert |
| `README.md:32-35` | `math_model`, `general_knowledge_model`, `safety_model`, `multilingual_model` | canonical HF repo path table (with `<your-org>` placeholder) |
| `README.md:215` | `cs-552-2026-<your-org>` | docs URL for EVAL_REPORT location |
| `README.md:131-141` | `General knowledge`, `Multilinguality`, `Safety` | domain descriptions |
| `evaluate/*` | `knowledge`, `multilingual`, `safety` | benchmark→method mapping (no domain-specific logic, only `\boxed{}` + choice-letter fallback) |

Hits for `henrotin`, `magnin`, `morgane`, `max_`, `mmlu`, `arc`, `mgsm`, `msvamp`, `safetybench`, `xstest`, `pku-safe`: **zero**. The only literal teammate references in the entire repo are the two illustrative names `Mathis` and `Julien` in `USAGE.md`. No HF repo IDs of the form `cs-552-2026-emainelpe/*` are hardcoded anywhere — only the templated `cs-552-2026-<your-org>/*` form in README.

**No README, NOTES, or doc file mentions teammates' planned adapters, training configs, or coordination beyond `USAGE.md`'s general "each teammate copies these two files" rule.**

---

## Section 4 — Merge infrastructure (current state)

Greps for `DARE`, `AdaMerging`, `TIES`, `task-arithmetic`, `task-vector`, `model-soup`, `mergekit`, `slerp`, `merge_and_unload`, `set_adapters`, `add_adapter`, `load_adapter`, `PeftModel.from_pretrained`:

| File:line | Match | Context |
|---|---|---|
| `lora.yaml:10` | `DARE + AdaMerging` | comment — "REQUIREMENT for DARE + AdaMerging to compose the adapters into the group model in Phase 3" |
| `USAGE.md:15` | `DARE + AdaMerging` | comment — "must be identical across all four experts for the Phase 3 merge (DARE + AdaMerging) to work mathematically" |

These are the **only two mentions**, both in documentation comments. **No merge infrastructure present — starting from scratch.** No Python that loads or composes adapters, no `mergekit` config, no task-vector arithmetic, no soup script.

---

## Section 5 — Eval harness (current state)

### Files

| Path | What it does | Inputs | Outputs |
|---|---|---|---|
| `evaluate/score.py` | CLI: scores a generations JSONL with the same extraction the CI uses | `--generations <jsonl>` `--benchmark {math,knowledge,multilingual,safety}` `[--output scored.json]` | stdout: `pass@1=…, pass@8=…`; optional detailed per-problem JSON |
| `evaluate/benchmarks.py` | Per-benchmark extraction + correctness (`extract_benchmark_answer`, `is_correct_benchmark_answer`) | text, method, reference | extracted string \| None; bool |
| `evaluate/extract_answer.py` | `\boxed{…}` parser (`last_boxed_only_string`, `remove_boxed`, `extract_boxed_answer`); math normalization (`normalize_final_answer`, `strip_string`, `_fix_fracs`, `_fix_a_slash_b`); equivalence (`is_equiv`) | strings | strings/bool |
| `evaluate/pass_at_k.py` | Unbiased Chen-2021 pass@k estimator + dataset aggregator | per-problem correct counts, n, k_values | dict like `{"pass@1": …, "pass@8": …}` |
| `evaluate/README.md` | Usage docs |  |  |
| `evaluate/__init__.py` | `__version__ = "1.0.0"` |  |  |

### Benchmark→method map (`evaluate/score.py:28-33`)

```python
BENCHMARK_TO_METHOD = {
    "math":         "boxed",
    "knowledge":    "knowledge",    # tries boxed → falls back to text/choice letter
    "multilingual": "boxed",
    "safety":       "boxed",
}
```

### Important facts

- **No OpenCompass / `mmengine` / `configs/datasets/` / `configs/models/` directories** anywhere. The math-track is *not* using OpenCompass in this repo. The header in `extract_answer.py:1-5` says the helpers were "ported from OpenCompass" but only the extraction/normalization functions, not the harness.
- **No inference / generation code.** `evaluate/` requires completions as input — it cannot run a model.
- **No non-math eval beyond the shared `\boxed{}` parser.** The "knowledge" method adds a choice-letter regex (`evaluate/benchmarks.py:62-76`) but no benchmark-specific code (no MMLU loader, no MGSM loader, no SafetyBench loader, no XSTest loader, etc.).
- **No math validation set beyond `validation_samples/math.jsonl` (10 problems).** No path to a larger held-out math eval set is referenced.

---

## Section 6 — Inference / vLLM / generation config

| Item | Status |
|---|---|
| `generation_config.json` files in repo | **NONE** (`find` returned nothing) |
| vLLM serving / inference scripts | **NONE** in repo |
| `\boxed{}` answer-extraction logic | `evaluate/extract_answer.py:16-68` |
| Chat-template handling | `chat_template.jinja` (root) — Qwen3 template, **thinking forced ON** |

### `\boxed{…}` extractor (exact logic, `evaluate/extract_answer.py:16-68`)

Not a regex — a brace-balanced scan from the **last** `\boxed` (or `\fbox` fallback) index:

```python
def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{": num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None: return None
    return string[idx : right_brace_idx + 1]
```

Then `remove_boxed(s)` strips the outer `\boxed{` / `\fbox{` and trailing `}` (`extract_answer.py:43-48`). With `strip_double_curly_brace=True` (used by both `boxed` and `knowledge` methods in `benchmarks.py:24,28`), a single-level `^\{(.*)\}$` regex match unwraps a nested brace.

### Chat-template (`chat_template.jinja:21`)

- `{%- set enable_thinking = true %}` — hard-coded ON; assistant turns are expected to emit `<think>…</think>` before the final boxed answer.
- Tool-call block exists (lines 22–32) but is gated on `if tools` and irrelevant for the CI prompt path (no tools passed).
- System message: lines 34–36 — `<|im_start|>system\n + system content + <|im_end|>\n` only when the first message has `role=='system'`. **No hard-coded system prompt baked into the template**; whatever system prompt is sent at training/inference is the only signal.

### Notes

- CI calls `tokenizer.apply_chat_template(messages, add_generation_prompt=True)` with no `enable_thinking` kwarg (README:47, 52-54). The template explicitly defends against that.
- `eos_token: "<|im_end|>"` is set in `lora.yaml:41` for parity with Qwen3.

---

## Section 7 — Training infrastructure

| Aspect | Status |
|---|---|
| Training entrypoint script for math v5 | **NOT IN THIS REPO** |
| Math v5 training config | **NOT IN THIS REPO** |
| Optimizer / scheduler / batch / grad accum / precision / packing | **NOT IN THIS REPO** |
| Liger Kernel usage | **NOT IN THIS REPO** (zero matches for `liger` across all files) |
| Run:AI launchers in repo | `docker/submit.sh` — interactive Jupyter Lab only (no training command baked in) |
| SLURM scripts | **NONE** |

### `docker/submit.sh` (the only launcher; `submit.sh:34-100`)

- Defaults: `GPUS=1`, `NODE=a100-40g`, image `ayushkumartarun/course-cs-552-standard:v1` (course standard, not a custom build).
- Currently has Julien's GASPAR/GROUP edits as **uncommitted changes** (only diff vs origin/main: `GASPAR="erbland"`, `GROUP="g65"`, plus `chmod +x` so file mode flipped from 100644 → 100755).
- Submits an *interactive* job that launches `jupyter lab` on port 8888 with token `cs552`. **It does not run training** — training has to be invoked from inside Jupyter or `runai bash`.
- Mounts: `/scratch` (team), `/shared-ro`, `/shared-rw` via three named PVCs.
- Env: `HF_HOME=/scratch/hf_cache`, `HF_HUB_ENABLE_HF_TRANSFER=1`, optional `HF_TOKEN`, optional `WANDB_API_KEY` (both read from caller's shell).

### `docker/Dockerfile` (optional custom image)

- Base `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`, Python 3.12.
- Pinned: `torch==2.10.0+cu128`, `vllm==0.19.1`, FlashAttention 2.8.3 (prebuilt wheel).
- `requirement_docker.txt` adds: `transformers`, `trl`, `peft`, `accelerate`, `deepspeed`, `bitsandbytes`, `datasets`, `wandb`, `lightning`, `hydra-core`, `tensorboardX`, `scikit-learn`, `scipy`, etc. No `liger-kernel`, no `mergekit`, no `mmengine`, no `opencompass`.

### `docker/#submit.sh#` (untracked)

- Emacs autosave; **byte-identical** to `docker/submit.sh` (verified via `diff`). Safe to ignore or delete; not a divergent edit.

---

## Section 8 — HF Hub publishing

| Item | Status |
|---|---|
| `huggingface_hub` `upload_folder` / `create_repo` / `HfApi` calls | **ZERO** matches across `*.py`, `*.sh`, `*.yaml`, `*.md` |
| `push_to_hub` calls | **ZERO** matches |
| Hard-coded `cs-552-2026-emainelpe/*` repo IDs | **ZERO** (all references in `README.md` use the `<your-org>` placeholder) |
| HF token / secret management | **No `.env`**, no secrets file; `HF_TOKEN` is read from the caller's shell env via `submit.sh:68` and `submit.sh:87` (`--environment HF_TOKEN="${HF_TOKEN:-}"`) |
| `huggingface_hub` in deps | Yes — `docker/requirement_docker.txt:4` (no version pin) |

The math v5 deploy to `cs-552-2026-emainelpe/math_model` must have been done from outside this repo (CLI, notebook, or the math repo) — there is no code path inside this repo that pushes anything to HF.

---

## Section 9 — Open questions / red flags

Listed in rough priority order for the group-model Stage-1 risk gate.

1. **CRITICAL — External LoRA-spec verification.** The locked spec lives only here. The math v5 adapter on `cs-552-2026-emainelpe/math_model` may or may not match. Until you `hf download adapter_config.json` for all four adapters and `diff` against `lora.yaml`, the merge plan is not safe to start. (`USAGE.md:55-58` outlines the exact `diff` commands.)
2. **CRITICAL — `modules_to_save` is silently absent.** `lora.yaml` does not set `modules_to_save`. If any teammate added it (e.g. for embedding tuning), their checkpoint will carry full-rank tensors that break additive merging. Confirm during the same external check above.
3. **HIGH — No teammate-side artifacts visible.** Nothing in this repo proves that anyone other than the math owner has even started training. The pre-merge alignment check `USAGE.md:51` schedules is for "≈ May 27"; today is 2026-05-15 so there are ≈12 days. Worth pinging teammates explicitly to confirm spec adoption *before* you start writing merge code.
4. **HIGH — No inference / generation path in this repo.** To smoke-test merged checkpoints locally against `validation_samples/*.jsonl`, you need to write (or import) a vLLM-based generator that produces n=8 completions per problem and feeds them into `evaluate.score`. That code does not exist yet.
5. **MEDIUM — `general_knowledge.jsonl` has 14 rows, not 10.** `README.md:172-174` says "10 problems per benchmark (40 total)". `wc -l` says 14. Either the README is stale or someone added 4 extra rows. Check `git log validation_samples/general_knowledge.jsonl` and decide whether to trust the file.
6. **MEDIUM — `validation_samples/README.md` is referenced but missing.** `README.md:177-183` shows a tree that includes `validation_samples/README.md`. The file does not exist in this repo.
7. **MEDIUM — Identity mismatch.** Git remote is `emainelpe_group_model`, repo content is `emainelpe-shared`. Either (a) the team plans to repurpose this repo as the group-model checkpoint repo (overwriting it with real model artifacts), or (b) the two things were conflated. Reconcile early — pushing `lora.yaml` to a HF "model" repo will fail validation (no `config.json` / weights at root).
8. **MEDIUM — Uncommitted `docker/submit.sh` change.** `GASPAR=""`/`GROUP="gxx"` → `GASPAR="erbland"`/`GROUP="g65"`. This is a personalized edit that shouldn't be committed to the shared repo as-is (every teammate will need their own values). Either git-ignore the file, commit it with placeholders, or keep it as a local-only override. The file-mode flip 100644 → 100755 is also part of the diff.
9. **LOW — `docker/#submit.sh#` is an emacs autosave.** Identical to `submit.sh` (verified by `diff`). It's currently `?? untracked`. Add to `.gitignore` (`*#*`, `.#*`) to avoid accidentally committing it.
10. **LOW — Two TODO/FIXME-adjacent hits.** Both are `RCP_GUIDE.md:83-84` (`hf_xxx`, `WANDB_API_KEY=xxx` token-placeholder examples). Neither is real technical debt.
11. **LOW — `huggingface_hub` is unpinned.** `docker/requirement_docker.txt:4` has no version constraint. Probably fine because the Hub API is stable, but worth pinning if merge code starts depending on a specific behavior.

No dead code, no competing eval implementations, no hardcoded paths beyond the standard `/scratch`, `/shared-ro`, `/shared-rw` mounts.

---

## Section 10 — `REPORT.md` context

**`REPORT.md` does not exist at the repo root.** (`find . -name REPORT.md` returns nothing.) No equivalent "decisions / commitments" doc exists either (no `NOTES.md`, no `TODO.md`, no `tasks/` directory). The closest equivalent is `USAGE.md` (the team-internal "shared-files" doc), which carries these load-bearing decisions for the group phase:

- **LoRA spec is locked at `r=32, α=64`**, all 7 linear projections (`USAGE.md:66-70`), justified by Shuttleworth et al. 2024 (full justification "lives in the team's literature review", not in this repo).
- **Thinking mode is committed ON** in the proposal — `<think>…</think>` reasoning traces are not optional (`USAGE.md:69`).
- **Output contract** — every assistant turn ends with `\boxed{...}` (`USAGE.md:71-72`).
- **Pre-merge alignment check scheduled ≈ May 27** (`USAGE.md:51`).
- **Phase 3 merge method** — DARE + AdaMerging (`USAGE.md:15`, `lora.yaml:10`).

There is **no commitment about Liger Kernel, batch size, optimizer, dataset choice, or how to handle `generation_config.json` at merge time** — `USAGE.md:35` explicitly lists `generation_config.json` as a "per-expert at training time; merge-time decision later" item, which is an open question carried into Stage 1.

---

## Artifacts written

`AUDIT_GROUP_PHASE.md` (this file). No other files modified.

Audit path: `/home/julienerbland/Documents/EPFL/Master/MA2/MNLP/emainelpe_group_model/AUDIT_GROUP_PHASE.md`

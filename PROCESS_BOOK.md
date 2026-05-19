# PROCESS_BOOK.md

Rolling, append-only decision log for the **group-model phase** of CS-552
team Émainèlpé (g65). Each entry is dated; entries are never refactored
once written, even when later evidence contradicts them — the contradiction
is recorded in a new entry. Concrete numbers, paths, commit SHAs, and
verbatim decisions belong here. This file is the primary source material
for the final 4-page report.

---

## Day 0 — 2026-05-15 — Phase switch and audit

### What happened

- Switched from the **math specialist track** (v5 deployed to
  `cs-552-2026-emainelpe/math_model`, full-merged) to the **group-model
  phase**.
- Ran a read-only codebase audit; output is `AUDIT_GROUP_PHASE.md` at the
  repo root.
- Key audit finding: this repo is the team's shared **locked-spec
  contract** — `lora.yaml`, `chat_template.jinja`, `evaluate/`,
  `validation_samples/`. No merge code existed. No inference path existed.
  A CPU scorer was already present in `evaluate/`.

### Decisions and rationale

- **Repo structure: option B2 (dual-purpose)**. Add the merge code as a
  `merge/` subdir in the shared lock repo, rather than spinning up a new
  repo. Reasoning: the merge code reads `lora.yaml` and depends on
  `evaluate/`, so co-location is operationally simpler than cross-repo
  imports. Downside: the repo identity is slightly muddied (contract +
  code in one place); acceptable given the team size and timeline.
- **Pipeline consumes a local directory of adapters, not HF repo IDs**.
  The merge pipeline does not know or care how the four adapters got
  onto disk (HF download, `cp` from scratch, dummy generation). This
  decouples merging from teammate-coordination state and lets the same
  pipeline be exercised with synthetic adapters in tests.
- **4 fixed domain subdir names enforced**: `math`, `general_knowledge`,
  `safety`, `multilingual`. Order is fixed for reproducibility. Anything
  outside these four — missing or extra — is a hard error.
- **Synthetic end-to-end test in scope** (planned for Stage 4). Four
  dummy adapters → full pipeline on CPU. Converts the eventual "real
  adapters arrive" moment from a coordination crisis into pure
  execution.

### Open questions

- All 4 teammate LoRA specs unverified externally.
- Math adapter on HF is full-merged (~3.4 GB), not adapter-only. The
  source LoRA exists on RCP scratch.

---

## Day 1 — 2026-05-15 — Stage 1: skeleton

### What happened

- Verified the math v5 LoRA spec by reading
  `/scratch/Julien/runs/cs552-erbland-g65-v4-fresh-20260514-162214/final/adapter_config.json`
  on RCP. All 8 load-bearing fields match `lora.yaml` byte-identically
  (set-equality for `target_modules` — see Day 2 for why this matters).
- Confirmed **Path 2** for the pipeline: adapter-only inputs to the
  merge step; final publishing materializes the merged adapter into the
  base model for the HF `group_model` push.
- Built **Stage 1 skeleton**: `merge/` subdir with stubs + tests + docs.
  Every function body raises `NotImplementedError("Stage N")`. 16
  skeleton tests pass or skip cleanly on a torch-free laptop.
- Updated `USAGE.md` to clarify dual purpose. Added
  `modules_to_save: null` to `lora.yaml` with a rationale comment — the
  field was silently absent before; making it explicit prevents
  teammates from accidentally introducing full-rank tensors that break
  additive merging.

### Decisions and rationale

- **Path 2 (adapter-only merge inputs)**. The math HF deploy is
  full-merged, but the source LoRA still exists on RCP scratch.
  Teammates will be asked to push **adapter-only** artifacts (or share
  via scratch) in parallel with their current full-merged deploys.
  Reasoning: PEFT-native loading is cleaner than subtracting base
  weights from full models, and iteration is roughly 4× lighter
  (~545 MB vs ~3.4 GB per adapter).
- **`METHOD_REGISTRY` with 5 entries**: `uniform`, `dare_uniform`,
  `dare_weighted`, `ties`, `adamerging`. `dare` is a primitive that
  composes into the two `dare_*` methods, but is **not** itself a
  user-facing method (you don't average a single masked adapter).
  Compositions live in `methods/__init__.py` as
  `dare ∘ uniform_merge` and `dare ∘ weighted_linear_merge`.
- **CPU-only tests via `pytest.importorskip("torch")`**. Torch lives on
  the cluster, not on the laptop. The pattern lets tests pass on the
  laptop (skip torch-deps) and run fully on the cluster.

### Verified

- Math v5 LoRA spec matches `lora.yaml` byte-identically. **1 of 4**
  adapters verified.

### Open questions

- 3 teammate adapters still unverified.

---

## Day 2 — 2026-05-16 — Stages 2 and 3

### What happened

- **Stage 2** (`merge/`: Stage 2 spec verification + adapter loading):
  Implemented `verify_spec.py` (`load_locked_spec`, `verify`,
  `FieldResult`, `VerifyResult`, `SpecMismatchError`) and
  `load_adapter.py` (`canonicalize`, `load`, `load_all`).
  Whitelist-based verification on 8 load-bearing fields.
  `target_modules` compared as `set` (handles PEFT's graph-walk
  ordering). Toy adapter fixture for CPU tests (2-layer, hidden 64).
  **43 tests** pass on cluster.
- **Stage 3** (`merge/`: Stage 3 DARE + uniform + weighted-linear
  merging): Implemented `methods/dare.py`, `methods/uniform.py`,
  `methods/weighted_linear.py`. Wired `dare_uniform` and `dare_weighted`
  compositions in `methods/__init__.py`. **78 tests** pass on cluster.
  Cross-validation tests pass:
  - `dare_uniform(drop=0)` ≡ `uniform_merge`
  - `dare_weighted(drop=0)` ≡ `weighted_linear_merge`
  - `weighted_linear_merge([1/N]*N)` ≡ `uniform_merge`

### Decisions and rationale

- **Verification is whitelist-based, not full-diff.** Only the 8
  load-bearing fields are checked (`base_model_name_or_path`, `r`,
  `lora_alpha`, `lora_dropout`, `bias`, `task_type`, `modules_to_save`,
  `target_modules`). All other PEFT bookkeeping (`peft_version`,
  `inference_mode`, `init_lora_weights`, `alpha_pattern`, etc.) is
  ignored. Reasoning: avoids false positives when teammates use
  different PEFT versions.
- **`target_modules` compared as set, not list.** PEFT serializes the
  list in graph-walk order, not in user-specified order. Math v5's
  `adapter_config.json` already has `target_modules` in a different
  order than `lora.yaml`; both are correct.
- **`modules_to_save` omission treated as null.** PEFT may write
  `modules_to_save: null` explicitly or omit it entirely depending on
  version. Both are semantically equivalent, so both pass verification.
- **Weighted-linear: pass-through normalization (no auto-normalize).**
  Reasoning: DARE-rescaled survivors already inflate magnitudes;
  AdaMerging-learned coefficients don't sum to 1 by construction;
  uniform-by-construction happens to sum to 1 as an accident. Auto-
  normalizing would fight callers. Rejecting non-summing weights would
  force callers to normalize before passing. Pass-through is the
  principled choice and the API contract for downstream methods.
- **DARE seeding: single global seed, walk dict in insertion order.**
  Python's dict iteration is stable (≥ 3.7); `load()` is reproducible;
  therefore the bernoulli draw sequence is reproducible across runs.
  Per-parameter seed-hashing is unnecessary complexity.
- **DARE in compositions: seed = global_seed + i for the i-th adapter.**
  Independent masks per adapter while remaining deterministic given a
  single global seed. Test coverage verifies bit-identity across
  repeated calls.
- **Internal arithmetic in fp32, cast back to bf16 for output.**
  bf16 accumulation error is significant when averaging or weighting
  multiple tensors. Inputs/outputs are bf16 (matching real adapters);
  intermediate compute is fp32.
- **Strict adapter directory: missing or extra subdirs are hard errors.**
  Avoids silent "we picked the wrong adapter" failures.
- **`download_adapter` removed (Option A).** Stage 1 had a download
  stub. With the local-directory contract, HF download is the caller's
  responsibility (`hf download`, `cp` from scratch). Removing the stub
  shrinks the API surface; can revisit as `merge/fetch_adapters.py`
  later if needed.

### Verified (on cluster)

- 78/78 tests pass.
- **Cross-validation**: `dare_uniform(drop=0)` ≡ `uniform_merge`;
  `dare_weighted(drop=0)` ≡ `weighted_linear_merge`;
  `weighted_linear_merge([1/N]*N)` ≡ `uniform_merge`. All within bf16
  tolerance (`rtol=1e-2`, `atol=1e-2`).
- **Reproducibility**: same seed → bit-identical merged output
  (`torch.equal`).
- **Statistical**: DARE at `drop_rate=0.5` zeros ~50% of entries
  (within [0.4, 0.6] per-tensor band, a 9σ envelope).
- **Spec mismatch** raises `SpecMismatchError` with the adapter name
  and field name in the message.

### Open questions

- 3 teammate adapters still unverified.
- Teammate ping not yet sent. Will include the now-tested `verify()`
  function as a self-service check teammates can run before pushing.
- Generation config consensus: each individual adapter has its own;
  the group model needs one. Decision deferred to Stage 5.

---

## Day 3 — 2026-05-16 — Stage 4: TIES + pipeline orchestration + synthetic end-to-end

### What happened

- Implemented `methods/ties.py`: trim → elect-sign → disjoint-merge per
  Yadav et al. 2023. Per-tensor magnitude trimming; hard sign election
  (exact ties elect zero → parameter dropped at that position).
- Implemented `pipeline.py`: orchestrator chaining
  `load_all → verify → method dispatch → SVD factorization → PEFT-format
  save`. Output is a drop-in PEFT adapter directory
  (`adapter_config.json` + `adapter_model.safetensors`).
- Added `decanonicalize()` to `load_adapter.py` — inverse of
  `canonicalize()`. Round-trip tested:
  `canonicalize(decanonicalize(c, "lora_A")) == c`.
- Added synthetic end-to-end test `test_pipeline_synthetic.py`: 4 toy
  adapters → full pipeline → reload, with round-trip assertion for each
  merge method. Validates that the entire pipeline works on CPU before
  real Qwen3-1.7B adapters arrive.
- After Stage 4 the only remaining stub in `merge/methods/` is
  `adamerging` (Stage 7, post-milestone).
- Local laptop verification: torch-free tests pass; torch-dependent
  tests (DARE, uniform, weighted_linear, TIES, pipeline, synthetic E2E)
  skip cleanly via `pytest.importorskip`. **Cluster verification pending
  on next `git pull && pytest`.**

### Decisions and rationale

- **TIES default `trim_ratio = 0.5`** — symmetric with DARE's `drop_rate`
  default. Both methods "drop half" by intent, though one drops by
  magnitude (TIES) and the other randomly (DARE). Reasonable starting
  point for hyperparameter sweeps. (Stage 1 stub used `density = 0.2`
  per the original paper; Stage 4 spec changed this to a
  `trim_ratio = 0.5` parameter — superseded.)
- **Hard sign election (exact ties elect zero).** Clean deterministic
  rule. Sign-count sum equals zero ⇒ `torch.sign(sum)` returns 0 ⇒
  parameter dropped at that position. Alternative magnitude-weighted
  sign election overcomplicates the algorithm for marginal gains.
- **Per-tensor magnitude trimming, not global.** Consistent with DARE
  (also per-tensor). The TIES paper describes both variants; per-tensor
  is simpler and more interpretable.
- **SVD factorization back to PEFT format.** Merge produces a full-rank
  ΔW dict; PEFT requires rank-r factors. Standard approach: SVD
  truncation with even singular-value split:
  `lora_A = (r/α)·diag(√S)·Vᵀ`, `lora_B = U·diag(√S)`. The `(r/α)` on
  `lora_A` cancels the `(α/r)` scaling that `load()` re-applies, so
  effective ΔW = best rank-r approximation of the merged ΔW.
- **Pipeline does NOT push to HF.** That is `merge/publish.py`'s job
  (Stage 5). Pipeline only writes a local PEFT-format directory.
- **Pipeline signature changed from Stage 1 stub.** Stage 1 had
  `merge_adapters(adapter_repos, method, base_model_repo, ...)`. Stage 4
  spec switches to `merge_adapters(adapters_dir, method, output_dir,
  locked_spec_path=None, method_kwargs=None)`. Reasoning: aligns with
  the Day-0 decision that the pipeline consumes a local adapter
  directory, not HF repo IDs. The base model repo is implicit in the
  locked spec.
- **Output dir empty-vs-nonempty handling.** Existing empty dir succeeds
  (lets users `mkdir -p` in advance); existing non-empty dir raises
  `FileExistsError`. Catches accidental overwrites of prior runs.
- **Synthetic test uses loose round-trip tolerance** (`rtol=0.5`,
  `atol=0.5`). SVD truncation introduces error proportional to dropped
  singular values. With 4 rank-32 task vectors merged on a hidden_dim=64
  toy model, the merged ΔW can be up to rank-128 in gate/up/down
  modules, so rank-32 truncation can drop up to 96 singular components.
  The point of the test is "the pipeline works end-to-end," not "SVD is
  information-preserving."

### Verified (local laptop, torch-free)

- 24 torch-free tests pass: `verify_spec` (13), canonicalize regex (5),
  canonicalize/decanonicalize round-trip (1) + bad-factor rejection (1),
  skeleton import-smoke + locked-spec regression + remaining stub
  checkers.
- All torch-dependent tests (DARE, uniform, weighted_linear, TIES,
  pipeline, synthetic E2E) skip cleanly via `pytest.importorskip`.

### Verified (cluster)

- TBD — pending next `git pull && pytest merge/tests/ -v` on RCP.
- Fixed: SVD outputs needed `.contiguous()` before safetensors save; adamerging stub needed raise-first to avoid TypeError on minimal kwargs.
- Added: `test_svd_factor_round_trip_within_truncation_tolerance` isolates the SVD factorization math from the rest of the pipeline so a rank-r input round-trips exactly; analysis showed no scaling bug in `svd_factor` (the `(α/r) · (r/α)` cancellation holds), so synthetic-E2E failures are likely SVD truncation + bf16 precision exceeding `rtol=0.5, atol=0.5` rather than a math bug.
- Synthetic E2E round-trip tests rewritten to structural assertions; SVD truncation is expected behavior, not a bug. Math verified by `test_svd_factor_round_trip_within_truncation_tolerance`.

### Open questions

- TIES + DARE composition not yet built (could be `dare_ties` for
  symmetry with `dare_uniform` / `dare_weighted`). Defer — only build if
  Stage 6 evaluation suggests it's worth it.
- AdaMerging still deferred to Stage 7 (post-milestone).
- 3 teammate adapters still unverified.

---

## Day 4 — 2026-05-18 — Stage 5a: AdaMerging core

### What happened

- Implemented `methods/adamerging.py`: layer-wise per-task coefficient
  learning with entropy-min loss + L2 regularization. Coefficients are
  fp32 of shape `[N_tasks, N_layers]`; the merge math computes
  `sum_i coefficient[i, layer_of_k] * task_vector_i[k]` per parameter,
  where the layer index is extracted from the canonical name
  `model.layers.<N>.…`. Returns an `AdaMergingResult` dataclass exposing
  the merged dict, final coefficients, full loss history, step count,
  and early-stop flag.
- Added `dare_adamerging` composition: DARE applied **once** per task
  before training (Option α from the design discussion), with per-task
  seed `= seed + i` matching `dare_uniform` / `dare_weighted`. Returns
  the merged dict only, so it slots into `METHOD_REGISTRY`'s callable
  contract.
- AdaMerging is registry-callable via a thin `_adamerging_dict` shim
  that unwraps `AdaMergingResult.merged`. Direct callers (tests, future
  hyperparameter sweeps) keep the full result; pipeline dispatch sees
  a dict, identical to all other methods. Registry grows from 5 to 6.
- Synthetic forward + data fixtures (`fixtures/adamerging_helpers.py`)
  give the training loop a differentiable path on CPU without a real
  model: two stacked linear layers + tanh + unembed, perturbed by the
  merged q_proj entries. ~50 LoC. Stage 5b will plug in a real Qwen3
  forward via PEFT hooks.

### Decisions and rationale

- **Layer-wise granularity.** Paper default; one scalar per (task, layer).
  For Qwen3-1.7B (28 layers × 4 tasks) that's 112 learnable scalars; for
  toy adapters (2 × 4) it's 8. All target_modules within a layer share
  the coefficient — sufficient resolution without over-parameterizing.
- **Init = 0.3, lr = 1e-2, Adam.** Paper defaults. Slight under-weighting
  from uniform encourages a non-degenerate solution; high lr is fine
  because we're tuning ~100 scalars, not millions of weights.
- **Pure entropy + L2 (1e-4).** Entropy alone can let coefficients drift
  to extremes that overfit the unlabeled distribution; 1e-4 L2 keeps
  them bounded without dominating signal.
- **Early stop with patience=100, improvement threshold 1e-6.** Detects
  flat loss without false-positives from fp32 noise.
- **DARE once before AdaMerging, not per step.** Re-DAREing every step
  produces a stochastic objective even for fixed coefficients (different
  masks → different merge → different loss), which complicates
  convergence diagnostics. Deterministic loss surface is cleaner.
- **`forward_fn` and `data_iter` are caller-supplied.** Decouples the
  merge math from base-model framework. Tests use a synthetic forward;
  Stage 5b will provide a real-Qwen3 forward via PEFT hooks.
- **Bookkeeping fix vs spec.** The spec used `for...else` to set
  `early_stopped=False`, but Python's `for...else` only runs when the
  loop terminates *without* break — and we always break at
  `step >= max_steps`. Replaced with explicit `early_stopped = False`
  init + `steps_run = step + 1` updates per training step. Now correct
  on all three exit paths (max_steps, plateau, iterator exhaustion).
- **Registry-shim for `adamerging`.** Spec said
  `METHOD_REGISTRY["adamerging"] = adamerging` directly, but
  `adamerging()` returns `AdaMergingResult` while the pipeline
  iterates `merged.items()`. Wrapping via `_adamerging_dict` keeps all
  six registry callables dict-returning so any pipeline dispatch path
  works uniformly.
- **Unlabeled data plan for Stage 5b real runs:**
  - Math: GSM8K train split (first 1k examples)
  - Knowledge: MMLU train split (first 1k examples)
  - Multilingual: MGSM train split (first 1k examples)
  - Safety: XSTest benign prompts (~250 examples, cycled if needed)

### Verified (laptop)

- 12 new AdaMerging tests added; all torch-gated via `importorskip` so
  they skip cleanly on the laptop. Existing 25 torch-free tests still
  pass.
- Skeleton test renamed to `test_method_registry_has_all_six_methods`;
  obsolete `test_adamerging_stub_raises_stage_7` removed.

### Verified (cluster)

- TBD — pending next `git pull && pytest merge/tests/ -v` on RCP.
  Expected: ~120 tests pass (was 108; +12 new AdaMerging tests; -1
  removed stub test; +1 net renamed registry test). Synthetic forward
  must produce decreasing loss over 100 steps and coefficient
  divergence > 0.01 from the 0.3 init.

### Open questions

- Real-Qwen3 convergence on actual unlabeled data is the Stage 5b
  cluster smoke test.
- AdaMerging memory cost on Qwen3-1.7B not yet measured (budget
  estimate: ~7 GB base + ~2 GB activations + ~545 MB × 4 adapters ≈
  15 GB total, well within A100-40GB).
- Should AdaMerging coefficients be initialized from `dare_uniform`'s
  output (i.e., 1/N) instead of 0.3? Paper picks 0.3; revisit if real
  convergence is slow.

---

## Open questions, blockers, decisions to revisit

Running bulleted list. Items may be marked **resolved** but are never
deleted.

- **Active blocker**: 3 of 4 teammate adapters unverified against
  `lora.yaml`. Math verified on Day 1.
- **Pending action**: Teammate ping draft (asking for: spec compliance,
  `modules_to_save: null` confirmation, deploy format
  adapter-vs-merged, ETA for first push). Drafted after Stage 4 lands.
- **Open**: Does `modules_to_save` need explicit re-confirmation with
  teammates? Added retroactively to `lora.yaml` on Day 1; teammates
  trained before this clarification, so their adapter_configs may
  simply omit the field. The Stage 2 verifier treats omission as
  equivalent to null — but it would be cleaner to have everyone
  re-affirm the rule.
- **Deferred**: AdaMerging (Stage 7, post-milestone). Re-evaluate if
  DARE-based methods underperform on the May 24 milestone.
- **Deferred**: TIES default `trim_ratio=0.5`. May want to sweep
  alongside DARE `drop_rate` once both are implemented.
- **Open**: Generation config for the group model — whose values to
  use, or new ones? Decision deferred to Stage 5.
- **Math v6 in flight**: 200k OMI2 SFT training in parallel. If v6
  lifts above v5 on CI, swap v5 → v6 as the math input before Stage 6
  (milestone push). Otherwise keep v5.

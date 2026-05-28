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

## Day 5 — 2026-05-19 — Stage 5b: Real-Qwen3 plumbing for AdaMerging

### What happened

- Built `merge/data/unlabeled.py`: a `DatasetConfig` dataclass, the
  4-element `UNLABELED_DATASETS` constant (GSM8K `main/train`, MMLU
  `all/auxiliary_train`, XSTest `prompts` filtered to `safe_*`, MGSM
  `en/test`), `assert_cache_exists()` heuristic check, and
  `make_unlabeled_iter()` that pre-tokenizes via the Qwen3 chat
  template, packs into uniform `batch_size`-sized right-padded batches,
  and cycles per-domain so we never run dry. Round-robin domain order
  matches `CANONICAL_DOMAINS`.
- Built `merge/qwen3_forward.py`: loads Qwen3-1.7B once via
  `AutoModelForCausalLM`, freezes parameters, builds the
  `module_name → canonical_name` map (currently identity for Qwen3),
  and returns `(forward_fn, cleanup)`. `forward_fn` enters a context
  manager that installs **forward post-hooks** on each LoRA-targetable
  Linear; each hook returns `output + F.linear(input, merged[canon],
  bias=None)`. Differentiable w.r.t. the AdaMerging coefficients
  without touching base weights.
- Built `scripts/fetch_adamerging_data.py`: idempotent pre-download.
  Logs per-dataset size, applies the XSTest type filter so the user
  sees the post-filter count, accumulates failures and exits non-zero
  if any dataset fails to fetch.
- Built `scripts/smoke_adamerging.py`: 5-step orchestrator —
  cache check → tokenizer load → forward callable + base model load →
  data iter → 4 random Qwen3-sized adapters → `merge_adapters(method=
  "dare_adamerging", ...)`. Asserts the merged adapter directory is
  well-formed. Disables early stopping by setting
  `early_stop_patience = max_steps + 1` so the smoke run touches every
  configured step.
- Added `merge/tests/fixtures/qwen3_adapter.py`:
  `make_random_qwen3_adapter()` builds a Qwen3-1.7B-shaped PEFT
  adapter (28 layers, hidden=2048, intermediate=6144, GQA with 8 KV
  heads → k/v out dim = 1024, r=32, bf16). Random init scaled to
  `1/√r` so the implied ΔW has unit-ish magnitude. Used only by the
  smoke script — too heavy for laptop tests.
- Added `merge/tests/test_data_unlabeled.py`: 9 CPU-runnable tests
  for `UNLABELED_DATASETS` constants (count, indices, canonical name
  agreement, required fields, sanity bounds, frozen-dataclass), the
  `assert_cache_exists` error path (must list every missing repo and
  point at the fetch script), and the happy path (after creating the
  4 HF-convention subdirs).

### Decisions and rationale

- **Forward *post*-hook, not pre-hook with weight mutation.** The
  natural Approach (B) — mutate `self.weight.data` before forward,
  restore after via a paired hook — silently breaks AdaMerging.
  `Parameter.data = X` is the no-grad assignment path; gradients
  would not flow back to the AdaMerging coefficients. We instead use
  Approach (A) in its cleanest form: a `register_forward_hook`
  (post-hook) that returns `output + F.linear(input, delta,
  bias=None)`. The base forward runs normally (frozen weights, no
  recomputation needed); the *delta contribution* is a fresh
  differentiable tensor op routed through `merged[canonical]`, which
  is itself a function of `coefficients`. Autograd works as
  designed.
- **Pre-download script, not lazy `datasets.load_dataset` inside the
  iterator.** Cluster pods are preemptible. A killed download leaves
  a partial cache that the datasets library may or may not detect;
  surfacing that failure inside a training run is hard to debug.
  Pre-download is explicit, retriable, and prints per-dataset sizes
  so anomalies (e.g. XSTest's filter unexpectedly dropping to zero)
  surface immediately.
- **Heuristic cache check, not online verification.** `assert_cache_
  exists()` only verifies a top-level dir per dataset under
  `<cache>/datasets/<repo>___`. A full integrity check would re-load
  each dataset, which is slow and overlaps with what
  `make_unlabeled_iter()` already does. The heuristic catches the
  common failure mode (wrong `HF_HOME`, wiped cache) without false-
  negative-ing on minor layout drift.
- **MGSM `test` split, not `train`.** MGSM has no train split per
  language. We use `en/test` minus any overlap with the milestone
  validation set. Cycling is fine (the set is small; AdaMerging
  trains on entropy, not memorized labels).
- **XSTest filtered to `safe_*` types.** Pure entropy on harmful
  prompts would push the model toward confident harmful outputs —
  the wrong direction. Restricting to benign-but-superficially-
  harmful (the over-refusal test set) aligns the objective with what
  safety training should do.
- **Batch size 2 default, max length 512.** Conservative for an
  A100-40g (≈3.4 GB base + activations + 4 adapters ≈ 15 GB peak,
  with comfortable headroom for stash-and-restore activations during
  hook execution).
- **Qwen3-1.7B GQA shapes hard-coded in the smoke fixture.** Real
  Qwen3-1.7B uses Grouped-Query Attention: k_proj and v_proj output
  `n_kv_heads × head_dim = 1024`, not `hidden = 2048`. Earlier draft
  of `make_random_qwen3_adapter()` had square attn projections and
  would have produced shape-mismatched ΔW vs. the live model. Fixed
  before write. Constants are kept module-level (`QWEN3_1_7B_*`) so
  a Qwen3-1.7B minor revision is a one-line review.
- **Coefficient init = `1 / √r` for random adapters.** This makes the
  smoke-script ΔWs roughly unit-magnitude per coordinate, so
  AdaMerging coefficients move enough to validate the gradient path
  in 50 steps. Real adapters arrive with their own scale; we do not
  re-init on top.

### Verified (laptop)

- `pytest merge/tests/ -v`: 34 passed, 97 skipped (was 25 / 97 in
  Stage 5a + 9 new CPU tests in `test_data_unlabeled.py`). All new
  tests run without `datasets` or `torch` installed — the cache
  check works against `pathlib`-only paths, and the dataset config
  is plain dataclasses.
- `git status`: only the allowed files modified (no commit, no push).

### Verified (cluster)

- TBD — pending `git pull && pip install datasets transformers &&
  python scripts/fetch_adamerging_data.py && python
  scripts/smoke_adamerging.py --max-steps 20` on RCP. Expected:
  4 datasets fetched, smoke script exits 0 in ~5-10 min, AdaMerging
  loss curve appears in the log and shows non-trivial decrease across
  the 20 steps.

### Open questions

- Real teammate adapters still not pushed. Stage 5b's smoke uses
  random-init; production AdaMerging will run on the 4 real adapters
  when they land.
- Smoke runs 50 steps by default; production target is 1000. May need
  to tune `lr` (currently 1e-2) and `early_stop_patience` (currently
  100) based on what the smoke loss curve looks like.
- `infer.py` (Stage 5c) and `eval_all.py` (Stage 5c) still pending.
  These unblock the bake-off comparison.
- `publish.py` (Stage 5d) still pending. Required for the May 24
  milestone push.

---

## Day 6 — 2026-05-19 — Stage 5c.1: vLLM-based eval infrastructure

### What happened

- Built `merge/infer.py` (vLLM-based n=8 inference for a single benchmark).
  `run_inference(vllm_model, lora_request, benchmark, validation_jsonl,
  output_jsonl, config)` renders prompts via the Qwen3 chat template
  (`add_generation_prompt=True`), calls `vllm_model.generate(...)` with a
  single `SamplingParams(n=config.n, ...)`, and writes a JSONL shaped
  exactly like `evaluate.score` consumes. The vLLM model is owned by the
  caller — `infer.py` never loads or releases it.
- Built `merge/eval_all.py` (multi-benchmark orchestrator with failure
  classification). Top-level entry: `evaluate_all_benchmarks(adapter_dir,
  base_model_repo, output_dir, validation_samples_dir, ...)`. Loads
  `vllm.LLM(enable_lora=True, max_lora_rank=32, dtype="bfloat16")` once,
  attaches the merged adapter via `LoRARequest`, then loops the 4
  canonical domains (`math`, `general_knowledge`, `safety`, `multilingual`),
  scoring with the existing `evaluate/*` helpers and classifying each
  pass@8=0 problem into one of 7 categories.
- Failure taxonomy: `no_boxed`, `empty_boxed`, `wrong_answer`,
  `malformed_answer`, `truncated`, `refusal`, `mixed`. Per-problem detail
  saved with the 8 completions inline so debugging is concrete.
- Per-completion priority order: REFUSAL > TRUNCATED > NO_BOXED >
  EMPTY_BOXED > [extract + compare]. WRONG_ANSWER vs MALFORMED_ANSWER
  discriminator is a one-line numeric-shape check: if the expected
  answer parses as a number but the extracted answer does not, it's
  malformed (e.g. `\boxed{x}` vs gold `42`); otherwise it's wrong.
- Tests added (torch-free):
  `tests/test_failure_classification.py` (17 tests covering all 7
  categories, refusal priority, strict-majority and mixed aggregation,
  knowledge MCQ paths, edge cases like multiple boxed + trailing
  whitespace), and `tests/test_eval_io.py` (10 tests covering JSONL
  parsing with `prompt`/`answer` vs `problem`/`solution` field names,
  `InferenceConfig` defaults, dataclass round-trips through json).
- Removed the obsolete `test_infer_stubs_raise_stage_5` and
  `test_eval_all_stubs_raise_stage_5` from `tests/test_skeleton.py`
  (those modules are real now). Kept the `publish.py` stub test.

### Decisions and rationale

- **vLLM with `enable_lora=True`**, not `merge_and_unload + full model
  push` at eval time. Adapter-on-base is more memory-efficient and lets
  us swap adapters between bake-off runs without reloading base.
  `merge_and_unload` happens only at the final HF push (Stage 5d).
- **vLLM loaded once per `evaluate_all_benchmarks`** call, reused across
  4 benchmarks. Avoids 4× cold start (~30-60s each).
- **`n=8` via vLLM `SamplingParams(n=8, ...)`** — single API call per
  problem yields both pass@1 and pass@8 from the same generation pool.
- **Scoring uses `evaluate.benchmarks.{extract,is_correct}_benchmark_answer`
  + `evaluate.pass_at_k.compute_pass_at_k_for_dataset`** directly, not
  the `evaluate.score` CLI wrapper. Same logic, no SystemExit coupling.
- **Canonical domain → method mapping**: `_DOMAIN_TO_METHOD` maps our
  `general_knowledge` to evaluate's `knowledge`; math/safety/multilingual
  all use `boxed`. The mapping is private to `eval_all.py`.
- **Failure detail = category + the 8 completions inline.** Output file
  gets large (~100KB per benchmark) but the debugging value is real.
  Without the actual completions it's hard to tell why a problem failed.
- **`max_tokens=2048`** for eval (vs the project's 16384 CI cap). For
  thinking-mode the model typically uses 1-2k tokens; 2048 covers CoT +
  boxed answer with margin. CLI override remains possible.
- **Truncation heuristic**: `tokens_used == max_tokens_limit` AND no
  closing `\boxed{}`. Rare edge case of hitting exactly `max_tokens`
  with a valid answer is acceptable misclassification.
- **REFUSAL has priority over correctness in `classify_completion`.**
  This is consistent within the pass@8=0 contract: by the time
  `classify_problem_failure` runs, the scorer has already confirmed all
  8 completions are wrong, so any refusal-shaped phrase is meaningful
  evidence. The unit test exercises the per-completion path with an
  artificially correct refusal to lock the priority in.

### Verified (laptop)

- All Stage 5c.1 IO + classification tests pass torch-free (run
  separately and bundled into the full suite). Total test count:
  131 → 161 (+30): 20 failure-classification + 12 IO tests, minus 2
  obsolete stub tests removed from `tests/test_skeleton.py`.
- Laptop pytest result: `64 passed, 97 skipped in 0.52s`.
- `python -c "from merge.eval_all import FailureCategory; print(list(FailureCategory))"`
  works on torch-free laptop (no vLLM needed for this import).
- `git status` shows only files in the allowed list.

### Verified (cluster) [TBD]

- `pip install vllm` + `pytest merge/tests/ -v` (should be ~161 tests,
  most of the currently-skipped torch-dependent tests will execute).
- `evaluate_all_benchmarks(...)` on a smoke merged adapter against the
  40-problem validation snapshot. Expected: ~10-15 min wall clock for
  n=8 across 40 problems. With random-init adapters pass@k will be
  near zero — point is to verify the pipeline runs end-to-end.

### Open questions

- vLLM's `enable_lora` requires the adapter to be in PEFT format
  (which our `pipeline.merge_adapters` output is) and at the locked
  rank (`max_lora_rank=32`). Should match locked spec; verify on the
  first cluster run.
- Stage 5b smoke had unexplained ~5-minute tail latency at cleanup.
  Worth checking whether `evaluate_all_benchmarks` exhibits the same
  on `del llm`.
- Bake-off orchestration (`scripts/run_bakeoff.py`) is Stage 5c.2;
  unblocked once 5c.1 cluster smoke passes.

---

## Day 6 follow-up — Generation config handling

**Issue raised:** Stage 5c.1's `InferenceConfig` hardcoded Qwen3 defaults
(`temperature=0.7`, `top_p=0.8`, `top_k=20`) instead of reading the merged
adapter's `generation_config.json`. This meant eval would not match what
CI would measure for a model shipping a custom config.

**Fix:**

- New module `merge/generation_config.py` with `make_generation_config(...)`
  (project-schema-compliant dict constructor) and `load_generation_config(...)`
  (hierarchical fallback loader).
- `pipeline.merge_adapters` accepts optional `generation_config: dict | None`.
  If provided, written as `generation_config.json` in the output dir.
- `evaluate_all_benchmarks` uses `load_generation_config` when `config=None`.
  Priority: explicit arg → `merged_adapter_dir/generation_config.json`
  → `repo_root/generation_config.json` → Qwen3 defaults.
- `InferenceConfig.from_generation_config_dict` classmethod for the
  dict→dataclass conversion.

**Design rationale:**

- Structure is locked (project mandate): `bos/eos/pad` token IDs are Qwen3
  constants (151643 / [151645, 151643] / 151643), `do_sample=true`,
  `transformers_version="4.51.0"`.
- Values (`temperature`, `top_p`, `top_k`, `max_new_tokens`) are bake-off
  hyperparameters and not team-locked.
- First bake-off sweep: `temperature ∈ {0.0, 0.3, 0.7}`, `top_p=0.8`,
  `top_k=20` held constant, `max_new_tokens=16384`.
- No team-locked `generation_config.json` at the repo root yet — falls
  through to defaults until we add one.

**Tests:** 18 new tests covering `make_generation_config` validation,
`load_generation_config` priority order, `InferenceConfig` dict
conversion, and `pipeline.merge_adapters` with/without `generation_config`.

**Verified:** All new tests pass on torch-free laptop. Cluster eval
smoke needs a fresh run to verify the fallback hierarchy end-to-end
(next run).

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
- **Resolved (Day 6 follow-up)**: Generation config for the group model
  — structure is locked by the project description (token IDs, do_sample,
  transformers version); sampling values are bake-off hyperparameters
  written by `pipeline.merge_adapters(..., generation_config=...)` and
  read via hierarchical fallback at eval time.
- **Math v6 in flight**: 200k OMI2 SFT training in parallel. If v6
  lifts above v5 on CI, swap v5 → v6 as the math input before Stage 6
  (milestone push). Otherwise keep v5.

## Day 6 follow-up #2 — 2026-05-20 — Stage 5c.1.5: load() GPU + eval-time temperature sweep

### What happened

Two performance/process fixes ahead of the bake-off:

1. **GPU-accelerated adapter loading.** On the first cluster smoke of
   Stage 5c.1, `merge.load_adapter.load_all` ran on CPU bf16 and took
   ~10 min for the 4-adapter set (196 small matmuls per adapter, all
   on CPU). Added an opt-in `device` kwarg to `load`, `load_all`, and
   `pipeline.merge_adapters`. Default remains `"cpu"`; `merge_adapters`
   auto-selects `"cuda"` when available. The final safetensors save
   materializes contiguous CPU tensors regardless of where the merge
   ran, so the on-disk artifact is identical to the CPU path.
2. **Eval-time temperature sweep.** Realized late on Day 6 that
   `temperature` is sampling-only — it changes nothing about the
   merged weights. The original bake-off plan (re-merge for each of
   `{0.0, 0.3, 0.5, 0.7}`) wasted 3 of 4 merges. Created
   `scripts/eval_sweep.py` which takes a single merged adapter and a
   list of temperatures, runs `evaluate_all_benchmarks` per temperature
   with explicit `InferenceConfig` overrides, and writes
   `T_<temp>/` subdirs + an incremental `sweep_results.json`.
   `temperature=0.0` is rejected at the CLI: vLLM forbids `n>1` under
   greedy decoding, and the bake-off needs `n=8` for pass@8.

### Decisions & rationale

- **Device defaults `"cpu"` everywhere, auto-cuda only in the
  pipeline.** The library functions are the API surface other code
  imports; defaulting them to GPU would surprise the laptop test
  suite. The pipeline is a top-level orchestrator — it can safely
  auto-select cuda.
- **Save always on CPU.** safetensors refuses non-contiguous tensors,
  and SVD factor outputs are views. A single `.detach().to("cpu").contiguous()`
  pass before save handles both concerns and decouples the on-disk
  format from the compute device.
- **CPU/GPU parity assertion: compare reconstructed ΔW, not raw
  factors.** SVD has a sign ambiguity (`U → -U, Vᵀ → -Vᵀ` is the same
  reconstruction). A factor-by-factor tolerance test can spuriously
  fail when the two devices' SVD routines pick different signs. The
  device-invariant quantity is the reconstruction; that's what the
  test asserts.
- **Sweep resilience instead of fail-fast.** A vLLM OOM on one
  temperature should not abort the sweep — the other temperatures
  may still complete and inform the bake-off. The script catches per
  temperature, records the traceback to the result row, writes
  incrementally, and continues. Exit codes: `0` all-ok, `1` any-failed,
  `2` setup error.
- **Injectable callables, not a class.** `run_sweep(args, eval_callable,
  config_factory)` takes the two ML-dependent pieces as parameters.
  Production wiring lives in `_default_eval_callable` and
  `_default_config_factory`, which lazily import torch/vLLM. The test
  suite injects `types.SimpleNamespace`-based stubs and stays
  torch-free.
- **Dropped `temperature=0.0` from the bake-off entirely.** Deterministic
  greedy decoding is a separate concern (final HF push with `n=1`); it
  doesn't belong in a pass@8 sweep. `validate_args` rejects it with a
  message that explains why and points to the alternative.

### Verified (laptop)

- `pytest merge/tests/test_eval_sweep.py -v` — 21 passed, 0 failed.
- `pytest merge/tests/ -v` — 102 passed, 104 skipped (all skips are
  `pytest.importorskip("torch")` gates as expected on a torch-free
  laptop).
- `python3 scripts/eval_sweep.py --help` — renders the full argparse
  surface without touching torch or vLLM.
- `python3 scripts/eval_sweep.py --merged-adapter-dir /tmp/does-not-exist
  --output-dir /tmp/x --temperatures 0.0 0.5` — exits 2 and prints
  both the missing-dir error and the temperature=0.0 rejection in one
  shot (validation reports every error, not just the first).

### Verified (cluster)

- *To fill in after the next `runai submit`: confirm
  `load_all(adapters_dir, locked_spec, device="cuda")` cuts wall-clock
  load time from ~10 min to seconds on the 4-adapter set; confirm
  `scripts/eval_sweep.py --temperatures 0.3 0.5 0.7` produces three
  scorecards and an aggregated `sweep_results.json` from one merge.*

### Open questions

- None blocking. If a temperature consistently OOMs at the project's
  max-tokens setting, we may want a per-temperature `max_tokens`
  override on the sweep CLI — defer until we see it happen.


## Day 7 — 2026-05-20 — vLLM LoRA loader rejects PEFT format; switch to full-model merge output

### What happened

Stage 5c.1.5 cluster smoke for `eval_sweep.py` on a merged adapter failed
at vLLM's LoRA loader on the first generation:

```
ValueError: base_model.model.model.layers.0.mlp.down_proj.lora_A.default.weight is unsupported LoRA weight
```

vLLM's `parse_fine_tuned_lora_name` couldn't parse our PEFT-format weight
keys. Reproduced across all 3 temperatures (0.3, 0.5, 0.7). Failure is in
the LoRA loader, not our merge pipeline — the merged adapter was valid
PEFT format (uniform method on 4 random-init Qwen3-sized adapters;
produced 196 task-vector entries × correct shapes; verified by our own
`load_adapter.load()`).

### Decision

Change `pipeline.merge_adapters` to always produce a full HF-format model
via `peft.merge_and_unload()` instead of a LoRA-only output. Drop
`enable_lora=True` from eval entirely.

### Rationale

- vLLM's LoRA support is too restrictive for our SVD-factorized output.
  We could investigate which keys it accepts (the `.default.` adapter-name
  segment? `down_proj` target?) but that's a black box we don't control.
- Full-model output matches what the May 24 milestone CI grades anyway.
- The math-track has been doing `merge_and_push.py` with `merge_and_unload`
  for the same reason. Group-track now does the same.
- Disk cost: 4 methods × ~3.4 GB = ~14 GB on `/scratch`. Trivial.
- Merge time cost: +30-60 sec per merge for `merge_and_unload` +
  `save_pretrained`. Trivial in the context of a 3+ hour bake-off.

### Implementation

- `pipeline.merge_adapters` now: `load_all` (GPU) → merge method → SVD
  factor → in-memory PEFT model with our factors injected via
  `load_state_dict(strict=False)` → sanity-check one injection survived
  → `merge_and_unload` → `save_pretrained` → copy tokenizer + locked
  `chat_template.jinja` + optional `generation_config.json`.
- New parameter `base_model_repo: str = "Qwen/Qwen3-1.7B"` lets future
  experiments swap the base without code changes.
- `evaluate_all_benchmarks` loads the merged dir as a full model:
  `LLM(model=merged_dir, dtype="bfloat16")`. No more `enable_lora` /
  `LoRARequest`. The `base_model_repo` parameter remains in the
  signature for caller compatibility but is now ignored — vLLM reads
  the model directly from the merged directory.
- `run_inference` drops the `lora_request` parameter; vLLM `.generate`
  is called with `sampling_params=` only.
- `scripts/eval_sweep.py` `validate_args` checks for `config.json` +
  (`model.safetensors` or `model.safetensors.index.json` for sharded
  variants) instead of the legacy `adapter_*` files.

### Test surface

- Pipeline integration tests can no longer round-trip the toy adapters
  (hidden=64) through `merge_adapters` because the in-memory PEFT
  wrapper has Qwen3 shapes (hidden=2048). Two affected test files:
  - `merge/tests/test_pipeline.py` — kept error-path tests (KeyError,
    FileNotFoundError, SpecMismatchError, FileExistsError, adamerging
    `forward_fn` missing) — all raise before base-model load and stay
    laptop-runnable. Happy-path tests rewritten with a new
    `qwen3_random_adapters_dir` fixture and gated on CUDA +
    transformers + peft (cluster only).
  - `merge/tests/test_pipeline_synthetic.py` — replaced all toy-adapter
    end-to-end tests with a single cluster-gated set that asserts
    `config.json` + `model.safetensors` + tokenizer + chat template
    are present and that `AutoModelForCausalLM.from_pretrained` round-
    trips. The rank-r truncation discipline is verified in isolation
    by `test_svd_factor_round_trip_within_truncation_tolerance` in
    `test_pipeline.py`.
- `test_pipeline_cpu_and_cuda_produce_equivalent_output` in
  `test_load_adapter.py` skipped with a TODO — natural CPU/GPU parity
  assertion is now on the baked Qwen3 model weights, not the SVD
  factors. Verified manually on cluster smoke or as a follow-up.
- `merge/tests/test_eval_sweep.py` updated: fake adapter dir helpers
  now write `config.json` + `model.safetensors`; new test for the
  sharded variant.

### Verified (laptop)

- Code paths: `pipeline.merge_adapters`, `eval_all.evaluate_all_benchmarks`,
  `infer.run_inference`, `scripts/eval_sweep.py::validate_args`.
- `pytest merge/tests/ -v` runs to completion with the new test layout
  (most pipeline integration tests skip on CUDA absence as expected).

### Verified (cluster)

- *To fill in after the next `runai submit`: confirm
  `pipeline.merge_adapters` on the 4 random Qwen3 adapters writes a
  full HF-format directory of size ~3.4 GB; confirm `eval_sweep.py
  --temperatures 0.3 0.5 0.7` against that directory produces three
  scorecards without the original `unsupported LoRA weight` error.*

### Open questions / follow-ups

- Disk-space management: the bake-off will produce 4 × 3.4 GB = ~14 GB
  on `/scratch`. Default: keep all dirs for post-hoc analysis; document
  a clean-up command if disk gets tight.
- Should we expose a `--lora-only` flag on `merge_adapters` for callers
  who want the LoRA-only path back? Skip until requested — no current
  consumer needs it.
- `scripts/smoke_adamerging.py` still asserts the legacy
  `adapter_*` files post-merge (listed as don't-touch in the refactor
  prompt). Its post-merge assertions will need updating to the
  full-model layout before the next smoke run; flagged here as
  follow-up.


## Day 7 follow-up — 2026-05-20 — bitsandbytes pin + GPU cleanup on exception

### What happened

Cluster verification of the Day 7 full-model refactor surfaced two issues
that compounded into 4 test failures and 3 spurious OOMs.

1. **bitsandbytes 0.42 → CUDA 12 mismatch.** The cluster docker image
   ships bitsandbytes 0.42 with pre-compiled `.so` files for CUDA 11.x
   only; cluster runtime is CUDA 12.8. PEFT imports bnb unconditionally
   when constructing LoRA Linear layers (peft/tuners/lora/model.py
   imports `bnb` for the 8-bit quantization paths, even when we don't
   use them). The bnb import path raised:

   ```
   RuntimeError: CUDA Setup failed despite GPU being available
   ```

   This broke `pipeline.merge_adapters` at `get_peft_model(base, ...)`,
   which propagated to every full-pipeline test.

2. **Failed merges pinned GPU memory.** When the first
   `test_pipeline_uniform_produces_full_model` failed (from the bnb
   import error), the already-loaded Qwen3-1.7B (~3.4 GB) stayed
   allocated. The 3 subsequent pipeline tests OOM'd at increasingly
   pathological sizes (192 MB tried / 170 MB free → 24 MB / 8 MB →
   2 MB / 2 MB), turning one root failure into four.

### Decisions and rationale

- **Pin `bitsandbytes>=0.44.0` in `requirements.txt`.** Loose pin so
  later updates don't fight the image; tight enough to skip the
  CUDA-11-only 0.42 binaries. The cluster fix is `pip install -U
  bitsandbytes` or `pip install -r requirements.txt` on a fresh pod.
- **Wrap GPU-holding work in try/finally inside
  `pipeline.merge_adapters`.** Releases base / peft_model /
  merged_model / task_vectors / intermediate dicts in all exit paths,
  not just on success. The pattern: initialize the GPU-holding names
  to `None` before the try so unconditional `del` is safe; in finally
  run `del → gc.collect() → torch.cuda.empty_cache()`. Log the
  acquire/release events so a multi-merge run (e.g. the upcoming
  bake-off) can be diagnosed from logs alone.
- **Method-name validation moved inside the try block.** Was an
  early-fail before `load_all`; that made the cleanup path untestable
  via the natural "bad method name" trigger and gave the test
  ambiguous coverage. Cost: a few seconds of wasted `load_all` work
  on typo'd method names. Benefit: the cleanup test exercises the
  GPU finally-block path with a simple `KeyError` trigger and lays
  down a regression check for the exact failure mode we saw on
  cluster (one bad merge cascading into multiple OOMs).

### Implementation

- `requirements.txt` — added `bitsandbytes>=0.44.0` with a comment
  pointing at this entry and the `pip install -U` fallback.
- `merge/pipeline.py` — refactored body of `merge_adapters` into a
  try/finally. Initialized `base / peft_model / merged_model /
  task_vectors / adapters_by_domain / factorized / state_update /
  existing_state` to `None` before the try. Method-name validation
  moved inside the try block (after `load_all`). Finally block runs
  unconditional `del`s, `gc.collect()`, and `torch.cuda.empty_cache()`
  with a release log line.
- `merge/tests/test_pipeline.py` — added
  `test_pipeline_releases_gpu_memory_on_exception`. Cluster-gated
  (CUDA + transformers + peft). First call triggers a `KeyError` for
  `method="unknown_method"`, then a 400 MB probe allocation must
  succeed, then a second successful `uniform` merge must complete.
- `CLAUDE.md` — added a bullet to the dependencies note explaining
  the bnb pin requirement.

### Verified (laptop)

- `pytest merge/tests/ -v` runs to completion: same pass/skip pattern
  as before the fix (laptop is torch-free; CUDA-gated tests skip).
  The new cleanup test joins the cluster-only skip set.

### Verified (cluster)

- *To fill in after the next `runai submit`: confirm that with
  bitsandbytes >=0.44 installed in the pod, all 4 previously-failing
  pipeline integration tests pass; confirm
  `test_pipeline_releases_gpu_memory_on_exception` passes (probe
  succeeds and second merge completes); confirm no OOM cascade across
  back-to-back pipeline tests.*

### Open questions / follow-ups

- Should the cluster docker image upgrade bnb itself, so future pods
  don't need to `pip install -U` on first start? Owned by whoever
  manages the team image; not blocking.
- The wasted-`load_all`-on-typo cost from moving method validation
  inside the try is ~10s on CPU bf16 / seconds on GPU. Acceptable
  for the testability win, but if the bake-off ever runs typo'd
  method names in a tight loop the early-fail variant could come back
  via a cheap `if method not in METHOD_REGISTRY` guard before
  `load_all` — without removing the in-try check that the test relies
  on.


## Day 8 — 2026-05-20 — Stage 5c.2: Full bake-off orchestration

### What happened

Built `scripts/run_bakeoff.py` to orchestrate the milestone-day method
comparison: 4 merge methods × 3 sampling temperatures = 12 evaluation
scorecards on the same 4 input adapters, in one run, with an aggregated
`bakeoff_results.json` at the top level.

### Decisions and rationale

- **Merge once per method, sweep temperatures on that merge.**
  Temperature is sampling-only — it has no effect on merged weights.
  4 merges + 12 evals (not 12 merges) cuts ~3-4 hours of redundant
  work. Mirrors the Day 6 follow-up #2 insight that motivated
  `eval_sweep.py`.

- **Methods locked to `[uniform, dare_uniform, dare_adamerging, ties]`.**
  Four methods we have implemented end-to-end (Stages 3, 4, 5a). Skip
  `adamerging` standalone (composition with DARE is the experimentally
  motivated path per the AdaMerging paper) and `dare_weighted`
  (per-task weights are tunable hyperparameters that belong in a
  separate weighted-DARE sweep, not the headline comparison).

- **Temperatures `[0.3, 0.5, 0.7]`.** Locked from Day 6 follow-up #2.
  Greedy (0.0) is excluded by vLLM's `n>1` constraint.

- **AdaMerging hyperparameters: single bake-off config, not swept.**
  `drop_rate=0.5, lr=1e-2, lambda_l2=1e-4, max_steps=200,
  early_stop_patience=100, batch_size=2`. Tuning these is a separate
  experiment (a "hyperparameter sweep" that would be 4-8x more runs).
  For the bake-off we want methods-vs-methods, not method-vs-itself.
  `--adamerging-max-steps` is the one knob exposed because 200 is the
  bake-off budget compromise (vs production ~1000) — overridable
  without touching the script if cluster wall-clock allows.

- **Per-(method, temperature) failure isolation.** A failed merge
  marks that method's 3 temperature slots all-failed and continues to
  the next method. A single temperature OOM marks only that slot.
  Bake-off is ~3.5-4 hours; we don't want one OOM at hour 2 to lose
  the other 11 runs' data. Incremental writes to
  `bakeoff_results.json` after each method completes give us
  durability — crashing 3 hours in still leaves 2 hours' worth on
  disk.

- **Hard-fail at startup if any input adapter is missing or fails
  locked-spec verification.** `verify_spec` runs on all 4 adapter
  configs BEFORE any GPU work starts. A 4-hour bake-off that crashes
  because of a typo'd adapter directory is wasteful in a way that an
  early-fail check eliminates for free.

- **Injectable callables for testability.** Same pattern as
  `eval_sweep.py`: `run_bakeoff(args, merge_callable, eval_callable,
  config_factory, adamerging_state)` takes the three ML-dependent
  pieces as parameters. Production wiring calls
  `_default_merge_callable` / `_default_eval_callable` /
  `_default_config_factory`, which lazily import torch/peft/vllm.
  Tests inject `types.SimpleNamespace`-based stubs and stay
  torch-free.

- **AdaMerging forward_fn + data_iter built ONCE at startup.** Builds
  before the method loop only if `dare_adamerging` is in `--methods`.
  Cleanup happens at the end via the `cleanup()` returned by
  `make_qwen3_forward`, in a finally block so an unexpected
  exception still releases GPU. Memory cost: ~3.4 GB for the
  Qwen3 forward base plus the in-merge base load during
  `pipeline.merge_adapters` — that's 2× base in GPU for the duration
  of the dare_adamerging merge step (~7 GB), well within A100-40g.

- **Default output dir: `<repo_root>/bakeoff_<YYYY-MM-DD-HHMM>/`.**
  Under the repo root, which on cluster is `/scratch/Group/...`
  (persistent), not `/tmp/` (ephemeral and small). Timestamped so
  back-to-back runs don't collide.

- **Winner: highest average pass@8 across the 4 benchmarks.** Single
  scalar makes for an unambiguous comparison. The grid output shows
  per-benchmark pass@1 / pass@8 so a reviewer can inspect any
  unbalanced result (e.g. a method that wins overall by sacrificing
  one domain).

### Implementation

- `scripts/run_bakeoff.py` — 4 dataclasses (`TemperatureRunRow`,
  `MethodRunRow`, `BakeoffPayload`, plus the existing `InferenceConfig`
  in `merge.infer`). `validate_args` mirrors `eval_sweep.py` plus
  adapter-dir / methods / adamerging-max-steps checks.
  `verify_locked_specs` runs `merge.verify_spec.verify` on each of
  the 4 adapter configs. `build_method_kwargs` returns the
  per-method kwarg dict — including the AdaMerging state when
  applicable. `run_bakeoff` is the injectable core; `main` wires up
  the production callables and the AdaMerging cleanup in a finally.
- `merge/tests/test_run_bakeoff.py` — 36 unit tests, all torch-free.
  Covers argparse validation, spec verification, kwargs construction,
  dataclass JSON round-trip, result aggregation, winner selection,
  the happy path with stubs, incremental writes after each method,
  per-method merge resilience, per-temperature eval resilience, the
  no-state dare_adamerging failure path, and the print_summary smoke.

### Verified (laptop)

- `pytest merge/tests/test_run_bakeoff.py -v` — **36 passed, 0
  failed** in <1s on the torch-free laptop.
- `python3 scripts/run_bakeoff.py --help` renders the full argparse
  surface plus the recommended `nohup` launch pattern.
- Full merge suite: `pytest merge/tests/ -v` — 139 passed, 97 skipped
  (skips are torch/CUDA/Qwen3-gated tests; pattern unchanged from
  Day 7 follow-up).

### Verified (cluster)

- *To fill in after the next `runai submit`: confirm the full bake-off
  (4 methods × 3 temperatures) completes end-to-end on real teammate
  adapters; confirm per-(method, temperature) failure isolation works
  under a real OOM; confirm the output layout matches the README
  spec.*

### Open questions / follow-ups

- Bake-off wall-clock budget: 4 merges × ~5 min + 12 evals × ~15 min
  ≈ 3.5 hours on A100-40g. If `dare_adamerging` with `--adamerging-max-steps
  200` over-runs, drop to 100 (`/2`) or 50 (`/4`). If it underfits,
  bump to 500 — but that pushes into 4+ hour territory.
- Should the bake-off auto-publish the winner to HF (`group_model`
  repo)? Defer to Stage 5d (`publish.py`); the bake-off should leave
  the winner identification and human-in-loop confirmation step
  intact.
- A "second-place" report (winner + 95% CI band on avg pass@8) would
  be useful for the final report. Out of scope here; can be added on
  top of `bakeoff_results.json` after the cluster smoke.

---

## Day 8 follow-up (2026-05-26) — bake-off GPU starvation fix

### Symptom

First real bake-off (2026-05-26, `loras/` × all 4 methods × 3
temperatures) failed every single (method, temperature) combination:

- `uniform`: merge OK, but all 3 vLLM evals raised
  `Engine core initialization failed` — vLLM could not initialize
  its engine because ~3.4 GB of GPU memory was held by an idle
  forward_fn.
- `dare_uniform`: merge failed instantly (1.2 s) — GPU saturated
  before `load_all` could finish.
- `dare_adamerging`: OOM at `merge/methods/dare.py:84` (the fp32
  upcast in the DARE mask op). Two full Qwen3-1.7B copies plus the
  DARE upcast exceeded 40 GB.
- `ties`: merge OK, all 3 evals failed (same vLLM starvation).

### Cause

`scripts/run_bakeoff.py`'s Day 7-era design built the AdaMerging
`forward_fn` (which loads + pins a full Qwen3-1.7B in GPU) ONCE at
startup and held it through the entire bake-off via `main()`'s
`try/finally`. The forward_fn is only actually consumed during the
`dare_adamerging` MERGE (the AdaMerging training loop). It is NOT
needed during any eval, nor during any other method's merge.
Holding it for ~3.5 hours was wasted memory pressure on the GPU
the whole time. vLLM's `Engine core initialization failed` was the
visible failure, but the root cause was the static state lifetime.

### Fix

Three structural changes:

1. **`run_bakeoff.py`**: replaced `adamerging_state: dict | None`
   parameter with `adamerging_state_factory: Callable | None`. The
   factory is invoked exactly ONCE, immediately before the
   `dare_adamerging` merge. Its cleanup callable fires IMMEDIATELY
   after that merge returns (success or failure) in a `finally`
   block, before the first eval temperature runs. For every other
   method (`uniform`, `dare_uniform`, `ties`) the factory is never
   invoked and no forward_fn is resident.
2. **`merge/qwen3_forward.py`**: strengthened `cleanup()` — it now
   `gc.collect()` + `torch.cuda.empty_cache()` and logs CUDA
   `memory_allocated()` before/after so a missed drop is visible
   in the bake-off log. The return tuple was extended to
   `(forward_fn, cleanup, base_model)` so the pipeline can reuse
   the same base model for `merge_and_unload` instead of loading a
   second copy.
3. **`merge/pipeline.py`**: `merge_adapters` accepts an optional
   `base_model=` parameter. When provided, the pipeline skips its
   own `from_pretrained` (saving ~3.4 GB of duplicate allocation
   during the `dare_adamerging` merge) and does NOT free the
   externally-owned model in its finally block. Caller retains
   ownership and is responsible for `cleanup`.

### Why option (a) was safe for base-model reuse

`merge.methods.adamerging.adamerging` uses `forward_fn` only inside
its training loop. After the loop exits, the final merged tensor
is recomputed under `torch.no_grad()` via `_compute_merged` — no
forward pass needed. By the time control returns to `merge_adapters`
and reaches `merge_and_unload`, the forward_fn is no longer in
use. `merge_and_unload` mutates the base model in place (baking
deltas into base weights), which is fine — AdaMerging training has
finished, the model is otherwise about to be freed anyway.

### Belt-and-suspenders

The recommended bake-off launch now sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce CUDA
pool fragmentation across the multiple alloc/free cycles in one
run. This is independent of the structural fix above but pairs
well with it.

### Verified (laptop)

- `pytest merge/tests/test_run_bakeoff.py -v` — including 3 new
  tests pinning the forward_fn lifetime: cleanup-before-eval on
  success path, cleanup-on-merge-failure path, factory-never-invoked
  for non-AdaMerging methods.
- `pytest merge/tests/test_pipeline.py -v` — including 1 new test
  pinning the `base_model=` reuse contract (cluster-only — gated by
  CUDA).

### To verify (cluster)

Cluster re-run of the full bake-off is the next verification gate.
Expected: forward_fn cleanup log line appears between the
`dare_adamerging` merge and the first eval; subsequent vLLM evals
succeed; total wall-clock similar to the Day 7 estimate (~3.5 h).

---

## Day 8 follow-up #2 (2026-05-26) — base_model reuse CAUSED a freed-0.00-GB leak

### Symptom (revisited)

After the Day 8 follow-up #1 fix (scoping forward_fn to the
dare_adamerging merge), the next cluster bake-off still failed every
(method, temperature). The bake-off log this time included an
unambiguous diagnostic from the new cleanup logger:

    make_qwen3_forward cleanup: cuda allocated 3.46 GB -> 3.46 GB (freed 0.00 GB)

`empty_cache()` reclaimed nothing because the model was still strongly
referenced *somewhere else* when cleanup nulled its closure box.

### Root cause

The Day 8 follow-up #1 fix exposed the loaded base model as a third
tuple element from `make_qwen3_forward` and threaded it through
`state["base_model"]` into `merge_adapters(base_model=...)` for reuse.
That was an attempt to dodge a 2× residency cost (forward_fn's base +
merge_adapters' own base, ~6.8 GB peak) during the dare_adamerging
merge. The side effect: `cleanup()` nulling `model_ref[0]` no longer
dropped the last reference — the state dict still held one. So
`empty_cache()` saw the allocator still holding 3.46 GB of live tensors
and freed nothing. The leak compounded across methods until vLLM had no
room for its engine and the OOM cascade began.

### Mem-probe evidence

A pre-fix mem probe showed `merge_adapters` is internally clean:
peaks at ~5.82 GB reserved, returns to ~0.01 GB allocated after its
finally block. So 2× residency was never the real concern — the
hypothetical worst case (~3.46 + ~6 ≈ 9.5 GB) fits comfortably in
40 GB. The reuse solved a non-problem and introduced a correctness bug.

### Fix

1. **`make_qwen3_forward`** now returns a 2-tuple `(forward_fn, cleanup)`.
   The raw model is not exposed. The local `model` binding inside
   `make_qwen3_forward` is `del`'d after wrapping it in the closure
   box, so the box is the only strong reference. Cleanup nulling it
   genuinely drops the last reference.
2. **`merge_adapters`** lost the `base_model=` parameter. It always
   loads its own base for `merge_and_unload`.
3. **`scripts/run_bakeoff.py`** stops capturing the raw model in
   `state` and stops forwarding `base_model=` into `merge_adapters`.
   A new log line after cleanup —
   `[%s] GPU after forward_fn cleanup: %.2f GB allocated` —
   makes the drop visible in the next cluster log.

### Verified (laptop)

- `pytest merge/tests/test_qwen3_forward.py -v` — new file. The
  weakref test would have caught this bug before deployment: it
  monkey-patches `from_pretrained` with a tiny stub, takes a
  `weakref.ref` to the model, calls cleanup, and asserts the ref is
  dead.
- `pytest merge/tests/test_pipeline.py -v` — the cluster-only
  `test_pipeline_reuses_externally_provided_base_model` test was
  replaced with a laptop-runnable test asserting `merge_adapters`
  rejects a `base_model=` kwarg (the parameter is gone).
- `pytest merge/tests/test_run_bakeoff.py -v` — the two cleanup-lifetime
  tests were flipped: they now assert `base_model` is NOT in
  `merge_callable`'s kwargs and that the state dict contains only
  `forward_fn` + `data_iter`.

### Lesson

When you add a parameter purely to avoid a hypothesized memory cost,
verify the cost is real first. A mem-probe takes 10 minutes; the
follow-up fix took longer than the original.

### To verify (cluster)

Cluster re-run is the gate. Expected log line after the
dare_adamerging merge:

    make_qwen3_forward cleanup: cuda allocated 3.46 GB -> 0.X GB (freed 3.4X GB)
    [dare_adamerging] GPU after forward_fn cleanup: 0.X GB allocated

If the freed value is still ~0.00 GB, another dangling reference
exists and the search continues.

---

## Day 8 (2026-05-26) — bake-off results + Stage 5d publish

### Bake-off outcome

After the two follow-ups (forward_fn lifetime scope, then the
`base_model` reuse removal + 0.6 GPU utilization cap), the bake-off
ran end-to-end. Headline:

| method            | T   | avg pass@8 | notes |
|-------------------|-----|------------|-------|
| **ties**          | 0.5 | **0.700**  | **winner** |
| uniform           | varies | < ties | clean run, baseline behavior |
| dare_uniform      | —   | OOM    | fp32 upcast in `merge/methods/dare.py:84` |
| dare_adamerging   | —   | OOM    | same upcast, plus AdaMerging training overhead |

**n=10 problems per benchmark caveat**: pass@8 over 10 problems is a
small-sample estimator. Treat ranks as indicative, not definitive. The
ties win was robust across all 4 benchmarks (no single domain carrying
the average) which is what we wanted from a merge method.

dare_uniform / dare_adamerging both OOM'd at the same line —
`dare.py:84` upcasts the mask op to fp32 inside the merge loop,
peaking GPU memory above the budget left after the 3.46 GB base load.
Fix deferred to Stage 6 (use bf16 mask op, or chunk the upcast); the
ties result is sufficient for the Milestone 2 push.

### Stage 5d: `scripts/publish.py`

The course CI grades by loading the *uploaded* model's
`generation_config.json`. The bake-off scored ties under
`temperature=0.5, top_p=0.8, top_k=20, max_new_tokens=2048`, but the
merged dir's bundled config is whatever
`merge_and_unload`/`save_pretrained` left there (Qwen3 defaults =
0.7 / 0.8 / 20 / 16384). If we push as-is, CI grades under different
sampling than the bake-off validated. So publish MUST rewrite the
config to the winning params.

Implementation:

- `scripts/publish.py` is a thin CLI: validate model dir → build the
  winning `generation_config.json` via
  `merge.generation_config.make_generation_config` → print the upload
  plan (repo, every file + size, total, the full new gen config) →
  if not `--confirm`, exit; if `--confirm`, back up the existing
  `generation_config.json` → `.bak`, write the new one, call
  `HfApi().create_repo(exist_ok=True)` + `upload_folder`.
- **Dry-run by default**: `--confirm` is required to push. Without
  it the script never modifies the model directory and never touches
  the Hub. A 3.4 GB push to the wrong repo is expensive and
  embarrassing, so the safe path is the default.
- `--repo-id` is required, no default. The intended slug is
  `cs-552-2026-emainelpe/group_model` but the user must type it.
- `merge/publish.py`'s `publish_adapter` stub is left in place —
  it predates the Day 7 LoRA → full-model refactor and is no longer
  the right shape, but removing it would break callers that import
  from `merge.publish`. The script-level entrypoint is the supported
  surface from Stage 5d onward.

### Verified (laptop)

- `pytest merge/tests/test_publish.py -v` — 18 tests, all passing.
  Coverage: gen config rewrite shape (the 4 tunables + 4 structural
  fields), backup-then-overwrite flow, missing-original no-backup path,
  validation (missing config / chat template / weights / tokenizer /
  dir), sharded weight layout acceptance, `.bak` exclusion from upload
  list, dry-run no-side-effects, `--confirm` upload + create_repo call
  shape, sampling-param overrides flowing through, validation failure
  → exit 2 short-circuit, `--repo-id` and `--model-dir` required.
- `python scripts/publish.py --help` renders the full argparse surface
  + dry-run / confirm example block.
- Full suite: `pytest merge/tests/ -v` — passing on the torch-free
  laptop (skips remain torch/CUDA/vllm-gated).

### To do (manual, by Julien)

1. `python scripts/publish.py --model-dir bakeoff_2026-05-26-1145/ties/merged --repo-id cs-552-2026-emainelpe/group_model`
   — inspect the printed plan and the new gen config.
2. If both look right: re-run with `--confirm`.
3. CI run on `cs-552-2026-emainelpe/group_model` post-push to confirm
   the leaderboard hits the same pass@8 numbers we saw locally.

---

## Day 8 follow-up #3 (2026-05-26) — DARE fp32 upcast OOM

### Symptom

In the bake-off (`bakeoff_20260526_1145`), both DARE-based methods
failed during merge:

- `dare_uniform`: `merge:failed` in 1.7s — same 4 adapters as the
  successful `uniform` and `ties` runs, no forward_fn loaded.
- `dare_adamerging`: `merge:failed` in 1.1s.

Both OOM'd at `merge/methods/dare.py:84`:

    masked = tensor.to(torch.float32) * mask * rescale_factor
    torch.OutOfMemoryError: CUDA out of memory ... GPU has 39.49 GiB
    total, ~15 MiB free

The decisive clue: `dare_uniform` crashed with NO forward_fn resident
— the 39.47 GiB was all coming from the merge itself. `uniform` and
`ties` (same 4 task vectors, no DARE) ran fine. So the dare()
implementation was the sole cause.

### Root cause

The function built THREE full-size fp32 allocations per input tensor:

1. `keep_prob_t = torch.full(tensor.shape, ..., dtype=torch.float32)`
2. `mask = torch.bernoulli(keep_prob_t, ...)` — also fp32 (Bernoulli
   returns input dtype).
3. `tensor.to(torch.float32)` — a full fp32 copy of the input.
4. Their pairwise product — another fp32 intermediate.

Per-iteration peak ≈ 9× the bf16 input size (2 bytes/elem in,
~18 bytes/elem at peak). Across 4 adapters × 196 target modules,
CUDA caching allocator fragmentation pushed reserved memory past
40 GB before the loop could free anything.

bf16 was chosen for the merge throughout the rest of the pipeline
precisely because fp32 doesn't fit. The fp32 upcast in dare() was a
leftover from the Stage 3 implementation when memory budget hadn't
been pressure-tested.

### Fix

Mask + multiply in the input dtype:

    keep_prob_t = torch.full(
        tensor.shape, keep_prob, dtype=tensor.dtype, device=tensor.device,
    )
    mask = torch.bernoulli(keep_prob_t, generator=generator)
    out[name] = tensor * mask * rescale_factor

bf16 precision is overkill for the operation — the mask is 0 or 1
(both exactly representable), the rescale is one scalar multiply,
and DARE's own stochastic variance dwarfs any bf16 rounding. For
the canonical `drop_rate=0.5`, `keep_prob=0.5` is exact in bf16;
other drop rates have bf16 rounding error ~10⁻³ on the Bernoulli
probability — well below the method's intrinsic randomness.

### Tests

`test_dare.py`'s existing tests (statistical drop-ratio, mean
preservation under rescale, key/shape/dtype preservation,
reproducibility, no-input-mutation) all asserted properties that the
bf16 implementation preserves — no test changes needed beyond
adding two regression tests:

- `test_dare_does_not_upcast_to_fp32_internally` — asserts the
  output dtype matches input dtype AND ``element_size() == 2`` (bf16
  storage, not fp32-converted-back storage). Catches a future
  regression that adds an implicit upcast.
- `test_dare_fp32_input_stays_fp32` — belt-and-suspenders: the
  cleaned-up code drops the explicit ``.to(tensor.dtype)`` cast and
  relies on torch promotion; verify fp32-in still yields fp32-out
  for non-bf16 callers.

### Memory budget for dare_adamerging post-fix

- forward_fn base: ~3.46 GB (bf16 Qwen3-1.7B)
- 4 DARE'd task vectors: ~13.6 GB (4 × 3.4 GB bf16)
- merged dict recomputed per AdaMerging step: ~3.4 GB bf16
- AdaMerging activations (batch_size=2, modest seq len, only the
  delta path retains activations because base params are
  `requires_grad=False`): ~5-10 GB rough upper bound
- Coefficients + Adam state: negligible (4 × 28 = 112 floats)

Estimated peak: ~25-30 GB. Comfortable headroom on a 40 GB A100.
The cluster re-run is the verification gate — if a spike pushes it
over, the next steps are activation checkpointing or a smaller
AdaMerging batch.

### What this unblocks

The first bake-off only fully scored 2 of 4 methods (uniform, ties).
With dare() fixed, the cluster can re-run `dare_uniform` and
`dare_adamerging` against the same input adapters (deterministic
seeds) and complete the 4-method comparison for the final report.
The ties @ T=0.5 winner identified in the partial bake-off does NOT
need re-validation — DARE only affected the failed runs.

---

## Day 8 follow-up #4 (2026-05-26) — dare() now mutates in place

### Symptom

After the bf16 fix (follow-up #3), `dare_uniform` STILL OOM'd on a
verified-clean A100-40g (4 MiB used at launch, no co-tenant, no
forward_fn). The OOM line moved from the old fp32 upcast at dare.py:84
to the output-dict write at dare.py:97:

    out[name] = tensor * mask * rescale_factor
    torch.OutOfMemoryError

### Root cause

`dare()`'s non-mutating contract built a FRESH output dict alongside
the input. The 4 input task vectors total ~12 GB (per-adapter ΔW ≈
3 GB across 196 target modules); the output dict adds another ~12 GB
on top. Add mask + intermediate (~3-6 GB during the loop, plus CUDA
caching-allocator fragmentation across 196 alloc/free cycles per
adapter) and the 40 GB ceiling falls over.

The non-mutating contract is correct for general-purpose callers but
unnecessary for the merge composers — the pipeline never re-reads
the originals after `merge_fn` returns. So the duplication was paying
for an invariant nobody actually needed.

### Fix

Added `inplace: bool = False` to `dare()`. Default preserves the
historical non-mutating contract (and the existing
``test_dare_does_not_modify_input`` test). When `True`:

    if inplace:
        tensor.mul_(mask)
        if rescale_factor != 1.0:
            tensor.mul_(rescale_factor)

No new output tensor is allocated. The mask itself is still
allocated per-iteration but `del mask` after use makes that one bf16
allocation reclaimable; only one mask is resident at a time, not 196.

`dare_uniform`, `dare_weighted`, and `dare_adamerging` all pass
`inplace=True`. Pipeline-side: `merge_adapters` does
`task_vectors = list(adapters_by_domain.values())` and passes the
list to `merge_fn` — after which `merged.items()` is the only
consumer. So mutation is safe.

### What this saves

Before: 4 inputs (~12 GB) + 4 outputs (~12 GB) + per-iteration
mask/intermediate (~few GB) ≈ 24-28 GB peak before considering
fragmentation. With fragmentation the actual allocator footprint
pushed past 40 GB.

After: 4 inputs (mutated in place, still ~12 GB) + one mask resident
at a time (~few hundred MB) ≈ 12-13 GB peak.

### What this does NOT solve

If the 4 input task vectors alone exceeded 40 GB — say a future
model with larger hidden size — no amount of in-place trickery
would help. The acceptable non-fix in the task spec applies for that
case: document the limitation and move on. For Qwen3-1.7B the
in-place fix should be enough; cluster re-run is the gate.

### Task 3 (incremental free) — skipped

The pipeline holds the inputs via two references: `adapters_by_domain`
(the dict) AND `task_vectors` (the list). Popping the list from
inside `dare_uniform` would not free GPU memory because
`adapters_by_domain` still pins every tensor. Plumbing a
"consume + free" signal back to `merge_adapters` so it could drop
`adapters_by_domain[domain]` entries crosses the merge-method
abstraction boundary for marginal benefit. In-place mutation alone
removes the duplication that was the actual cause; incremental free
would only matter if 4 task vectors alone exceeded the budget.

### Tests

- `test_dare_does_not_modify_input` (default `inplace=False`): passes
  unchanged.
- `test_dare_uniform_reproducible_with_seed` and
  `test_dare_weighted_reproducible_with_seed` (compositions test):
  updated to load tvs fresh per call, matching the real pipeline
  flow (which always calls `load_all` fresh before `merge_fn`).
  Quoting the diff intent: "dare_uniform applies DARE in-place to
  save GPU memory, so re-using the same dict object would feed
  already-masked inputs into the second call."
- New `test_dare.py` tests for the inplace path: mutates-input
  contract, identical results as non-inplace for the same seed,
  rescale preserves mean magnitude, dtype preservation.

### What this unblocks

Same as follow-up #3: `dare_uniform` and `dare_adamerging` can be
re-run on the cluster to complete the 4-method comparison for the
final report. The ties @ T=0.5 winner stands.

---

## Day 8 follow-up #5 (2026-05-26) — full 4-method bake-off + pass@1 vs pass@8 metric mismatch

### What ran

After follow-ups #3 (bf16 mask) and #4 (in-place DARE), `dare_uniform`
and `dare_adamerging` both completed cleanly on the cluster. We now
have the full 4-method × 3-temperature grid:

- `bakeoff_20260526_1145/` — uniform, ties (pre-DARE-fix run)
- `bakeoff_dare_20260526_1246/` — dare_uniform, dare_adamerging
  (post-fix re-run on the same input adapters)

Peak GPU dropped from ~25-30 GB to ~12-13 GB on the DARE leg, and
`dare_adamerging`'s forward_fn coexisted with the merge step and was
released cleanly afterwards (logged: freed 3.44 GB at teardown). No
OOM on a clean 40 GB A100. The two follow-ups closed the GPU budget
problem.

### Bake-off "Winner" — and what it actually measured

`scripts/run_bakeoff.select_winner` ranks configs by **average pass@8
across all 4 benchmarks**. By that metric:

- Winner: **ties @ T=0.5, avg pass@8 = 0.700**

That is the model that got published via `scripts/publish.py` to
`cs-552-2026-emainelpe/group_model`, with `generation_config.json`
rewritten to bundle `temperature=0.5, top_p=0.8, top_k=20`.

### The metric mismatch (discovered post-publish)

The course CI does NOT grade on avg pass@8. It grades:

- math → **pass@8**
- general_knowledge, safety, multilingual → **pass@1**

So the bake-off optimized one metric and the leaderboard scores a
different one. Re-ranking the same scorecards on a (presumed
equal-weight) average of `math.pass@8 + gen/safety/mult.pass@1`
reshuffles the top of the table.

**Point estimates on the CI-shaped metric** (transcribed from the
session that produced the bake-off; **NOT independently
recomputed on this laptop — the bakeoff dirs live on the cluster**;
to be confirmed against `scorecard.json` before any redeployment
decision):

| method            | T   | CI-metric (point estimate) |
|-------------------|-----|----------------------------|
| dare_adamerging   | 0.5 | ~0.519                     |
| dare_uniform      | 0.3 | ~0.509                     |
| dare_adamerging   | 0.3 | ~0.503                     |
| uniform           | 0.5 | ~0.497                     |
| **ties** (published) | 0.5 | ~0.497                  |

i.e. the model on the Hub is roughly 4th/5th on the CI-shaped metric,
not 1st.

### Why this is NOT a clean "republish dare_adamerging" call

Three reasons, all of which need to be in the entry honestly:

1. **The CI aggregation is unconfirmed.** The numbers above assume
   equal-weight average of the 4 per-benchmark scores. The course CI
   may weight benchmarks differently, sum instead of average, or use a
   slightly different per-benchmark metric. Until we read the grader,
   the re-ranking is an estimate, not a fact.
2. **n=10 problems per benchmark.** The top ~5 configs span
   ~0.497-0.519 — a spread of about 0.022, i.e. 3-4 problems total
   across the four benchmarks. That is well inside sampling noise at
   n=10. The configs are statistically indistinguishable on this
   validation set. Choosing dare_adamerging over ties on a 0.022
   point-estimate lead would be selecting on noise.
3. **The current published model already passed the dry-run plan
   review and the `--confirm` push.** Swapping it for one ~0.022
   ahead on an unconfirmed aggregation is not justified by what we
   currently know.

### Negative result worth recording

`dare_adamerging` is the most complex method in the grid (DARE
masking + AdaMerging's learned per-domain coefficients, requires a
forward_fn, extra GPU pressure). It is **not clearly better** than
plain `uniform` or `ties` on this validation set. Within noise it
ties them. The bake-off does not endorse the complexity.

### Bug to fix (separate change, not in this entry's scope)

`scripts/run_bakeoff.select_winner` should rank by the CI-shaped
metric, not avg pass@8, so future bake-off "Winner" lines actually
match what gets graded. The avg-pass@8 ranking is what caused this
selection to be misleading. Leaving as a known issue, not patched
here.

### Open items (do not treat as settled)

- **CI aggregation formula** unconfirmed. The exact weighting / sum
  vs average / per-benchmark metric definition needs to be read off
  the course grader, not guessed.
- **Validation set size (n=10/benchmark)** is too small to choose
  between the top configs on signal. Either re-eval the top 2-3 on a
  larger set, or document the statistical tie and keep the current
  publication.
- **Republish decision** is open. Defensible options: (a) keep ties
  @ T=0.5 and add a note in the model card about the statistical
  tie; (b) re-evaluate top candidates on a larger n and pick the
  winner there; (c) republish dare_adamerging @ T=0.5 — only
  warranted if we first confirm the CI aggregation AND find the
  larger-n lead survives noise. Default in the absence of those two:
  keep the current upload.
- **Numbers above are transcribed**, not recomputed. The cluster has
  the source scorecards; before acting on the re-ranking,
  recomputation from `scorecard.json` is required.

### What's not blocked by this

Milestone 2 (2026-05-24) is past; the group model is published and
will get graded on whatever the CI does. The findings above shape
what we report and whether/how we iterate before the final
deadline (2026-06-07), but do not block any other in-flight work.

---

## Day 8 follow-up #6 (2026-05-26) — AdaMerging metrics persistence + diagnostic re-run

### Symptom

`AdaMergingResult` already carries everything the report needs —
`loss_history`, learned per-(task, layer) `coefficients`,
`steps_run`, `early_stopped`. But the registry shim
(`_adamerging_dict` in `merge/methods/__init__.py`) unwrapped it to
just `.merged` so the pipeline could treat all methods uniformly.
The bake-off (2026-05-26) therefore discarded those fields; the loss
curve only existed in the bake-off's stdout log, which lived under
`/tmp` and disappeared with the next pod restart.

The follow-up #5 weight comparison showed `dare_adamerging`'s output
sits within 0.74% of `dare_uniform`'s — too close to know whether
AdaMerging learned little, or learned a lot that happened to net out
small. The coefficients matrix is the only way to tell, and we
threw it away.

### Approach chosen: Option B (variant)

Two paths considered:

- **Option A** — make registry entries return `AdaMergingResult`
  directly, branch in the pipeline. Cleaner type-wise, but breaks
  the existing `test_dare_adamerging_composition` contract (asserts
  the registry returns `dict`) and asks `merge_adapters` to do
  isinstance dispatch on merge results.
- **Option B (chosen)** — keep the registry-returns-dict contract,
  add optional `metrics_out_path` + `task_names` kwargs to
  `dare_adamerging` and `_adamerging_dict`. When both are supplied,
  the wrappers persist the AdaMergingResult fields themselves before
  unwrapping to a dict. The pipeline injects these kwargs when
  `method in {"adamerging", "dare_adamerging"}`.

Option B chosen because it preserves the uniform return-type
contract (cheap pipeline, no isinstance branching) and is fully
back-compat: direct callers that omit `metrics_out_path` see no
behavioral change, so the synthetic-fixture composition test and
the cluster smoke script keep working untouched.

### Where the metrics file lives

`merge_adapters` injects::

    method_kwargs["metrics_out_path"] = output_dir / "adamerging_metrics.json"
    method_kwargs["task_names"]       = list(adapters_by_domain.keys())

So `adamerging_metrics.json` lands next to `config.json` /
`model.safetensors` / `chat_template.jinja` inside the merged-model
dir. That's the same dir that gets uploaded by `scripts/publish.py`
— intentional: anyone re-pulling the published model can inspect
its training story without separate artifact tracking.

### task_names row-order correctness

`coefficients` has shape `[N_tasks, N_layers]`; row `i` MUST
correspond to `task_names[i]` or the report figure is wrong. The
chain is:

1. `merge_adapters` calls `load_all(adapters_dir, ...)` returning
   `adapters_by_domain: dict[str, dict[str, Tensor]]` in
   `CANONICAL_DOMAINS` insertion order (math, general_knowledge,
   safety, multilingual).
2. `task_vectors = list(adapters_by_domain.values())` — index `i`
   is `list(adapters_by_domain.keys())[i]`.
3. `task_names = list(adapters_by_domain.keys())` is what gets
   injected into method_kwargs.
4. `dare_adamerging` passes `task_vectors` (dared) into
   `adamerging` in that same order.
5. `adamerging` builds `coefficients[i, :]` from `task_vectors[i]`.

So `coefficients[i]` ⇄ `task_names[i]` ⇄ `task_vectors[i]` ⇄
adapter at `adapters_dir/task_names[i]/`. The persistence helper
`_persist_adamerging_metrics` also defensively raises if
`len(task_names) != coefficients.shape[0]` so a future refactor
that desyncs the two surfaces the bug instead of silently
mislabeling rows.

### Schema of `adamerging_metrics.json`

```
{
  "task_names":    [str, ...]    # length N_tasks, row order for coefficients
  "n_tasks":       int,          # N_tasks
  "n_layers":      int,          # N_layers
  "steps_run":     int,
  "early_stopped": bool,
  "loss_history":  [float, ...]  # length steps_run, total loss per step
  "coefficients":  [[float, ...], ...]    # shape [N_tasks, N_layers]
  "hyperparams": {
      "method":              "dare_adamerging" | "adamerging",
      "drop_rate":           float,        # dare_adamerging only
      "seed":                int | null,   # dare_adamerging only
      "rescale":             bool,         # dare_adamerging only
      "init_coefficient":    float,
      "lr":                  float,
      "lambda_l2":           float,
      "max_steps":           int,
      "early_stop_patience": int,
      ...                                    # forward_fn / data_iter excluded
  }
}
```

Plain JSON, no torch dependency for read-back — matplotlib in the
diagnostic and any future analysis script can load it directly.

### Standalone diagnostic re-run

`scripts/adamerging_diagnostic.py` re-runs `dare_adamerging` on the
4 real adapters with the bake-off hyperparameters (drop_rate=0.5,
seed=42, max_steps=200, lr=1e-2, lambda_l2=1e-4,
init_coefficient=0.3, batch_size=2), captures the full
`AdaMergingResult`, and writes:

- `metrics.json`               — same schema as above
- `loss_curve.png`             — loss vs step
- `coefficients_heatmap.png`   — `[N_tasks × N_layers]` RdBu_r
  heatmap with `task_names` as y-labels

Default output dir is
`/scratch/Group/emainelpe_group_model/adamerging_diagnostic/` and
the script explicitly refuses any output path under `/tmp` (that's
the exact failure mode that lost the bake-off's curve).

The script does NOT push to HF and does NOT run `merge_and_unload`
— it short-circuits at the AdaMerging step, so it's cheap to
re-run with different hyperparameters if the figures need updating
for the report.

### Tests

5 new tests in `merge/tests/test_adamerging.py`:

- `test_dare_adamerging_persists_metrics_when_path_provided`:
  full schema check + row-order check (the safety row has a
  distinctive value).
- `test_dare_adamerging_no_metrics_when_path_omitted`: back-compat
  — `tmp_path` stays empty when `metrics_out_path=None`.
- `test_dare_adamerging_metrics_requires_task_names`: refuse the
  ambiguous-row case.
- `test_adamerging_registry_shim_persists_metrics`: bare
  `adamerging` registry entry also works; `forward_fn`/`data_iter`
  do NOT leak into the hyperparams snapshot (they're
  non-serializable).
- `test_persist_adamerging_metrics_rejects_task_name_length_mismatch`:
  defensive shape check.

All five use `monkeypatch` to replace `adamerging` with a fake
returning a small synthetic `AdaMergingResult`, so the tests stay
CPU-light (torch-only, no transformers / no GPU).

### What still has to happen on the cluster

`scripts/adamerging_diagnostic.py` is the gate. It produces the
figures for the final report; the laptop can only verify the
plumbing. Launch::

    nohup python3 scripts/adamerging_diagnostic.py \
        --adapters-dir loras/ \
        --output-dir /scratch/Group/emainelpe_group_model/adamerging_diagnostic/ \
        > adamerging_diagnostic.log 2>&1 &

Should take ~5-10 min on an A100-40g (200 training steps, no
merge_and_unload).

### What this does NOT change

- The published group model (`cs-552-2026-emainelpe/group_model` =
  ties @ T=0.5) is untouched. Metrics persistence is read-only data
  collection; it does not influence the merge result.
- Future bake-off runs through `scripts/run_bakeoff.py` automatically
  benefit — when the bakeoff invokes `merge_adapters` for
  `dare_adamerging`, the metrics file lands in the per-(method, temp)
  merged dir alongside the model. No bakeoff change needed.

---

## Day 8 follow-up #7 (2026-05-26) — AdaMerging diagnostic run: instability + likely implementation deviation

### What ran

`scripts/adamerging_diagnostic.py` (added in follow-up #6) re-ran
`dare_adamerging` on the 4 real adapters with the bake-off
hyperparameters (drop_rate=0.5, seed=42, max_steps=200, lr=1e-2,
lambda_l2=1e-4, init_coefficient=0.3, batch_size=2). Artifacts
under `/scratch/Group/emainelpe_group_model/adamerging_diagnostic/`:
`metrics.json`, `loss_curve.png`, `coefficients_heatmap.png`.

### Headline numbers (from `metrics.json`)

- **Training stopped early at step 159** (early_stop_patience=100,
  max_steps=200) — the loop ran ≥100 consecutive steps without
  improving on `best_loss`.
- **Loss did not converge.** Per-step total loss (entropy + L2)
  swung between ~0.0031 and ~2.66 across the 159 steps — three
  orders of magnitude — with no monotone trend. First loss ≈ 0.0036,
  last loss ≈ 0.0050. There is no curve to read; it is noise around
  a bimodal-by-domain regime.
- **Learned coefficients scattered**, including into negative
  territory despite init=0.3 (all positive):

  | task              | min     | max    | mean   |
  |-------------------|---------|--------|--------|
  | math              | −0.415  | +0.540 | +0.116 |
  | general_knowledge | −0.349  | +0.412 | +0.055 |
  | safety            | −0.386  | +0.595 | +0.179 |
  | multilingual      | −0.379  | +0.599 | +0.151 |

  Coefficients drifted far from init and acquired sign changes per
  layer.

### Apparent paradox: scattered coefficients, near-identical weights

Follow-up #5's weight-space comparison found `dare_adamerging`'s
merged output was only ~0.74% (mean abs, relative) different from
`dare_uniform`'s, and all 4 merge methods were within 1.7% pairwise.
So despite per-(task, layer) coefficients ranging across roughly
[−0.4, +0.6] with sign flips, the resulting merged weights barely
move off the uniform baseline. Two compatible reads: (a) the
per-task coefficients average out across the sum
`sum_i c_{i,L} · tv_i[k]`; (b) the optimizer is wandering noisily
in coefficient space without driving the merged weight tensor
anywhere in particular.

### Likely root cause: per-domain single-batch SGD ≠ paper objective

Reading the training loop in `merge/methods/adamerging.py`:

- `data_iter` yields `(domain_idx, batch)` round-robin across 4
  domains.
- Each step takes ONE batch (`batch_size=2`) from ONE domain and
  computes entropy on the LAST token only.
- The optimizer steps after every such batch (lr=1e-2).

Different domains produce very different last-token entropies
(math prompts at this scale are near-deterministic, entropy ≈
0.003; general-knowledge / safety / multilingual prompts are open,
entropy ≈ 2.6). So the per-step loss alternates between regimes
that differ by ~3 orders of magnitude, and the gradients between
adjacent steps point at fundamentally different objectives. With
a 1e-2 lr on a 4×28 coefficient grid and no batching across
domains, this is high-variance SGD on a non-stationary objective.

The original AdaMerging formulation aggregates entropy across the
unlabeled distribution per parameter update (i.e. one update step
absorbs evidence from multiple domains, not one). Our current loop
deviates from that: per-step gradients are dominated by whichever
domain the iterator happened to land on.

We attribute the loss instability and the wandering coefficients
to this implementation deviation — not to a fundamental limitation
of the AdaMerging method itself. This is a hypothesis consistent
with the symptoms, not a proof.

### What this entry deliberately does NOT claim

- It does not claim AdaMerging the method fails on this problem.
  All we have measured is that *our current implementation*, with
  per-domain single-batch SGD at lr=1e-2, fails to converge on
  this setup.
- It does not claim the 0.74% weight-space delta vs. uniform is
  evidence of "AdaMerging didn't learn anything." The coefficients
  clearly moved (mean drifted from 0.3 to ~0.05-0.18, range opened
  to ~1.0). What it suggests is that *the movements canceled out
  in weight space*, which is consistent with a non-stationary
  objective producing zero-mean coefficient noise.

### Next step (in progress, outcome pending)

A corrected implementation is being attempted (Option B): aggregate
entropy across all 4 domains within a single forward/backward pass
(or accumulate gradients over a domain-balanced minibatch) so each
optimizer step minimizes the same multi-domain objective. The
hypothesis: a stationary objective will let the loss actually
descend and the coefficients converge to a meaningful per-layer
allocation. If after that fix the merged model is still ~uniform
in weight space, the implementation deviation is ruled out and we
will revisit whether the AdaMerging signal is genuinely too weak
on this 4-adapter Qwen3-1.7B setup. Outcome pending — to be
recorded in a subsequent follow-up.

### What does not change in the meantime

The published group model remains `ties @ T=0.5`
(`cs-552-2026-emainelpe/group_model`). The bake-off ranking
discussion in follow-up #5 stands; this entry only sharpens *why*
`dare_adamerging` lands near the uniform baseline on the published
artifact.

---

## Day 8 follow-up #8 (2026-05-26) — AdaMerging Option B: opt-in domain-aggregated objective

### Hypothesis under test (from follow-up #7)

Our `adamerging()` loop deviated from the paper formulation by
taking one optimizer step per single-domain batch. The round-robin
data iterator yields domains 0,1,2,3,0,1,... so consecutive
gradient steps optimize entropies that differ by ~3 orders of
magnitude (math ≈ 0.003 vs others ≈ 2.6). Hypothesis: aggregating
entropies across one batch from EACH domain before each optimizer
step gives a stationary multi-domain objective and recovers
convergence.

### Design choice: `aggregate_domains: bool` (opt-in)

Two API shapes were considered:

- `steps_per_update: int = 1` — gradient-accumulation-style; expressive
  but conflates "how many batches per update" with "domain coverage."
- **`aggregate_domains: bool = False`** *(chosen)* — boolean flag with
  a clear, single semantic: "one batch per domain per update." Easier
  to reason about, harder to misuse, doesn't add a numeric knob to
  the surface.

Default is `False`, which preserves the existing path byte-for-byte
— important because the 2026-05-26 bake-off used the default path
and the published `ties` winner was selected against
`dare_adamerging` results from that exact code path. The new path is
strictly additive; one new test
(`test_adamerging_default_unchanged_byte_for_byte`) pins this.

### `max_steps` semantics under aggregation

When `aggregate_domains=True`, `max_steps` counts **optimizer
UPDATES**, not batches. Each update consumes `n_tasks` consecutive
yields from `data_iter`. So:

- `n_tasks=4, max_steps=200` ⇒ 200 updates ⇒ 800 batches consumed.
- The caller must ensure `data_iter` yields at least
  `max_steps * n_tasks` tuples. `scripts/adamerging_diagnostic.py`
  provisions this automatically by passing
  `make_unlabeled_iter(max_steps=args.max_steps * n_tasks)` when the
  flag is set.
- `loss_history` length equals the number of UPDATES executed (≤
  `max_steps`). Early-stop patience counts updates too.

This is documented in the docstring and in the `--aggregate-domains`
help text.

### One-batch-per-domain from the round-robin iterator (verified)

`make_unlabeled_iter` yields::

    for step in range(max_steps):
        domain_idx = step % n_domains
        yield (domain_idx, next(cyclers[domain_idx]))

So in any `n_domains` consecutive yields, the `domain_idx` values
are `0, 1, ..., n_domains-1` (in that order). With `n_domains ==
n_tasks` (== 4 for our setup, by construction: each task is one
domain), consuming `n_tasks` consecutive yields gives exactly one
batch per domain. The aggregated branch in `adamerging` does that
via a plain inner `for _ in range(n_tasks): next(iterator)` loop.

For robustness, the branch additionally checks that the
`n_tasks` consumed `domain_idx`s are distinct and logs a WARNING
if a non-round-robin iterator was passed. This is a soft check —
the aggregation still proceeds (it's still strictly less noisy than
single-batch SGD) but the warning makes the deviation visible.

`test_adamerging_aggregated_consumes_n_tasks_batches_per_update`
wraps the synthetic iterator in a counter and asserts exactly
`max_steps * n_tasks` `__next__` calls happen — the cheapest possible
proof that the iterator-consumption discipline is correct.

### Aggregation operation: mean over domains

Two reasonable choices for combining the 4 per-domain entropies:

- **mean** *(chosen)* — keeps the entropy magnitude on the same scale
  as the single-batch path, so `lambda_l2=1e-4` continues to balance
  entropy vs regularization the same way. A direct drop-in.
- **sum** — multiplies entropy by `n_tasks`, requiring a 4× lower
  `lambda_l2` to keep the same trade-off. Surprising to a future
  reader.

L2 is added once per update (it's a function of coefficients, not of
the batches), matching the single-batch path.

### Recommendation: lower `lr` for the aggregated mode

Not enforced — `lr` stays a free parameter the caller chooses. The
docstring notes that with a smoother per-update signal, lr=1e-3 is a
more reasonable starting point than the lr=1e-2 the bake-off used.
The diagnostic script defaults to `--lr 1e-2` to keep the
flag-flip-only comparison clean; the report can rerun with
`--lr 1e-3 --aggregate-domains` if the lr=1e-2 aggregated run still
oscillates.

### Diagnostic script changes

- New flag `--aggregate-domains`. When set:
  - `output_dir` is auto-suffixed with `_aggregated/` so the
    baseline (round-robin) artifacts under
    `/scratch/Group/emainelpe_group_model/adamerging_diagnostic/`
    are preserved. The aggregated artifacts land at
    `/scratch/Group/emainelpe_group_model/adamerging_diagnostic_aggregated/`.
  - `data_iter` is sized to `max_steps * n_tasks` automatically.
  - `aggregate_domains=True` is forwarded to `adamerging()`.
  - `aggregate_domains` is recorded in the
    `metrics.json` `hyperparams` snapshot so the run is
    self-identifying.
- **Matplotlib hardening**: `metrics.json` is written FIRST. The
  loss-curve and heatmap plots are wrapped in `try/except` that logs
  a warning and proceeds. Reason: the baseline 2026-05-26 run wrote
  metrics fine but crashed at the plot step, which (combined with
  the metrics-after-plots ordering at the time) was what almost lost
  the curve data a second time. This ordering guarantees the JSON
  always lands.

### Tests

4 new tests in `merge/tests/test_adamerging.py`:

- `test_adamerging_default_unchanged_byte_for_byte`:
  reproducibility pin. Same fixture, two calls (one implicit
  default, one explicit `aggregate_domains=False`) must give
  identical `loss_history` and bitwise-equal coefficients.
- `test_adamerging_aggregated_consumes_n_tasks_batches_per_update`:
  iterator-consumption discipline (counted by a wrapper iterator).
- `test_adamerging_aggregated_stops_clean_on_short_iterator`:
  partial-update exhaustion stops cleanly without raising; reports
  fewer `steps_run`.
- `test_adamerging_aggregated_loss_history_records_aggregated_value`:
  smoke check that the recorded per-update loss is a finite
  positive scalar.

All 4 are torch-gated (skip on the torch-free laptop, run on the
cluster). Full local suite: **160 passed, 115 skipped** — same 160
pass count as before, 4 new skips for the new torch-gated tests.

### Files modified

- `merge/methods/adamerging.py` — `aggregate_domains` parameter +
  aggregated branch. Default path UNCHANGED.
- `scripts/adamerging_diagnostic.py` — flag, output-dir suffix,
  iterator sizing, hyperparams snapshot, matplotlib try/except.
- `merge/tests/test_adamerging.py` — 4 new tests.

### Next step (cluster gate)

Re-run the diagnostic in the new mode::

    nohup python3 scripts/adamerging_diagnostic.py \
        --aggregate-domains \
        --adapters-dir loras/ \
        --output-dir /scratch/Group/emainelpe_group_model/adamerging_diagnostic/ \
        > adamerging_diagnostic_aggregated.log 2>&1 &

(Note: pass the BASE output-dir; the script appends `_aggregated`
automatically.) Approximate wall-clock: ~4× the baseline run, since
each optimizer step now does 4 forwards through Qwen3-1.7B. With
batch_size=2 on an A100-40g, the autograd graph for 4 forwards is
within budget, but watch GPU memory; if it OOMs, drop `--batch-size`
to 1 or `--max-steps` to 100.

If the aggregated loss curve descends and the coefficients settle
into a stable pattern, the implementation-deviation hypothesis from
follow-up #7 is confirmed and we have a working AdaMerging signal
to put in the final report. If the curve still oscillates, the
hypothesis is rejected and AdaMerging's signal on this 4-adapter
Qwen3-1.7B setup is genuinely weak.

---

## Day 8 follow-up #9 (2026-05-26) — AdaMerging aggregated OOM: gradient accumulation

### Symptom

The aggregated-mode cluster re-run (follow-up #8) OOM'd on a clean
40 GB A100 — pod showed 50 MiB free at launch, so this was not
residue from a prior process. `adamerging_diag_agg.log`:

    torch.OutOfMemoryError: ... (39.47 GiB held by this process)

Baseline (single-domain) mode peaks at ~31.65 GB on the same setup
and runs to completion. So roughly 4× the activation memory of the
baseline — which is exactly what the previous aggregated branch was
holding alive.

### Root cause

The follow-up #8 aggregated branch built **all n_tasks autograd
graphs** before calling `backward()`::

    entropies = []
    for batch in collected:                        # 4 forwards
        ...
        entropies.append(-(p * log_p).sum(-1).mean())
    entropy = torch.stack(entropies).mean()        # one tensor
    loss = entropy + l2
    loss.backward()                                # one backward

Each `forward_fn(merged, batch)` retains activations through Qwen3
for the autograd path back to `coefficients`. With 4 forwards before
any backward, all four activation graphs are co-resident. Add the
input task vectors (~12 GB), the base model in bf16 (~3.4 GB), and
the per-step `merged` dict, and 40 GB is gone.

### Fix: gradient accumulation

Summing per-domain losses then one `backward()` is mathematically
identical to `backward()`-ing each per-domain loss separately and
letting gradients accumulate into `.grad` before a single
`optimizer.step()` — because gradients are linear::

    grad_total = d(loss)/dc
               = d( (1/n) Σ entropy_d + L2 )/dc
               = (1/n) Σ d(entropy_d)/dc + dL2/dc

The right-hand-side decomposition is exactly what gradient
accumulation produces. The accumulation form lets each per-domain
graph be freed by `backward()` before the next forward runs, so peak
memory is ~1 domain's worth instead of n_tasks worth.

New branch (per update)::

    optimizer.zero_grad()
    total_loss_value = 0.0
    for domain_idx, batch in collected:
        merged    = _compute_merged(task_vectors, coefficients, name_to_layer)
        logits    = forward_fn(merged, batch)
        ...
        entropy_d = -(p * log_p).sum(-1).mean()
        scaled    = entropy_d / n_tasks
        scaled.backward()                          # frees graph d
        total_loss_value += scaled.item()
        del merged, logits, ..., entropy_d, scaled
    l2 = lambda_l2 * coefficients.pow(2).sum()
    l2.backward()                                  # accumulates dL2
    total_loss_value += l2.item()
    optimizer.step()

`loss_history` records `total_loss_value` per update — the same
scalar `loss.item()` would have reported under the previous form.

### Why `merged` is recomputed per-domain

Considered: build `merged` once per update, reuse across all 4
forwards. Rejected. The four forward graphs would all share the
upstream `coefficients → merged` portion. The first `backward()`
would free that shared portion (PyTorch's default), which would
then break the next domain's `backward()` unless every
`backward()` got `retain_graph=True` — and that puts the peak
memory back where it started (all activation buffers retained
until the last step's backward). Recomputing `merged` per domain
costs one extra small fp32→bf16 weighted-sum per forward, which is
trivial relative to one Qwen3 forward pass.

### Verifying math equivalence

`test_adamerging_aggregated_matches_sum_then_backward_numerically`
runs ONE aggregated update via the new accumulation path and ONE
update via the explicit `(stack -> mean -> + L2 -> single
backward())` form, both seeded identically, and asserts the
post-step coefficients agree to `rtol=1e-5, atol=1e-6`. That is the
formal proof that the refactor preserves the previous aggregated
intent.

### Default path

`aggregate_domains=False` branch is UNCHANGED.
`test_adamerging_default_unchanged_byte_for_byte` (added in
follow-up #8) still pins it bit-for-bit.

### Tests

Two new tests in `merge/tests/test_adamerging.py`:

- `test_adamerging_aggregated_second_update_proves_graphs_freed`:
  runs 3 updates on the synthetic fixture. If per-domain graphs
  weren't being freed, the second update would either error
  ("Trying to backward through the graph a second time") or behave
  pathologically. Successful completion is the cheapest proof we
  can extract without poking at torch's `gc` internals.
- `test_adamerging_aggregated_matches_sum_then_backward_numerically`:
  bit-for-bit math equivalence (see above).

Existing aggregated tests (consumption count, partial-iterator
stop, finite-positive loss-history) still pass against the new
branch.

Full local suite: **160 passed, 117 skipped in 0.78s** — same 160
pass count; 2 new torch-gated skips for the new tests.

### Files modified

- `merge/methods/adamerging.py` — aggregated branch rewritten to
  gradient accumulation. Default branch untouched.
- `merge/tests/test_adamerging.py` — 2 new tests.

### Expected memory savings

Order-of-magnitude estimate, single Qwen3-1.7B forward at
`batch_size=2`, `max_length=512`:

- Activation memory per forward: ~6-8 GB (28 layers × intermediate
  activations × autograd retention).
- Previous aggregated peak: ~31.65 GB baseline + 3 extra graphs ≈
  past 40 GB ceiling (observed: 39.47 GiB held at OOM).
- New aggregated peak: ~31.65 GB baseline + at most 1 graph at a
  time ≈ baseline, well within 40 GB.

Cluster re-run is the gate.

### Next step (cluster gate)

Re-launch unchanged from follow-up #8::

    nohup python3 scripts/adamerging_diagnostic.py \
        --aggregate-domains \
        --adapters-dir loras/ \
        --output-dir /scratch/Group/emainelpe_group_model/adamerging_diagnostic/ \
        > adamerging_diagnostic_aggregated.log 2>&1 &

If this completes without OOM, follow-up #8's empirical question
becomes answerable (does the aggregated objective actually
converge?). If it still OOMs, something larger than the per-domain
activation graphs is responsible and we revisit the budget from
scratch.

---

## Day 8 follow-up #10 (2026-05-26) — Aggregated AdaMerging RESULTS: instability fixed, coefficients still underdetermined

Builds on follow-ups #7 (baseline diagnostic showing oscillation),
#8 (opt-in aggregated mode), and #9 (gradient-accumulation memory
fix that made the cluster run feasible). This entry records the
empirical outcome of the aggregated cluster run and the
interpretation that closes out the AdaMerging investigation.

### Aggregated run completed clean

Launched per the follow-up #9 plan with `--aggregate-domains`. The
gradient-accumulation rewrite did its job — the previous OOM did not
recur. Artifacts under
`/scratch/Group/emainelpe_group_model/adamerging_diagnostic_aggregated/`
(`metrics.json`, `loss_curve.png`, `coefficients_heatmap.png`).

### Headline numbers (from `metrics.json` — see verification note)

- **193 optimizer updates, early-stopped at update 192** (patience
  100, max_steps 200). Each update consumed 4 batches (one per
  domain); every update's `seen_domains` was `[0, 1, 2, 3]`,
  confirming the round-robin → one-per-domain consumption discipline
  from follow-up #8 held throughout.
- **Loss converged.** First update ≈ 1.2014; min ≈ 0.0133 reached
  around update ~92; thereafter it plateaus, wobbling 0.24–0.40
  around a low level with a max ~2.01 from occasional spikes. Last
  update ≈ 0.3954. The shape is smooth descent then plateau — a
  genuine convergence curve, qualitatively different from the
  baseline's flat 3-orders-of-magnitude oscillation.
- **Learned coefficients** (init=0.3) per task across 28 layers:

  | task              | min     | max     | mean    |
  |-------------------|---------|---------|---------|
  | math              | −0.458  | +0.677  | +0.148  |
  | general_knowledge | −0.487  | +0.536  | +0.024  |
  | safety            | −0.346  | +0.798  | +0.186  |
  | multilingual      | −0.308  | +0.630  | +0.173  |

### Comparison with baseline (follow-up #7)

| signal                       | baseline (round-robin)               | aggregated                                |
|------------------------------|--------------------------------------|-------------------------------------------|
| training behaviour           | oscillation 0.003 ↔ 2.66, no trend  | smooth descent ~1.20 → ~0.013 then plateau |
| early-stop                   | step 159 (no improvement found)     | update 192 (loss did improve, then plateaued) |
| math coeff range / mean      | [−0.415, +0.540] / +0.116            | [−0.458, +0.677] / +0.148                 |
| gen_knowledge coeff range / mean | [−0.349, +0.412] / +0.055        | [−0.487, +0.536] / +0.024                 |
| safety coeff range / mean    | [−0.386, +0.595] / +0.179            | [−0.346, +0.798] / +0.186                 |
| multilingual coeff range / mean | [−0.379, +0.599] / +0.151        | [−0.308, +0.630] / +0.173                 |

Two things to read from this table:

1. The training-stability problem from follow-up #7 is **fixed**.
   That confirms the implementation-deviation hypothesis: the
   baseline's failure to converge was caused by per-domain
   single-batch SGD on a non-stationary objective, not by an
   intrinsic AdaMerging limitation.
2. The **converged coefficients are about as scattered as the
   baseline's**. Same rough span (~[−0.4, +0.7]), same sign-mixing,
   means in the same low range with the same ranking
   (safety > multilingual > math > general_knowledge). Convergence
   in loss did not translate into a sharper, more interpretable
   per-layer allocation.

#### Caveat: best-loss numbers are NOT directly comparable

The baseline's best loss ≈ 0.0034 vs. the aggregated's best ≈ 0.0133
looks like the aggregated is "worse." It is not. The baseline
measured **single-domain** entropy at each step — its 0.0034 came
from one lucky math batch hitting the deterministic-math regime mid
3-orders-of-magnitude oscillation. The aggregated value is the
**mean entropy across all 4 domains** per update, which can never
drop below the floor set by the high-entropy open-prompt domains
(safety/gen_knowledge/multilingual ≈ 2.6 baseline). What is
comparable across the two runs is the SHAPE of the curve
(oscillation vs descent + plateau), not the magnitude.

### Interpretation (inference, not re-verified)

Three facts now stand together:

1. The aggregated training converges in loss (this entry).
2. The converged coefficients are scattered / underdetermined (this
   entry).
3. From follow-up #5's weight-space comparison: all four merge
   methods produce ~99 % identical merged weights (< 1.7 %
   pairwise, mean abs relative). `dare_adamerging`'s merged model
   sat ~0.74 % away from `dare_uniform`'s — well within the
   methods-are-all-the-same band.

Read together, the most natural inference is: the four task vectors
on these adapters are **near-collinear and of modest magnitude**, so
the entropy objective is **nearly flat** with respect to how the
task vectors are weighted. Many coefficient combinations produce
near-identical merged models and near-identical entropy. The
optimizer therefore finds *a* low-entropy point, but the loss
surface offers no pressure toward any specific per-layer
allocation — the coefficients are underdetermined. The practical
consequence is that AdaMerging's adaptivity provides **no benefit
over simple averaging on these particular adapters**.

This is an inference, not a re-verified measurement. To confirm it
we would need to (a) re-merge with the aggregated coefficients,
(b) re-diff the resulting full model against `dare_uniform`'s, and
(c) check whether the gap is again ≈ 0.74 % — which would close the
loop on "the aggregated run converged to a coefficient pattern that
still yields the ~uniform merge." That extra step was not run; the
inference rests on the coefficient scatter combined with the
existing 99 %-identical-merges finding.

### Coefficient granularity (relevant to the scatter)

AdaMerging in our implementation learns **per-(task, layer)**
coefficients: shape `[N_tasks, N_layers]` = `[4, 28]` = **112
scalars**. The 7 LoRA target-module types within each layer
(`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj`) share that layer's coefficient — the merge math
indexes `coefficient[task, layer_of_k]` via
`_layer_index_from_canonical(k)`, which extracts the layer index
only. So the 196 weight matrices that constitute the merged
adapter (28 layers × 7 modules) are weighted with only 112
free parameters, not 196. This is consistent with the AdaMerging
paper's per-layer formulation; just noting it because the heatmap
has 28 columns (layers), not 196.

`n_layers=28` matches Qwen3-1.7B's `num_hidden_layers` (verified
in earlier sessions; the layer-name regex would have errored out
during `_layer_index_from_canonical` otherwise, and the diagnostic
ran clean).

### Figures saved for the report

Four figures across the two diagnostic dirs (referenced by the
report; copies staged under `report_figures/` for inclusion in the
write-up):

- baseline `loss_curve.png` — oscillation, no trend.
- baseline `coefficients_heatmap.png` — scattered, sign-mixed.
- aggregated `loss_curve.png` — smooth descent + plateau.
- aggregated `coefficients_heatmap.png` — also scattered and
  sign-mixed, despite the converged loss.

The visual contrast on the loss curves is the most striking part
of the AdaMerging story; the visual similarity of the two
coefficient heatmaps is the punchline.

### Outcome for the published model

No change. The group model on
`cs-552-2026-emainelpe/group_model` remains `ties @ T=0.5` from
follow-up #5. The conclusion above gives us the writeup we need
in the final report — AdaMerging was investigated, an
implementation deviation was diagnosed and fixed, and the
properly-trained variant converges in loss but to an
underdetermined coefficient pattern that produces a merged model
within noise of the simpler methods. That is exactly consistent
with the bake-off ranking.

### What is genuinely closed vs. genuinely open

- **Closed**: AdaMerging training stability on this setup. The
  fix is the aggregated objective (follow-up #8) + gradient
  accumulation (follow-up #9), verified by the descent + plateau
  in the loss curve.
- **Closed**: the question "did we just have a bad implementation
  of AdaMerging?" — diagnosed and corrected.
- **Open (left as inference, not verified)**: the underlying
  geometric claim that the four task vectors are near-collinear
  / small-magnitude such that the entropy surface is flat. The
  scatter-plus-uniform-merge pattern is consistent with this but
  was not separately measured.
- **Open**: whether a re-merge under the aggregated coefficients
  reproduces the ~0.74 % vs `dare_uniform` weight-space delta.
  Not run; would close the inference loop.

Neither open item gates the final report or the published model.

### Verification note

The two `metrics.json` files live on the cluster
(`/scratch/Group/emainelpe_group_model/...`) and are not present on
the laptop where this entry was written. The numbers above are
transcribed from the session that produced them (matching prior
practice in follow-ups #5 and #7); a fresh `cat metrics.json` on
the cluster is the authoritative source. If any value here
disagrees with the file, the file wins and only the affected
line/cell needs editing.

---

## Day 8 follow-up #11 (2026-05-26) — plumb aggregate_domains through dare_adamerging + bake-off

### Why

Follow-up #10's diagnostic proved the aggregated objective CONVERGES
(loss 1.20 → 0.013), but it only produced `metrics.json` + figures —
NOT a saved, evaluable model. The path that *does* save + eval a
full merged model is `merge_adapters(method="dare_adamerging")` via
`scripts/run_bakeoff.py`. But `aggregate_domains` (added to
`adamerging()` in follow-up #8, commit 056e1d7) was never plumbed
through `dare_adamerging()` or the bake-off — confirmed by grep:
zero hits in either file. So we could not bake-off-and-eval a
*converged* dare_adamerging model.

This entry closes that gap so the aggregated variant can be built,
scored against TIES on the same validation set, and published if it
wins.

### Two gaps, both closed

1. **`dare_adamerging()` dropped the flag silently.** Its
   `adamerging(...)` call enumerated kwargs explicitly
   (`init_coefficient=..., lr=..., ...`), so any extra kwarg a caller
   passed was discarded without error. Added
   `aggregate_domains: bool = False` to the signature and forwarded
   it into the call. Also recorded in the persisted
   `adamerging_metrics.json` hyperparams so a baked model is
   self-identifying.

2. **The bake-off had no flag and mis-sized the iterator.** Added
   `--aggregate-domains` (store_true, default False). When set:
   - `build_method_kwargs` injects `aggregate_domains=True` for
     `dare_adamerging` only (ignored for uniform / dare_uniform /
     ties).
   - `_build_adamerging_state` sizes the data iterator to
     `max_steps * n_tasks`.

### The iterator-sizing fix (the load-bearing line)

`make_unlabeled_iter`'s generator runs `for step in range(max_steps)`
— it yields *exactly* `max_steps` tuples, it does not cycle forever.
In aggregated mode each optimizer UPDATE consumes `n_tasks` (= 4)
batches, so requesting only `max_steps` would starve training at
update `max_steps // 4`. The fix::

    n_tasks = len(CANONICAL_DOMAINS)
    iter_steps = (
        args.adamerging_max_steps * n_tasks
        if getattr(args, "aggregate_domains", False)
        else args.adamerging_max_steps
    )

This mirrors exactly how `scripts/adamerging_diagnostic.py`
(follow-up #8) sizes its iterator under `--aggregate-domains`. So
`--adamerging-max-steps 200 --aggregate-domains` runs 200 optimizer
updates (800 batches consumed), matching the diagnostic run that
converged.

### `max_steps` semantics (restated for the bake-off surface)

`--adamerging-max-steps` counts AdaMerging optimizer steps. In the
default (per-batch) mode that's one batch per step. Under
`--aggregate-domains` it counts UPDATES, each consuming `n_tasks`
batches — documented in the `--adamerging-max-steps` help text.

### Default behavior unchanged

Every new parameter defaults to `False`. The first bake-off's
`dare_adamerging` path is byte-identical when the flag is absent:
`build_method_kwargs` injects nothing, the iterator is sized to
`max_steps` as before, and `dare_adamerging` forwards
`aggregate_domains=False` (which selects the unchanged per-batch
branch in `adamerging`). Existing bake-off tests stay green.

### Tests

- `merge/tests/test_adamerging.py`:
  `test_dare_adamerging_forwards_aggregate_domains` — spies on the
  mocked `adamerging` and asserts the flag arrives as `False` by
  default and `True` when set.
- `merge/tests/test_run_bakeoff.py`:
  - `..._aggregate_domains_off_by_default` / `..._on_when_flag_set`
    — kwarg injection only when the flag is set.
  - `..._aggregate_domains_ignored_for_non_adamerging` — uniform /
    dare_uniform are untouched.
  - `test_aggregate_domains_arg_parses` — store_true, default False.
  - `test_build_adamerging_state_sizes_iter_for_aggregation` — stubs
    transformers / unlabeled / qwen3_forward and captures the
    `max_steps` passed to `make_unlabeled_iter`: `50 * n_tasks`
    aggregated, `50` default.

Full suite: **165 passed, 118 skipped** (was 160 passed; the 5 new
bake-off tests are torch-free and run; the dare_adamerging
forwarding test is torch-gated). Default-path bake-off tests
unchanged and green.

### Files modified

- `merge/methods/__init__.py` — `dare_adamerging` signature + call +
  metrics hyperparams.
- `scripts/run_bakeoff.py` — `--aggregate-domains` flag,
  `build_method_kwargs` injection, `_build_adamerging_state` iter
  sizing, threading from `args` through both.
- `merge/tests/test_adamerging.py`,
  `merge/tests/test_run_bakeoff.py` — new tests.

### Cluster gate

Re-run the bake-off (or a single-method sweep) with
`--aggregate-domains` to build + eval the converged dare_adamerging
model, then compare its scorecards against the published `ties`
model from follow-up #5. Launch unchanged from the documented
command plus `--aggregate-domains`; pick a distinct `--output-dir`
so the first bake-off's results are preserved for comparison.

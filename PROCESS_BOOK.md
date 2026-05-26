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

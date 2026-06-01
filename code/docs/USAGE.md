# emainelpe-shared (+ group-merge)

This repo has a **dual purpose**:

1. **Locked-spec contract** for all four expert tracks — `lora.yaml` and
   `chat_template.jinja` are byte-identical across every teammate's expert
   repo. This is what makes the Phase-3 group merge mathematically valid.
2. **Group-merge code** for the team — `merge/` is a self-contained subdir
   that composes the four specialist adapters into the group model.
   Teammates only need to care about purpose (1); Julien owns purpose (2)
   for team Émainèlpé (g65).

The two files locked across all four experts:

| File | What it is |
|---|---|
| `lora.yaml` | LoRA shape (rank, alpha, target modules) and other locked training parameters |
| `chat_template.jinja` | Qwen3 chat template with thinking mode forced ON |

## Why these two files are shared

Both files determine the *shape and behavior* of the trained adapter, and that
shape must be identical across all four experts for the Phase 3 merge
(DARE + AdaMerging) to work mathematically.

- `lora.yaml` defines the adapter's tensor shapes. If Mathis trains at `r=16`
  and Julien trains at `r=32`, their adapters cannot be linearly combined.
- `chat_template.jinja` gets baked into each expert's tokenizer at training
  time. Different templates would produce inconsistent prompt formatting in
  the merged group model.

Both lock-ins happen at training time, not merge time. There is no
"fix it later" option — that's why these files are shared from day one.

## What is **not** in this repo

Things that should **diverge** between experts (and so live in each
person's own repo):

- Datasets and data preparation code
- Training scripts (SFT, RLVR / DPO loops)
- Per-expert hyperparameters: learning rate, epochs, batch size, scheduler
- Evaluation scripts and validation snapshots
- `generation_config.json` (per-expert at training time; merge-time decision later)

## Workflow

1. **Initial setup.** Each teammate copies these two files into their expert
   repo. Recommended path inside each expert repo:
   ```
   <expert-repo>/
   ├── configs/lora.yaml          ← copy of lora.yaml
   └── chat_template/chat_template.jinja  ← copy of chat_template.jinja
   ```
   Add a comment at the top of each copy:
   `# Source: emainelpe-shared. DO NOT EDIT LOCALLY.`

2. **No local edits.** If a teammate finds a bug or needs a change, propose
   it on this repo first. Get team sign-off. Then everyone updates their
   copy.

3. **Pre-merge alignment check (≈ May 20).** Before Phase 3 begins, the team
   verifies all four expert repos have byte-identical copies of these files.
   The date is set **before** the May 24 milestone so divergences leave time
   for a retrain rather than blocking the deadline:
   ```bash
   diff <expert-repo>/configs/lora.yaml emainelpe-shared/lora.yaml
   diff <expert-repo>/chat_template/chat_template.jinja \
        emainelpe-shared/chat_template.jinja
   ```
   Any divergence at this point means whoever diverged needs to retrain.

## Locked decisions encoded here

For reference (full justification lives in the team's literature review):

- **r=32, α=64**: closes the LoRA-vs-full-FT gap on reasoning tasks
  (Shuttleworth et al. 2024)
- **All seven linear projections targeted**: attention-only LoRA
  underperforms on reasoning at the 1.7B scale
- **`modules_to_save` MUST stay unset / null**. Setting it (e.g. to
  `["embed_tokens", "lm_head"]` for embedding or head tuning) introduces
  full-rank tensors that **break additive task-vector merging**
  (DARE / AdaMerging / TIES). If anyone needs to tune the embedding or
  head, escalate to the team — it is a merge-method decision, not a
  per-expert decision. The skeleton test in `merge/tests/test_skeleton.py`
  (`test_locked_spec_unchanged`) regresses on this.
- **Thinking mode ON**: the team's proposal commits to `<think>...</think>`
  reasoning traces; the CI does not pass `enable_thinking` as a kwarg
- **Output contract**: every assistant turn ends with `\boxed{...}` for
  automated answer extraction

## The `merge/` subdir (group-merge code)

`merge/` is a self-contained subpackage that consumes the locked-spec contract
files (`lora.yaml`, `chat_template.jinja`, `evaluate/`, `validation_samples/`)
and produces the merged group adapter pushed to
`cs-552-2026-emainelpe/group_model`.

Teammates do **not** need to read or run anything in `merge/`. New commits
there will not touch `lora.yaml`, `chat_template.jinja`, `evaluate/`, or
`validation_samples/` — those remain stable contracts. See `merge/README.md`
for the pipeline overview and the stage-by-stage implementation status.

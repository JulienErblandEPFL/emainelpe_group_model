# `adapters/` — Teammate LoRA Handoff

This directory is the drop zone for the four specialist LoRA adapters
that the group merge pipeline (Stage 5c.2 bake-off) consumes. One subdir
per teammate, all locked to the same shape.

If you're delivering an adapter, you only need this page. Drop your
files into the right subdir, run the verification snippet in §3, and
you're done.

## 1. Where to put your adapter

Exact layout — the four subdir names are canonical and the merge code
will reject any variation:

```
adapters/
├── math/                          # Julien
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── general_knowledge/             # Max
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── safety/                        # Morgane
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── multilingual/                  # Mathis
    ├── adapter_config.json
    └── adapter_model.safetensors
```

The names `math`, `general_knowledge`, `safety`, `multilingual` are the
strings the pipeline indexes on (see
[`merge/load_adapter.py`](../merge/load_adapter.py)'s
`CANONICAL_DOMAINS`). `general-knowledge`, `gen_knowledge`, `knowledge`,
`general`, etc. are NOT accepted — `load_all` raises before any GPU work
starts.

## 2. Required adapter format

Save your adapter with PEFT's standard call:

```python
peft_model.save_pretrained("path/to/your/adapter")
```

This produces the two files we need: `adapter_config.json` and
`adapter_model.safetensors`. Any other layout (custom serializer, `.bin`
weights, sharded checkpoints) is not supported — we only read
`adapter_model.safetensors`.

Your adapter MUST match [`lora.yaml`](../lora.yaml) on these 8 fields:

| Field | Required value |
|---|---|
| `base_model_name_or_path` | `Qwen/Qwen3-1.7B` |
| `r` | `32` |
| `lora_alpha` | `64` |
| `lora_dropout` | `0.05` |
| `bias` | `"none"` |
| `task_type` | `"CAUSAL_LM"` |
| `modules_to_save` | `null` (or absent) |
| `target_modules` | `{q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj}` |

`target_modules` is compared as a SET — PEFT serializes the list in
graph-walk order, which is not the order in `lora.yaml`, and that's
fine. What matters is the set is exactly those 7 names, no more, no
less.

Every other PEFT bookkeeping field (`peft_version`,
`init_lora_weights`, `alpha_pattern`, `inference_mode`, …) is ignored.
Version drift between our PEFT installs is fine.

## 3. How to verify before handoff

Before you hand off, run this from the repo root:

```bash
python -c "
from pathlib import Path
from merge.verify_spec import verify, load_locked_spec

DOMAIN = 'math'  # <-- change to your domain

spec = load_locked_spec(Path('lora.yaml'))
adapter_cfg = Path('adapters') / DOMAIN / 'adapter_config.json'
result = verify(adapter_cfg, spec)

if result.passed:
    print('Adapter passes spec.')
else:
    print(result.summary)
    for fr in result.field_results:
        if not fr.passed:
            print(f'  {fr.field}: expected={fr.expected!r}, got={fr.actual!r}')
    raise SystemExit(1)
"
```

`Adapter passes spec.` and exit code 0 means you're done. Anything else
means a load-bearing field disagrees with `lora.yaml`; the printout
tells you which one and what to fix.

Run this on the cluster (or wherever you trained) before
`git push`-ing or copying the files over — fixing a divergence after
delivery is a coordination cost we don't need.

## 4. Common pitfalls

- **Wrong rank.** If you fine-tuned with `r=16` or `r=64`, the merge
  rejects the whole adapter. The spec is `r=32`. Same for
  `lora_alpha=64` — both are locked.
- **Wrong target modules.** Some training scripts default to attention
  only (`q_proj`, `v_proj`). Our locked spec requires all 7 modules
  (4 attention + 3 MLP). Attention-only LoRA consistently underperforms
  on reasoning at the 1.7B scale, and the merge is additive across
  modules, so a partial-module adapter would contribute zero to the
  modules it didn't train.
- **`modules_to_save` set to a list.** Should be `null` (or absent).
  Setting it to `["lm_head"]` or similar introduces full-rank tensors
  that break additive task-vector merging — the verifier rejects.
- **`.bin` weights instead of `.safetensors`.** Older PEFT versions
  default to the legacy `torch.save` format. We only read
  `adapter_model.safetensors`. If yours is `.bin`, re-save with
  `peft_model.save_pretrained(path, safe_serialization=True)`.
- **Wrong base model.** The adapter must declare
  `"base_model_name_or_path": "Qwen/Qwen3-1.7B"`. If you fine-tuned on
  a fork or a quantized checkpoint, set this string explicitly before
  saving.

## 5. What NOT to include

The merge pipeline doesn't read these and they only clutter the diff /
upload:

- `generation_config.json` — bake-off uses an explicit eval-time
  `InferenceConfig` override; any teammate-bundled gen config is
  ignored.
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
  `vocab.txt`, … — the bake-off uses `Qwen/Qwen3-1.7B`'s tokenizer
  (loaded fresh from the base repo).
- `chat_template.jinja` — the repo has the team-locked one at the
  root; teammate copies are ignored.
- Optimizer state, training logs, intermediate checkpoints,
  TensorBoard files, `*.pt`/`*.bin` lone tensors. Anything other than
  the two files in §1.

Two files. That's it.

## 6. Once all 4 are delivered

When the four subdirs are populated, anyone can kick off the bake-off:

```bash
python -u scripts/run_bakeoff.py --adapters-dir adapters/
```

This runs 4 merge methods × 3 sampling temperatures = 12 evaluations on
the same set of adapters and writes an aggregated `bakeoff_results.json`
plus per-(method, temperature) scorecards. Expected wall-clock on
A100-40g: ~3.5–4 hours. See
[`merge/README.md`](../merge/README.md#stage-5c2-full-bake-off) for the
full output layout and the resilience semantics (a failed merge or a
single OOM doesn't kill the run).

## 7. References

- [`../lora.yaml`](../lora.yaml) — the locked LoRA spec. Authoritative
  source of truth for the 8 required fields.
- [`../merge/verify_spec.py`](../merge/verify_spec.py) — the verifier
  used by both the snippet in §3 and the pipeline's hard-fail
  startup check.
- [`PROCESS_BOOK.md`](../../PROCESS_BOOK.md) — design log (at the repo
  root, one level above `code/`). Day 0 through Day 8 cover the why behind
  every locked choice on this page (decision B2 on repo structure,
  modules_to_save rationale, etc.).
- [`../merge/README.md`](../merge/README.md) — pipeline internals if
  you want to dig deeper.
- Team channel (Émainèlpé, g65) for anything not covered here.

"""Runnable scripts for the group-model phase.

Currently contains:

- ``fetch_adamerging_data.py`` — pre-download the 4 unlabeled datasets
  consumed by AdaMerging training.
- ``smoke_adamerging.py`` — cluster smoke test: random-init Qwen3-sized
  LoRA adapters → ``dare_adamerging`` through the pipeline → structural
  validation of the merged output.
"""

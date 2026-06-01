"""Data infrastructure for the group-model merge phase.

Currently only one module: :mod:`merge.data.unlabeled`, which provides the
round-robin (domain_idx, batch) iterator AdaMerging consumes during its
entropy-minimization training loop.
"""

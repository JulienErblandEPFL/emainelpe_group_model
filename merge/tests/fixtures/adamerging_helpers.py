"""
Synthetic test fixtures for AdaMerging.

Provides a tiny "fake transformer" forward and a synthetic data iterator
so the AdaMerging training loop can be exercised on CPU without a real
base model. Real-Qwen3 validation happens in Stage 5b on the cluster.

The synthetic forward simulates the bare minimum needed for gradient flow
from a scalar loss back to the AdaMerging coefficients: a stack of linear
layers whose weights are perturbed by the merged task vector's q_proj
entries. It is NOT a real transformer (no attention, no MLP, no positional
encoding) — its sole purpose is to provide a differentiable path.
"""
from __future__ import annotations

from typing import Callable, Iterator

import torch


def make_synthetic_forward_fn(
    hidden_dim: int = 64,
    intermediate_dim: int = 128,  # unused; kept for interface parity with toy_adapter
    vocab_size: int = 100,
    n_layers: int = 2,
    seed: int = 0,
) -> Callable[[dict[str, torch.Tensor], dict], torch.Tensor]:
    """Build a synthetic forward function mimicking a tiny transformer.

    The returned callable:

    1. Embeds ``input_ids`` into hidden states ``[B, T, H]``.
    2. For each layer, uses the canonical key
       ``model.layers.{i}.self_attn.q_proj`` from the merged task vector
       as a perturbation to a random base weight matrix. The hidden state
       is multiplied by ``(base + delta).T`` and passed through ``tanh``.
    3. Multiplies the final hidden state by a random unembed matrix to
       produce logits ``[B, T, V]``.

    All base tensors are fp32; the delta is cast from input dtype to fp32
    via ``.float()`` inside the forward, so gradients flow from the loss
    back to the coefficients through the bf16 merged tensor.

    This is a TOY simulation. It exists only to give the AdaMerging loop
    something differentiable to optimize against.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    embedding = torch.randn(vocab_size, hidden_dim, generator=g, dtype=torch.float32)
    base_layer_weights = [
        torch.randn(hidden_dim, hidden_dim, generator=g, dtype=torch.float32)
        for _ in range(n_layers)
    ]
    unembed = torch.randn(vocab_size, hidden_dim, generator=g, dtype=torch.float32)

    def forward_fn(merged: dict[str, torch.Tensor], batch: dict) -> torch.Tensor:
        input_ids = batch["input_ids"]
        hidden = embedding[input_ids]
        for layer_idx in range(n_layers):
            key = f"model.layers.{layer_idx}.self_attn.q_proj"
            layer_delta = merged[key].float()
            effective_weight = base_layer_weights[layer_idx] + layer_delta
            hidden = torch.matmul(hidden, effective_weight.T)
            hidden = torch.tanh(hidden)
        logits = torch.matmul(hidden, unembed.T)
        return logits

    return forward_fn


def make_synthetic_data_iter(
    n_tasks: int = 4,
    batch_size: int = 2,
    seq_len: int = 8,
    vocab_size: int = 100,
    n_batches: int = 200,
    seed: int = 0,
) -> Iterator[tuple[int, dict]]:
    """Yield ``(domain_idx, batch)`` tuples for synthetic AdaMerging tests.

    Domains rotate round-robin (0, 1, ..., N-1, 0, 1, ...). Each batch is
    fully random; the synthetic forward does not care about ``input_ids``
    semantics, only that they index into the embedding matrix.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    def iterator() -> Iterator[tuple[int, dict]]:
        for i in range(n_batches):
            domain_idx = i % n_tasks
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
            attention_mask = torch.ones_like(input_ids)
            batch = {"input_ids": input_ids, "attention_mask": attention_mask}
            yield (domain_idx, batch)

    return iterator()

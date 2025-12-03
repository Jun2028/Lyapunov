"""
Shared helpers for RL fine-tuning.
"""

from __future__ import annotations

from typing import List

import torch


def slice_batch_sequences(tensor: torch.Tensor, lengths: torch.Tensor) -> List[torch.Tensor]:
    """
    Extract per-sample sequences from a padded batch tensor.

    Args:
        tensor: (seq_len, batch) tensor of token ids.
        lengths: (batch,) valid sequence lengths.

    Returns:
        List of 1D CPU tensors (one per batch element) trimmed to the provided lengths.
    """

    sequences = []
    for idx in range(lengths.size(0)):
        L = int(lengths[idx].item())
        sequences.append(tensor[:L, idx].detach().cpu())
    return sequences

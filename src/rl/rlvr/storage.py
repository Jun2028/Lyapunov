from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


class SequenceRolloutStorage:
    """Buffer for per-sequence rollout statistics."""

    def __init__(self, pad_values: Optional[Dict[str, float]] = None) -> None:
        self._chunks: List[Dict[str, torch.Tensor]] = []
        self.pad_values = pad_values or {}

    def __len__(self) -> int:
        # Each chunk stores rollout tensors keyed by names like "actions", "token_log_probs", etc.
        # "actions" is always present, so use it to count stored sequences.
        return sum(chunk["actions"].shape[0] for chunk in self._chunks)

    def clear(self) -> None:
        self._chunks.clear()

    def add(self, **tensors: torch.Tensor) -> None:
        if not tensors:
            raise ValueError("SequenceRolloutStorage.add requires tensors")
        batch = next(iter(tensors.values()))
        size = batch.shape[0]
        for name, tensor in tensors.items():
            if tensor.shape[0] != size:
                raise ValueError(f"Tensor {name} has mismatched batch size {tensor.shape[0]} != {size}")
        self._chunks.append(tensors)

    def concat(self) -> Dict[str, torch.Tensor]:
        if not self._chunks:
            raise RuntimeError("SequenceRolloutStorage is empty")
        keys = self._chunks[0].keys()
        concatenated: Dict[str, torch.Tensor] = {}
        for key in keys:
            tensors = [chunk[key] for chunk in self._chunks]
            if key in self.pad_values:
                tensors = self._pad_to_max_len(tensors, self.pad_values[key])
            concatenated[key] = torch.cat(tensors, dim=0)
        return concatenated

    @staticmethod
    def _pad_to_max_len(tensors: List[torch.Tensor], value: float) -> List[torch.Tensor]:
        if not tensors:
            return tensors
        max_len = max(tensor.shape[1] for tensor in tensors)
        if all(tensor.shape[1] == max_len for tensor in tensors):
            return tensors
        padded: List[torch.Tensor] = []
        for tensor in tensors:
            pad_len = max_len - tensor.shape[1]
            if pad_len <= 0:
                padded.append(tensor)
                continue
            if tensor.dim() == 2:
                padded.append(F.pad(tensor, (0, pad_len), value=value))
            elif tensor.dim() == 3:
                padded.append(F.pad(tensor, (0, 0, 0, pad_len), value=value))
            elif tensor.dim() == 4:
                padded.append(F.pad(tensor, (0, 0, 0, pad_len), value=value))
            else:
                raise ValueError(f"Unsupported tensor rank {tensor.dim()} for padding")
        return padded

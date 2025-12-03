"""
Buffer utilities for collecting PPO rollouts.
generate rollouts → storage.add(**rollout.as_dict()) → once have enough data, call storage.iter_minibatches(...) during the PPO update loop. 
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional

import torch


REQUIRED_KEYS = ["input_ids", "attention_mask", "actions", "log_probs", "values", "rewards", "dones", "action_mask"]


@dataclass
class RolloutChunk:
    """
    Container for a batch of trajectories gathered during rollout.
    Assumes tensors share the same batch dimension and sequence length.
    """

    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None: #check that all the required keys are present and the tensor shapes
        if not self.tensors:
            raise ValueError("RolloutChunk requires at least one tensor.")
        missing = [key for key in REQUIRED_KEYS if key not in self.tensors]
        if missing:
            raise KeyError(f"Missing rollout tensors: {missing}")

        # Sanity-check shapes on the first tensor.
        first = next(iter(self.tensors.values()))
        batch_shape = first.shape[:2]
        for name, tensor in self.tensors.items():
            if tensor.shape[:2] != batch_shape:
                raise ValueError(f"Tensor {name} has mismatched leading shape {tensor.shape[:2]} != {batch_shape}")

    @property
    def device(self) -> torch.device:
        return next(iter(self.tensors.values())).device

    def to(self, device: torch.device) -> "RolloutChunk":
        """
        Move all tensors to the provided device.
        """

        moved = {k: v.to(device) for k, v in self.tensors.items()}
        return RolloutChunk(moved)

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return self.tensors


class RolloutStorage:
    """
    Accumulates RolloutChunk objects and provides minibatch iterators for PPO updates.
    """

    def __init__(self) -> None:
        self._chunks: List[RolloutChunk] = []

    def __len__(self) -> int:
        return sum(chunk.tensors[REQUIRED_KEYS[0]].shape[0] for chunk in self._chunks)

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()

    def add(self, **tensors: torch.Tensor) -> None:
        """
        Append a new rollout chunk.
        """

        chunk = RolloutChunk(tensors)
        self._chunks.append(chunk)

    def extend(self, chunks: Iterable[RolloutChunk]) -> None:
        for chunk in chunks:
            if not isinstance(chunk, RolloutChunk):
                raise TypeError(f"Expected RolloutChunk, received {type(chunk)}")
            self._chunks.append(chunk)

    def concat(self) -> Dict[str, torch.Tensor]:
        """
        Concatenate stored tensors along the batch dimension.
        """

        if not self._chunks:
            raise RuntimeError("RolloutStorage is empty.")
        cat_tensors: Dict[str, List[torch.Tensor]] = {key: [] for key in self._chunks[0].tensors.keys()}
        for chunk in self._chunks:
            for key, tensor in chunk.tensors.items():
                cat_tensors[key].append(tensor)
        return {key: torch.cat(value, dim=0) for key, value in cat_tensors.items()}

    def iter_minibatches(
        self,
        num_minibatches: int,
        shuffle: bool = True,
        data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Yield flattened minibatches for PPO updates.
        Optionally accepts a precomputed tensor dict (e.g., with extra keys like advantages).
        """

        if data is None:
            data = self.concat()
        batch_size = data[next(iter(data.keys()))].shape[0]
        if num_minibatches <= 0:
            raise ValueError("num_minibatches must be >= 1")
        minibatch_size = batch_size // num_minibatches
        if minibatch_size == 0:
            minibatch_size = batch_size
            num_minibatches = 1

        indices = torch.arange(batch_size, device=data[next(iter(data.keys()))].device)
        if shuffle:
            indices = indices[torch.randperm(batch_size, device=indices.device)]

        for start in range(0, batch_size, minibatch_size):
            end = min(start + minibatch_size, batch_size)
            batch_idx = indices[start:end]
            yield {key: value.index_select(0, batch_idx) for key, value in data.items()} # yield the selected minibatch

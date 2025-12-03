"""
Critic utilities for PPO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ValueLoss:
    mse: torch.Tensor #mean squared error
    clipped: Optional[torch.Tensor] = None

    @property
    def total(self) -> torch.Tensor:
        if self.clipped is None:
            return self.mse
        return torch.max(self.mse, self.clipped)


class ValueHead(nn.Module):
    """
    Scalar projection used for the value baseline.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__() #necessary for any nn.Module subclass
        self.proj = nn.Linear(hidden_dim, 1) #project hidden states to a single value

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states).squeeze(-1) #.squeeze(-1) drops the last dimension, which is size 1


def value_loss(
    new_values: torch.Tensor,
    target_returns: torch.Tensor,
    old_values: Optional[torch.Tensor],
    clip_range: Optional[float],
) -> torch.Tensor:
    """
    Compute PPO's clipped value loss.

    Args:
        new_values: Current critic predictions for each token / timestep,
            shaped (batch, seq_len) to match the rollout layout.
        target_returns: Bootstrapped returns used as regression targets,
            same shape as `new_values`.
        old_values: Critic outputs that were recorded during rollout
            collection (batch, seq_len); required for clipping.
        clip_range: Symmetric bound applied to (new - old). If None,
            reverts to plain MSE.

    Returns:
        Scalar tensor with the mean of max(L_unclipped, L_clipped), mirroring
        the policy clip objective so the critic cannot move farther than
        `clip_range` away from `old_values` in a single update.
    """

    value_pred = new_values
    if clip_range is not None and old_values is not None:
        clipped_values = old_values + (new_values - old_values).clamp(-clip_range, clip_range) #bound the change in value predictions, prevent overfitting to noise
        loss_unclipped = F.mse_loss(value_pred, target_returns, reduction="none")
        loss_clipped = F.mse_loss(clipped_values, target_returns, reduction="none")
        return torch.max(loss_unclipped, loss_clipped).mean() 
    return F.mse_loss(value_pred, target_returns) #reduction="mean" by default

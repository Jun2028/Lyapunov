"""
Advantage and return utilities for PPO.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .config import PPOConfig


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    config: PPOConfig,
    next_value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimator (GAE-Lambda).

    Args:
        rewards: Tensor (batch, seq_len).
        values: Tensor (batch, seq_len).
        dones: Tensor (batch, seq_len) with 1 where episode ended.
        config: PPOConfig containing gamma / gae_lambda.
        next_value: Tensor (batch,) bootstrap value from the last state.

    Returns:
        advantages, returns (matching shape of rewards).
    """

    gamma = config.gamma
    lam = config.gae_lambda

    batch, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)
    if mask is None:
        mask = torch.ones_like(rewards)

    for t in reversed(range(seq_len)):
        mask_t = mask[:, t]
        not_done = 1.0 - dones[:, t]
        if t < seq_len - 1:
            value_next = values[:, t + 1]
            mask_next = mask[:, t + 1]
        else:
            value_next = next_value
            mask_next = torch.zeros_like(mask_t)
        delta = rewards[:, t] + gamma * value_next * mask_next * not_done - values[:, t]
        last_adv = delta + gamma * lam * mask_next * not_done * last_adv
        advantages[:, t] = last_adv * mask_t

    returns = (advantages + values) * mask
    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Normalize advantages across the flattened batch.
    """

    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + epsilon)

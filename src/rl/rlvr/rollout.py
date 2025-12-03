from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..policy_api import TransformerPolicy
from ..rollout import RolloutGenerator as TokenRolloutGenerator


class _ZeroReward:
    def __call__(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return torch.zeros(sequences.size(0), device=sequences.device, dtype=sequences.dtype)


@dataclass
class SequenceRolloutResult:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    actions: torch.Tensor
    action_mask: torch.Tensor
    values: torch.Tensor
    lengths: torch.Tensor


class SequenceRolloutGenerator:
    """Sample full sequences and aggregate log-prob / value statistics."""

    def __init__(
        self,
        policy: TransformerPolicy,
        max_length: int,
        eos_token_id: int,
        pad_token_id: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> None:
        self.policy = policy
        self.generator = TokenRolloutGenerator(
            policy=policy,
            reward_fn=_ZeroReward(),
            max_length=max_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            temperature=temperature,
            top_k=top_k,
        )

    @torch.no_grad()
    def generate(
        self,
        prompts: torch.Tensor,
        src_enc: Optional[torch.Tensor] = None,
        src_len: Optional[torch.Tensor] = None,
    ) -> SequenceRolloutResult:
        token_result = self.generator.generate(prompts=prompts, src_enc=src_enc, src_len=src_len)
        lengths = torch.clamp(token_result.action_mask.sum(dim=1).long(), min=1)
        batch = token_result.values.size(0)
        batch_idx = torch.arange(batch, device=token_result.values.device)
        last_pos = lengths - 1
        values = token_result.values[batch_idx, last_pos]
        return SequenceRolloutResult(
            input_ids=token_result.input_ids,
            attention_mask=token_result.attention_mask,
            actions=token_result.actions,
            action_mask=token_result.action_mask,
            values=values,
            lengths=lengths,
        )

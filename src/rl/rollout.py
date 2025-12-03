"""
Token-level rollout utilities for PPO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .policy_api import TransformerPolicy, PolicyOutput

RewardFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor] #defines a function type that takes in two tensors and returns a tensor


@dataclass
class RolloutResult:
    """
    Container for a batch of sampled sequences and per-step statistics.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    action_mask: torch.Tensor


class RolloutGenerator:
    """
    Autoregressively sample from the policy to build PPO rollouts.
    """

    def __init__(
        self,
        policy: TransformerPolicy,
        reward_fn: RewardFn,
        max_length: int,
        eos_token_id: int,
        pad_token_id: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> None:
        self.policy = policy 
        self.reward_fn = reward_fn
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.temperature = temperature
        self.top_k = top_k

    @torch.no_grad()
    def generate(
        self,
        prompts: torch.Tensor, 
        src_enc: Optional[torch.Tensor] = None,
        src_len: Optional[torch.Tensor] = None,
    ) -> RolloutResult:
        """
        Args:
            prompts: Initial tokens (seq_len, batch). probably torch.full((1, batch_size), bos_id, device=device) somewhere
            src_enc/src_len: Optional encoder context for seq2seq tasks.
        """

        device = prompts.device
        max_length = self.max_length
        seq = prompts.clone() # (seq_len, batch)
        batch = seq.size(1)
        lengths = torch.full((batch,), seq.size(0), dtype=torch.long, device=device)
        actions_full = torch.full((batch, max_length), self.pad_token_id, device=device, dtype=seq.dtype)
        initial_tokens = min(prompts.size(0), max_length)
        if initial_tokens > 0:
            actions_full[:, :initial_tokens] = prompts[:initial_tokens].transpose(0, 1)
        log_probs_full = torch.zeros(batch, max_length, device=device)
        values_full = torch.zeros(batch, max_length, device=device)
        action_mask = torch.zeros(batch, max_length, device=device)

        done_flags = torch.zeros(batch, dtype=torch.bool, device=device)

        while seq.size(0) < max_length and not done_flags.all():
            lengths = torch.where(done_flags, lengths, torch.full_like(lengths, seq.size(0)))
            active_mask = (~done_flags).float()
            output = self.policy(
                input_ids=seq, 
                lengths=lengths,
                src_enc=src_enc,
                src_len=src_len,
                causal=True,
            ) #module(*args, **kwargs) is just syntactic sugar for module.forward(*args, **kwargs)
            next_logits = output.logits[:, -1, :] / max(self.temperature, 1e-6)
            next_values = output.values[:, -1]
            sampled_tokens, token_log_probs = self._sample_tokens(next_logits)
            raw_tokens = sampled_tokens
            sampled_tokens = torch.where(done_flags, torch.full_like(sampled_tokens, self.pad_token_id), sampled_tokens)
            token_log_probs = torch.where(done_flags, torch.zeros_like(token_log_probs), token_log_probs)
            next_values = torch.where(done_flags, torch.zeros_like(next_values), next_values)
            seq = torch.cat([seq, sampled_tokens.unsqueeze(0)], dim=0) #[seq[0], seq[1], ..., seq[T-1], sampled_tokens]
            position = seq.size(0) - 1
            actions_full[:, position] = sampled_tokens
            log_probs_full[:, position] = token_log_probs
            values_full[:, position] = next_values
            action_mask[:, position] = active_mask
            newly_finished = (~done_flags) & raw_tokens.eq(self.eos_token_id)
            done_flags |= newly_finished #done_flags is a boolean tensor marking which batch items have already hit EOS. |= is an in-place logical OR operation

        if seq.size(0) < max_length:
            pad_len = max_length - seq.size(0)
            pad = torch.full((pad_len, batch), self.pad_token_id, device=device, dtype=seq.dtype)
            seq = torch.cat([seq, pad], dim=0)

        input_ids = seq  # (max_length, batch)
        attention_mask = (input_ids != self.pad_token_id).long().transpose(0, 1) #.long() to convert bool to int (0/1) (batch, seq_len)

        traj_lengths = self._compute_lengths(input_ids)
        returns = self.reward_fn(input_ids.transpose(0, 1), traj_lengths) #first tensor has shape (batch, max_length), each row is one sampled trajectory
        rewards = torch.zeros(batch, max_length, device=device)
        dones = torch.zeros_like(rewards)
        for idx in range(batch):
            final_step = traj_lengths[idx] - 1
            rewards[idx, final_step] = returns[idx] #only the final step has non-zero reward
            dones[idx, final_step] = 1.0

        return RolloutResult(
            input_ids=input_ids.transpose(0, 1),
            attention_mask=attention_mask,
            actions=actions_full,
            log_probs=log_probs_full,
            values=values_full,
            rewards=rewards,
            dones=dones,
            action_mask=action_mask,
        )

    def _sample_tokens(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.top_k is not None and self.top_k > 0:
            top_k = min(self.top_k, logits.size(-1))
            top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_logits, dim=-1)
            sampled_idx = torch.multinomial(probs, 1).squeeze(-1)
            sampled_tokens = top_indices.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)
            log_probs = torch.log(probs.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1) + 1e-12)
            return sampled_tokens, log_probs
        dist = torch.distributions.Categorical(logits=logits)
        sampled_tokens = dist.sample()
        log_probs = dist.log_prob(sampled_tokens)
        return sampled_tokens, log_probs

    def _compute_lengths(self, input_ids: torch.Tensor) -> torch.Tensor:
        eos_mask = input_ids.eq(self.eos_token_id)
        eos_positions = eos_mask.float().argmax(dim=0)
        eos_exists = eos_mask.any(dim=0)
        default_len = torch.full((input_ids.size(1),), input_ids.size(0), device=input_ids.device, dtype=torch.long)
        lengths = torch.where(eos_exists, eos_positions.long() + 1, default_len)
        return lengths

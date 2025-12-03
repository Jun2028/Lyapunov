from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
from logging import getLogger

import torch
from torch import nn, optim

from ..policy_api import TransformerPolicy
from ..reward import RewardFunction
from .config import RLVRConfig
from .rollout import SequenceRolloutGenerator
from .storage import SequenceRolloutStorage

logger = getLogger(__name__)


@dataclass
class RLVRStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    valid_rate: float


class RLVRTrainer:
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        reward_fn: RewardFunction,
        config: RLVRConfig,
        optimizer: Optional[optim.Optimizer] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        debug_stats: bool = False,
    ) -> None:
        self.config = config
        self.policy = TransformerPolicy(decoder)
        self.encoder = encoder
        self.reward_fn = reward_fn
        pad_values = {
            "src_tokens": float(self.policy.decoder_module.pad_index),
            "src_enc": 0.0,
            "token_log_probs": 0.0,
        }
        self.storage = SequenceRolloutStorage(pad_values=pad_values)
        params = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = optimizer or optim.AdamW(params, lr=1e-4)
        self.generator = SequenceRolloutGenerator(
            policy=self.policy,
            max_length=config.rollout_length,
            eos_token_id=self.policy.decoder_module.eos_index,
            pad_token_id=self.policy.decoder_module.pad_index,
            temperature=temperature,
            top_k=top_k,
        )
        self.debug_stats = debug_stats

    def collect_rollouts(self, prompts: torch.Tensor, src_enc: torch.Tensor, src_len: torch.Tensor, src_tokens: torch.Tensor) -> None:
        result = self.generator.generate(prompts=prompts, src_enc=src_enc, src_len=src_len)
        rewards = self.reward_fn(result.input_ids, result.lengths)
        with torch.no_grad():
            lengths = result.attention_mask.sum(dim=1)
            tf_output = self.policy(
                input_ids=result.input_ids.transpose(0, 1),
                lengths=lengths,
                src_enc=src_enc,
                src_len=src_len,
            )
            token_log_probs = self.policy.compute_log_probs(tf_output.logits, result.actions)
        self.storage.add(
            input_ids=result.input_ids,
            attention_mask=result.attention_mask,
            actions=result.actions,
            action_mask=result.action_mask,
            token_log_probs=token_log_probs.detach(),
            values=result.values,
            rewards=rewards,
            src_tokens=src_tokens.transpose(0, 1).contiguous().detach(),
            src_enc=src_enc.detach(),
            src_len=src_len.clone(),
        )

    def update(self) -> RLVRStats:
        """
        Run one PPO update sweep over the rollouts currently in storage.

        Steps:
        1) Concatenate stored trajectories and build per-sample advantages as (reward - value). Optionally normalize advantages.
        2) Set returns to the observed rewards and compute a simple valid_rate (fraction of rewards > 0) for logging.
        3) For each epoch, randomly permute the batch, split into minibatches, and call `_update_minibatch` to apply gradients. Each minibatch applies PPO clipping, entropy bonus, and (if enabled) a differentiable KL penalty to the loss.
        4) Clear rollout storage and return the stats from the last minibatch plus the valid_rate.

        Returns:
            RLVRStats: policy/value losses, entropy, approx KL, and valid_rate from the processed rollouts.
        """
        data = self.storage.concat()
        rewards = data["rewards"]
        advantages = rewards - data["values"]
        if self.config.normalize_advantages:
            mean = advantages.mean()
            std = advantages.std()
            advantages = (advantages - mean) / (std + self.config.advantage_epsilon)
        data["advantages"] = advantages
        data["returns"] = rewards
        valid_rate = (rewards > 0).float().mean().item()
        stats = []
        batch_size = data["actions"].shape[0]
        first_tensor = next(iter(data.values()))
        device = first_tensor.device
        minibatch_size = max(batch_size // self.config.num_minibatches, 1)
        for _ in range(self.config.update_epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)
                idx = indices[start:end]
                batch = {key: value.index_select(0, idx) for key, value in data.items()}
                batch_stats = self._update_minibatch(batch)
                stats.append(batch_stats)
        self.storage.clear()
        if stats:
            final = stats[-1]
            return RLVRStats(final.policy_loss, final.value_loss, final.entropy, final.approx_kl, valid_rate)
        return RLVRStats(0, 0, 0, 0, valid_rate)

    def _update_minibatch(self, batch: Dict[str, torch.Tensor]) -> RLVRStats:
        lengths = batch["attention_mask"].sum(dim=1)
        src_enc = batch.get("src_enc")
        src_len = batch.get("src_len")
        decoder_device = next(self.policy.decoder_module.parameters()).device
        if src_len is not None:
            src_len = src_len.to(decoder_device)
        if src_enc is not None:
            src_enc = src_enc.to(decoder_device)
            if src_len is not None:
                max_ctx = int(src_len.max().item())
                if src_enc.size(1) > max_ctx:
                    src_enc = src_enc[:, :max_ctx, :]
        new_output = self.policy(
            input_ids=batch["input_ids"].transpose(0, 1),
            lengths=lengths,
            src_enc=src_enc,
            src_len=src_len,
        )
        new_token_log_probs = self.policy.compute_log_probs(new_output.logits, batch["actions"])
        old_token_log_probs = batch["token_log_probs"].to(new_token_log_probs.dtype)
        mask = batch["action_mask"].to(new_token_log_probs.dtype)
        valid_tokens = mask.sum().clamp_min(1.0)
        log_ratio = (new_token_log_probs - old_token_log_probs) * mask
        ratio = torch.exp(log_ratio) * mask + (1.0 - mask)
        advantages = batch["advantages"].to(new_token_log_probs.dtype).unsqueeze(1)
        advantages = advantages * mask
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * advantages
        policy_loss = -(torch.min(surr1, surr2) * mask).sum() / valid_tokens
        seq_values = self._gather_final_values(new_output.values, batch["action_mask"])

        value_error = seq_values - batch["returns"]
        value_loss = value_error.pow(2).mean()
        entropy = self.policy.entropy(new_output.logits, mask=batch["action_mask"])
        diff = (old_token_log_probs - new_token_log_probs).to(torch.float32) * mask
        approx_kl = 0.5 * diff.pow(2).sum() / valid_tokens
        approx_kl_value = approx_kl.detach().item()
        kl_penalty = 0.0
        if self.config.use_kl_penalty and self.config.kl_coef > 0:
            kl_penalty = self.config.kl_coef * approx_kl

        self.optimizer.zero_grad()
        loss = policy_loss + self.config.value_coef * value_loss - self.config.ent_coef * entropy + kl_penalty #loss term
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        if self.debug_stats:
            logger.info(
                "[rlvr-debug] batch=%d policy_loss=%.4f value_loss=%.4f entropy=%.4f kl=%.4f ratio_mean=%.4f",
                batch["actions"].shape[0],
                policy_loss.item(),
                value_loss.item(),
                entropy.item(),
                approx_kl_value,
                ratio.mean().item(),
            )
        return RLVRStats(
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy=entropy.item(),
            approx_kl=approx_kl_value,
            valid_rate=0.0,
        )

    @staticmethod
    def _gather_final_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = torch.clamp(mask.sum(dim=1).long(), min=1)
        idx = lengths - 1
        batch_idx = torch.arange(values.size(0), device=values.device)
        return values[batch_idx, idx]

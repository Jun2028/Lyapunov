"""
High-level PPO trainer tying together rollouts, advantage estimation, and optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional
from logging import getLogger

import torch
from torch import nn, optim

from .config import PPOConfig, build_ppo_config
from .policy_api import TransformerPolicy
from .rollout import RolloutGenerator
from .storage import RolloutStorage
from .advantage import compute_gae, normalize_advantages
from .reward import RewardFunction

logger = getLogger(__name__)


@dataclass
class PPOStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float


class PPOTrainer:
    def __init__(
        self,
        decoder: nn.Module,
        reward_fn: RewardFunction,
        config: PPOConfig,
        optimizer: Optional[optim.Optimizer] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        debug_stats: bool = False,
    ) -> None:
        self.config = config
        self.policy = TransformerPolicy(decoder)
        self.reward_fn = reward_fn
        self.rollout_storage = RolloutStorage()
        params = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = optimizer or optim.AdamW(params, lr=1e-4)
        self.sample_temperature = temperature
        self.sample_top_k = top_k
        self.debug_stats = debug_stats

    def collect_rollouts(self, prompts: torch.Tensor, **rollout_kwargs) -> None:
        generator = RolloutGenerator(
            policy=self.policy,
            reward_fn=self.reward_fn,
            max_length=self.config.rollout_length,
            eos_token_id=self.policy.decoder_module.eos_index,
            pad_token_id=self.policy.decoder_module.pad_index,
            temperature=self.sample_temperature,
            top_k=self.sample_top_k,
        )
        result = generator.generate(prompts=prompts, **rollout_kwargs)
        self.rollout_storage.add(
            input_ids=result.input_ids,
            attention_mask=result.attention_mask,
            actions=result.actions,
            log_probs=result.log_probs,
            values=result.values,
            rewards=result.rewards,
            dones=result.dones,
            action_mask=result.action_mask,
        )

    def update(self) -> PPOStats:
        data = self.rollout_storage.concat()
        advantages, returns = compute_gae(
            rewards=data["rewards"],
            values=data["values"],
            dones=data["dones"],
            config=self.config,
            next_value=torch.zeros_like(data["values"][:, -1]),
            mask=data["action_mask"],
        )
        if self.config.normalize_advantages:
            advantages = normalize_advantages(advantages, self.config.advantage_epsilon)
        data["advantages"] = advantages
        data["returns"] = returns
        if self.debug_stats:
            self._log_rollout_stats(data, advantages, returns)

        stats = []
        for _ in range(self.config.update_epochs):
            for batch in self.rollout_storage.iter_minibatches(self.config.num_minibatches, data=data):
                mask = batch["action_mask"]
                valid_tokens = mask.sum().clamp_min(1.0)
                batch_adv = batch["advantages"] * mask
                new_output = self.policy(
                    input_ids=batch["input_ids"].transpose(0, 1),
                    lengths=batch["attention_mask"].sum(dim=1),
                )
                new_log_probs = self.policy.compute_log_probs(new_output.logits, batch["actions"])
                ratio = torch.exp(new_log_probs - batch["log_probs"])
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * batch_adv
                policy_loss = - (torch.min(surr1, surr2) * mask).sum() / valid_tokens

                value_error = (new_output.values - batch["returns"])
                value_loss = (value_error.pow(2) * mask).sum() / valid_tokens
                entropy = self.policy.entropy(new_output.logits, mask=mask)
                approx_kl = 0.5 * (((batch["log_probs"] - new_log_probs) ** 2) * mask).sum() / valid_tokens
                approx_kl_value = approx_kl.detach().item()
                kl_penalty = 0.0
                if self.config.use_kl_penalty and self.config.kl_coef > 0:
                    kl_penalty = self.config.kl_coef * approx_kl
                self.optimizer.zero_grad()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.ent_coef * entropy
                    + kl_penalty
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                if self.debug_stats:
                    self._log_minibatch_stats(batch, mask, ratio, approx_kl_value)
                stats.append(
                    PPOStats(
                        policy_loss=policy_loss.item(),
                        value_loss=value_loss.item(),
                        entropy=entropy.item(),
                        approx_kl=approx_kl_value,
                    )
                )
        self.rollout_storage.clear()
        return stats[-1] if stats else PPOStats(0, 0, 0, 0)

    def _log_rollout_stats(self, data: Dict[str, torch.Tensor], advantages: torch.Tensor, returns: torch.Tensor) -> None:
        mask = data["action_mask"]
        token_count = mask.sum().item()
        rewards_stats = self._masked_stats(data["rewards"], mask)
        values_stats = self._masked_stats(data["values"], mask)
        adv_stats = self._masked_stats(advantages, mask)
        return_stats = self._masked_stats(returns, mask)
        logger.info(
            "[ppo-debug] rollout stats | tokens=%.0f rewards(mean=%.4f,std=%.4f,min=%.4f,max=%.4f) "
            "advantages(mean=%.4f,std=%.4f,min=%.4f,max=%.4f) values(mean=%.4f,std=%.4f,min=%.4f,max=%.4f) "
            "returns(mean=%.4f,std=%.4f,min=%.4f,max=%.4f)",
            token_count,
            *rewards_stats,
            *adv_stats,
            *values_stats,
            *return_stats,
        )

    def _log_minibatch_stats(
        self,
        batch: Dict[str, torch.Tensor],
        mask: torch.Tensor,
        ratio: torch.Tensor,
        approx_kl_value: float,
    ) -> None:
        mask_tokens = mask.sum().item()
        ratio_min = ratio.min().item()
        ratio_max = ratio.max().item()
        ratio_mean = ratio.mean().item()
        ratio_std = ratio.std(unbiased=False).item()
        adv_abs = batch["advantages"].abs()
        adv_max = adv_abs.max().item()
        logger.info(
            "[ppo-debug] minibatch stats | tokens=%.0f ratio(min=%.4f,max=%.4f,mean=%.4f,std=%.4f) "
            "adv_abs_max=%.4f approx_kl=%.4f",
            mask_tokens,
            ratio_min,
            ratio_max,
            ratio_mean,
            ratio_std,
            adv_max,
            approx_kl_value,
        )

    @staticmethod
    def _masked_stats(tensor: torch.Tensor, mask: torch.Tensor) -> tuple[float, float, float, float]:
        masked = tensor[mask.bool()]
        if masked.numel() == 0:
            return 0.0, 0.0, 0.0, 0.0
        return (
            masked.mean().item(),
            masked.std(unbiased=False).item(),
            masked.min().item(),
            masked.max().item(),
        )

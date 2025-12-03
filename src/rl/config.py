"""
Configuration helpers for PPO fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PPOConfig:
    """
    Container for PPO hyperparameters.
    """

    rollout_length: int = 1024
    num_envs: int = 1
    update_epochs: int = 6
    num_minibatches: int = 6
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: Optional[float] = None
    kl_coef: float = 0.0
    use_kl_penalty: bool = False
    reward_clip: Optional[float] = None
    normalize_advantages: bool = True
    advantage_epsilon: float = 1e-8


def register_rl_args(parser) -> None:
    """
    Add RL-specific options to the global argument parser.
    """

    group = parser.add_argument_group("ppo")
    group.add_argument("--rl_rollout_length", type=int, default=PPOConfig.rollout_length, help="Tokens per PPO rollout.")
    group.add_argument("--rl_num_envs", type=int, default=PPOConfig.num_envs, help="Independent rollout workers.")
    group.add_argument("--rl_update_epochs", type=int, default=PPOConfig.update_epochs, help="Gradient passes per batch.")
    group.add_argument("--rl_num_minibatches", type=int, default=PPOConfig.num_minibatches, help="Minibatches per rollout batch.")
    group.add_argument("--rl_gamma", type=float, default=PPOConfig.gamma, help="Discount factor.")
    group.add_argument("--rl_gae_lambda", type=float, default=PPOConfig.gae_lambda, help="GAE lambda.")
    group.add_argument("--rl_clip_range", type=float, default=PPOConfig.clip_range, help="Clipping range for PPO.")
    group.add_argument(
        "--rl_clip_range_vf",
        type=float,
        default=PPOConfig.clip_range_vf if PPOConfig.clip_range_vf is not None else -1.0,
        help="Optional value function clipping range (-1 to disable).",
    )
    group.add_argument("--rl_ent_coef", type=float, default=PPOConfig.ent_coef, help="Entropy regularization coefficient.")
    group.add_argument("--rl_value_coef", type=float, default=PPOConfig.value_coef, help="Value loss coefficient.")
    group.add_argument("--rl_max_grad_norm", type=float, default=PPOConfig.max_grad_norm, help="Gradient clipping norm.")
    group.add_argument(
        "--rl_target_kl",
        type=float,
        default=PPOConfig.target_kl if PPOConfig.target_kl is not None else -1.0,
        help="Target KL for early stopping (-1 to disable).",
    )
    group.add_argument("--rl_kl_coef", type=float, default=PPOConfig.kl_coef, help="Initial KL penalty coefficient.")
    group.add_argument(
        "--rl_use_kl_penalty",
        type=int,
        default=int(PPOConfig.use_kl_penalty),
        help="1 to enable KL-penalty PPO, 0 for clipping.",
    )
    group.add_argument(
        "--rl_reward_clip",
        type=float,
        default=PPOConfig.reward_clip if PPOConfig.reward_clip is not None else -1.0,
        help="Clamp rewards to [-clip, clip] (-1 to disable).",
    )
    group.add_argument(
        "--rl_normalize_advantages",
        type=int,
        default=int(PPOConfig.normalize_advantages),
        help="1 to normalize advantage estimates within each batch.",
    )
    group.add_argument(
        "--rl_advantage_epsilon",
        type=float,
        default=PPOConfig.advantage_epsilon,
        help="Stability epsilon for advantage normalization.",
    )


def build_ppo_config(params) -> PPOConfig:
    """
    Instantiate a PPOConfig from the global argparse Namespace.
    """

    clip_range_vf = None if getattr(params, "rl_clip_range_vf", -1.0) < 0 else params.rl_clip_range_vf
    target_kl = None if getattr(params, "rl_target_kl", -1.0) < 0 else params.rl_target_kl
    reward_clip = None if getattr(params, "rl_reward_clip", -1.0) < 0 else params.rl_reward_clip

    return PPOConfig(
        rollout_length=params.rl_rollout_length,
        num_envs=params.rl_num_envs,
        update_epochs=params.rl_update_epochs,
        num_minibatches=params.rl_num_minibatches,
        gamma=params.rl_gamma,
        gae_lambda=params.rl_gae_lambda,
        clip_range=params.rl_clip_range,
        clip_range_vf=clip_range_vf,
        ent_coef=params.rl_ent_coef,
        value_coef=params.rl_value_coef,
        max_grad_norm=params.rl_max_grad_norm,
        target_kl=target_kl,
        kl_coef=params.rl_kl_coef,
        use_kl_penalty=bool(params.rl_use_kl_penalty),
        reward_clip=reward_clip,
        normalize_advantages=bool(params.rl_normalize_advantages),
        advantage_epsilon=params.rl_advantage_epsilon,
    )

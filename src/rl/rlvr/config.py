from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RLVRConfig:
    rollout_length: int = 1024
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_range: float = 0.1
    ent_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    normalize_advantages: bool = True
    advantage_epsilon: float = 1e-8
    use_kl_penalty: bool = False
    kl_coef: float = 0.0


def register_rlvr_args(parser) -> None:
    group = parser.add_argument_group("rlvr")
    group.add_argument("--rl_total_updates", type=int, default=100000, help="Number of RL updates to run.")
    group.add_argument("--rl_rollouts_per_update", type=int, default=1, help="How many rollout batches before each update.")
    group.add_argument("--rl_rollout_length", type=int, default=RLVRConfig.rollout_length, help="Maximum sampled tokens per trajectory.")
    group.add_argument("--rl_update_epochs", type=int, default=RLVRConfig.update_epochs, help="Gradient passes per update batch.")
    group.add_argument("--rl_num_minibatches", type=int, default=RLVRConfig.num_minibatches, help="Minibatches per rollout batch.")
    group.add_argument("--rl_clip_range", type=float, default=RLVRConfig.clip_range, help="Clipping range for PPO ratios.")
    group.add_argument("--rl_ent_coef", type=float, default=RLVRConfig.ent_coef, help="Entropy bonus coefficient.")
    group.add_argument("--rl_value_coef", type=float, default=RLVRConfig.value_coef, help="Value loss coefficient.")
    group.add_argument("--rl_max_grad_norm", type=float, default=RLVRConfig.max_grad_norm, help="Gradient clipping norm.")
    group.add_argument("--rl_lr", type=float, default=1e-4, help="Learning rate for RL optimizer.")
    group.add_argument("--rl_reward_scale", type=float, default=1.0, help="Scalar applied to raw rewards.")
    group.add_argument("--rl_use_kl_penalty", type=int, default=0, help="Set to 1 to enable KL penalty.")
    group.add_argument("--rl_kl_coef", type=float, default=0.0, help="Coefficient for KL penalty term.")
    group.add_argument("--rl_normalize_advantages", type=int, default=1, help="Normalize advantages within each batch.")
    group.add_argument("--rl_advantage_epsilon", type=float, default=RLVRConfig.advantage_epsilon, help="Epsilon for advantage normalization.")
    group.add_argument("--rl_save_period", type=int, default=1000, help="Checkpoint save period (0 to disable).")
    group.add_argument("--rl_debug_stats", type=int, default=0, help="Enable verbose RLVR stats logging.")


def build_rlvr_config(params) -> RLVRConfig:
    return RLVRConfig(
        rollout_length=params.rl_rollout_length,
        num_minibatches=params.rl_num_minibatches,
        update_epochs=params.rl_update_epochs,
        clip_range=params.rl_clip_range,
        ent_coef=params.rl_ent_coef,
        value_coef=params.rl_value_coef,
        max_grad_norm=params.rl_max_grad_norm,
        normalize_advantages=bool(params.rl_normalize_advantages),
        advantage_epsilon=params.rl_advantage_epsilon,
        use_kl_penalty=bool(params.rl_use_kl_penalty),
        kl_coef=params.rl_kl_coef,
    )

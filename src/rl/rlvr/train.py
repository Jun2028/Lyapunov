"""Entry point for single-step RLVR fine-tuning."""

from __future__ import annotations

import json
import os

import torch
from torch import nn, optim

import src
from train import get_parser as get_base_parser
from src.pbs import init_distributed_mode, init_signal_handler
from src.utils import initialize_exp, to_cuda
from src.model import check_model_params, build_modules
from src.envs import build_env

from ..lora import LoRAConfig, apply_lora_to_decoder, only_lora_parameters, lora_state_dict, load_lora_state_dict
from ..reward import LyapunovReward
from ..utils import slice_batch_sequences
from ..evaluator import RLEvaluator
from .config import RLVRConfig, build_rlvr_config, register_rlvr_args
from .trainer import RLVRTrainer


def get_parser():
    parser = get_base_parser()
    register_rlvr_args(parser)
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank of LoRA adapters.")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="Scaling factor for LoRA adapters.")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout applied inside LoRA adapters.")
    parser.add_argument("--rollout_temperature", type=float, default=1.0, help="Sampling temperature for rollouts.")
    parser.add_argument("--rollout_top_k", type=int, default=0, help="Top-k sampling (0 to disable).")
    parser.add_argument("--reload_rlvr_checkpoint", type=str, default="", help="Resume RLVR training from checkpoint path.")
    parser.add_argument("--rl_eval_period", type=int, default=0, help="Run RL evaluation every N updates (0 to disable).")
    parser.add_argument("--rl_eval_split", type=str, default="valid", help="Dataset split to evaluate on (valid/test).")
    return parser


def build_data_path(params):
    if params.reload_data == "":
        return None
    segments = [x.split(",") for x in params.reload_data.split(";") if x]
    data_path = {el[0]: tuple(el[1:]) for el in segments}
    for paths in data_path.values():
        assert all(os.path.isfile(path) for path in paths)
    for task in params.tasks:
        assert task in data_path
    return data_path


def build_iterators(env, params, data_path):
    if params.env_base_seed < 0:
        params.env_base_seed = torch.randint(0, 1_000_000_000, ()).item()
    return {task: iter(env.create_train_iterator(task, data_path, params)) for task in params.tasks}


def fetch_batch(task, env, params, data_path, iterators):
    try:
        return next(iterators[task])
    except StopIteration:
        iterators[task] = iter(env.create_train_iterator(task, data_path, params))
        return next(iterators[task])


def save_checkpoint(params, modules, trainer: RLVRTrainer, step: int):
    if not params.is_master:
        return
    checkpoint = {
        "step": step,
        "lora": lora_state_dict(modules["decoder"]),
        "value_head": trainer.policy.value_head.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "command": getattr(params, "command", ""),
        "config": trainer.config.__dict__,
    }
    path = os.path.join(params.dump_path, f"rlvr_step_{step:06d}.pth")
    torch.save(checkpoint, path)


def load_checkpoint(path: str, modules, trainer: RLVRTrainer) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    load_lora_state_dict(modules["decoder"], checkpoint["lora"])
    trainer.policy.value_head.load_state_dict(checkpoint["value_head"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("step", 0))


def main(params):
    init_distributed_mode(params)
    logger = initialize_exp(params)
    init_signal_handler()
    src.utils.CUDA = not params.cpu

    env = build_env(params)
    modules = build_modules(env, params)
    encoder, decoder = modules["encoder"], modules["decoder"]
    encoder.eval()
    
    decoder.eval()  # disable dropout

    for p in encoder.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False

    lora_cfg = LoRAConfig(rank=params.lora_rank, alpha=params.lora_alpha, dropout=params.lora_dropout)
    apply_lora_to_decoder(decoder, lora_cfg)

    if params.multi_gpu:
        for key in modules.keys():
            modules[key] = nn.parallel.DistributedDataParallel(
                modules[key], device_ids=[params.local_rank], output_device=params.local_rank, broadcast_buffers=True
            )
        encoder, decoder = modules["encoder"], modules["decoder"]

    reward_fn = LyapunovReward(env, reward_scale=params.rl_reward_scale)
    rlvr_config = build_rlvr_config(params)
    trainer = RLVRTrainer(
        encoder=encoder,
        decoder=decoder,
        reward_fn=reward_fn,
        config=rlvr_config,
        temperature=params.rollout_temperature,
        top_k=(params.rollout_top_k if params.rollout_top_k > 0 else None),
        debug_stats=bool(params.rl_debug_stats),
    )

    try:
        decoder_device = next(trainer.policy.decoder_module.parameters()).device
        trainer.policy.value_head.to(decoder_device)
    except StopIteration:
        pass

    trainable = list({id(p): p for p in only_lora_parameters(trainer.policy.decoder_module) + list(trainer.policy.value_head.parameters())}.values())
    trainer.optimizer = optim.AdamW(trainable, lr=params.rl_lr)

    completed = 0
    if params.reload_rlvr_checkpoint:
        completed = load_checkpoint(params.reload_rlvr_checkpoint, modules, trainer)
        logger.info(f"Reloaded RLVR checkpoint from {params.reload_rlvr_checkpoint} (completed_updates={completed}).")

    data_path = build_data_path(params)
    iterators = build_iterators(env, params, data_path)
    rl_evaluator = None
    if params.rl_eval_period > 0:
        if data_path is None:
            if params.is_master:
                logger.warning("RL evaluation requested but no reload_data provided; disabling evaluation.")
        else:
            rl_evaluator = RLEvaluator(modules, env, params, data_path)

    updates = params.rl_total_updates
    if completed >= updates:
        logger.info("Requested total updates already completed. Exiting.")
        return
    rollouts_per_update = params.rl_rollouts_per_update

    for update_idx in range(completed, updates):
        for rollout_idx in range(rollouts_per_update):
            task = params.tasks[(update_idx * rollouts_per_update + rollout_idx) % len(params.tasks)]
            (x1, len1), (x2, len2), _ = fetch_batch(task, env, params, data_path, iterators)
            len1_cpu = len1.clone()
            len2_cpu = len2.clone()
            src_refs = slice_batch_sequences(x1, len1_cpu)
            tgt_refs = slice_batch_sequences(x2, len2_cpu)
            reward_fn.set_reference_batch(src_refs, tgt_refs)
            x1, len1, x2, len2 = to_cuda(x1, len1, x2, len2)
            encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
            prompts = x2[:1, :].clone()
            src_enc = encoded.transpose(0, 1)
            trainer.collect_rollouts(prompts=prompts, src_enc=src_enc, src_len=len1, src_tokens=x1)

        stats = trainer.update()
        if params.is_master:
            logger.info(
                json.dumps(
                    {
                        "update": update_idx + 1,
                        "policy_loss": stats.policy_loss,
                        "value_loss": stats.value_loss,
                        "entropy": stats.entropy,
                        "approx_kl": stats.approx_kl,
                        "valid_rate": stats.valid_rate,
                    }
                )
            )
        if params.rl_save_period > 0 and (update_idx + 1) % params.rl_save_period == 0:
            save_checkpoint(params, modules, trainer, update_idx + 1)
        if (
            rl_evaluator is not None
            and params.rl_eval_period > 0
            and (update_idx + 1) % params.rl_eval_period == 0
            and params.is_master
        ):
            eval_scores = rl_evaluator.run(data_type=params.rl_eval_split)
            payload = {"update": update_idx + 1}
            payload.update(eval_scores)
            logger.info(json.dumps(payload))


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    check_model_params(params)
    main(params)

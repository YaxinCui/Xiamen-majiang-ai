#!/usr/bin/env python3
"""Measure a training-only privileged critic against an actor value baseline.

The actor is frozen throughout.  A critic is calibrated on one independent
rollout set, then the exact same later wall/opponent samples are collected
twice: once with the actor's public value head and once with the critic.  This
reports only aggregate residual statistics and never writes the critic, hidden
features, opponent hands, or wall composition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit("请使用 .venv/bin/python 运行；该脚本需要 PyTorch") from error

from scripts.train_torch_ppo import (
    PrivilegedCritic,
    PpoStep,
    collect_rollouts_batched,
    train_privileged_critic,
)
from xiamen_mahjong.torch_policy import (
    ARCHITECTURE_CANDIDATE_MLP,
    TorchPolicyValueAgent,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=("classic", "core"), default="classic")
    parser.add_argument("--calibration-episodes", type=int, default=512)
    parser.add_argument("--evaluation-episodes", type=int, default=512)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--critic-epochs", type=int, default=8)
    parser.add_argument("--critic-batch-size", type=int, default=512)
    parser.add_argument("--critic-learning-rate", type=float, default=0.0002)
    parser.add_argument("--critic-hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--opponent-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--teacher-opponent-probability", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ppo-baseline-variance.json"),
    )
    return parser.parse_args()


def baseline_statistics(steps: Sequence[PpoStep]) -> dict[str, float]:
    if not steps:
        raise ValueError("baseline 统计没有 PPO step")
    residuals = [step.reward - step.old_value for step in steps]
    mean = sum(residuals) / len(residuals)
    mean_square = sum(value * value for value in residuals) / len(residuals)
    variance = sum((value - mean) ** 2 for value in residuals) / len(residuals)
    return {
        "decisions": float(len(residuals)),
        "residual_mean": mean,
        "residual_std": variance**0.5,
        "residual_mean_square": mean_square,
    }


def _same_actor_trajectory(
    public_steps: Sequence[PpoStep], critic_steps: Sequence[PpoStep]
) -> bool:
    return len(public_steps) == len(critic_steps) and all(
        public.action_index == critic.action_index
        and public.reward == critic.reward
        and public.decision.state == critic.decision.state
        and public.decision.legal_actions == critic.decision.legal_actions
        for public, critic in zip(public_steps, critic_steps)
    )


def main() -> None:
    args = parse_args()
    if (
        args.calibration_episodes <= 0
        or args.evaluation_episodes <= 0
        or args.rollout_batch_size <= 0
        or args.critic_epochs <= 0
        or args.critic_batch_size <= 0
        or args.critic_learning_rate <= 0
        or args.critic_hidden_size <= 0
        or not 0.0 <= args.teacher_opponent_probability <= 1.0
    ):
        raise ValueError("critic 方差 A/B 超参数不合法")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但当前 PyTorch 无可用 GPU")
    actor = TorchPolicyValueAgent.load(args.checkpoint, device=args.device)
    if actor.architecture != ARCHITECTURE_CANDIDATE_MLP:
        raise ValueError("critic 方差 A/B 仅支持 candidate_mlp checkpoint")
    opponents: list[tuple[str, TorchPolicyValueAgent]] = []
    for checkpoint in args.opponent_checkpoint:
        opponent = TorchPolicyValueAgent.load(checkpoint, device=args.device)
        if opponent.architecture != ARCHITECTURE_CANDIDATE_MLP:
            raise ValueError("critic 方差 A/B 对手池仅支持 candidate_mlp checkpoint")
        opponents.append((str(checkpoint), opponent))
    if args.teacher_opponent_probability < 1.0 and not opponents:
        raise ValueError("混入非 Teacher 对手时必须提供 opponent checkpoint")

    torch.manual_seed(args.seed)
    if actor.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    critic = PrivilegedCritic(args.critic_hidden_size).to(actor.device)
    calibration_steps, calibration_rollout = collect_rollouts_batched(
        actor,
        episodes=args.calibration_episodes,
        profile=args.profile,
        seed=args.seed,
        reward_scale=80.0,
        opponents=opponents,
        teacher_opponent_probability=args.teacher_opponent_probability,
        rollout_batch_size=args.rollout_batch_size,
        privileged_critic=critic,
    )
    calibration = train_privileged_critic(
        critic,
        calibration_steps,
        device=actor.device,
        batch_size=args.critic_batch_size,
        epochs=args.critic_epochs,
        learning_rate=args.critic_learning_rate,
        seed=args.seed + 1,
    )
    evaluation_seed = args.seed + 1_000_000
    public_steps, public_rollout = collect_rollouts_batched(
        actor,
        episodes=args.evaluation_episodes,
        profile=args.profile,
        seed=evaluation_seed,
        reward_scale=80.0,
        opponents=opponents,
        teacher_opponent_probability=args.teacher_opponent_probability,
        rollout_batch_size=args.rollout_batch_size,
    )
    critic_steps, critic_rollout = collect_rollouts_batched(
        actor,
        episodes=args.evaluation_episodes,
        profile=args.profile,
        seed=evaluation_seed,
        reward_scale=80.0,
        opponents=opponents,
        teacher_opponent_probability=args.teacher_opponent_probability,
        rollout_batch_size=args.rollout_batch_size,
        privileged_critic=critic,
    )
    trajectories_match = _same_actor_trajectory(public_steps, critic_steps)
    if not trajectories_match:
        raise RuntimeError("critic-on/off A/B 的 actor 轨迹不一致，拒绝报告方差比较")
    public_stats = baseline_statistics(public_steps)
    critic_stats = baseline_statistics(critic_steps)
    reduction = 1.0 - critic_stats["residual_mean_square"] / max(
        public_stats["residual_mean_square"], 1e-12
    )
    report: dict[str, Any] = {
        "algorithm": "fixed_actor_privileged_critic_baseline_ab_v1",
        "profile": args.profile,
        "actor_checkpoint": str(args.checkpoint),
        "device": args.device,
        "actor_frozen": True,
        "critic": {
            "training_only": True,
            "serialized": False,
            "feature_scope": "complete_simulator_state_in_memory_only",
            "hidden_size": args.critic_hidden_size,
            "calibration_epochs": args.critic_epochs,
            "calibration_batch_size": args.critic_batch_size,
            "calibration_learning_rate": args.critic_learning_rate,
        },
        "opponent_pool": {
            "teacher_probability": args.teacher_opponent_probability,
            "frozen_checkpoints": [str(path) for path in args.opponent_checkpoint],
        },
        "calibration": {"rollout": calibration_rollout.payload(), "update": calibration},
        "evaluation": {
            "episodes": args.evaluation_episodes,
            "seed": evaluation_seed,
            "same_actor_trajectory": trajectories_match,
            "public_value": {"rollout": public_rollout.payload(), **public_stats},
            "privileged_critic": {"rollout": critic_rollout.payload(), **critic_stats},
            "residual_mean_square_reduction": reduction,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

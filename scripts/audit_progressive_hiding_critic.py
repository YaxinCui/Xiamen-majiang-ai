#!/usr/bin/env python3
"""Falsify a progressive-hiding value-calibration curriculum before PPO.

This deliberately does *not* train, save, or evaluate a playable actor.  A
fresh random legal-action policy produces one fixed in-memory set of terminal
rollouts.  Two identically initialized training-only critics then receive the
same training steps:

* direct baseline: visible information for the entire equal epoch budget;
* curriculum: oracle, then hidden-wall, then final visible information.

Only independent final-visible Huber and MAE are reported.  The report cannot
support a strength claim and contains neither hidden feature vectors nor a
critic checkpoint.  A curriculum can merely pass this narrow calibration gate
when it improves *both* metrics; policy training remains a separate future
experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit("请使用 .venv/bin/python 运行；该脚本需要 PyTorch") from error

from scripts.train_torch_ppo import (
    PrivilegedCritic,
    collect_rollouts_batched,
    evaluate_privileged_critic,
    source_revision,
    train_privileged_critic,
    train_progressive_hiding_critic,
)
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent


HIDING_SCHEDULE = (("oracle", 3), ("hide_wall", 3), ("visible", 3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("core", "classic"), default="core")
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--critic-hidden-size", type=int, default=32)
    parser.add_argument("--critic-batch-size", type=int, default=128)
    parser.add_argument("--epochs-per-stage", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--reward-scale", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=202608560)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/progressive-hiding-core-smoke-v1.json"),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if (
        args.train_episodes <= 0
        or args.validation_episodes <= 0
        or args.rollout_batch_size <= 0
        or args.critic_hidden_size <= 0
        or args.critic_batch_size <= 0
        or args.epochs_per_stage <= 0
        or args.learning_rate <= 0
        or args.reward_scale <= 0
    ):
        raise ValueError("progressive-hiding 校准超参数必须为正数")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但当前 PyTorch 无可用 GPU")


def _schedule(epochs_per_stage: int) -> tuple[tuple[str, int], ...]:
    return tuple((stage, epochs_per_stage) for stage, _epochs in HIDING_SCHEDULE)


def _rounded(payload: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 10) for key, value in payload.items()}


def main() -> None:
    args = parse_args()
    _validate_args(args)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # This actor is intentionally fresh and remains untouched after collection.
    # It is not a candidate model, nor is it written anywhere.
    actor = TorchPolicyValueAgent(feature_version=3, hidden_size=32, device=args.device)
    collector_critic = PrivilegedCritic(args.critic_hidden_size).to(device)
    train_steps, train_summary = collect_rollouts_batched(
        actor,
        episodes=args.train_episodes,
        profile=args.profile,
        seed=args.seed,
        reward_scale=args.reward_scale,
        rollout_batch_size=args.rollout_batch_size,
        privileged_critic=collector_critic,
    )
    validation_steps, validation_summary = collect_rollouts_batched(
        actor,
        episodes=args.validation_episodes,
        profile=args.profile,
        seed=args.seed + 100_000,
        reward_scale=args.reward_scale,
        rollout_batch_size=args.rollout_batch_size,
        privileged_critic=collector_critic,
    )

    # Both comparisons start with bit-identical critic parameters.  The
    # collection critic is not one of them and affects no candidate action.
    torch.manual_seed(args.seed + 1)
    direct = PrivilegedCritic(args.critic_hidden_size).to(device)
    curriculum = PrivilegedCritic(args.critic_hidden_size).to(device)
    curriculum.load_state_dict(copy.deepcopy(direct.state_dict()))
    schedule = _schedule(args.epochs_per_stage)
    direct_training = train_privileged_critic(
        direct,
        train_steps,
        device=device,
        batch_size=args.critic_batch_size,
        epochs=sum(epochs for _stage, epochs in schedule),
        learning_rate=args.learning_rate,
        seed=args.seed + 10,
        hiding_stage="visible",
    )
    curriculum_training = train_progressive_hiding_critic(
        curriculum,
        train_steps,
        schedule=schedule,
        device=device,
        batch_size=args.critic_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed + 10,
    )
    direct_validation = evaluate_privileged_critic(
        direct, validation_steps, device=device, hiding_stage="visible"
    )
    curriculum_validation = evaluate_privileged_critic(
        curriculum, validation_steps, device=device, hiding_stage="visible"
    )
    improves_huber = curriculum_validation["huber"] < direct_validation["huber"]
    improves_mae = curriculum_validation["mae"] < direct_validation["mae"]
    passed = improves_huber and improves_mae
    payload: dict[str, Any] = {
        "experiment": "progressive_hiding_critic_calibration_core_smoke_v1",
        "source_revision": source_revision(),
        "protocol": {
            "profile": args.profile,
            "fresh_random_actor": True,
            "actor_optimized": False,
            "teacher_opponents_only": True,
            "train_episodes": args.train_episodes,
            "validation_episodes": args.validation_episodes,
            "rollout_batch_size": args.rollout_batch_size,
            "reward_scale": args.reward_scale,
            "seed": args.seed,
            "device": str(device),
            "critic_hidden_size": args.critic_hidden_size,
            "critic_batch_size": args.critic_batch_size,
            "learning_rate": args.learning_rate,
            "direct_hiding_stage": "visible",
            "direct_epochs": sum(epochs for _stage, epochs in schedule),
            "curriculum_schedule": [
                {"stage": stage, "epochs": epochs} for stage, epochs in schedule
            ],
            "same_initial_critic_parameters": True,
            "same_in_memory_train_rollouts": True,
            "independent_validation_walls": True,
        },
        "rollouts": {
            "train": train_summary.payload(),
            "validation": validation_summary.payload(),
        },
        "training": {
            "direct_visible": _rounded(direct_training),
            "progressive_hiding": [_rounded(metric) for metric in curriculum_training],
        },
        "validation_visible_only": {
            "direct_visible": _rounded(direct_validation),
            "progressive_hiding": _rounded(curriculum_validation),
            "delta_progressive_minus_direct": {
                "huber": round(
                    curriculum_validation["huber"] - direct_validation["huber"], 10
                ),
                "mae": round(
                    curriculum_validation["mae"] - direct_validation["mae"], 10
                ),
            },
        },
        "selection_gate": {
            "pass_requires": "independent final-visible Huber and MAE both strictly lower than equal-budget direct-visible baseline",
            "improves_huber": improves_huber,
            "improves_mae": improves_mae,
            "passed": passed,
            "status": (
                "calibration_pass_only_no_policy_or_strength_claim"
                if passed
                else "rejected_before_policy_training"
            ),
        },
        "privacy": {
            "hidden_feature_vectors_exported": False,
            "critic_checkpoint_saved": False,
            "actor_checkpoint_saved": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

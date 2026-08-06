#!/usr/bin/env python3
"""Collect safe public-state legal-action targets from counterfactual rollouts.

Each exported decision has ``action_values`` aligned with its engine-legal
actions.  The values are candidate terminal net scores under the frozen
continuation policy, not features and not an export of the wall or opponents'
concealed hands.  Use this as a policy-improvement data source after a policy
checkpoint has already been warmed up on rule Teacher trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import (
    collect_counterfactual_action_value_trajectories,
    split_trajectories_by_hand,
    trajectory_manifest,
    write_trajectory_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=("classic", "core"), default="classic")
    parser.add_argument(
        "--seed-count",
        type=int,
        default=100,
        help="物理牌墙数；每副牌轮换四个候选席位",
    )
    parser.add_argument(
        "--samples-per-hand",
        type=int,
        default=1,
        help="每个候选席位局面最多抽取的多动作决策数",
    )
    parser.add_argument(
        "--rollouts-per-action",
        type=int,
        default=1,
        help="同一动作在冻结对手混合下的重复 continuation 数",
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=1,
        help="并行推进的独立反事实分支数；仅批量化支持 scores_batch 的策略推理",
    )
    parser.add_argument(
        "--response-sample-probability",
        type=float,
        default=0.35,
        help="存在响应局面时，从响应局面抽样的概率",
    )
    parser.add_argument("--seed", type=int, default=20266808)
    parser.add_argument("--split-salt", default="counterfactual-action-value-v1")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--opponent-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="可重复指定；作为冻结对手池的 checkpoint",
    )
    parser.add_argument(
        "--teacher-opponent-probability",
        type=float,
        default=1.0,
        help="每个非候选座位使用规则 Teacher 的概率；其余均匀取冻结对手池",
    )
    parser.add_argument(
        "--belief-resample",
        action="store_true",
        help="每个 rollout 从本家可见信息重采样未知牌墙、暗手、花与对手暗杠牌面",
    )
    parser.add_argument(
        "--belief-latest-discard-particles",
        type=int,
        default=0,
        help=(
            "启用最新 normal draw→discard 的局部 SIR 条件化所用粒子数；"
            "需同时设置 --belief-resample，0 表示关闭"
        ),
    )
    parser.add_argument(
        "--belief-latest-discard-likelihood-power",
        type=float,
        default=0.25,
        help="局部弃牌行为似然幂次；小于 1 时保守地向公开先验回缩",
    )
    parser.add_argument(
        "--belief-latest-discard-min-ess-fraction",
        type=float,
        default=0.5,
        help="局部 SIR 的最低预重采样 ESS 比例；未达标的局面会被跳过",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/counterfactual-action-value-classic-v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.seed_count <= 0
        or args.samples_per_hand <= 0
        or args.rollouts_per_action <= 0
        or args.rollout_batch_size <= 0
    ):
        raise ValueError(
            "seed-count、samples-per-hand、rollouts-per-action、rollout-batch-size 必须为正数"
        )
    if not 0.0 <= args.teacher_opponent_probability <= 1.0:
        raise ValueError("teacher-opponent-probability 必须在 0 和 1 之间")
    if args.teacher_opponent_probability < 1.0 and not args.opponent_checkpoint:
        raise ValueError("混入非 Teacher 对手时必须提供 --opponent-checkpoint")
    if args.belief_latest_discard_particles < 0:
        raise ValueError("belief-latest-discard-particles 不能为负数")
    if args.belief_latest_discard_particles and not args.belief_resample:
        raise ValueError("latest-discard 条件化需要 --belief-resample")
    policy = TorchPolicyValueAgent.load(args.checkpoint, device=args.device)
    opponents = [
        (
            f"frozen:{path.name}",
            TorchPolicyValueAgent.load(path, device=args.device),
        )
        for path in args.opponent_checkpoint
    ]
    trajectories, summary = collect_counterfactual_action_value_trajectories(
        policy,
        seed_count=args.seed_count,
        profile=args.profile,
        seed=args.seed,
        samples_per_hand=args.samples_per_hand,
        rollouts_per_action=args.rollouts_per_action,
        response_sample_probability=args.response_sample_probability,
        opponents=opponents,
        teacher_opponent_probability=args.teacher_opponent_probability,
        belief_resample=args.belief_resample,
        belief_latest_discard_particles=args.belief_latest_discard_particles,
        belief_latest_discard_likelihood_power=args.belief_latest_discard_likelihood_power,
        belief_latest_discard_min_ess_fraction=args.belief_latest_discard_min_ess_fraction,
        rollout_batch_size=args.rollout_batch_size,
    )
    partitions = split_trajectories_by_hand(
        trajectories,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        split_salt=args.split_salt,
    )
    empty_partitions = [name for name, records in partitions.items() if not records]
    if empty_partitions:
        raise ValueError(
            "动作价值数据切分存在空分区："
            f"{', '.join(empty_partitions)}；请增加 --seed-count 或更换 --split-salt"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partition_reports = {}
    for name, records in partitions.items():
        filename = f"{name}.trajectories.jsonl"
        partition_reports[name] = {
            "file": filename,
            "hands": write_trajectory_jsonl(records, args.output_dir / filename),
            "manifest": trajectory_manifest(records),
        }
    report = {
        "source": "counterfactual_action_value_rollout",
        "profile": args.profile,
        "candidate_checkpoint": str(args.checkpoint),
        "candidate_architecture": policy.architecture,
        "inference_device": args.device,
        "seed_count": args.seed_count,
        "samples_per_hand": args.samples_per_hand,
        "rollouts_per_action": args.rollouts_per_action,
        "rollout_batch_size": args.rollout_batch_size,
        "response_sample_probability": args.response_sample_probability,
        "opponent_pool": {
            "teacher_probability": args.teacher_opponent_probability,
            "frozen_checkpoints": [str(path) for path in args.opponent_checkpoint],
        },
        "target_semantics": {
            "kind": "candidate_terminal_net_score_q_pi",
            "observation": "actor_hand_plus_public_information_only",
            "private_simulator_state_exported": False,
            "continuation": "candidate_checkpoint_and_frozen_opponent_mixture",
            "belief_resample": args.belief_resample,
            "belief_latest_discard": {
                "particles": args.belief_latest_discard_particles,
                "likelihood_power": args.belief_latest_discard_likelihood_power,
                "minimum_ess_fraction": args.belief_latest_discard_min_ess_fraction,
                "scope": "latest opponent normal draw then discard; not full history posterior",
            },
        },
        "summary": summary.payload(),
        "dataset": trajectory_manifest(trajectories),
        "split": {
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "split_salt": args.split_salt,
            "wall_rotations_grouped": True,
        },
        "partitions": partition_reports,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the single frozen 100-wall Teacher screen for reference-KL PPO v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.agents import HeuristicTeacherAgent
from xiamen_mahjong.evaluation import evaluate_against_teacher, paired_score_comparison
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent


PROTOCOL = "teacher_clone_reference_kl_small_mlp_ppo_v2"
PROFILE = "classic"
PHYSICAL_WALLS = 100
SELECTION_SEED = 202647500
BASE_CHECKPOINT_SHA256 = (
    "b087a0e496e266de4c9e116fb9a4b5c3950964abdefff24843c5f43ea2aa09df"
)
TRAINING_SOURCE_TREE_SHA256 = (
    "2858e7665f48f7099da5185af03b2cf0118c7f6b31661fdba4266fa6b3588fa6"
)
FINAL_REFERENCE_KL_MAXIMUM = 0.02
FINAL_ARGMAX_DISAGREEMENT_MINIMUM = 0.005
FINAL_ARGMAX_DISAGREEMENT_MAXIMUM = 0.10


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_training_config() -> dict[str, Any]:
    return {
        "iterations": 4,
        "episodes_per_iteration": 1024,
        "rollout_batch_size": 64,
        "ppo_epochs": 2,
        "batch_size": 512,
        "learning_rate": 0.00005,
        "policy_head_learning_rate_multiplier": 10.0,
        "reference_kl_weight": 0.05,
        "clip_ratio": 0.15,
        "value_weight": 0.25,
        "entropy_weight": 0.002,
        "reward_scale": 80.0,
        "seed": 202646000,
        "profile": PROFILE,
        "device": "cuda",
        "teacher_opponent_probability": 1.0,
        "self_play_opponent_probability": 0.0,
        "teacher_prior_margin": 0.0,
        "opponent_checkpoints": [],
        "privileged_critic": False,
        "privileged_critic_hidden_size": 128,
        "privileged_critic_weight": 0.25,
    }


def validate_training_report(
    report: dict[str, Any], *, checkpoint: Path
) -> dict[str, Any]:
    """Fail closed unless the report proves the pre-registered mechanics gate."""

    if report.get("algorithm") != "legal_action_ppo_terminal_score_v1":
        raise ValueError("训练算法标识不匹配")
    if report.get("profile") != PROFILE:
        raise ValueError("训练规则 profile 不匹配")
    if report.get("checkpoint_source_sha256") != BASE_CHECKPOINT_SHA256:
        raise ValueError("强启动 checkpoint 哈希不匹配")
    if report.get("source_tree_sha256") != TRAINING_SOURCE_TREE_SHA256:
        raise ValueError("训练源码树哈希不匹配")
    if report.get("training_config") != expected_training_config():
        raise ValueError("训练超参数与冻结协议不完全一致")
    final_checkpoint = report.get("final_checkpoint")
    if not isinstance(final_checkpoint, str) or (
        (ROOT / final_checkpoint).resolve() != checkpoint.resolve()
        and Path(final_checkpoint).resolve() != checkpoint.resolve()
    ):
        raise ValueError("训练报告 final checkpoint 与候选不一致")

    iterations = report.get("iterations")
    if not isinstance(iterations, list) or len(iterations) != 4:
        raise ValueError("训练必须恰好包含四轮")
    total_decisions = 0
    for index, item in enumerate(iterations, start=1):
        if not isinstance(item, dict) or item.get("iteration") != index:
            raise ValueError("训练轮次编号不连续")
        if item.get("rollout_seed") != 202646000 + index * 100_000:
            raise ValueError("rollout seed 不匹配")
        if item.get("update_seed") != 202646000 + index:
            raise ValueError("update seed 不匹配")
        rollout = item.get("rollout")
        if not isinstance(rollout, dict) or rollout.get("episodes") != 1024:
            raise ValueError("每轮必须恰好采样 1,024 局")
        decisions = rollout.get("decisions")
        action_counts = rollout.get("action_counts")
        if (
            isinstance(decisions, bool)
            or not isinstance(decisions, int)
            or decisions <= 0
            or not isinstance(action_counts, dict)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in action_counts.values()
            )
            or sum(action_counts.values()) != decisions
        ):
            raise ValueError("rollout 动作计数不完整")
        if rollout.get("opponent_profile_counts") != {"heuristic_teacher": 3072}:
            raise ValueError("对手并非三家 frozen Teacher")
        wins = rollout.get("wins")
        draws = rollout.get("draws")
        if (
            isinstance(wins, bool)
            or not isinstance(wins, int)
            or isinstance(draws, bool)
            or not isinstance(draws, int)
            or wins < 0
            or draws < 0
            or wins + draws > 1024
        ):
            raise ValueError("rollout 胜局/流局统计不合法")
        total_decisions += decisions

    update = iterations[-1].get("update")
    if not isinstance(update, dict):
        raise ValueError("最终 update 指标缺失")
    kl = update.get("final_reference_kl_mean")
    disagreement = update.get("final_reference_argmax_disagreement_rate")
    if not isinstance(kl, (int, float)) or not math.isfinite(float(kl)):
        raise ValueError("最终 reference KL 无效")
    if not isinstance(disagreement, (int, float)) or not math.isfinite(
        float(disagreement)
    ):
        raise ValueError("最终 argmax 分歧率无效")
    mechanics_passes = (
        0.0 <= float(kl) <= FINAL_REFERENCE_KL_MAXIMUM
        and FINAL_ARGMAX_DISAGREEMENT_MINIMUM
        <= float(disagreement)
        <= FINAL_ARGMAX_DISAGREEMENT_MAXIMUM
    )
    return {
        "passes": mechanics_passes,
        "total_candidate_decisions": total_decisions,
        "action_accounting_complete": True,
        "legal_action_mask_enforced_by_collector": True,
        "corrected_candidate_win_semantics": "game.winner == candidate_seat",
        "final_reference_kl_mean": float(kl),
        "maximum_final_reference_kl_mean": FINAL_REFERENCE_KL_MAXIMUM,
        "final_reference_argmax_disagreement_rate": float(disagreement),
        "allowed_final_reference_argmax_disagreement_rate": [
            FINAL_ARGMAX_DISAGREEMENT_MINIMUM,
            FINAL_ARGMAX_DISAGREEMENT_MAXIMUM,
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("selection 输出已存在；禁止覆盖或重复读取同一墙集")
    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    mechanics = validate_training_report(report, checkpoint=args.checkpoint)
    if not mechanics["passes"]:
        raise ValueError("mechanics gate 未通过；禁止读取 selection 墙")

    raw_checkpoint = __import__("torch").load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    metadata = raw_checkpoint.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("latest_iteration") != 4
        or metadata.get("source_tree_sha256") != TRAINING_SOURCE_TREE_SHA256
        or metadata.get("training_config") != expected_training_config()
    ):
        raise ValueError("checkpoint 内嵌 provenance 与训练报告不一致")

    candidate = TorchPolicyValueAgent.load(args.checkpoint, device=args.device)
    candidate_result = evaluate_against_teacher(
        candidate, hands=PHYSICAL_WALLS, profile=PROFILE, seed=SELECTION_SEED
    )
    teacher_result = evaluate_against_teacher(
        HeuristicTeacherAgent(),
        hands=PHYSICAL_WALLS,
        profile=PROFILE,
        seed=SELECTION_SEED,
    )
    paired = paired_score_comparison(candidate_result, teacher_result)
    passes = paired["paired_seed_score_delta_95pct_low"] > 0.0
    payload = {
        "protocol": PROTOCOL,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "training_report": str(args.training_report),
        "training_report_sha256": file_sha256(args.training_report),
        "mechanics_gate": mechanics,
        "selection": {
            "physical_walls": PHYSICAL_WALLS,
            "seat_rotations_per_wall": 4,
            "first_seed": SELECTION_SEED,
            "last_seed_inclusive": SELECTION_SEED + PHYSICAL_WALLS - 1,
            "candidate": candidate_result.payload(),
            "teacher_baseline": teacher_result.payload(),
            "paired_against_heuristic_teacher": paired,
            "promotion_gate": "paired_seed_score_delta_95pct_low > 0",
            "passes": passes,
        },
        "status": (
            "selection_passed_ready_for_250k_training_not_deployed"
            if passes
            else "selection_rejected_configuration_frozen_not_deployed"
        ),
        "additional_terminal_wall_stage": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

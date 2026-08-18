#!/usr/bin/env python3
"""Audit a response-Q checkpoint against its frozen policy on held-out labels.

This is an offline diagnostic only.  It scores the Q choice and the frozen
policy choice against per-action terminal-score targets that were produced by
the rule engine, reports paired optimal-action and regret differences, and
refuses to treat the result as deployment authorization.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import read_trajectory_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="可重复指定；多个独立 Q checkpoint 将按逐动作均值集成。",
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        required=True,
        help="可重复指定；仅追加按物理牌墙隔离的 action_values 留出集。",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--policy-top-k",
        type=int,
        help="只在 reference policy 概率最高的 k 个合法 response 内选择 Q；省略时全候选。",
    )
    parser.add_argument(
        "--minimum-q-advantage-points",
        type=float,
        default=0.0,
        help="Q 覆盖 policy 前，Q 预测值必须至少高出的终局分数点数。",
    )
    parser.add_argument(
        "--q-score-scale",
        type=float,
        default=80.0,
        help="Q checkpoint 的归一化终局分数尺度。",
    )
    parser.add_argument("--policy-match-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def best_index(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def is_optimal(targets: Sequence[float], index: int, *, tolerance: float = 1e-6) -> bool:
    return targets[index] >= max(targets) - tolerance


def paired_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("配对审计需要至少一个决策")
    mean = sum(values) / len(values)
    variance = (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1
        else 0.0
    )
    stderr = math.sqrt(variance / len(values))
    return {
        "mean": mean,
        "stderr": stderr,
        "95pct_low": mean - 1.96 * stderr,
        "95pct_high": mean + 1.96 * stderr,
    }


def main() -> None:
    args = parse_args()
    if args.policy_match_tolerance < 0:
        raise ValueError("policy-match-tolerance 不能为负数")
    if args.policy_top_k is not None and args.policy_top_k <= 0:
        raise ValueError("policy-top-k 必须为正数")
    if args.minimum_q_advantage_points < 0 or args.q_score_scale <= 0:
        raise ValueError("minimum-q-advantage-points 不能为负数，q-score-scale 必须为正数")
    candidates = [
        TorchPolicyValueAgent.load(path, device=args.device) for path in args.checkpoint
    ]
    reference = TorchPolicyValueAgent.load(args.reference, device=args.device)
    q_optimal: list[float] = []
    policy_optimal: list[float] = []
    q_regrets: list[float] = []
    policy_regrets: list[float] = []
    q_minus_policy_optimal: list[float] = []
    policy_minus_q_regret: list[float] = []
    target_ties = 0
    decisions = 0
    actions = 0
    max_policy_delta = 0.0
    overrides = 0
    for path in args.data:
        for trajectory in read_trajectory_jsonl(path):
            for decision in trajectory.decisions:
                targets = decision.action_values
                if decision.state.get("phase") != "response":
                    raise ValueError("response-Q 审计拒绝非 response 决策")
                if targets is None:
                    raise ValueError("输入必须含 action_values")
                q_member_scores = [
                    candidate.action_value_scores(decision) for candidate in candidates
                ]
                if any(scores is None for scores in q_member_scores):
                    raise ValueError("checkpoint 必须有 Q head")
                safe_member_scores = [
                    scores for scores in q_member_scores if scores is not None
                ]
                q_scores = [
                    sum(scores[index] for scores in safe_member_scores)
                    / len(safe_member_scores)
                    for index in range(len(targets))
                ]
                if len(targets) != len(q_scores) or not targets:
                    raise ValueError("Q 预测与 action_values 长度不匹配")
                reference_logits, _reference_value = reference.policy_value(decision)
                for candidate in candidates:
                    candidate_logits, _candidate_value = candidate.policy_value(decision)
                    if len(candidate_logits) != len(reference_logits):
                        raise ValueError("候选与 reference policy 动作数不一致")
                    max_policy_delta = max(
                        max_policy_delta,
                        *(
                            abs(left - right)
                            for left, right in zip(candidate_logits, reference_logits)
                        ),
                    )
                policy_index = best_index(reference_logits)
                eligible_indices = (
                    sorted(
                        range(len(targets)),
                        key=lambda index: (reference_logits[index], -index),
                        reverse=True,
                    )[: args.policy_top_k]
                    if args.policy_top_k is not None
                    else list(range(len(targets)))
                )
                q_index = max(
                    eligible_indices, key=lambda index: (q_scores[index], -index)
                )
                q_advantage_points = (
                    q_scores[q_index] - q_scores[policy_index]
                ) * args.q_score_scale
                if q_index != policy_index and (
                    q_advantage_points < args.minimum_q_advantage_points
                ):
                    q_index = policy_index
                overrides += q_index != policy_index
                target_maximum = max(targets)
                target_ties += sum(
                    abs(value - target_maximum) <= 1e-6 for value in targets
                ) > 1
                q_is_optimal = float(is_optimal(targets, q_index))
                policy_is_optimal = float(is_optimal(targets, policy_index))
                q_regret = target_maximum - targets[q_index]
                policy_regret = target_maximum - targets[policy_index]
                q_optimal.append(q_is_optimal)
                policy_optimal.append(policy_is_optimal)
                q_regrets.append(q_regret)
                policy_regrets.append(policy_regret)
                q_minus_policy_optimal.append(q_is_optimal - policy_is_optimal)
                policy_minus_q_regret.append(policy_regret - q_regret)
                decisions += 1
                actions += len(targets)
    if decisions <= 0:
        raise ValueError("没有可审计的 action_values 决策")
    payload = {
        "status": "audit_only_not_authorized_for_action_selection",
        "checkpoints": [str(path) for path in args.checkpoint],
        "ensemble_members": len(candidates),
        "reference": str(args.reference),
        "data": [str(path) for path in args.data],
        "decision_phase": "response_required_by_dataset_contract",
        "q_selection": (
            "all_legal_actions"
            if args.policy_top_k is None
            else f"reference_policy_top_{args.policy_top_k}"
        ),
        "minimum_q_advantage_points": args.minimum_q_advantage_points,
        "q_score_scale": args.q_score_scale,
        "q_override_rate": overrides / decisions,
        "decisions": decisions,
        "actions": actions,
        "target_optimum_tie_rate": target_ties / decisions,
        "policy_identity": {
            "max_abs_policy_logit_delta": max_policy_delta,
            "tolerance": args.policy_match_tolerance,
            "matches_reference": max_policy_delta <= args.policy_match_tolerance,
        },
        "absolute_metrics": {
            "q_optimal_action_rate": sum(q_optimal) / decisions,
            "policy_optimal_action_rate": sum(policy_optimal) / decisions,
            "q_mean_regret_points": sum(q_regrets) / decisions,
            "policy_mean_regret_points": sum(policy_regrets) / decisions,
        },
        "paired_q_minus_policy": {
            "optimal_action_rate_difference": paired_summary(q_minus_policy_optimal),
            "regret_improvement_points": paired_summary(policy_minus_q_regret),
        },
        "warning": (
            "This reports synthetic held-out counterfactual targets, not a real-game "
            "strength result. A positive lower bound here is necessary but never "
            "sufficient for response-Q deployment."
        ),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

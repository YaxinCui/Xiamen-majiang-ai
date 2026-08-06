#!/usr/bin/env python3
"""Audit a one-response Teacher override with grouped IPS and DR estimates.

The input must be held-out trajectories collected with
``collect_candidate_teacher_dagger_trajectories.py --teacher-base``.  Every
accepted row has exactly one randomized response action and then returns to
the frozen Teacher.  The estimate therefore covers *one* possible override,
not a policy that changes every response in a hand.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.off_policy import LoggedIntervention, intervention_estimates
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import TeacherDecision, read_trajectory_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outcome-checkpoint",
        type=Path,
        action="append",
        required=True,
        help="至少两个仅用训练/验证墙训练的 afterstate outcome checkpoint",
    )
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        required=True,
        help="仅输入从 outcome checkpoint 训练墙隔离的 intervention test JSONL",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--value-scale", type=float, default=80.0)
    parser.add_argument("--score-lcb-z", type=float, default=1.0)
    parser.add_argument(
        "--minimum-lcb-advantage",
        type=float,
        default=0.0,
        help="只有 LCB 比 Teacher 标签动作高至少该分数时才提出一次 override",
    )
    parser.add_argument(
        "--minimum-effective-sample-size",
        type=float,
        default=30.0,
        help="报告 ready_for_single_override_game_screen 所需的每侧最小 IPS ESS",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("均值需要至少一个值")
    return sum(values) / len(values)


def population_std(values: Sequence[float]) -> float:
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def best_index(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def teacher_epsilon_propensities(
    *, action_count: int, teacher_index: int, epsilon: float
) -> tuple[float, ...]:
    """Exact ``epsilon × uniform + (1-epsilon) × Teacher`` probabilities."""

    if action_count <= 0 or not 0 <= teacher_index < action_count:
        raise ValueError("Teacher 行为概率的动作集合无效")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("OPE 只接受实际随机化的 epsilon，必须在 (0, 1)")
    probabilities = [epsilon / action_count] * action_count
    probabilities[teacher_index] += 1.0 - epsilon
    return tuple(probabilities)


def assert_same_policy_logits(
    agents: Sequence[TorchPolicyValueAgent], decision: TeacherDecision
) -> None:
    reference, _value = agents[0].policy_value(decision)
    for agent in agents[1:]:
        logits, _value = agent.policy_value(decision)
        if len(logits) != len(reference) or any(
            abs(left - right) > 1e-6 for left, right in zip(logits, reference)
        ):
            raise ValueError(
                "outcome ensemble 的冻结 policy logits 不一致；"
                "不能把不同基策略的 outcome 头混合用于 OPE"
            )


def lcb_target_index(
    decision: TeacherDecision,
    agents: Sequence[TorchPolicyValueAgent],
    *,
    value_scale: float,
    score_lcb_z: float,
    minimum_lcb_advantage: float,
) -> tuple[int, tuple[float, ...]]:
    """Return a conservative response target and mean direct score estimates."""

    outputs = [agent.afterstate_outcomes(decision) for agent in agents]
    if any(output is None for output in outputs):
        raise ValueError("outcome checkpoint 缺少 afterstate outcome head")
    safe_outputs = [output for output in outputs if output is not None]
    assert_same_policy_logits(agents, decision)
    score_by_action = [
        [output[0][index] * value_scale for output in safe_outputs]
        for index in range(len(decision.legal_actions))
    ]
    score_means = tuple(mean(values) for values in score_by_action)
    score_lcbs = [
        score_means[index] - score_lcb_z * population_std(values)
        for index, values in enumerate(score_by_action)
    ]
    teacher_index = decision.chosen_index
    proposed = best_index(score_lcbs)
    target = (
        proposed
        if score_lcbs[proposed]
        >= score_lcbs[teacher_index] + minimum_lcb_advantage
        else teacher_index
    )
    return target, score_means


def selected_interventions(
    paths: Iterable[Path],
    agents: Sequence[TorchPolicyValueAgent],
    *,
    value_scale: float,
    score_lcb_z: float,
    minimum_lcb_advantage: float,
) -> tuple[list[LoggedIntervention], Counter[str], int]:
    """Build OPE rows while validating the Teacher intervention contract."""

    observations: list[LoggedIntervention] = []
    target_kinds: Counter[str] = Counter()
    scanned = 0
    for path in paths:
        for trajectory in read_trajectory_jsonl(path):
            metadata = trajectory.source_metadata
            if metadata.get("behavior_policy") != "single_intervention_epsilon_uniform":
                raise ValueError("OPE 输入不是单点 epsilon 干预轨迹")
            if metadata.get("base_policy") != "heuristic_teacher":
                raise ValueError("此评测器只接受 --teacher-base 收集的轨迹")
            epsilon = metadata.get("uniform_exploration_probability")
            if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
                raise ValueError("Teacher 干预轨迹缺少 epsilon")
            scores = trajectory.outcome.get("scores")
            if not isinstance(scores, list) or len(scores) != 4:
                raise ValueError("训练轨迹缺少四家终局得分")
            if not trajectory.split_group_id:
                raise ValueError("Teacher 干预轨迹缺少按物理墙隔离的 split_group_id")
            for decision in trajectory.decisions:
                scanned += 1
                if decision.state.get("phase") != "response":
                    continue
                if (
                    decision.executed_index is None
                    or decision.executed_probability is None
                    or decision.executed_probability >= 1.0
                ):
                    continue
                propensities = teacher_epsilon_propensities(
                    action_count=len(decision.legal_actions),
                    teacher_index=decision.chosen_index,
                    epsilon=float(epsilon),
                )
                logged = decision.executed_index
                if not math.isclose(
                    propensities[logged], decision.executed_probability, abs_tol=1e-8
                ):
                    raise ValueError("记录的 executed_probability 与 Teacher epsilon 行为不一致")
                target, direct_values = lcb_target_index(
                    decision,
                    agents,
                    value_scale=value_scale,
                    score_lcb_z=score_lcb_z,
                    minimum_lcb_advantage=minimum_lcb_advantage,
                )
                observations.append(
                    LoggedIntervention(
                        group_id=trajectory.split_group_id,
                        logged_index=logged,
                        propensities=propensities,
                        reward=float(scores[decision.seat]),
                        baseline_index=decision.chosen_index,
                        target_index=target,
                        direct_values=direct_values,
                    )
                )
                target_kinds[decision.legal_actions[target].kind] += 1
    return observations, target_kinds, scanned


def main() -> None:
    args = parse_args()
    if len(args.outcome_checkpoint) < 2:
        raise ValueError("保守 outcome LCB 至少需要两个独立 checkpoint")
    if (
        args.value_scale <= 0
        or args.score_lcb_z < 0
        or args.minimum_lcb_advantage < 0
        or args.minimum_effective_sample_size <= 0
    ):
        raise ValueError("OPE 数值参数不合法")
    agents = [
        TorchPolicyValueAgent.load(path, device=args.device)
        for path in args.outcome_checkpoint
    ]
    observations, target_kinds, scanned = selected_interventions(
        args.data,
        agents,
        value_scale=args.value_scale,
        score_lcb_z=args.score_lcb_z,
        minimum_lcb_advantage=args.minimum_lcb_advantage,
    )
    estimates = intervention_estimates(observations)
    ips = estimates["ips"]
    dr = estimates.get("doubly_robust")
    support = estimates["support"]
    assert isinstance(ips, dict) and isinstance(support, dict)
    assert isinstance(dr, dict)
    ready = (
        float(ips["95pct_low"]) > 0.0
        and float(dr["95pct_low"]) > 0.0
        and float(support["target_effective_sample_size"])
        >= args.minimum_effective_sample_size
        and float(support["baseline_effective_sample_size"])
        >= args.minimum_effective_sample_size
    )
    payload = {
        "status": "audit_only_one_response_teacher_override",
        "outcome_checkpoints": [str(path) for path in args.outcome_checkpoint],
        "data": [str(path) for path in args.data],
        "inference_device": args.device,
        "candidate": {
            "base": "heuristic_teacher",
            "target": "max_afterstate_score_lcb_or_teacher",
            "score_lcb_z": args.score_lcb_z,
            "minimum_lcb_advantage": args.minimum_lcb_advantage,
            "protocol": "one randomized response position; Teacher before and after",
        },
        "scanned_candidate_decisions": scanned,
        "randomized_response_interventions": len(observations),
        "target_action_kinds": dict(sorted(target_kinds.items())),
        "estimates": estimates,
        "ready_for_single_override_game_screen": ready,
        "screen_gate": {
            "requires": "both grouped IPS and grouped DR 95% lower bounds positive",
            "minimum_effective_sample_size_per_side": args.minimum_effective_sample_size,
        },
        "warning": (
            "This estimate identifies only one response replacement followed by the "
            "same Teacher suffix under the logged intervention protocol. It is not "
            "evidence for a policy that changes multiple decisions, browser deployment, "
            "or strength against humans. Outcome checkpoints must be trained without "
            "these held-out wall groups."
        ),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

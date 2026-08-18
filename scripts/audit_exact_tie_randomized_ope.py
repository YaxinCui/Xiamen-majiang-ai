#!/usr/bin/env python3
"""Audit a fixed exact-tie ensemble on high-support one-action interventions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.agents import HeuristicTeacherAgent
from xiamen_mahjong.human_review import _ActorVisibleReviewGame
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import TeacherDecision, read_trajectory_jsonl


def teacher_epsilon_propensities(
    *, action_count: int, teacher_index: int, epsilon: float
) -> tuple[float, ...]:
    if action_count <= 1 or not 0.0 < epsilon < 1.0:
        raise ValueError("exact-tie OPE action_count/epsilon 无效")
    if not 0 <= teacher_index < action_count:
        raise ValueError("exact-tie OPE Teacher index 越界")
    uniform = epsilon / action_count
    return tuple(
        uniform + (1.0 - epsilon if index == teacher_index else 0.0)
        for index in range(action_count)
    )


def deterministic_policy_difference_contribution(
    *,
    reward: float,
    logged_index: int,
    baseline_index: int,
    target_index: int,
    propensities: Sequence[float],
) -> float:
    """Unbiased target-minus-baseline contribution for one randomized row."""

    if len(propensities) <= 1 or any(value <= 0.0 for value in propensities):
        raise ValueError("exact-tie OPE propensity 无效")
    for index in (logged_index, baseline_index, target_index):
        if not 0 <= index < len(propensities):
            raise ValueError("exact-tie OPE action index 越界")
    if target_index == baseline_index:
        return 0.0
    contribution = 0.0
    if logged_index == target_index:
        contribution += reward / propensities[target_index]
    if logged_index == baseline_index:
        contribution -= reward / propensities[baseline_index]
    return contribution


def _exact_tie_indices(decision: TeacherDecision) -> tuple[int, ...]:
    if decision.state.get("phase") != "discard":
        return ()
    game = _ActorVisibleReviewGame({"state": decision.state})
    ranked = HeuristicTeacherAgent().explain_discard(game, 0)
    if len(ranked) < 2:
        return ()
    top_score = float(ranked[0]["score"])
    tied_tiles = {
        int(row["tile"]) for row in ranked if float(row["score"]) == top_score
    }
    indices = tuple(
        index
        for index, action in enumerate(decision.legal_actions)
        if action.kind == "discard" and action.tile in tied_tiles
    )
    return indices if decision.chosen_index in indices and len(indices) >= 2 else ()


def _ensemble_target_index(
    policies: Sequence[TorchPolicyValueAgent],
    decision: TeacherDecision,
    tie_indices: Sequence[int],
) -> int:
    tie_actions = tuple(decision.legal_actions[index] for index in tie_indices)
    teacher_local = tie_indices.index(decision.chosen_index)
    local = TeacherDecision(
        profile=decision.profile,
        seed=None,
        seat=0,
        state=decision.state,
        legal_actions=tie_actions,
        chosen_index=teacher_local,
        reference_teacher_index=teacher_local,
    )
    votes = [policy.predict_index(local) for policy in policies]
    selected_local = votes[0] if len(set(votes)) == 1 else teacher_local
    return int(tie_indices[selected_local])


def _interval(values: Sequence[float]) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("exact-tie OPE 至少需要两个 wall groups")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    stderr = math.sqrt(variance / len(values))
    return {
        "wall_groups": len(values),
        "mean": mean,
        "stderr": stderr,
        "95pct_low": mean - 1.96 * stderr,
        "95pct_high": mean + 1.96 * stderr,
    }


def audit(paths: Sequence[Path], policies: Sequence[TorchPolicyValueAgent]) -> dict:
    if not paths or len(policies) != 5:
        raise ValueError("exact-tie OPE 需要数据和五个 ensemble members")
    group_values: dict[str, list[float]] = defaultdict(list)
    group_seats: dict[str, set[int]] = defaultdict(set)
    issues: Counter[str] = Counter()
    randomized = 0
    exact_ties = 0
    target_overrides = 0
    supported_target_or_teacher = 0
    for path in paths:
        for trajectory in read_trajectory_jsonl(path):
            metadata = trajectory.source_metadata
            epsilon = metadata.get("uniform_exploration_probability")
            candidate_seat = metadata.get("candidate_seat")
            if (
                metadata.get("behavior_policy") != "single_intervention_epsilon_uniform"
                or metadata.get("base_policy") != "heuristic_teacher"
                or metadata.get("intervention_phase") != "discard"
                or isinstance(epsilon, bool)
                or not isinstance(epsilon, (int, float))
                or not 0.0 < float(epsilon) < 1.0
                or isinstance(candidate_seat, bool)
                or not isinstance(candidate_seat, int)
                or not 0 <= candidate_seat < 4
            ):
                issues["invalid_trajectory_metadata"] += 1
                continue
            group = trajectory.split_group_id
            scores = trajectory.outcome.get("scores")
            if not isinstance(group, str) or not group or not isinstance(scores, list) or len(scores) != 4:
                issues["invalid_group_or_outcome"] += 1
                continue
            interventions = [
                decision
                for decision in trajectory.decisions
                if decision.state.get("phase") == "discard"
                and decision.executed_index is not None
                and decision.executed_probability is not None
                and decision.executed_probability < 1.0
            ]
            if len(interventions) > 1:
                issues["multiple_randomized_decisions"] += 1
                continue
            contribution = 0.0
            if interventions:
                randomized += 1
                decision = interventions[0]
                tie_indices = _exact_tie_indices(decision)
                if tie_indices:
                    exact_ties += 1
                    target = _ensemble_target_index(policies, decision, tie_indices)
                    if target != decision.chosen_index:
                        target_overrides += 1
                        propensities = teacher_epsilon_propensities(
                            action_count=len(decision.legal_actions),
                            teacher_index=decision.chosen_index,
                            epsilon=float(epsilon),
                        )
                        assert decision.executed_index is not None
                        assert decision.executed_probability is not None
                        expected = propensities[decision.executed_index]
                        if abs(expected - decision.executed_probability) > 1e-9:
                            issues["recorded_propensity_mismatch"] += 1
                            continue
                        if decision.executed_index in {
                            target,
                            decision.chosen_index,
                        }:
                            supported_target_or_teacher += 1
                        contribution = deterministic_policy_difference_contribution(
                            reward=float(scores[candidate_seat]),
                            logged_index=decision.executed_index,
                            baseline_index=decision.chosen_index,
                            target_index=target,
                            propensities=propensities,
                        )
            group_values[group].append(contribution)
            group_seats[group].add(candidate_seat)
    complete = [
        sum(values) / 4.0
        for group, values in group_values.items()
        if len(values) == 4 and group_seats[group] == {0, 1, 2, 3}
    ]
    incomplete = len(group_values) - len(complete)
    if incomplete:
        issues["incomplete_wall_groups"] += incomplete
    interval = _interval(complete)
    return {
        "status": "diagnostic_exact_tie_randomized_ope_no_deployment_selection",
        "data": [str(path) for path in paths],
        "randomized_discard_decisions": randomized,
        "exact_tie_randomized_decisions": exact_ties,
        "fixed_ensemble_target_overrides": target_overrides,
        "logged_target_or_teacher_assignments": supported_target_or_teacher,
        "issues": dict(sorted(issues.items())),
        "paired_wall_ht_difference": interval,
        "estimand": (
            "one randomly positioned discard intervention followed by frozen Teacher; "
            "not the value of repeated deployment overrides"
        ),
        "gate": {
            "passes_positive_lower_bound": interval["95pct_low"] > 0.0 and not issues,
            "deployment_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policies = [
        TorchPolicyValueAgent.load(path, device=args.device) for path in args.checkpoint
    ]
    report = audit(args.data, policies)
    if args.output.exists():
        raise ValueError("拒绝覆盖 exact-tie randomized OPE 报告")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

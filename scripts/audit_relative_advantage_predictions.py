#!/usr/bin/env python3
"""Audit one relative-advantage model against validation pseudo-labels only."""

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

from scripts.audit_teacher_response_intervention_ope import mean, teacher_epsilon_propensities
from scripts.select_teacher_relative_advantage_override import best_index
from xiamen_mahjong.off_policy import (
    LoggedIntervention,
    winsorized_doubly_robust_action_advantages,
)
from xiamen_mahjong.relative_advantage import RelativeAdvantageAgent
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import read_trajectory_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--outcome-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--maximum-abs-correction", type=float, required=True)
    parser.add_argument("--huber-delta", type=float, default=16.0)
    parser.add_argument("--value-scale", type=float, default=80.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def huber(error: float, *, delta: float) -> float:
    magnitude = abs(error)
    return 0.5 * magnitude**2 / delta if magnitude <= delta else magnitude - 0.5 * delta


def main() -> None:
    args = parse_args()
    if (
        len(args.outcome_checkpoint) != 3
        or args.maximum_abs_correction <= 0
        or args.huber_delta <= 0
        or args.value_scale <= 0
    ):
        raise ValueError("v3 需要三个 direct checkpoint 与正 C/delta/value-scale")
    agent = RelativeAdvantageAgent.load(args.checkpoint, device=args.device)
    direct_agents = [
        TorchPolicyValueAgent.load(path, device=args.device)
        for path in args.outcome_checkpoint
    ]
    errors: list[float] = []
    target_values: list[float] = []
    prediction_values: list[float] = []
    target_best_matches = 0
    teacher_abs_max = 0.0
    decisions = 0
    wall_groups: set[str] = set()
    for trajectory in read_trajectory_jsonl(args.data):
        metadata = trajectory.source_metadata
        if (
            metadata.get("behavior_policy") != "single_intervention_epsilon_uniform"
            or metadata.get("base_policy") != "heuristic_teacher"
            or metadata.get("intervention_phase") != "discard"
        ):
            raise ValueError("validation 输入不符合 Teacher discard intervention 契约")
        epsilon = metadata.get("uniform_exploration_probability")
        scores = trajectory.outcome.get("scores")
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not isinstance(scores, list)
            or len(scores) != 4
        ):
            raise ValueError("validation 输入缺少 epsilon 或得分")
        for decision in trajectory.decisions:
            if (
                decision.state.get("phase") != "discard"
                or decision.executed_index is None
                or decision.executed_probability is None
                or decision.executed_probability >= 1.0
            ):
                continue
            outputs = [item.afterstate_outcomes(decision) for item in direct_agents]
            if any(item is None for item in outputs):
                raise ValueError("direct checkpoint 缺少 afterstate outcome head")
            safe_outputs = [item for item in outputs if item is not None]
            direct_values = tuple(
                mean([item[0][index] * args.value_scale for item in safe_outputs])
                for index in range(len(decision.legal_actions))
            )
            propensities = teacher_epsilon_propensities(
                action_count=len(decision.legal_actions),
                teacher_index=decision.chosen_index,
                epsilon=float(epsilon),
            )
            if not math.isclose(
                propensities[decision.executed_index],
                decision.executed_probability,
                abs_tol=1e-9,
            ):
                raise ValueError("validation propensity 与 epsilon 契约不一致")
            targets = winsorized_doubly_robust_action_advantages(
                LoggedIntervention(
                    group_id=trajectory.split_group_id,
                    logged_index=decision.executed_index,
                    propensities=propensities,
                    reward=float(scores[decision.seat]),
                    baseline_index=decision.chosen_index,
                    target_index=decision.chosen_index,
                    direct_values=direct_values,
                ),
                maximum_abs_correction=args.maximum_abs_correction,
            )
            predictions = agent.relative_advantages(decision)
            teacher = decision.chosen_index
            teacher_abs_max = max(teacher_abs_max, abs(predictions[teacher]))
            target_best_matches += best_index(targets) == best_index(predictions)
            for index, target in enumerate(targets):
                if index == teacher:
                    continue
                prediction = predictions[index]
                errors.append(prediction - target)
                target_values.append(target)
                prediction_values.append(prediction)
            decisions += 1
            wall_groups.add(trajectory.split_group_id)
    if not errors:
        raise ValueError("没有可审计的 validation intervention")
    payload = {
        "status": "validation_diagnostic_not_a_candidate_selection",
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "outcome_checkpoints": [str(path) for path in args.outcome_checkpoint],
        "maximum_abs_correction": args.maximum_abs_correction,
        "huber_delta": args.huber_delta,
        "decisions": decisions,
        "wall_groups": len(wall_groups),
        "non_teacher_actions": len(errors),
        "prediction_vs_crossfit_pseudo_label": {
            "mae": mean([abs(error) for error in errors]),
            "huber": mean([huber(error, delta=args.huber_delta) for error in errors]),
            "target_mean": mean(target_values),
            "prediction_mean": mean(prediction_values),
            "target_best_action_match_rate": target_best_matches / decisions,
            "teacher_prediction_absolute_max": teacher_abs_max,
        },
        "warning": (
            "Validation comparison is against explicitly biased pseudo-labels, not true "
            "counterfactual outcomes. It cannot select C/T or justify deployment."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

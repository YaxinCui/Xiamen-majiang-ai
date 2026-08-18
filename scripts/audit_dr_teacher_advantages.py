#!/usr/bin/env python3
"""Audit DR pseudo-advantages versus Teacher on an untouched validation split.

This is deliberately an aggregate-only diagnostic.  It consumes one-action
Teacher-suffix intervention records and direct outcome models trained without
their wall groups, then reports the scale and tail weight of the per-action
doubly-robust advantage targets.  It neither fits an advantage learner nor
selects a Mahjong action.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_teacher_response_intervention_ope import (
    mean,
    teacher_epsilon_propensities,
)
from xiamen_mahjong.off_policy import (
    LoggedIntervention,
    doubly_robust_action_advantages,
    winsorized_doubly_robust_action_advantages,
)
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import read_trajectory_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--intervention-phase", choices=("discard", "response"), required=True)
    parser.add_argument("--value-scale", type=float, default=80.0)
    parser.add_argument(
        "--maximum-abs-correction",
        type=float,
        help=(
            "可选：对每项 importance-weighted residual 作 winsorization。"
            "仅用于有偏训练标签的尾部审计，不能替代未缩减 OPE。"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values or not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile 输入为空或 fraction 越界")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("DR 优势审计没有样本")
    center = mean(values)
    return {
        "count": len(values),
        "mean": center,
        "std": math.sqrt(sum((value - center) ** 2 for value in values) / len(values)),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "minimum": min(values),
        "maximum": max(values),
    }


def iter_randomized_phase_decisions(
    paths: Iterable[Path], *, phase: str
):
    for path in paths:
        for trajectory in read_trajectory_jsonl(path):
            metadata = trajectory.source_metadata
            if metadata.get("behavior_policy") != "single_intervention_epsilon_uniform":
                raise ValueError("DR 优势审计只接受单点 epsilon 干预轨迹")
            if metadata.get("base_policy") != "heuristic_teacher":
                raise ValueError("DR 优势审计只接受 Teacher 基线轨迹")
            if metadata.get("intervention_phase") != phase:
                raise ValueError("干预数据 phase 与 DR 优势审计不一致")
            epsilon = metadata.get("uniform_exploration_probability")
            if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
                raise ValueError("DR 优势审计缺少 epsilon")
            scores = trajectory.outcome.get("scores")
            if not isinstance(scores, list) or len(scores) != 4:
                raise ValueError("DR 优势审计缺少四家终局分数")
            for decision in trajectory.decisions:
                if (
                    decision.state.get("phase") == phase
                    and decision.executed_index is not None
                    and decision.executed_probability is not None
                    and decision.executed_probability < 1.0
                ):
                    yield trajectory, decision, float(epsilon), float(scores[decision.seat])


def main() -> None:
    args = parse_args()
    if len(args.outcome_checkpoint) < 2 or args.value_scale <= 0:
        raise ValueError("至少需要两个 outcome checkpoint，且 value-scale 必须为正数")
    if (
        args.maximum_abs_correction is not None
        and args.maximum_abs_correction <= 0
    ):
        raise ValueError("maximum-abs-correction 必须为正数")
    agents = [
        TorchPolicyValueAgent.load(path, device=args.device)
        for path in args.outcome_checkpoint
    ]
    all_nonbaseline: list[float] = []
    logged: list[float] = []
    direct_nonbaseline: list[float] = []
    propensity_values: list[float] = []
    wall_groups: set[str] = set()
    decision_count = 0
    for trajectory, decision, epsilon, reward in iter_randomized_phase_decisions(
        args.data, phase=args.intervention_phase
    ):
        outputs = [agent.afterstate_outcomes(decision) for agent in agents]
        if any(output is None for output in outputs):
            raise ValueError("outcome checkpoint 缺少 afterstate outcome head")
        safe_outputs = [output for output in outputs if output is not None]
        direct_values = tuple(
            mean([output[0][index] * args.value_scale for output in safe_outputs])
            for index in range(len(decision.legal_actions))
        )
        baseline = decision.chosen_index
        assert decision.executed_index is not None
        propensities = teacher_epsilon_propensities(
            action_count=len(decision.legal_actions),
            teacher_index=baseline,
            epsilon=epsilon,
        )
        observation = LoggedIntervention(
            group_id=trajectory.split_group_id,
            logged_index=decision.executed_index,
            propensities=propensities,
            reward=reward,
            baseline_index=baseline,
            target_index=baseline,
            direct_values=direct_values,
        )
        advantages = (
            doubly_robust_action_advantages(observation)
            if args.maximum_abs_correction is None
            else winsorized_doubly_robust_action_advantages(
                observation,
                maximum_abs_correction=args.maximum_abs_correction,
            )
        )
        logged.append(advantages[decision.executed_index])
        for index, advantage in enumerate(advantages):
            if index == baseline:
                continue
            all_nonbaseline.append(advantage)
            direct_nonbaseline.append(direct_values[index] - direct_values[baseline])
            propensity_values.append(propensities[index])
        decision_count += 1
        wall_groups.add(trajectory.split_group_id)
    payload = {
        "status": "diagnostic_only_no_advantage_learner_or_action_selection",
        "intervention_phase": args.intervention_phase,
        "outcome_checkpoints": [str(path) for path in args.outcome_checkpoint],
        "data": [str(path) for path in args.data],
        "value_scale": args.value_scale,
        "decisions": decision_count,
        "wall_groups": len(wall_groups),
        "direct_model_scope": "all listed outcome models must exclude every audited wall group",
        "pseudo_label": {
            "kind": (
                "unshrunk_doubly_robust_advantage"
                if args.maximum_abs_correction is None
                else "winsorized_doubly_robust_advantage_explicitly_biased"
            ),
            "maximum_abs_correction": args.maximum_abs_correction,
        },
        "pseudo_advantage_points": {
            "non_teacher_actions": summary(all_nonbaseline),
            "logged_actions": summary(logged),
            "direct_model_non_teacher_gap": summary(direct_nonbaseline),
        },
        "non_teacher_propensity": summary(propensity_values),
        "warning": (
            "These are per-row DR pseudo-outcome diagnostics, not an OPE result. "
            "Winsorized labels are intentionally biased. Neither form can select actions "
            "or justify a strength claim."
        ),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

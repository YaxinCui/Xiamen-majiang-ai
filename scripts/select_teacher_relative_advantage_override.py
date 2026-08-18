#!/usr/bin/env python3
"""Select a pre-registered Teacher-relative override on held-out wall groups.

This is an evaluator, not a deployment entry point.  It estimates a single
discard replacement followed by a frozen Teacher suffix.  Selection reads only
the selector walls.  Terminal walls are opened exactly once, and only when a
candidate clears both unshrunk IPS and DR lower-bound gates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_teacher_response_intervention_ope import mean, teacher_epsilon_propensities
from xiamen_mahjong.off_policy import LoggedIntervention, intervention_estimates
from xiamen_mahjong.relative_advantage import RelativeAdvantageAgent
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import TeacherDecision, read_trajectory_jsonl


@dataclass(frozen=True)
class Candidate:
    maximum_abs_correction: float
    agent: RelativeAdvantageAgent
    checkpoint: Path
    report: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="格式 C=relative-advantage.pt；必须完整提供 20、40、80。",
    )
    parser.add_argument(
        "--candidate-report",
        action="append",
        required=True,
        help="格式 C=report.json；用来核验固定训练契约。",
    )
    parser.add_argument("--outcome-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--selection-data", type=Path, required=True)
    parser.add_argument("--terminal-data", type=Path, required=True)
    parser.add_argument("--minimum-advantage", type=float, action="append", required=True)
    parser.add_argument("--minimum-effective-sample-size", type=float, default=75.0)
    parser.add_argument("--value-scale", type=float, default=80.0)
    parser.add_argument("--intervention-phase", choices=("discard",), default="discard")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _parse_mapping(raw: Sequence[str], *, label: str) -> dict[float, Path]:
    parsed: dict[float, Path] = {}
    for item in raw:
        key, separator, value = item.partition("=")
        if not separator or not key or not value:
            raise ValueError(f"{label} 必须使用 C=path 格式")
        try:
            correction = float(key)
        except ValueError as error:
            raise ValueError(f"{label} 的 C 必须为数字") from error
        if correction in parsed:
            raise ValueError(f"{label} 中的 C 不能重复")
        parsed[correction] = Path(value)
    return parsed


def load_candidates(args: argparse.Namespace) -> tuple[Candidate, ...]:
    checkpoints = _parse_mapping(args.candidate, label="candidate")
    reports = _parse_mapping(args.candidate_report, label="candidate-report")
    required = {20.0, 40.0, 80.0}
    if set(checkpoints) != required or set(reports) != required:
        raise ValueError("必须恰好提供 C=20、40、80 三个预注册候选及报告")
    candidates = []
    for correction in sorted(required):
        report_path = reports[correction]
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        contract = payload.get("training_contract")
        if (
            payload.get("model") != "teacher_relative_advantage_centered_mlp"
            or payload.get("status") != "diagnostic_only_not_authorized_for_action_selection"
            or not isinstance(contract, dict)
            or float(contract.get("maximum_abs_correction")) != correction
            or contract.get("teacher_action_prediction") != "exactly_zero_by_centering"
            or contract.get("validation_or_heldout_read") is not False
            or contract.get("terminal_outcomes_in_input") is not False
        ):
            raise ValueError(f"C={correction:g} 的相对优势训练契约不满足")
        candidates.append(
            Candidate(
                maximum_abs_correction=correction,
                agent=RelativeAdvantageAgent.load(checkpoints[correction], device=args.device),
                checkpoint=checkpoints[correction],
                report=report_path,
            )
        )
    return tuple(candidates)


def best_index(values: Sequence[float]) -> int:
    if not values:
        raise ValueError("候选动作为空")
    return max(range(len(values)), key=lambda index: (values[index], -index))


def target_index(
    decision: TeacherDecision, candidate: Candidate, *, minimum_advantage: float
) -> int:
    values = candidate.agent.relative_advantages(decision)
    teacher = decision.chosen_index
    if len(values) != len(decision.legal_actions) or abs(values[teacher]) > 1e-6:
        raise ValueError("relative advantage 模型未保持 Teacher 零点")
    proposed = best_index(values)
    return proposed if proposed != teacher and values[proposed] > minimum_advantage else teacher


def _direct_values(
    decision: TeacherDecision, agents: Sequence[TorchPolicyValueAgent], *, value_scale: float
) -> tuple[float, ...]:
    outputs = [agent.afterstate_outcomes(decision) for agent in agents]
    if any(output is None for output in outputs):
        raise ValueError("outcome checkpoint 缺少 afterstate outcome head")
    safe_outputs = [output for output in outputs if output is not None]
    return tuple(
        mean([output[0][index] * value_scale for output in safe_outputs])
        for index in range(len(decision.legal_actions))
    )


def selected_interventions(
    paths: Iterable[Path],
    *,
    candidate: Candidate,
    minimum_advantage: float,
    outcome_agents: Sequence[TorchPolicyValueAgent],
    value_scale: float,
    phase: str,
) -> tuple[list[LoggedIntervention], Counter[str], int]:
    observations: list[LoggedIntervention] = []
    target_kinds: Counter[str] = Counter()
    scanned = 0
    for path in paths:
        for trajectory in read_trajectory_jsonl(path):
            metadata = trajectory.source_metadata
            if metadata.get("behavior_policy") != "single_intervention_epsilon_uniform":
                raise ValueError("OPE 输入不是单点 epsilon 干预轨迹")
            if metadata.get("base_policy") != "heuristic_teacher":
                raise ValueError("OPE 输入不是 Teacher 基线")
            if metadata.get("intervention_phase") != phase:
                raise ValueError("OPE 输入 phase 与候选不一致")
            epsilon = metadata.get("uniform_exploration_probability")
            scores = trajectory.outcome.get("scores")
            if (
                isinstance(epsilon, bool)
                or not isinstance(epsilon, (int, float))
                or not 0.0 < float(epsilon) < 1.0
                or not isinstance(scores, list)
                or len(scores) != 4
            ):
                raise ValueError("OPE 轨迹缺少 epsilon 或终局得分")
            for decision in trajectory.decisions:
                if (
                    decision.state.get("phase") != phase
                    or decision.executed_index is None
                    or decision.executed_probability is None
                    or decision.executed_probability >= 1.0
                ):
                    continue
                scanned += 1
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
                    raise ValueError("OPE recorded propensity 与 epsilon 契约不一致")
                selected = target_index(
                    decision, candidate, minimum_advantage=minimum_advantage
                )
                observations.append(
                    LoggedIntervention(
                        group_id=trajectory.split_group_id,
                        logged_index=decision.executed_index,
                        propensities=propensities,
                        reward=float(scores[decision.seat]),
                        baseline_index=decision.chosen_index,
                        target_index=selected,
                        direct_values=_direct_values(
                            decision, outcome_agents, value_scale=value_scale
                        ),
                    )
                )
                target_kinds[decision.legal_actions[selected].kind] += 1
    if not observations:
        raise ValueError("OPE 没有随机干预记录")
    return observations, target_kinds, scanned


def gate_summary(
    estimates: dict[str, object], *, minimum_effective_sample_size: float
) -> dict[str, float | bool]:
    ips = estimates.get("ips")
    dr = estimates.get("doubly_robust")
    support = estimates.get("support")
    if not isinstance(ips, dict) or not isinstance(dr, dict) or not isinstance(support, dict):
        raise ValueError("候选 OPE 缺少 IPS、DR 或支持度")
    ips_low = float(ips["95pct_low"])
    dr_low = float(dr["95pct_low"])
    target_ess = float(support["target_effective_sample_size"])
    baseline_ess = float(support["baseline_effective_sample_size"])
    return {
        "ips_95pct_low": ips_low,
        "dr_95pct_low": dr_low,
        "minimum_lower_bound": min(ips_low, dr_low),
        "target_effective_sample_size": target_ess,
        "baseline_effective_sample_size": baseline_ess,
        "passes": (
            ips_low > 0.0
            and dr_low > 0.0
            and target_ess >= minimum_effective_sample_size
            and baseline_ess >= minimum_effective_sample_size
        ),
    }


def audit_one(
    *,
    data: Path,
    candidate: Candidate,
    minimum_advantage: float,
    outcome_agents: Sequence[TorchPolicyValueAgent],
    value_scale: float,
    minimum_effective_sample_size: float,
    phase: str,
) -> dict[str, Any]:
    observations, target_kinds, scanned = selected_interventions(
        (data,),
        candidate=candidate,
        minimum_advantage=minimum_advantage,
        outcome_agents=outcome_agents,
        value_scale=value_scale,
        phase=phase,
    )
    estimates = intervention_estimates(observations)
    return {
        "maximum_abs_correction": candidate.maximum_abs_correction,
        "minimum_advantage": minimum_advantage,
        "scanned_candidate_decisions": scanned,
        "randomized_interventions": len(observations),
        "target_action_kinds": dict(sorted(target_kinds.items())),
        "estimates": estimates,
        "gate": gate_summary(
            estimates, minimum_effective_sample_size=minimum_effective_sample_size
        ),
    }


def choose_candidate(audits: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [item for item in audits if item["gate"]["passes"]]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            float(item["gate"]["minimum_lower_bound"]),
            float(item["minimum_advantage"]),
            -float(item["maximum_abs_correction"]),
        ),
    )


def main() -> None:
    args = parse_args()
    thresholds = tuple(sorted(set(args.minimum_advantage)))
    if (
        thresholds != (0.0, 8.0, 16.0)
        or args.minimum_effective_sample_size != 75.0
        or args.value_scale <= 0
    ):
        raise ValueError("只接受 v3 预注册阈值 {0,8,16}、ESS=75 与正 value-scale")
    candidates = load_candidates(args)
    outcome_agents = [
        TorchPolicyValueAgent.load(path, device=args.device)
        for path in args.outcome_checkpoint
    ]
    if len(outcome_agents) != 3:
        raise ValueError("v3 必须使用三个固定 cross-fit direct outcome 模型")
    selection_audits = [
        audit_one(
            data=args.selection_data,
            candidate=candidate,
            minimum_advantage=threshold,
            outcome_agents=outcome_agents,
            value_scale=args.value_scale,
            minimum_effective_sample_size=args.minimum_effective_sample_size,
            phase=args.intervention_phase,
        )
        for candidate in candidates
        for threshold in thresholds
    ]
    chosen = choose_candidate(selection_audits)
    payload: dict[str, Any] = {
        "status": "selection_rejected_terminal_unread",
        "protocol": {
            "base": "heuristic_teacher",
            "intervention_phase": args.intervention_phase,
            "policy": "one discard override followed by Teacher suffix",
            "pre_registered_corrections": [20.0, 40.0, 80.0],
            "pre_registered_minimum_advantages": thresholds,
            "minimum_effective_sample_size": args.minimum_effective_sample_size,
            "selection_data": str(args.selection_data),
            "terminal_data": str(args.terminal_data),
            "terminal_read": False,
        },
        "candidates": [
            {
                "maximum_abs_correction": candidate.maximum_abs_correction,
                "checkpoint": str(candidate.checkpoint),
                "report": str(candidate.report),
            }
            for candidate in candidates
        ],
        "outcome_checkpoints": [str(path) for path in args.outcome_checkpoint],
        "selection_audits": selection_audits,
    }
    if chosen is not None:
        terminal = audit_one(
            data=args.terminal_data,
            candidate=next(
                candidate
                for candidate in candidates
                if candidate.maximum_abs_correction
                == float(chosen["maximum_abs_correction"])
            ),
            minimum_advantage=float(chosen["minimum_advantage"]),
            outcome_agents=outcome_agents,
            value_scale=args.value_scale,
            minimum_effective_sample_size=args.minimum_effective_sample_size,
            phase=args.intervention_phase,
        )
        payload["status"] = "terminal_audited_not_authorized_for_deployment"
        payload["chosen_on_selection"] = chosen
        payload["terminal_audit"] = terminal
        payload["protocol"]["terminal_read"] = True
        payload["ready_for_single_override_game_screen"] = bool(terminal["gate"]["passes"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

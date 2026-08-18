#!/usr/bin/env python3
"""Select and terminal-audit a pre-registered stochastic Teacher residual policy.

The target only replaces one discard decision, then resumes the frozen Teacher.
It is evaluated as a stochastic contextual-bandit policy with exact logged
propensities; it is never a full Mahjong policy or a web deployment entry.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_teacher_response_intervention_ope import teacher_epsilon_propensities
from scripts.select_teacher_relative_advantage_override import (
    Candidate,
    _direct_values,
    load_candidates,
)
from xiamen_mahjong.off_policy import (
    LoggedStochasticIntervention,
    stochastic_intervention_estimates,
)
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import TeacherDecision, read_trajectory_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--candidate-report", action="append", required=True)
    parser.add_argument("--outcome-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--outcome-report", type=Path, action="append", required=True)
    parser.add_argument("--selection-data", type=Path, required=True)
    parser.add_argument("--terminal-data", type=Path, required=True)
    parser.add_argument("--beta", type=float, action="append", required=True)
    parser.add_argument("--temperature", type=float, action="append", required=True)
    parser.add_argument("--minimum-effective-sample-size", type=float, default=100.0)
    parser.add_argument("--value-scale", type=float, default=80.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stochastic_probabilities(
    decision: TeacherDecision,
    candidate: Candidate,
    *,
    beta: float,
    temperature: float,
) -> tuple[float, ...]:
    """Return ``(1-beta) Teacher + beta softmax(clipped relative advantage)``."""

    if not 0.0 < beta < 1.0 or temperature <= 0.0:
        raise ValueError("beta 必须在 (0,1)，temperature 必须为正")
    values = candidate.agent.relative_advantages(decision)
    teacher = decision.chosen_index
    if len(values) != len(decision.legal_actions) or abs(values[teacher]) > 1e-6:
        raise ValueError("relative advantage 模型未保持 Teacher 零点")
    logits = [max(-32.0, min(32.0, value)) / temperature for value in values]
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = sum(exponentials)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("relative advantage softmax 无效")
    residual = [value / total for value in exponentials]
    target = [beta * value for value in residual]
    target[teacher] += 1.0 - beta
    if not math.isclose(sum(target), 1.0, abs_tol=1e-8) or any(
        not math.isfinite(value) or value < 0.0 for value in target
    ):
        raise RuntimeError("stochastic target policy 未归一化")
    return tuple(target)


def _validate_outcome_reports(paths: Sequence[Path]) -> None:
    if len(paths) != 3:
        raise ValueError("v4 需要三个 cross-fit direct outcome report")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        boundary = payload.get("behavior_boundary")
        selection = payload.get("checkpoint_selection")
        if (
            payload.get("status") != "diagnostic_only_not_authorized_for_action_selection"
            or not isinstance(boundary, dict)
            or boundary.get("fixed_epochs_without_validation") is not True
            or boundary.get("terminal_test_read") is not False
            or not isinstance(selection, dict)
            or selection.get("metric") != "fixed_pre_registered_epoch"
        ):
            raise ValueError(f"direct report 不满足固定 cross-fit 契约：{path}")


def selected_interventions(
    paths: Iterable[Path],
    *,
    candidate: Candidate,
    beta: float,
    temperature: float,
    outcome_agents: Sequence[TorchPolicyValueAgent],
    value_scale: float,
) -> tuple[list[LoggedStochasticIntervention], Counter[str], int, float]:
    rows: list[LoggedStochasticIntervention] = []
    target_kinds: Counter[str] = Counter()
    scanned = 0
    expected_override_probability = 0.0
    for path in paths:
        for trajectory in read_trajectory_jsonl(path):
            metadata = trajectory.source_metadata
            if (
                metadata.get("behavior_policy") != "single_intervention_epsilon_uniform"
                or metadata.get("base_policy") != "heuristic_teacher"
                or metadata.get("intervention_phase") != "discard"
            ):
                raise ValueError("OPE 输入不符合 Teacher discard 单点干预契约")
            epsilon = metadata.get("uniform_exploration_probability")
            scores = trajectory.outcome.get("scores")
            if (
                isinstance(epsilon, bool)
                or not isinstance(epsilon, (int, float))
                or not 0.0 < float(epsilon) < 1.0
                or not isinstance(scores, list)
                or len(scores) != 4
            ):
                raise ValueError("OPE 输入缺少有效 epsilon 或终局分数")
            for decision in trajectory.decisions:
                if (
                    decision.state.get("phase") != "discard"
                    or decision.executed_index is None
                    or decision.executed_probability is None
                    or decision.executed_probability >= 1.0
                ):
                    continue
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
                    raise ValueError("记录 propensity 与 epsilon 行为契约不一致")
                target = stochastic_probabilities(
                    decision, candidate, beta=beta, temperature=temperature
                )
                baseline = tuple(
                    1.0 if index == decision.chosen_index else 0.0
                    for index in range(len(decision.legal_actions))
                )
                rows.append(
                    LoggedStochasticIntervention(
                        group_id=trajectory.split_group_id,
                        logged_index=decision.executed_index,
                        propensities=propensities,
                        reward=float(scores[decision.seat]),
                        baseline_probabilities=baseline,
                        target_probabilities=target,
                        direct_values=_direct_values(
                            decision, outcome_agents, value_scale=value_scale
                        ),
                    )
                )
                target_kinds[decision.legal_actions[decision.chosen_index].kind] += 1
                expected_override_probability += 1.0 - target[decision.chosen_index]
                scanned += 1
    if not rows:
        raise ValueError("OPE 没有随机干预决策")
    return rows, target_kinds, scanned, expected_override_probability / len(rows)


def gate_summary(
    estimates: dict[str, object], *, minimum_effective_sample_size: float
) -> dict[str, float | bool]:
    ips = estimates.get("ips")
    dr = estimates.get("doubly_robust")
    support = estimates.get("support")
    if not isinstance(ips, dict) or not isinstance(dr, dict) or not isinstance(support, dict):
        raise ValueError("stochastic candidate 缺少 IPS、DR 或支持度")
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
    beta: float,
    temperature: float,
    outcome_agents: Sequence[TorchPolicyValueAgent],
    value_scale: float,
    minimum_effective_sample_size: float,
) -> dict[str, Any]:
    rows, target_kinds, scanned, override_probability = selected_interventions(
        (data,),
        candidate=candidate,
        beta=beta,
        temperature=temperature,
        outcome_agents=outcome_agents,
        value_scale=value_scale,
    )
    estimates = stochastic_intervention_estimates(rows)
    return {
        "maximum_abs_correction": candidate.maximum_abs_correction,
        "beta": beta,
        "temperature": temperature,
        "scanned_candidate_decisions": scanned,
        "expected_override_probability": override_probability,
        "teacher_action_kinds": dict(sorted(target_kinds.items())),
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
            -float(item["beta"]),
            float(item["temperature"]),
            -float(item["maximum_abs_correction"]),
        ),
    )


def main() -> None:
    args = parse_args()
    betas = tuple(sorted(set(args.beta)))
    temperatures = tuple(sorted(set(args.temperature)))
    if (
        betas != (0.05, 0.1)
        or temperatures != (8.0, 16.0)
        or args.minimum_effective_sample_size != 100.0
        or args.value_scale <= 0
    ):
        raise ValueError("只接受 v4 预注册 beta、temperature、ESS 与正 value-scale")
    candidates = load_candidates(args)
    _validate_outcome_reports(args.outcome_report)
    if len(args.outcome_checkpoint) != 3:
        raise ValueError("v4 必须使用三个 cross-fit direct outcome checkpoint")
    outcome_agents = [
        TorchPolicyValueAgent.load(path, device=args.device)
        for path in args.outcome_checkpoint
    ]
    selection_audits = [
        audit_one(
            data=args.selection_data,
            candidate=candidate,
            beta=beta,
            temperature=temperature,
            outcome_agents=outcome_agents,
            value_scale=args.value_scale,
            minimum_effective_sample_size=args.minimum_effective_sample_size,
        )
        for candidate in candidates
        for beta in betas
        for temperature in temperatures
    ]
    chosen = choose_candidate(selection_audits)
    payload: dict[str, Any] = {
        "status": "selection_rejected_terminal_unread",
        "protocol": {
            "base": "heuristic_teacher",
            "policy": "one stochastic discard mixture followed by Teacher suffix",
            "pre_registered_corrections": [20.0, 40.0, 80.0],
            "pre_registered_betas": betas,
            "pre_registered_temperatures": temperatures,
            "minimum_effective_sample_size": args.minimum_effective_sample_size,
            "selection_data": str(args.selection_data),
            "terminal_data": str(args.terminal_data),
            "terminal_read": False,
        },
        "selection_audits": selection_audits,
    }
    if chosen is not None:
        candidate = next(
            item
            for item in candidates
            if item.maximum_abs_correction == float(chosen["maximum_abs_correction"])
        )
        terminal = audit_one(
            data=args.terminal_data,
            candidate=candidate,
            beta=float(chosen["beta"]),
            temperature=float(chosen["temperature"]),
            outcome_agents=outcome_agents,
            value_scale=args.value_scale,
            minimum_effective_sample_size=args.minimum_effective_sample_size,
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

#!/usr/bin/env python3
"""Choose a pre-registered one-phase override, then audit it once.

Candidate thresholds are compared only on ``--selection-data``.  If none
passes the grouped IPS/DR gate, the script intentionally never reads
``--terminal-data``.  If one does pass, the deterministic winner is evaluated
once on terminal wall groups.  This still estimates one Teacher phase
override followed by a Teacher suffix, not a full Mahjong policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_teacher_response_intervention_ope import selected_interventions
from xiamen_mahjong.off_policy import intervention_estimates
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--selection-data", type=Path, required=True)
    parser.add_argument("--terminal-data", type=Path, required=True)
    parser.add_argument(
        "--minimum-lcb-advantage",
        type=float,
        action="append",
        required=True,
        help="预注册的非负分数阈值；可重复指定一个有限网格",
    )
    parser.add_argument("--score-lcb-z", type=float, default=1.0)
    parser.add_argument("--value-scale", type=float, default=80.0)
    parser.add_argument("--minimum-effective-sample-size", type=float, default=30.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--intervention-phase",
        choices=("discard", "response"),
        default="response",
        help="必须与训练 outcome 模型和 selection/terminal 轨迹的 collector phase 一致",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def gate_summary(
    estimates: dict[str, object], *, minimum_effective_sample_size: float
) -> dict[str, float | bool]:
    ips = estimates["ips"]
    dr = estimates.get("doubly_robust")
    support = estimates["support"]
    if not isinstance(ips, dict) or not isinstance(dr, dict) or not isinstance(support, dict):
        raise ValueError("候选 OPE 缺少 IPS、DR 或支持度统计")
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


def choose_candidate(audits: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Select by worst positive lower bound, breaking ties more conservatively."""

    eligible = [audit for audit in audits if audit["gate"]["passes"]]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda audit: (
            float(audit["gate"]["minimum_lower_bound"]),
            float(audit["minimum_lcb_advantage"]),
        ),
    )


def audit_one(
    *,
    data: Path,
    agents: Sequence[TorchPolicyValueAgent],
    value_scale: float,
    score_lcb_z: float,
    minimum_lcb_advantage: float,
    minimum_effective_sample_size: float,
    intervention_phase: str = "response",
) -> dict[str, Any]:
    observations, target_kinds, scanned = selected_interventions(
        (data,),
        agents,
        value_scale=value_scale,
        score_lcb_z=score_lcb_z,
        minimum_lcb_advantage=minimum_lcb_advantage,
        intervention_phase=intervention_phase,
    )
    estimates = intervention_estimates(observations)
    return {
        "minimum_lcb_advantage": minimum_lcb_advantage,
        "intervention_phase": intervention_phase,
        "scanned_candidate_decisions": scanned,
        "randomized_interventions": len(observations),
        "target_action_kinds": dict(sorted(target_kinds.items())),
        "estimates": estimates,
        "gate": gate_summary(
            estimates,
            minimum_effective_sample_size=minimum_effective_sample_size,
        ),
    }


def main() -> None:
    args = parse_args()
    thresholds = tuple(sorted(set(args.minimum_lcb_advantage)))
    if len(args.outcome_checkpoint) < 2:
        raise ValueError("保守 outcome LCB 至少需要两个独立 checkpoint")
    if (
        not thresholds
        or thresholds[0] < 0
        or args.score_lcb_z < 0
        or args.value_scale <= 0
        or args.minimum_effective_sample_size <= 0
    ):
        raise ValueError("预注册网格或 OPE 数值参数不合法")
    agents = [
        TorchPolicyValueAgent.load(path, device=args.device)
        for path in args.outcome_checkpoint
    ]
    selection_audits = [
        audit_one(
            data=args.selection_data,
            agents=agents,
            value_scale=args.value_scale,
            score_lcb_z=args.score_lcb_z,
            minimum_lcb_advantage=threshold,
            minimum_effective_sample_size=args.minimum_effective_sample_size,
            intervention_phase=args.intervention_phase,
        )
        for threshold in thresholds
    ]
    chosen = choose_candidate(selection_audits)
    payload: dict[str, Any] = {
        "status": "selection_rejected_terminal_unread",
        "outcome_checkpoints": [str(path) for path in args.outcome_checkpoint],
        "inference_device": args.device,
        "protocol": {
            "base": "heuristic_teacher",
            "intervention_phase": args.intervention_phase,
            "policy": f"one {args.intervention_phase} override followed by Teacher suffix",
            "score_lcb_z": args.score_lcb_z,
            "pre_registered_minimum_lcb_advantages": thresholds,
            "selection_data": str(args.selection_data),
            "terminal_data": str(args.terminal_data),
            "terminal_read": False,
            "selection_rule": "maximize min(IPS lower bound, DR lower bound); tie -> higher threshold",
        },
        "selection": selection_audits,
        "chosen": None,
        "terminal": None,
        "warning": (
            "No terminal result is complete-policy, browser, or human-strength evidence. "
            "The terminal corpus is read only if selection passes both grouped lower-bound gates."
        ),
    }
    if chosen is not None:
        terminal = audit_one(
            data=args.terminal_data,
            agents=agents,
            value_scale=args.value_scale,
            score_lcb_z=args.score_lcb_z,
            minimum_lcb_advantage=float(chosen["minimum_lcb_advantage"]),
            minimum_effective_sample_size=args.minimum_effective_sample_size,
            intervention_phase=args.intervention_phase,
        )
        payload.update(
            {
                "status": "terminal_audit_complete",
                "chosen": chosen["minimum_lcb_advantage"],
                "terminal": terminal,
            }
        )
        payload["protocol"]["terminal_read"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Select a pre-registered public-defense Teacher candidate on fresh walls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.agents import HeuristicTeacherAgent, RiskAwareTeacherAgent
from xiamen_mahjong.evaluation import (
    PolicyEvaluation,
    evaluate_against_teacher,
    paired_score_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--risk-weight", type=float, action="append", required=True)
    parser.add_argument("--selection-hands", type=int, default=160)
    parser.add_argument("--selection-seed", type=int, default=202611000)
    parser.add_argument("--terminal-hands", type=int, default=400)
    parser.add_argument("--terminal-seed", type=int, default=202611500)
    parser.add_argument("--profile", choices=("classic",), default="classic")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audit_candidate(
    *,
    risk_weight: float,
    baseline: PolicyEvaluation,
    hands: int,
    seed: int,
    profile: str,
) -> dict[str, Any]:
    candidate = evaluate_against_teacher(
        RiskAwareTeacherAgent(risk_weight=risk_weight),
        hands=hands,
        seed=seed,
        profile=profile,
    )
    comparison = paired_score_comparison(candidate, baseline)
    return {
        "risk_weight": risk_weight,
        "candidate": candidate.payload(),
        "paired_against_heuristic_teacher": comparison,
        "passes": comparison["paired_seed_score_delta_95pct_low"] > 0.0,
    }


def choose_candidate(audits: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [item for item in audits if item["passes"]]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            float(item["paired_against_heuristic_teacher"]["paired_seed_score_delta_95pct_low"]),
            -float(item["risk_weight"]),
        ),
    )


def baseline_evaluation(*, hands: int, seed: int, profile: str) -> PolicyEvaluation:
    return evaluate_against_teacher(
        HeuristicTeacherAgent(), hands=hands, seed=seed, profile=profile
    )


def main() -> None:
    args = parse_args()
    weights = tuple(sorted(set(args.risk_weight)))
    if (
        weights != (0.0, 1.0, 2.0, 4.0, 8.0)
        or args.selection_hands != 160
        or args.selection_seed != 202611000
        or args.terminal_hands != 400
        or args.terminal_seed != 202611500
    ):
        raise ValueError("只接受 v5 预注册的权重网格、墙数与 seed 范围")
    selection_baseline = baseline_evaluation(
        hands=args.selection_hands, seed=args.selection_seed, profile=args.profile
    )
    selection_audits = [
        audit_candidate(
            risk_weight=weight,
            baseline=selection_baseline,
            hands=args.selection_hands,
            seed=args.selection_seed,
            profile=args.profile,
        )
        for weight in weights
    ]
    chosen = choose_candidate(selection_audits)
    payload: dict[str, Any] = {
        "status": "selection_rejected_terminal_unread",
        "protocol": {
            "profile": args.profile,
            "risk_weights": weights,
            "selection_hands": args.selection_hands,
            "selection_seed": args.selection_seed,
            "terminal_hands": args.terminal_hands,
            "terminal_seed": args.terminal_seed,
            "terminal_read": False,
            "promotion_gate": "paired_seed_score_delta_95pct_low > 0",
        },
        "selection_baseline": selection_baseline.payload(),
        "selection_audits": selection_audits,
    }
    if chosen is not None:
        terminal_baseline = baseline_evaluation(
            hands=args.terminal_hands, seed=args.terminal_seed, profile=args.profile
        )
        terminal_audit = audit_candidate(
            risk_weight=float(chosen["risk_weight"]),
            baseline=terminal_baseline,
            hands=args.terminal_hands,
            seed=args.terminal_seed,
            profile=args.profile,
        )
        payload["status"] = "terminal_audited_not_authorized_for_deployment"
        payload["chosen_on_selection"] = chosen
        payload["terminal_baseline"] = terminal_baseline.payload()
        payload["terminal_audit"] = terminal_audit
        payload["protocol"]["terminal_read"] = True
        payload["ready_for_1000_wall_screen"] = terminal_audit["passes"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

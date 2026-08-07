#!/usr/bin/env python3
"""Screen the fixed exact one-draw-tenpai Teacher on fresh rotated walls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.agents import (
    ExactOneDrawTenpaiTieBreakTeacherAgent,
    HeuristicTeacherAgent,
)
from xiamen_mahjong.evaluation import (
    PolicyEvaluation,
    evaluate_against_teacher,
    paired_score_comparison,
)


SCORE_MARGIN = 2.0
MINIMUM_LIVE_ADVANTAGE = 1
SELECTION_HANDS = 160
SELECTION_SEED = 202614500
TERMINAL_HANDS = 400
TERMINAL_SEED = 202614800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-hands", type=int, default=SELECTION_HANDS)
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--terminal-hands", type=int, default=TERMINAL_HANDS)
    parser.add_argument("--terminal-seed", type=int, default=TERMINAL_SEED)
    parser.add_argument("--profile", choices=("classic",), default="classic")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fixed_candidate() -> ExactOneDrawTenpaiTieBreakTeacherAgent:
    return ExactOneDrawTenpaiTieBreakTeacherAgent(
        score_margin=SCORE_MARGIN,
        minimum_live_advantage=MINIMUM_LIVE_ADVANTAGE,
    )


def audit_candidate(
    *, baseline: PolicyEvaluation, hands: int, seed: int, profile: str
) -> dict[str, Any]:
    candidate = evaluate_against_teacher(
        fixed_candidate(), hands=hands, seed=seed, profile=profile
    )
    comparison = paired_score_comparison(candidate, baseline)
    return {
        "candidate": candidate.payload(),
        "paired_against_heuristic_teacher": comparison,
        "passes": comparison["paired_seed_score_delta_95pct_low"] > 0.0,
    }


def baseline_evaluation(*, hands: int, seed: int, profile: str) -> PolicyEvaluation:
    return evaluate_against_teacher(
        HeuristicTeacherAgent(), hands=hands, seed=seed, profile=profile
    )


def main() -> None:
    args = parse_args()
    if (
        args.selection_hands != SELECTION_HANDS
        or args.selection_seed != SELECTION_SEED
        or args.terminal_hands != TERMINAL_HANDS
        or args.terminal_seed != TERMINAL_SEED
    ):
        raise ValueError("只接受 v1 预注册的墙数与 seed 范围")
    selection_baseline = baseline_evaluation(
        hands=args.selection_hands, seed=args.selection_seed, profile=args.profile
    )
    selection_audit = audit_candidate(
        baseline=selection_baseline,
        hands=args.selection_hands,
        seed=args.selection_seed,
        profile=args.profile,
    )
    payload: dict[str, Any] = {
        "status": "selection_rejected_terminal_unread",
        "protocol": {
            "profile": args.profile,
            "score_margin": SCORE_MARGIN,
            "minimum_live_advantage": MINIMUM_LIVE_ADVANTAGE,
            "selection_hands": args.selection_hands,
            "selection_seed": args.selection_seed,
            "terminal_hands": args.terminal_hands,
            "terminal_seed": args.terminal_seed,
            "terminal_read": False,
            "promotion_gate": "paired_seed_score_delta_95pct_low > 0",
        },
        "selection_baseline": selection_baseline.payload(),
        "selection_audit": selection_audit,
    }
    if selection_audit["passes"]:
        terminal_baseline = baseline_evaluation(
            hands=args.terminal_hands, seed=args.terminal_seed, profile=args.profile
        )
        terminal_audit = audit_candidate(
            baseline=terminal_baseline,
            hands=args.terminal_hands,
            seed=args.terminal_seed,
            profile=args.profile,
        )
        payload["status"] = "terminal_audited_not_authorized_for_deployment"
        payload["terminal_baseline"] = terminal_baseline.payload()
        payload["terminal_audit"] = terminal_audit
        payload["protocol"]["terminal_read"] = True
        payload["ready_for_1000_wall_screen"] = terminal_audit["passes"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

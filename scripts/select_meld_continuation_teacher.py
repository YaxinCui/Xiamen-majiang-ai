#!/usr/bin/env python3
"""Select a pre-registered response-continuation Teacher on fresh walls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.agents import HeuristicTeacherAgent, MeldContinuationTeacherAgent
from xiamen_mahjong.evaluation import (
    PolicyEvaluation,
    evaluate_against_teacher,
    paired_score_comparison,
)


GAINS = (0.0, 16.0, 32.0)
SELECTION_HANDS = 160
SELECTION_SEED = 202613200
TERMINAL_HANDS = 400
TERMINAL_SEED = 202613500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-claim-gain", type=float, action="append", required=True)
    parser.add_argument("--selection-hands", type=int, default=SELECTION_HANDS)
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--terminal-hands", type=int, default=TERMINAL_HANDS)
    parser.add_argument("--terminal-seed", type=int, default=TERMINAL_SEED)
    parser.add_argument("--profile", choices=("classic",), default="classic")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audit_candidate(
    *,
    minimum_claim_gain: float,
    baseline: PolicyEvaluation,
    hands: int,
    seed: int,
    profile: str,
) -> dict[str, Any]:
    candidate = evaluate_against_teacher(
        MeldContinuationTeacherAgent(minimum_claim_gain=minimum_claim_gain),
        hands=hands,
        seed=seed,
        profile=profile,
    )
    comparison = paired_score_comparison(candidate, baseline)
    return {
        "minimum_claim_gain": minimum_claim_gain,
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
            -float(item["minimum_claim_gain"]),
        ),
    )


def baseline_evaluation(*, hands: int, seed: int, profile: str) -> PolicyEvaluation:
    return evaluate_against_teacher(
        HeuristicTeacherAgent(), hands=hands, seed=seed, profile=profile
    )


def main() -> None:
    args = parse_args()
    gains = tuple(sorted(set(args.minimum_claim_gain)))
    if (
        gains != GAINS
        or args.selection_hands != SELECTION_HANDS
        or args.selection_seed != SELECTION_SEED
        or args.terminal_hands != TERMINAL_HANDS
        or args.terminal_seed != TERMINAL_SEED
    ):
        raise ValueError("只接受 v1 预注册的阈值网格、墙数与 seed 范围")
    selection_baseline = baseline_evaluation(
        hands=args.selection_hands, seed=args.selection_seed, profile=args.profile
    )
    selection_audits = [
        audit_candidate(
            minimum_claim_gain=gain,
            baseline=selection_baseline,
            hands=args.selection_hands,
            seed=args.selection_seed,
            profile=args.profile,
        )
        for gain in gains
    ]
    chosen = choose_candidate(selection_audits)
    payload: dict[str, Any] = {
        "status": "selection_rejected_terminal_unread",
        "protocol": {
            "profile": args.profile,
            "minimum_claim_gains": gains,
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
            minimum_claim_gain=float(chosen["minimum_claim_gain"]),
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
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

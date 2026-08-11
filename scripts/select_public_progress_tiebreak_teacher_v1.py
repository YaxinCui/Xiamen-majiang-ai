#!/usr/bin/env python3
"""Screen exact-score public-progress tiebreaks on fresh rotated walls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.agents import (
    HeuristicTeacherAgent,
    PublicProgressTieBreakTeacherAgent,
)
from xiamen_mahjong.evaluation import (
    PolicyEvaluation,
    evaluate_against_teacher,
    paired_score_comparison,
)


SELECTION_HANDS = 100
SELECTION_SEED = 202638300


class AuditedPublicProgressTieBreakTeacher(
    PublicProgressTieBreakTeacherAgent
):
    """Fixed candidate with aggregate-only exact-tie diagnostics."""

    def __init__(self) -> None:
        super().__init__()
        self.discard_decisions = 0
        self.exact_top_ties = 0
        self.overrides = 0
        self.shanten_improvements = 0
        self.ukeire_improvement_total = 0
        self.immediate_win_copy_improvement_total = 0

    def explain_discard(self, game, player_id: int) -> list[dict[str, object]]:
        frozen = HeuristicTeacherAgent.explain_discard(self, game, player_id)
        self.discard_decisions += bool(frozen)
        if frozen:
            top_score = float(frozen[0]["score"])
            self.exact_top_ties += int(
                sum(
                    abs(float(row["score"]) - top_score) <= 1e-9
                    for row in frozen
                )
                >= 2
            )
        ranked = super().explain_discard(game, player_id)
        if not ranked or not ranked[0].get(
            "selected_by_public_progress_tiebreak"
        ):
            return ranked

        frozen_tile = int(frozen[0]["tile"])
        frozen_row = next(
            row for row in ranked if int(row["tile"]) == frozen_tile
        )
        selected = ranked[0]
        shanten_delta = int(frozen_row["regular_hand_shanten"]) - int(
            selected["regular_hand_shanten"]
        )
        ukeire_delta = int(selected["public_improving_live_copies"]) - int(
            frozen_row["public_improving_live_copies"]
        )
        win_copy_delta = int(
            selected["public_immediate_winning_live_copies"]
        ) - int(frozen_row["public_immediate_winning_live_copies"])
        if shanten_delta < 0 or ukeire_delta < 0:
            raise AssertionError("exact-tie 候选违反公开 Pareto 约束")
        if shanten_delta == 0 and ukeire_delta == 0:
            raise AssertionError("exact-tie 候选没有严格公开进展")
        self.overrides += 1
        self.shanten_improvements += int(shanten_delta > 0)
        self.ukeire_improvement_total += ukeire_delta
        self.immediate_win_copy_improvement_total += win_copy_delta
        return ranked

    def coverage_payload(self) -> dict[str, int | float]:
        return {
            "discard_decisions": self.discard_decisions,
            "exact_top_ties": self.exact_top_ties,
            "exact_top_tie_rate": (
                self.exact_top_ties / self.discard_decisions
                if self.discard_decisions
                else 0.0
            ),
            "overrides": self.overrides,
            "override_rate": (
                self.overrides / self.discard_decisions
                if self.discard_decisions
                else 0.0
            ),
            "shanten_improvements": self.shanten_improvements,
            "ukeire_improvement_total": self.ukeire_improvement_total,
            "immediate_win_copy_improvement_total": (
                self.immediate_win_copy_improvement_total
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=SELECTION_HANDS)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--profile", choices=("classic",), default="classic")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fixed_candidate() -> AuditedPublicProgressTieBreakTeacher:
    return AuditedPublicProgressTieBreakTeacher()


def _timed_evaluation(agent: Any, *, hands: int, seed: int, profile: str):
    started = time.monotonic()
    result = evaluate_against_teacher(
        agent, hands=hands, seed=seed, profile=profile
    )
    return result, time.monotonic() - started


def baseline_evaluation(*, hands: int, seed: int, profile: str) -> PolicyEvaluation:
    return evaluate_against_teacher(
        HeuristicTeacherAgent(), hands=hands, seed=seed, profile=profile
    )


def main() -> None:
    args = parse_args()
    if args.hands != SELECTION_HANDS or args.seed != SELECTION_SEED:
        raise ValueError("只接受 v1 预注册的 100 墙与 seed 范围")

    baseline, baseline_seconds = _timed_evaluation(
        HeuristicTeacherAgent(),
        hands=args.hands,
        seed=args.seed,
        profile=args.profile,
    )
    candidate = fixed_candidate()
    candidate_result, candidate_seconds = _timed_evaluation(
        candidate,
        hands=args.hands,
        seed=args.seed,
        profile=args.profile,
    )
    comparison = paired_score_comparison(candidate_result, baseline)
    passes = comparison["paired_seed_score_delta_95pct_low"] > 0.0
    payload: dict[str, Any] = {
        "status": (
            "selection_passed_strength_gate_requires_efficiency_review"
            if passes
            else "selection_rejected"
        ),
        "protocol": {
            "profile": args.profile,
            "selection_hands": args.hands,
            "selection_seed": args.seed,
            "promotion_gate": "paired_seed_score_delta_95pct_low > 0",
            "candidate_family": "exact_teacher_score_tie_public_progress_pareto",
            "terminal_read": False,
            "parameter_sweep": False,
        },
        "runtime": {
            "baseline_seconds": baseline_seconds,
            "candidate_seconds": candidate_seconds,
            "candidate_to_baseline_ratio": (
                candidate_seconds / baseline_seconds
                if baseline_seconds
                else None
            ),
        },
        "baseline": baseline.payload(),
        "candidate": candidate_result.payload(),
        "coverage": candidate.coverage_payload(),
        "paired_against_heuristic_teacher": comparison,
        "passes": passes,
        "authorization": (
            "profile_and_optimize_exact-tie computation; no deployment yet"
            if passes
            else "none"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

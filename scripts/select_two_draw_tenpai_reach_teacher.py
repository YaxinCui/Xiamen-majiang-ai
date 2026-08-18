#!/usr/bin/env python3
"""Screen the fixed two-draw tenpai reach SlowExpert on fresh walls."""

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
    HeuristicTeacherAgent,
    TwoDrawTenpaiReachTeacherAgent,
)
from xiamen_mahjong.evaluation import (
    evaluate_against_teacher,
    paired_score_comparison,
)


SCORE_MARGIN = 2.0
MINIMUM_PROBABILITY_ADVANTAGE = 0.05
STRATIFIED_SCENARIOS = 32
SELECTION_HANDS = 100
SELECTION_SEED = 202630000


class AuditedTwoDrawTenpaiReachTeacher(TwoDrawTenpaiReachTeacherAgent):
    def __init__(self) -> None:
        super().__init__(
            score_margin=SCORE_MARGIN,
            minimum_probability_advantage=MINIMUM_PROBABILITY_ADVANTAGE,
        )
        if self.STRATIFIED_SCENARIOS != STRATIFIED_SCENARIOS:
            raise ValueError("预注册场景数与实现不一致")
        self.discard_decisions = 0
        self.overrides = 0
        self.minimum_observed_advantage: float | None = None
        self.shanten_worsenings = 0

    def explain_discard(self, game, player_id: int) -> list[dict[str, object]]:
        ranked = super().explain_discard(game, player_id)
        self.discard_decisions += bool(ranked)
        if not ranked or not ranked[0].get("selected_by_two_draw_tenpai_reach"):
            return ranked
        frozen = HeuristicTeacherAgent.explain_discard(self, game, player_id)[0]
        frozen_row = next(
            row for row in ranked if int(row["tile"]) == int(frozen["tile"])
        )
        advantage = float(ranked[0]["two_draw_tenpai_probability"]) - float(
            frozen_row["two_draw_tenpai_probability"]
        )
        if advantage + 1e-12 < MINIMUM_PROBABILITY_ADVANTAGE:
            raise AssertionError("两摸听牌覆盖没有达到预注册概率门槛")
        if int(ranked[0]["regular_hand_shanten"]) > int(
            frozen_row["regular_hand_shanten"]
        ):
            self.shanten_worsenings += 1
            raise AssertionError("两摸听牌覆盖增加了精确向听数")
        self.overrides += 1
        self.minimum_observed_advantage = (
            advantage
            if self.minimum_observed_advantage is None
            else min(self.minimum_observed_advantage, advantage)
        )
        return ranked

    def coverage_payload(self) -> dict[str, int | float | None]:
        return {
            "discard_decisions": self.discard_decisions,
            "overrides": self.overrides,
            "override_rate": (
                self.overrides / self.discard_decisions
                if self.discard_decisions
                else 0.0
            ),
            "minimum_observed_advantage": self.minimum_observed_advantage,
            "shanten_worsenings": self.shanten_worsenings,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=SELECTION_HANDS)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fixed_candidate() -> AuditedTwoDrawTenpaiReachTeacher:
    return AuditedTwoDrawTenpaiReachTeacher()


def main() -> None:
    args = parse_args()
    if args.hands != SELECTION_HANDS or args.seed != SELECTION_SEED:
        raise ValueError("只接受 v1 预注册的 100 墙与 seed 范围")
    baseline = evaluate_against_teacher(
        HeuristicTeacherAgent(),
        hands=args.hands,
        seed=args.seed,
        profile="classic",
    )
    candidate = fixed_candidate()
    candidate_result = evaluate_against_teacher(
        candidate,
        hands=args.hands,
        seed=args.seed,
        profile="classic",
    )
    comparison = paired_score_comparison(candidate_result, baseline)
    passes = comparison["paired_seed_score_delta_95pct_low"] > 0.0
    payload: dict[str, Any] = {
        "status": (
            "selection_passed_not_authorized_for_deployment"
            if passes
            else "selection_rejected"
        ),
        "protocol": {
            "profile": "classic",
            "score_margin": SCORE_MARGIN,
            "minimum_probability_advantage": MINIMUM_PROBABILITY_ADVANTAGE,
            "stratified_scenarios": STRATIFIED_SCENARIOS,
            "selection_hands": args.hands,
            "selection_seed": args.seed,
            "promotion_gate": "paired_seed_score_delta_95pct_low > 0",
        },
        "baseline": baseline.payload(),
        "candidate": candidate_result.payload(),
        "coverage": candidate.coverage_payload(),
        "paired_against_heuristic_teacher": comparison,
        "passes": passes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

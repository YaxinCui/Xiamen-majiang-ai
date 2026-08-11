#!/usr/bin/env python3
"""Screen the fixed v1-proposal plus exact-DP confirmation Teacher v2."""

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
    ExactTwoDrawTenpaiReachTeacherAgent,
    HeuristicTeacherAgent,
)
from xiamen_mahjong.evaluation import (
    evaluate_against_teacher,
    paired_score_comparison,
)


SCORE_MARGIN = 2.0
MINIMUM_PROBABILITY_ADVANTAGE = 0.05
STRATIFIED_PROPOSAL_SCENARIOS = 32
SELECTION_HANDS = 100
SELECTION_SEED = 202636100


class AuditedExactTwoDrawTeacher(ExactTwoDrawTenpaiReachTeacherAgent):
    def __init__(self) -> None:
        super().__init__(
            score_margin=SCORE_MARGIN,
            minimum_probability_advantage=MINIMUM_PROBABILITY_ADVANTAGE,
        )
        if self._stratified_proposal.STRATIFIED_SCENARIOS != (
            STRATIFIED_PROPOSAL_SCENARIOS
        ):
            raise ValueError("v2 proposal 场景数与预注册协议不一致")
        self.discard_decisions = 0
        self.stratified_proposals = 0
        self.exact_confirmed = 0
        self.exact_rejected = 0
        self.minimum_exact_advantage: float | None = None
        self.shanten_worsenings = 0

    def explain_discard(self, game, player_id: int) -> list[dict[str, object]]:
        ranked = super().explain_discard(game, player_id)
        self.discard_decisions += bool(ranked)
        diagnostics = self.last_exact_diagnostics
        proposed = bool(diagnostics["stratified_proposal"])
        confirmed = bool(diagnostics["exact_confirmed"])
        self.stratified_proposals += proposed
        self.exact_confirmed += confirmed
        self.exact_rejected += proposed and not confirmed
        if not confirmed:
            if ranked and ranked[0].get(
                "selected_by_exact_two_draw_tenpai_reach"
            ):
                raise AssertionError("未确认状态却改变了 Teacher")
            return ranked

        if not ranked[0].get("selected_by_exact_two_draw_tenpai_reach"):
            raise AssertionError("精确确认没有置顶唯一 proposal")
        frozen = HeuristicTeacherAgent.explain_discard(self, game, player_id)[0]
        frozen_row = next(
            row for row in ranked if int(row["tile"]) == int(frozen["tile"])
        )
        advantage = float(ranked[0]["exact_two_draw_tenpai_probability"]) - float(
            frozen_row["exact_two_draw_tenpai_probability"]
        )
        if advantage + 1e-12 < MINIMUM_PROBABILITY_ADVANTAGE:
            raise AssertionError("v2 override 没有达到精确概率门槛")
        if int(ranked[0]["regular_hand_shanten"]) > int(
            frozen_row["regular_hand_shanten"]
        ):
            self.shanten_worsenings += 1
            raise AssertionError("v2 override 增加了精确向听数")
        self.minimum_exact_advantage = (
            advantage
            if self.minimum_exact_advantage is None
            else min(self.minimum_exact_advantage, advantage)
        )
        return ranked

    def coverage_payload(self) -> dict[str, int | float | None]:
        return {
            "discard_decisions": self.discard_decisions,
            "stratified_proposals": self.stratified_proposals,
            "stratified_proposal_rate": (
                self.stratified_proposals / self.discard_decisions
                if self.discard_decisions
                else 0.0
            ),
            "exact_confirmed": self.exact_confirmed,
            "exact_confirmation_rate_given_proposal": (
                self.exact_confirmed / self.stratified_proposals
                if self.stratified_proposals
                else 0.0
            ),
            "exact_rejected": self.exact_rejected,
            "minimum_exact_advantage": self.minimum_exact_advantage,
            "shanten_worsenings": self.shanten_worsenings,
        }


def fixed_candidate() -> AuditedExactTwoDrawTeacher:
    return AuditedExactTwoDrawTeacher()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=SELECTION_HANDS)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hands != SELECTION_HANDS or args.seed != SELECTION_SEED:
        raise ValueError("只接受 v2 预注册的 100 墙与 seed 范围")
    if args.output.exists():
        raise ValueError("v2 selection 输出已存在，拒绝覆盖")

    baseline_started = time.perf_counter()
    baseline = evaluate_against_teacher(
        HeuristicTeacherAgent(),
        hands=args.hands,
        seed=args.seed,
        profile="classic",
    )
    baseline_seconds = time.perf_counter() - baseline_started

    candidate = fixed_candidate()
    candidate_started = time.perf_counter()
    candidate_result = evaluate_against_teacher(
        candidate,
        hands=args.hands,
        seed=args.seed,
        profile="classic",
    )
    candidate_seconds = time.perf_counter() - candidate_started
    comparison = paired_score_comparison(candidate_result, baseline)
    passes = comparison["paired_seed_score_delta_95pct_low"] > 0.0
    payload: dict[str, Any] = {
        "status": (
            "selection_passed_ready_for_offline_label_protocol_only"
            if passes
            else "selection_rejected"
        ),
        "protocol": {
            "profile": "classic",
            "candidate_family": (
                "v1_stratified_proposal_then_exact_two_draw_confirmation"
            ),
            "score_margin": SCORE_MARGIN,
            "minimum_probability_advantage": (
                MINIMUM_PROBABILITY_ADVANTAGE
            ),
            "stratified_proposal_scenarios": (
                STRATIFIED_PROPOSAL_SCENARIOS
            ),
            "exact_solver_version": candidate.SOLVER_VERSION,
            "selection_hands": args.hands,
            "selection_seed": args.seed,
            "promotion_gate": "paired_seed_score_delta_95pct_low > 0",
            "terminal_stage": "none_by_100_wall_routine_policy",
            "browser_authorized": False,
        },
        "runtime_seconds": {
            "baseline": baseline_seconds,
            "candidate": candidate_seconds,
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

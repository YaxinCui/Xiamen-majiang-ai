#!/usr/bin/env python3
"""Screen the fixed public Pareto deficiency Teacher on fresh rotated walls."""

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
    KnowledgeAwareDeficiencyTeacherAgent,
    PublicParetoDeficiencyTeacherAgent,
)
from xiamen_mahjong.evaluation import (
    PolicyEvaluation,
    evaluate_against_teacher,
    paired_score_comparison,
)


SCORE_MARGIN = 2.0
SELECTION_HANDS = 100
SELECTION_SEED = 202638000


class AuditedPublicParetoDeficiencyTeacher(
    PublicParetoDeficiencyTeacherAgent
):
    """Fixed v2 candidate with proposal, rejection and coverage counters."""

    def __init__(self) -> None:
        super().__init__(score_margin=SCORE_MARGIN)
        self._v1_proposal = KnowledgeAwareDeficiencyTeacherAgent(
            score_margin=SCORE_MARGIN
        )
        self.discard_decisions = 0
        self.v1_proposals = 0
        self.pareto_rejections = 0
        self.overrides = 0
        self.shanten_reduction_total = 0
        self.ukeire_change_total = 0

    def explain_discard(self, game, player_id: int) -> list[dict[str, object]]:
        frozen = HeuristicTeacherAgent.explain_discard(self, game, player_id)
        v1 = self._v1_proposal.explain_discard(game, player_id)
        ranked = super().explain_discard(game, player_id)
        self.discard_decisions += bool(ranked)

        frozen_tile = int(frozen[0]["tile"]) if frozen else None
        v1_proposed = bool(v1) and int(v1[0]["tile"]) != frozen_tile
        v2_overrode = bool(ranked) and int(ranked[0]["tile"]) != frozen_tile
        self.v1_proposals += int(v1_proposed)

        if v1_proposed and not v2_overrode:
            self.pareto_rejections += 1
        if not v2_overrode:
            return ranked

        if not ranked[0].get("selected_by_knowledge_aware_deficiency"):
            raise AssertionError("Pareto 候选未标记 override 来源")
        frozen_row = next(
            row for row in ranked if int(row["tile"]) == frozen_tile
        )
        shanten_reduction = int(frozen_row["regular_hand_shanten"]) - int(
            ranked[0]["regular_hand_shanten"]
        )
        ukeire_change = int(ranked[0]["public_improving_live_copies"]) - int(
            frozen_row["public_improving_live_copies"]
        )
        if shanten_reduction <= 0:
            raise AssertionError("Pareto 候选没有严格降低向听")
        if ukeire_change < 0:
            raise AssertionError("Pareto 候选牺牲了公开有效进张")
        self.overrides += 1
        self.shanten_reduction_total += shanten_reduction
        self.ukeire_change_total += ukeire_change
        return ranked

    def coverage_payload(self) -> dict[str, int | float]:
        return {
            "discard_decisions": self.discard_decisions,
            "v1_proposals": self.v1_proposals,
            "pareto_rejections": self.pareto_rejections,
            "overrides": self.overrides,
            "override_rate": (
                self.overrides / self.discard_decisions
                if self.discard_decisions
                else 0.0
            ),
            "v1_confirmation_rate": (
                self.overrides / self.v1_proposals
                if self.v1_proposals
                else 0.0
            ),
            "shanten_reduction_total": self.shanten_reduction_total,
            "ukeire_change_total": self.ukeire_change_total,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=SELECTION_HANDS)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--profile", choices=("classic",), default="classic")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fixed_candidate() -> AuditedPublicParetoDeficiencyTeacher:
    return AuditedPublicParetoDeficiencyTeacher()


def baseline_evaluation(*, hands: int, seed: int, profile: str) -> PolicyEvaluation:
    return evaluate_against_teacher(
        HeuristicTeacherAgent(), hands=hands, seed=seed, profile=profile
    )


def main() -> None:
    args = parse_args()
    if args.hands != SELECTION_HANDS or args.seed != SELECTION_SEED:
        raise ValueError("只接受 v2 预注册的 100 墙与 seed 范围")

    baseline = baseline_evaluation(
        hands=args.hands, seed=args.seed, profile=args.profile
    )
    candidate = fixed_candidate()
    candidate_result = evaluate_against_teacher(
        candidate, hands=args.hands, seed=args.seed, profile=args.profile
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
            "profile": args.profile,
            "score_margin": SCORE_MARGIN,
            "selection_hands": args.hands,
            "selection_seed": args.seed,
            "promotion_gate": "paired_seed_score_delta_95pct_low > 0",
            "terminal_read": False,
            "candidate_family": "strict_shanten_plus_non_decreasing_public_ukeire",
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

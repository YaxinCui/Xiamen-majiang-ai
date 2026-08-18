#!/usr/bin/env python3
"""Screen the fixed knowledge-aware deficiency Teacher on fresh walls."""

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
)
from xiamen_mahjong.evaluation import (
    PolicyEvaluation,
    evaluate_against_teacher,
    paired_score_comparison,
)


SCORE_MARGIN = 2.0
SELECTION_HANDS = 100
SELECTION_SEED = 202625000


class AuditedKnowledgeAwareDeficiencyTeacher(
    KnowledgeAwareDeficiencyTeacherAgent
):
    """Fixed candidate plus actor-visible coverage counters for its report."""

    def __init__(self) -> None:
        super().__init__(score_margin=SCORE_MARGIN)
        self.discard_decisions = 0
        self.overrides = 0
        self.shanten_reduction_total = 0

    def explain_discard(self, game, player_id: int) -> list[dict[str, object]]:
        ranked = super().explain_discard(game, player_id)
        self.discard_decisions += bool(ranked)
        if not ranked or not ranked[0].get(
            "selected_by_knowledge_aware_deficiency"
        ):
            return ranked
        frozen = HeuristicTeacherAgent.explain_discard(self, game, player_id)
        frozen_tile = int(frozen[0]["tile"])
        frozen_row = next(row for row in ranked if int(row["tile"]) == frozen_tile)
        reduction = int(frozen_row["regular_hand_shanten"]) - int(
            ranked[0]["regular_hand_shanten"]
        )
        if reduction <= 0:
            raise AssertionError("向听候选没有严格改善 frozen Teacher")
        self.overrides += 1
        self.shanten_reduction_total += reduction
        return ranked

    def coverage_payload(self) -> dict[str, int | float]:
        return {
            "discard_decisions": self.discard_decisions,
            "overrides": self.overrides,
            "override_rate": (
                self.overrides / self.discard_decisions
                if self.discard_decisions
                else 0.0
            ),
            "shanten_reduction_total": self.shanten_reduction_total,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=SELECTION_HANDS)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--profile", choices=("classic",), default="classic")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fixed_candidate() -> AuditedKnowledgeAwareDeficiencyTeacher:
    return AuditedKnowledgeAwareDeficiencyTeacher()


def baseline_evaluation(*, hands: int, seed: int, profile: str) -> PolicyEvaluation:
    return evaluate_against_teacher(
        HeuristicTeacherAgent(), hands=hands, seed=seed, profile=profile
    )


def main() -> None:
    args = parse_args()
    if args.hands != SELECTION_HANDS or args.seed != SELECTION_SEED:
        raise ValueError("只接受 v1 预注册的 100 墙与 seed 范围")

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
            "additional_terminal_screen": False,
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

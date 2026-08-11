#!/usr/bin/env python3
"""Screen the fixed exact-deficiency response Teacher on fresh walls."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.agents import (
    DeficiencyMeldTeacherAgent,
    HeuristicTeacherAgent,
)
from xiamen_mahjong.evaluation import (
    evaluate_against_teacher,
    paired_score_comparison,
)


SELECTION_HANDS = 100
SELECTION_SEED = 202632000


class AuditedDeficiencyMeldTeacher(DeficiencyMeldTeacherAgent):
    """Fixed candidate with aggregate response-only coverage counters."""

    def __init__(self) -> None:
        self.response_decisions = 0
        self.overrides = 0
        self.override_directions: Counter[str] = Counter()
        self.special_state_overrides = 0

    def choose_response(self, game, player_id: int, options):
        frozen = HeuristicTeacherAgent.choose_response(
            self, game, player_id, options
        )
        selected = super().choose_response(game, player_id, options)
        self.response_decisions += 1
        if selected == frozen:
            return selected
        self.overrides += 1
        self.override_directions[f"{frozen.kind}->{selected.kind}"] += 1
        if (
            game.tour_state is not None
            or game.gold_discard_lock_seat == player_id
            or player_id in game.opening_wait_seats
        ):
            self.special_state_overrides += 1
            raise AssertionError("精确向听响应候选覆盖了冻结特殊状态")
        return selected

    def coverage_payload(self) -> dict[str, Any]:
        return {
            "response_decisions": self.response_decisions,
            "overrides": self.overrides,
            "override_rate": (
                self.overrides / self.response_decisions
                if self.response_decisions
                else 0.0
            ),
            "override_directions": dict(sorted(self.override_directions.items())),
            "special_state_overrides": self.special_state_overrides,
        }


def fixed_candidate() -> AuditedDeficiencyMeldTeacher:
    return AuditedDeficiencyMeldTeacher()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=SELECTION_HANDS)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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
            "selection_passed_ready_for_manual_review"
            if passes
            else "selection_rejected"
        ),
        "protocol": {
            "profile": "classic",
            "candidate": "deficiency_meld_teacher_v1",
            "ordering": [
                "minimum_exact_regular_hand_shanten",
                "maximum_direct_wait_faces",
                "maximum_frozen_hand_shape_score",
            ],
            "strictly_improves_frozen_profile": True,
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

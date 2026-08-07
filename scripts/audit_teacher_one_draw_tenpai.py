#!/usr/bin/env python3
"""Audit whether near-tied Teacher discards miss exact one-draw-tenpai routes.

This is a diagnostic only.  It never selects an action, writes a checkpoint,
or exposes a hand, wall, seed, opponent hand, or per-decision trace.  The
result is an aggregate used to decide whether a costly new expert candidate is
justified.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.agents import GameAction, HeuristicTeacherAgent
from xiamen_mahjong.game import XiamenMahjongGame
from xiamen_mahjong.hand import one_draw_tenpai_profile, wait_tiles
from xiamen_mahjong.rules import XiamenRules


@dataclass
class OneDrawAudit:
    score_margin: float
    discard_decisions: int = 0
    non_tenpai_decisions: int = 0
    near_tied_non_tenpai_decisions: int = 0
    teacher_one_draw_route_counts: list[int] = field(default_factory=list)
    better_near_tied_alternatives: int = 0
    route_advantage_histogram: Counter[int] = field(default_factory=Counter)

    def inspect(self, game: XiamenMahjongGame, player_id: int) -> None:
        ranked = HeuristicTeacherAgent().explain_discard(game, player_id)
        if not ranked:
            return
        self.discard_decisions += 1
        top_score = float(ranked[0]["score"])
        player = game.players[player_id]
        candidates: list[tuple[int, float, int]] = []
        for row in ranked:
            tile = int(row["tile"])
            score = float(row["score"])
            if top_score - score > self.score_margin:
                continue
            after_discard = list(player.hand)
            after_discard.remove(tile)
            waits = wait_tiles(
                after_discard,
                game.gold_tile,
                meld_count=len(player.melds),
                melds_required=game.rules.melds_required,
                allow_seven_pairs=game.rules.allow_seven_pairs,
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
            )
            if waits:
                # A direct tenpai is already strictly closer than the
                # one-draw oracle measures; skip it from this audit's scope.
                continue
            profile = one_draw_tenpai_profile(
                after_discard,
                game.gold_tile,
                meld_count=len(player.melds),
                melds_required=game.rules.melds_required,
                allow_seven_pairs=game.rules.allow_seven_pairs,
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
            )
            candidates.append((tile, score, len(profile)))
        if not candidates:
            return
        top_tile = int(ranked[0]["tile"])
        top_route_count = next(
            (routes for tile, _score, routes in candidates if tile == top_tile),
            None,
        )
        if top_route_count is None:
            return
        self.non_tenpai_decisions += 1
        self.near_tied_non_tenpai_decisions += 1
        self.teacher_one_draw_route_counts.append(top_route_count)
        best_route_count = max(routes for _tile, _score, routes in candidates)
        advantage = best_route_count - top_route_count
        self.route_advantage_histogram[advantage] += 1
        if advantage > 0:
            self.better_near_tied_alternatives += 1

    def payload(self, *, hands: int, seed: int, profile: str) -> dict[str, Any]:
        compared = self.near_tied_non_tenpai_decisions
        mean_routes = (
            sum(self.teacher_one_draw_route_counts) / len(self.teacher_one_draw_route_counts)
            if self.teacher_one_draw_route_counts
            else None
        )
        return {
            "status": "diagnostic_only_not_authorized_for_action_selection",
            "scope": "aggregate_near_tied_non_tenpai_teacher_discards_only",
            "profile": profile,
            "hands": hands,
            "first_seed": seed,
            "score_margin": self.score_margin,
            "discard_decisions": self.discard_decisions,
            "near_tied_non_tenpai_decisions": compared,
            "teacher_one_draw_route_count_mean": mean_routes,
            "better_near_tied_alternative_rate": (
                self.better_near_tied_alternatives / compared if compared else None
            ),
            "route_face_advantage_histogram": dict(
                sorted(self.route_advantage_histogram.items())
            ),
            "privacy": "no_hand_wall_seed_or_opponent_hidden_state_exported",
        }


class AuditingTeacher(HeuristicTeacherAgent):
    def __init__(self, audit: OneDrawAudit):
        self.audit = audit

    def choose_turn_action(self, game, player_id: int) -> GameAction:
        action = super().choose_turn_action(game, player_id)
        if action.kind == "discard":
            self.audit.inspect(game, player_id)
        return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=40)
    parser.add_argument("--seed", type=int, default=202614100)
    parser.add_argument("--score-margin", type=float, default=2.0)
    parser.add_argument("--profile", choices=("classic",), default="classic")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hands <= 0 or args.score_margin < 0:
        raise ValueError("hands 必须为正数，score-margin 不能为负数")
    rules = XiamenRules.from_profile(args.profile)
    audit = OneDrawAudit(score_margin=args.score_margin)
    teacher = AuditingTeacher(audit)
    for offset in range(args.hands):
        XiamenMahjongGame(
            seed=args.seed + offset,
            rules=rules,
            agents={seat: teacher for seat in range(rules.player_count)},
            human_seat=-1,
            auto_advance=True,
        )
    payload = audit.payload(hands=args.hands, seed=args.seed, profile=args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

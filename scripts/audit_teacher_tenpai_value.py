#!/usr/bin/env python3
"""Audit a narrow, public-information blind spot in the frozen Teacher.

The frozen discard Teacher rewards the number of wait *faces* but does not
compare how many publicly possible copies remain or the ordinary self-draw
settlement of those waits.  This script streams safe trajectory-v4 files and
asks a deliberately narrower question:

    Among near-tied discards that all leave the actor in tenpai, how often
    does a different discard have larger
    ``sum(public_live_copies * visible_score_lower_bound)``?

The audit is aggregate-only.  It never exports a hand, public history, wall,
seed, opponent concealed tile, or per-decision candidate.  The visible score
is a lower bound because trajectory-v4 records the actor's flower count but
not flower identities, so family-completion water cannot be reconstructed.
No action selector or playable model is produced by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.hand import hand_quality, wait_tiles
from xiamen_mahjong.rules import XiamenRules
from xiamen_mahjong.scoring import classic_score
from xiamen_mahjong.tiles import WHITE_DRAGON, is_base_tile
from xiamen_mahjong.training import TeacherDecision, TrainingTrajectory


def _bucket(value: float, boundaries: tuple[float, ...]) -> str:
    for boundary in boundaries:
        if value <= boundary:
            return f"<= {boundary:g}"
    return f"> {boundaries[-1]:g}"


def _actor_melds(state: dict[str, Any]) -> list[dict[str, Any]]:
    for player in state.get("public_players", []):
        if int(player.get("relative_seat", -1)) != 0:
            continue
        result = []
        for meld in player.get("melds", []):
            tiles = [int(tile) for tile in meld.get("tiles", [])]
            if not tiles:
                raise ValueError("本家副露在安全轨迹中不能隐藏牌面")
            result.append({"kind": str(meld["kind"]), "tiles": tiles})
        return result
    return []


def _public_visible_counts(state: dict[str, Any]) -> Counter[int]:
    visible: Counter[int] = Counter(
        int(tile) for tile in state["hand"] if is_base_tile(int(tile))
    )
    for player in state.get("public_players", []):
        visible.update(
            int(tile)
            for tile in player.get("discards", [])
            if is_base_tile(int(tile))
        )
        for meld in player.get("melds", []):
            # Other players' concealed-kong faces are deliberately redacted.
            visible.update(
                int(tile)
                for tile in meld.get("tiles", [])
                if is_base_tile(int(tile))
            )
    indicator = state.get("gold_indicator")
    if isinstance(indicator, int) and is_base_tile(indicator):
        visible[indicator] += 1
    return visible


def _discard_rows(decision: TeacherDecision) -> list[dict[str, float | int]]:
    """Recompute frozen scores plus a public immediate-win value diagnostic."""

    state = decision.state
    if state.get("phase") != "discard":
        return []
    if state.get("tour") is not None or state.get("gold_locked"):
        # Tour settlement and flower-family identities are not fully encoded
        # in the safe actor snapshot.  Keep the first audit strictly ordinary.
        return []
    rules = XiamenRules.from_profile(str(state.get("rules_profile", "classic")))
    if not rules.enable_complex_water_scoring:
        return []
    raw_gold = state.get("gold_tile")
    gold_tile = int(raw_gold) if isinstance(raw_gold, int) else None
    proxy_tile = (
        WHITE_DRAGON
        if rules.white_dragon_is_gold_proxy
        and gold_tile is not None
        and gold_tile != WHITE_DRAGON
        else None
    )
    wildcard_tiles = {gold_tile} if rules.gold_is_wildcard and gold_tile is not None else set()
    hand = [int(tile) for tile in state["hand"]]
    melds = _actor_melds(state)
    meld_count = len(melds)
    visible = _public_visible_counts(state)
    # Flower faces are intentionally unavailable in v4.  Negative placeholders
    # preserve the known per-flower water without fabricating a completed set.
    flowers = [-(index + 1) for index in range(int(state.get("flowers", 0)))]
    rows: list[dict[str, float | int]] = []
    for action in decision.legal_actions:
        if action.kind != "discard" or action.tile is None or action.tile not in hand:
            continue
        after = list(hand)
        after.remove(action.tile)
        waits = wait_tiles(
            after,
            gold_tile,
            meld_count=meld_count,
            melds_required=rules.melds_required,
            allow_seven_pairs=rules.allow_seven_pairs,
            wildcard_tiles=wildcard_tiles,
            proxy_tile=proxy_tile,
            proxy_as=gold_tile,
        )
        quality = hand_quality(
            after,
            gold_tile,
            meld_count=meld_count,
            melds_required=rules.melds_required,
            wildcard_tiles=wildcard_tiles,
            proxy_tile=proxy_tile,
            proxy_as=gold_tile,
        )
        teacher_score = quality + len(waits) * 18.0
        if action.tile == gold_tile:
            teacher_score -= 7.0
        live_copies = 0
        weighted_value = 0.0
        for wait in waits:
            remaining = max(0, 4 - visible[wait])
            if remaining == 0:
                continue
            winner = SimpleNamespace(
                hand=sorted([*after, wait]),
                flowers=flowers,
                melds=melds,
            )
            score = classic_score(
                winner,
                is_dealer=bool(state.get("is_dealer")),
                wildcard_tiles=wildcard_tiles,
                gold_tile=gold_tile,
                win_type="self_draw",
                rules=rules,
                dealer_streak=int(state.get("dealer_streak", 0)),
                proxy_tile=proxy_tile,
                proxy_as=gold_tile,
            )
            live_copies += remaining
            weighted_value += remaining * float(score.total)
        rows.append(
            {
                "tile": int(action.tile),
                "teacher_score": float(teacher_score),
                "wait_faces": len(waits),
                "live_copies": live_copies,
                "weighted_value": weighted_value,
                "mean_score": weighted_value / live_copies if live_copies else 0.0,
            }
        )
    return sorted(
        rows,
        key=lambda row: (-float(row["teacher_score"]), int(row["tile"])),
    )


@dataclass
class TenpaiValueAudit:
    score_margin: float
    trajectories: int = 0
    decisions: int = 0
    discard_decisions: int = 0
    ordinary_discard_decisions: int = 0
    teacher_tenpai_decisions: int = 0
    multiple_tenpai_choices: int = 0
    near_tied_tenpai_choices: int = 0
    better_value_alternatives: int = 0
    top_two_margin_histogram: Counter[str] = field(default_factory=Counter)
    improvement_histogram: Counter[str] = field(default_factory=Counter)
    reason_counts: Counter[str] = field(default_factory=Counter)
    improvement_sum: float = 0.0
    teacher_live_copy_sum: int = 0
    alternative_live_copy_sum: int = 0
    teacher_mean_score_sum: float = 0.0
    alternative_mean_score_sum: float = 0.0

    def inspect(self, decision: TeacherDecision) -> None:
        self.decisions += 1
        if decision.chosen_action.kind != "discard":
            return
        self.discard_decisions += 1
        rows = _discard_rows(decision)
        if not rows:
            return
        self.ordinary_discard_decisions += 1
        chosen_tile = decision.chosen_action.tile
        if chosen_tile is None:
            return
        self.inspect_rows(rows, teacher_tile=chosen_tile)

    def inspect_rows(
        self, rows: Iterable[dict[str, float | int]], *, teacher_tile: int
    ) -> None:
        ranked = sorted(
            (dict(row) for row in rows),
            key=lambda row: (-float(row["teacher_score"]), int(row["tile"])),
        )
        if len(ranked) >= 2:
            margin = float(ranked[0]["teacher_score"]) - float(
                ranked[1]["teacher_score"]
            )
            self.top_two_margin_histogram[_bucket(margin, (0.0, 1.0, 2.0, 4.0, 8.0, 16.0))] += 1
        teacher = next(
            (row for row in ranked if int(row["tile"]) == teacher_tile), None
        )
        if teacher is None or int(teacher["wait_faces"]) <= 0:
            return
        self.teacher_tenpai_decisions += 1
        tenpai_rows = [row for row in ranked if int(row["wait_faces"]) > 0]
        if len(tenpai_rows) < 2:
            return
        self.multiple_tenpai_choices += 1
        top_score = float(teacher["teacher_score"])
        comparable = [
            row
            for row in tenpai_rows
            if top_score - float(row["teacher_score"]) <= self.score_margin + 1e-9
        ]
        if len(comparable) < 2:
            return
        self.near_tied_tenpai_choices += 1
        best = max(
            comparable,
            key=lambda row: (
                float(row["weighted_value"]),
                int(row["live_copies"]),
                float(row["mean_score"]),
                -int(row["tile"]),
            ),
        )
        improvement = float(best["weighted_value"]) - float(
            teacher["weighted_value"]
        )
        if int(best["tile"]) == teacher_tile or improvement <= 1e-9:
            return
        self.better_value_alternatives += 1
        self.improvement_sum += improvement
        self.improvement_histogram[
            _bucket(improvement, (12.0, 24.0, 48.0, 96.0, 192.0, 384.0))
        ] += 1
        teacher_live = int(teacher["live_copies"])
        best_live = int(best["live_copies"])
        teacher_score = float(teacher["mean_score"])
        best_score = float(best["mean_score"])
        self.teacher_live_copy_sum += teacher_live
        self.alternative_live_copy_sum += best_live
        self.teacher_mean_score_sum += teacher_score
        self.alternative_mean_score_sum += best_score
        more_live = best_live > teacher_live
        more_score = best_score > teacher_score + 1e-9
        if more_live and more_score:
            reason = "more_live_and_higher_score"
        elif more_live:
            reason = "more_live_copies"
        elif more_score:
            reason = "higher_visible_score"
        else:
            reason = "different_live_score_tradeoff"
        self.reason_counts[reason] += 1

    def payload(self, *, profile: str, maximum_teacher_trajectories: int) -> dict[str, Any]:
        compared = self.near_tied_tenpai_choices
        changed = self.better_value_alternatives
        return {
            "status": "diagnostic_only_not_authorized_for_action_selection",
            "hypothesis": "near_tied_direct_tenpai_discards_may_differ_in_public_live_weighted_visible_score_lower_bound",
            "profile": profile,
            "score_margin": self.score_margin,
            "maximum_teacher_trajectories": maximum_teacher_trajectories,
            "trajectories": self.trajectories,
            "decisions": self.decisions,
            "discard_decisions": self.discard_decisions,
            "ordinary_discard_decisions": self.ordinary_discard_decisions,
            "teacher_tenpai_decisions": self.teacher_tenpai_decisions,
            "multiple_tenpai_choices": self.multiple_tenpai_choices,
            "near_tied_tenpai_choices": compared,
            "better_value_alternatives": changed,
            "better_value_alternative_rate_among_near_tied": (
                changed / compared if compared else None
            ),
            "better_value_alternative_rate_per_ordinary_discard": (
                changed / self.ordinary_discard_decisions
                if self.ordinary_discard_decisions
                else None
            ),
            "mean_weighted_value_improvement_when_changed": (
                self.improvement_sum / changed if changed else None
            ),
            "mean_teacher_live_copies_when_changed": (
                self.teacher_live_copy_sum / changed if changed else None
            ),
            "mean_alternative_live_copies_when_changed": (
                self.alternative_live_copy_sum / changed if changed else None
            ),
            "mean_teacher_visible_score_when_changed": (
                self.teacher_mean_score_sum / changed if changed else None
            ),
            "mean_alternative_visible_score_when_changed": (
                self.alternative_mean_score_sum / changed if changed else None
            ),
            "top_two_teacher_margin_histogram": dict(
                sorted(self.top_two_margin_histogram.items())
            ),
            "weighted_value_improvement_histogram": dict(
                sorted(self.improvement_histogram.items())
            ),
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "known_limitations": [
                "ordinary_non_tour_discard_states_only",
                "immediate_self_draw_value_proxy_not_full_game_Q",
                "flower_family_water_omitted_because_v4_only_stores_flower_count",
                "public_live_copies_do_not_identify_wall_vs_opponent_hands",
            ],
            "privacy": "aggregate_only_no_hand_history_wall_seed_or_candidate_trace_exported",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--profile", choices=("classic",), default="classic")
    parser.add_argument("--score-margin", type=float, default=2.0)
    parser.add_argument("--maximum-teacher-trajectories", type=int, default=1_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.score_margin < 0 or args.maximum_teacher_trajectories <= 0:
        raise ValueError("score-margin 不能为负，maximum-teacher-trajectories 必须为正")
    audit = TenpaiValueAudit(score_margin=args.score_margin)
    stop = False
    for path in args.input:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                trajectory = TrainingTrajectory.from_payload(json.loads(line))
                if trajectory.profile != args.profile:
                    continue
                if trajectory.source_metadata.get("collector") != "teacher_self_play":
                    continue
                audit.trajectories += 1
                for decision in trajectory.decisions:
                    audit.inspect(decision)
                if audit.trajectories >= args.maximum_teacher_trajectories:
                    stop = True
                    break
        if stop:
            break
    payload = audit.payload(
        profile=args.profile,
        maximum_teacher_trajectories=args.maximum_teacher_trajectories,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit public-fixed exact-one-draw overrides with 32 shared worlds.

The old v1 candidate had positive but inconclusive real-game point estimates.
This v2 label pilot first fixes its public-information boundary for opponent
concealed kongs, then evaluates every changed discard against frozen Teacher
on shared belief worlds. It exports aggregate label-readiness statistics only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_confidence_override_belief_advantage import paired_delta_summary
from xiamen_mahjong.agents import (
    ExactOneDrawTenpaiTieBreakTeacherAgent,
    GameAction,
    HeuristicTeacherAgent,
)
from xiamen_mahjong.game import XiamenMahjongGame
from xiamen_mahjong.rules import XiamenRules
from xiamen_mahjong.training import (
    _continue_counterfactual_rollout,
    _resample_private_world_for_actor,
    _run_candidate_base_hand,
    _turn_actions,
)


PHYSICAL_WALLS = 50
FIRST_SEED = 202622000
WORLDS_PER_STATE = 32
MAXIMUM_STATES = 40
CONFIDENCE_Z = 1.96
SCORE_MARGIN = 2.0
MINIMUM_LIVE_ADVANTAGE = 1


def candidate_override(
    candidate: Any,
    teacher: Any,
    game: XiamenMahjongGame,
    actor_seat: int,
    legal: tuple[GameAction, ...],
) -> tuple[GameAction, GameAction] | None:
    teacher_action = teacher.choose_turn_action(game, actor_seat)
    candidate_action = candidate.choose_turn_action(game, actor_seat)
    if teacher_action not in legal or candidate_action not in legal:
        raise RuntimeError("候选或 Teacher 产生了规则引擎非法动作")
    if (
        teacher_action == candidate_action
        or teacher_action.kind != "discard"
        or candidate_action.kind != "discard"
        or game.tour_state is not None
        or game.gold_discard_lock_seat == actor_seat
    ):
        return None
    return teacher_action, candidate_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-walls", type=int, default=PHYSICAL_WALLS)
    parser.add_argument("--seed", type=int, default=FIRST_SEED)
    parser.add_argument("--worlds-per-state", type=int, default=WORLDS_PER_STATE)
    parser.add_argument("--maximum-states", type=int, default=MAXIMUM_STATES)
    parser.add_argument("--confidence-z", type=float, default=CONFIDENCE_Z)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.physical_walls != PHYSICAL_WALLS
        or args.seed != FIRST_SEED
        or args.worlds_per_state != WORLDS_PER_STATE
        or args.maximum_states != MAXIMUM_STATES
        or args.confidence_z != CONFIDENCE_Z
    ):
        raise ValueError("只接受 v2 预注册的墙数、seed、world 数和置信参数")

    rules = XiamenRules.classic()
    teacher = HeuristicTeacherAgent()
    candidate = ExactOneDrawTenpaiTieBreakTeacherAgent(
        score_margin=SCORE_MARGIN,
        minimum_live_advantage=MINIMUM_LIVE_ADVANTAGE,
    )
    rng = random.Random(args.seed ^ 0xE1A0D2)
    state_summaries: list[dict[str, float | int | None]] = []
    scanned_candidate_decisions = 0
    requested_worlds = 0
    usable_worlds = 0
    skipped_worlds = 0
    action_kind_counts: Counter[str] = Counter()
    stop = False
    for hand_offset in range(args.physical_walls):
        hand_seed = args.seed + hand_offset
        for actor_seat in range(rules.player_count):
            opponents = {
                seat: ("heuristic_teacher", teacher)
                for seat in range(rules.player_count)
                if seat != actor_seat
            }
            game = XiamenMahjongGame(
                seed=hand_seed,
                rules=rules,
                auto_advance=False,
                human_seat=-1,
            )
            snapshots = _run_candidate_base_hand(
                game,
                candidate_seat=actor_seat,
                candidate_policy=teacher,
                opponents=opponents,
            )
            for snapshot in snapshots:
                if snapshot.game.phase != "discard":
                    continue
                legal = snapshot.legal_actions
                teacher_action = teacher.choose_turn_action(snapshot.game, actor_seat)
                if teacher_action.kind == "discard":
                    scanned_candidate_decisions += 1
                override = candidate_override(
                    candidate, teacher, snapshot.game, actor_seat, legal
                )
                if override is None:
                    continue
                teacher_action, alternative = override
                action_kind_counts[alternative.kind] += 1
                deltas: list[float] = []
                for _world_index in range(args.worlds_per_state):
                    requested_worlds += 1
                    world = _resample_private_world_for_actor(
                        snapshot.game,
                        actor_seat=actor_seat,
                        rng=rng,
                    )
                    if world is None:
                        skipped_worlds += 1
                        continue
                    world_legal = tuple(_turn_actions(world, actor_seat))
                    if world_legal != legal:
                        skipped_worlds += 1
                        continue
                    usable_worlds += 1
                    alternative_score = _continue_counterfactual_rollout(
                        copy.deepcopy(world),
                        candidate_seat=actor_seat,
                        candidate_policy=teacher,
                        opponents=opponents,
                        forced_action=alternative,
                    )
                    teacher_score = _continue_counterfactual_rollout(
                        copy.deepcopy(world),
                        candidate_seat=actor_seat,
                        candidate_policy=teacher,
                        opponents=opponents,
                        forced_action=teacher_action,
                    )
                    deltas.append(float(alternative_score - teacher_score))
                state_summaries.append(
                    paired_delta_summary(deltas, confidence_z=args.confidence_z)
                )
                if len(state_summaries) >= args.maximum_states:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    usable_states = [
        summary for summary in state_summaries if int(summary["worlds"] or 0) >= 2
    ]
    state_means = [float(summary["mean"]) for summary in usable_states]
    aggregate_raw = paired_delta_summary(
        state_means, confidence_z=args.confidence_z
    )
    aggregate = {
        "states": aggregate_raw["worlds"],
        "mean": aggregate_raw["mean"],
        "stderr": aggregate_raw["stderr"],
        "low": aggregate_raw["low"],
        "high": aggregate_raw["high"],
    }
    positive_lcb = sum(float(summary["low"]) > 0.0 for summary in usable_states)
    negative_ucb = sum(float(summary["high"]) < 0.0 for summary in usable_states)
    uncertain = len(usable_states) - positive_lcb - negative_ucb
    usable_rate = usable_worlds / requested_worlds if requested_worlds else 0.0
    label_readiness = bool(
        len(usable_states) >= 15
        and usable_rate >= 0.90
        and positive_lcb >= 5
        and negative_ucb >= 5
    )
    payload: dict[str, Any] = {
        "status": (
            "paired_label_pilot_ready_for_safe_state_export"
            if label_readiness
            else "paired_label_pilot_not_ready"
        ),
        "candidate": "exact_one_draw_tenpai_public_fix_v2",
        "protocol": {
            "profile": "classic",
            "physical_walls": args.physical_walls,
            "first_seed": args.seed,
            "seat_rotations": rules.player_count,
            "worlds_per_state": args.worlds_per_state,
            "maximum_states": args.maximum_states,
            "confidence_z": args.confidence_z,
            "score_margin": SCORE_MARGIN,
            "minimum_live_advantage": MINIMUM_LIVE_ADVANTAGE,
            "continuation": "heuristic_teacher",
            "shared_worlds_between_actions": True,
            "opponent_concealed_kong_face_redacted": True,
        },
        "coverage": {
            "scanned_candidate_discard_decisions": scanned_candidate_decisions,
            "override_states": len(state_summaries),
            "usable_states": len(usable_states),
            "requested_worlds": requested_worlds,
            "usable_worlds": usable_worlds,
            "skipped_worlds": skipped_worlds,
            "usable_world_rate": usable_rate,
            "alternative_action_kind_counts": dict(sorted(action_kind_counts.items())),
        },
        "paired_state_classification": {
            "positive_95pct_lcb": positive_lcb,
            "negative_95pct_ucb": negative_ucb,
            "uncertain": uncertain,
        },
        "aggregate_across_state_means": aggregate,
        "label_readiness_gate": {
            "minimum_usable_states": 15,
            "minimum_usable_world_rate": 0.90,
            "minimum_positive_lcb_states": 5,
            "minimum_negative_ucb_states": 5,
            "passes": label_readiness,
        },
        "privacy": "aggregate_only_no_state_hand_history_action_trace_private_world_wall_opponent_hand_or_rng_exported",
        "interpretation": "offline_information_set_label_quality_only_not_policy_strength",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

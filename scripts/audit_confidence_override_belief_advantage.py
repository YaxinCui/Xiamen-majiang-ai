#!/usr/bin/env python3
"""Audit failed confidence overrides with shared information-set worlds.

This is an offline label-quality pilot, not an online search policy.  It finds
the already-fixed run3 high-confidence ordinary-discard overrides in fresh
Teacher games.  For each public decision, it redraws private worlds from the
actor-visible information set and evaluates the model alternative and Teacher
action in the same worlds with frozen-Teacher continuation.  Only aggregate
paired statistics are exported.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.select_confidence_gated_teacher import (
    CHECKPOINT,
    CHECKPOINT_SHA256,
    MINIMUM_POLICY_ADVANTAGE,
    _sha256,
)
from xiamen_mahjong.agents import GameAction, HeuristicTeacherAgent
from xiamen_mahjong.game import XiamenMahjongGame
from xiamen_mahjong.rules import XiamenRules
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import (
    _continue_counterfactual_rollout,
    _resample_private_world_for_actor,
    _run_candidate_base_hand,
    _turn_actions,
)


PHYSICAL_WALLS = 50
FIRST_SEED = 202620000
WORLDS_PER_STATE = 32
MAXIMUM_STATES = 40
CONFIDENCE_Z = 1.96


def paired_delta_summary(
    deltas: Sequence[float], *, confidence_z: float = CONFIDENCE_Z
) -> dict[str, float | int | None]:
    if confidence_z < 0:
        raise ValueError("confidence_z 不能为负数")
    if not deltas:
        return {
            "worlds": 0,
            "mean": None,
            "stderr": None,
            "low": None,
            "high": None,
        }
    values = [float(value) for value in deltas]
    mean = sum(values) / len(values)
    variance = (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1
        else 0.0
    )
    stderr = math.sqrt(variance / len(values))
    return {
        "worlds": len(values),
        "mean": mean,
        "stderr": stderr,
        "low": mean - confidence_z * stderr,
        "high": mean + confidence_z * stderr,
    }


def _fixed_override(
    policy: TorchPolicyValueAgent,
    game: XiamenMahjongGame,
    actor_seat: int,
    legal: tuple[GameAction, ...],
    teacher_action: GameAction,
) -> tuple[GameAction, float] | None:
    if (
        teacher_action.kind != "discard"
        or game.tour_state is not None
        or game.gold_discard_lock_seat == actor_seat
        or len(legal) < 2
    ):
        return None
    decision = policy._decision_for_game(game, actor_seat, legal)
    logits, _value = policy.policy_value(decision)
    teacher_index = legal.index(teacher_action)
    alternative_index = max(
        (index for index in range(len(legal)) if index != teacher_index),
        key=lambda index: (float(logits[index]), -index),
    )
    gap = float(logits[alternative_index]) - float(logits[teacher_index])
    if gap <= MINIMUM_POLICY_ADVANTAGE:
        return None
    return legal[alternative_index], gap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-walls", type=int, default=PHYSICAL_WALLS)
    parser.add_argument("--seed", type=int, default=FIRST_SEED)
    parser.add_argument("--worlds-per-state", type=int, default=WORLDS_PER_STATE)
    parser.add_argument("--maximum-states", type=int, default=MAXIMUM_STATES)
    parser.add_argument("--confidence-z", type=float, default=CONFIDENCE_Z)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
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
        raise ValueError("只接受 v1 预注册的墙数、seed、world 数和置信参数")
    if _sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("预注册 run3 checkpoint 已发生变化")

    rules = XiamenRules.classic()
    teacher = HeuristicTeacherAgent()
    policy = TorchPolicyValueAgent.load(CHECKPOINT, device=args.device)
    rng = random.Random(args.seed ^ 0xB3113F)
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
                override = _fixed_override(
                    policy, snapshot.game, actor_seat, legal, teacher_action
                )
                if override is None:
                    continue
                alternative, _policy_gap = override
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
        "protocol": {
            "profile": "classic",
            "physical_walls": args.physical_walls,
            "first_seed": args.seed,
            "seat_rotations": rules.player_count,
            "worlds_per_state": args.worlds_per_state,
            "maximum_states": args.maximum_states,
            "confidence_z": args.confidence_z,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "minimum_policy_advantage": MINIMUM_POLICY_ADVANTAGE,
            "continuation": "heuristic_teacher",
            "shared_worlds_between_actions": True,
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
        "device": args.device,
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

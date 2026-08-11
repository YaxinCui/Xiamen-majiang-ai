"""Actor-visible expert correction review data for discard-only residuals."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence
from uuid import uuid4

from .agents import HeuristicTeacherAgent, TwoDrawTenpaiReachTeacherAgent
from .hand import hand_quality, wait_tiles
from .human_data import (
    _private_key_paths,
    is_eligible_human_teacher_discard_correction,
)
from .rules import XiamenRules
from .tiles import WHITE_DRAGON, is_honor
from .training import DATASET_VERSION, TeacherDecision, read_trajectory_jsonl


REVIEW_QUEUE_VERSION = "xiamen-human-correction-review-queue-v1"
REVIEW_LABEL_VERSION = "xiamen-human-correction-review-label-v1"
REVIEW_GROUP_VERSION = "xiamen-human-correction-review-group-v1"
REVIEW_SPLIT_VERSION = "xiamen-human-correction-review-split-v1"
REVIEW_PRIORITY_VERSION = "xiamen-human-correction-review-priority-v1"
REVIEW_SLOW_EXPERT_VERSION = "two-draw-tenpai-reach-v1-32-scenarios"
_REVIEW_GROUP_SALT = "xiamen-human-correction-review-group-v1"
_REVIEW_ITEM_SALT = "xiamen-human-correction-review-item-v1"
_REVIEW_SPLIT_SALT = "xiamen-human-correction-review-split-v1"
_CONFIDENCE_VALUES = frozenset({"confirmed", "uncertain"})


def _opaque_hash(salt: str, value: str) -> str:
    return hashlib.blake2b(
        f"{salt}|{value}".encode("utf-8"), digest_size=16
    ).hexdigest()


def _queue_item_digest(item: dict[str, Any]) -> str:
    encoded = json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


class _ActorVisibleReviewGame:
    """Minimal game facade reconstructed only from one safe review state."""

    def __init__(self, item: dict[str, Any]) -> None:
        state = item["state"]
        self.rules = XiamenRules.classic()
        self.gold_tile = int(state["gold_tile"])
        self.gold_indicator = int(state["gold_indicator"])
        self.gold_proxy_tile = (
            WHITE_DRAGON if self.gold_tile != WHITE_DRAGON else None
        )
        self.wildcard_tiles = (self.gold_tile,)
        self.tour_state = None
        self.gold_discard_lock_seat = 0 if state.get("gold_locked") else None
        self.players = []
        for public in state["public_players"]:
            seat = int(public["relative_seat"])
            self.players.append(
                SimpleNamespace(
                    seat=seat,
                    hand=list(state["hand"]) if seat == 0 else [],
                    discards=list(public.get("discards", [])),
                    melds=[dict(meld) for meld in public.get("melds", [])],
                )
            )
        self.players.sort(key=lambda player: player.seat)

    def _forced_follow_tiles(self, player_id: int) -> list[int]:
        if not self.rules.enable_forced_honor_follow:
            return []
        appeared = {
            tile
            for player in self.players
            for tile in player.discards
            if is_honor(tile) and tile not in {self.gold_tile, self.gold_proxy_tile}
        }
        hand = self.players[player_id].hand
        return sorted(
            tile
            for tile in set(hand)
            if is_honor(tile) and hand.count(tile) == 1 and tile in appeared
        )


def build_review_slow_expert_priority(
    queue: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a queue-bound active-review order without mutating queue items."""

    frozen_agent = HeuristicTeacherAgent()
    agent = TwoDrawTenpaiReachTeacherAgent(
        score_margin=2.0,
        minimum_probability_advantage=0.05,
    )
    provisional: list[tuple[tuple[object, ...], dict[str, Any]]] = []
    disagreements = 0
    for original_index, item in enumerate(queue):
        issues = validate_review_queue_item(item)
        if issues:
            raise ValueError("不能排序无效 review item：" + ",".join(issues))
        game = _ActorVisibleReviewGame(item)
        frozen = frozen_agent.explain_discard(game, 0)
        reference_index = int(item["reference_teacher_index"])
        reference_tile = int(item["legal_actions"][reference_index]["tile"])
        if not frozen or int(frozen[0]["tile"]) != reference_tile:
            raise ValueError("review item 的 Teacher 参考无法从可见状态复现")
        ranked = agent.explain_discard(game, 0)
        if not ranked:
            raise ValueError("SlowExpert 没有产生弃牌排序")
        selected_tile = int(ranked[0]["tile"])
        try:
            selected_index = next(
                index
                for index, action in enumerate(item["legal_actions"])
                if action.get("kind") == "discard"
                and int(action.get("tile")) == selected_tile
            )
        except StopIteration as error:
            raise ValueError("SlowExpert 产生了 queue 外动作") from error
        differs = selected_index != reference_index
        disagreements += differs
        probability_advantage = 0.0
        if ranked[0].get("selected_by_two_draw_tenpai_reach"):
            reference_row = next(
                row for row in ranked if int(row["tile"]) == reference_tile
            )
            probability_advantage = float(
                ranked[0]["two_draw_tenpai_probability"]
            ) - float(reference_row["two_draw_tenpai_probability"])
        record = {
            "version": REVIEW_PRIORITY_VERSION,
            "item_id": item["item_id"],
            "queue_item_digest": _queue_item_digest(item),
            "slow_expert_version": REVIEW_SLOW_EXPERT_VERSION,
            "slow_expert_index": selected_index,
            "slow_expert_differs_from_teacher": differs,
            "two_draw_probability_advantage": round(probability_advantage, 8),
            "original_queue_index": original_index,
        }
        priority_key = (
            not differs,
            -probability_advantage,
            original_index,
        )
        provisional.append((priority_key, record))
    records = []
    for priority_rank, (_key, record) in enumerate(
        sorted(provisional, key=lambda row: row[0])
    ):
        records.append({**record, "priority_rank": priority_rank})
    return records, {
        "status": "review_priority_ready",
        "version": REVIEW_PRIORITY_VERSION,
        "slow_expert_version": REVIEW_SLOW_EXPERT_VERSION,
        "queue_items": len(queue),
        "slow_expert_teacher_disagreements": disagreements,
        "priority": "slow_expert_disagreement_then_probability_advantage",
        "blindness": "priority_and_slow_expert_hidden_until_human_label",
        "queue_mutated": False,
    }


def validate_review_priority_against_queue(
    records: Sequence[dict[str, Any]], queue: Sequence[dict[str, Any]]
) -> list[str]:
    issues: list[str] = []
    by_id = {str(item["item_id"]): item for item in queue}
    seen: set[str] = set()
    ranks: set[int] = set()
    for record in records:
        if (
            not isinstance(record, dict)
            or record.get("version") != REVIEW_PRIORITY_VERSION
        ):
            issues.append("invalid_priority_version")
            continue
        item_id = record.get("item_id")
        if not isinstance(item_id, str) or item_id not in by_id:
            issues.append("priority_item_not_in_queue")
            continue
        if item_id in seen:
            issues.append("duplicate_priority_item")
        seen.add(item_id)
        item = by_id[item_id]
        if record.get("queue_item_digest") != _queue_item_digest(item):
            issues.append("priority_queue_digest_mismatch")
        if record.get("slow_expert_version") != REVIEW_SLOW_EXPERT_VERSION:
            issues.append("invalid_slow_expert_version")
        index = record.get("slow_expert_index")
        if not (
            isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < len(item["legal_actions"])
            and item["legal_actions"][index].get("kind") == "discard"
        ):
            issues.append("invalid_slow_expert_index")
        differs = record.get("slow_expert_differs_from_teacher")
        if not isinstance(differs, bool) or (
            isinstance(index, int)
            and not isinstance(index, bool)
            and differs != (index != int(item["reference_teacher_index"]))
        ):
            issues.append("invalid_slow_expert_teacher_disagreement")
        advantage = record.get("two_draw_probability_advantage")
        if not (
            isinstance(advantage, (int, float))
            and not isinstance(advantage, bool)
            and math.isfinite(float(advantage))
            and 0 <= float(advantage) <= 1
        ):
            issues.append("invalid_two_draw_probability_advantage")
        original_index = record.get("original_queue_index")
        if not (
            isinstance(original_index, int)
            and not isinstance(original_index, bool)
            and 0 <= original_index < len(queue)
            and queue[original_index]["item_id"] == item_id
        ):
            issues.append("invalid_original_queue_index")
        rank = record.get("priority_rank")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            issues.append("invalid_priority_rank")
        elif rank in ranks:
            issues.append("duplicate_priority_rank")
        else:
            ranks.add(rank)
    if seen != set(by_id):
        issues.append("priority_queue_coverage_mismatch")
    if ranks != set(range(len(queue))):
        issues.append("priority_rank_coverage_mismatch")
    return sorted(set(issues))


def write_review_priority(
    path: str | Path, records: Sequence[dict[str, Any]]
) -> None:
    destination = Path(path)
    if destination.exists():
        raise ValueError("review priority 输出已存在，拒绝覆盖")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_review_priority(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"review priority 第 {line_number} 行 JSON 无效"
                ) from error
            records.append(record)
    return records


def _action_payload(action: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": action.kind}
    if action.tile is not None:
        payload["tile"] = action.tile
    if action.tiles:
        payload["tiles"] = list(action.tiles)
    return payload


def _teacher_discard_scores(decision: TeacherDecision) -> dict[int, float]:
    """Reproduce the frozen Teacher score from a safe decision snapshot."""

    state = decision.state
    hand = list(state.get("hand", []))
    gold_tile = state.get("gold_tile")
    if not all(isinstance(tile, int) and not isinstance(tile, bool) for tile in hand):
        return {}
    if isinstance(gold_tile, bool) or not isinstance(gold_tile, int):
        return {}
    public_players = state.get("public_players")
    if not isinstance(public_players, list):
        return {}
    actor = next(
        (
            player
            for player in public_players
            if isinstance(player, dict) and player.get("relative_seat") == 0
        ),
        None,
    )
    if not isinstance(actor, dict) or not isinstance(actor.get("melds"), list):
        return {}
    meld_count = len(actor["melds"])
    proxy_tile = WHITE_DRAGON if gold_tile != WHITE_DRAGON else None
    scores: dict[int, float] = {}
    for index, action in enumerate(decision.legal_actions):
        if action.kind != "discard" or action.tile not in hand:
            continue
        candidate = list(hand)
        candidate.remove(action.tile)
        waits = wait_tiles(
            candidate,
            gold_tile,
            meld_count=meld_count,
            melds_required=5,
            allow_seven_pairs=False,
            wildcard_tiles=(gold_tile,),
            proxy_tile=proxy_tile,
            proxy_as=gold_tile,
        )
        score = hand_quality(
            candidate,
            gold_tile,
            meld_count=meld_count,
            melds_required=5,
            wildcard_tiles=(gold_tile,),
            proxy_tile=proxy_tile,
            proxy_as=gold_tile,
        )
        score += len(waits) * 18.0
        if action.tile == gold_tile:
            score -= 7.0
        scores[index] = round(score, 3)
    return scores


def _queue_source_eligible(decision: TeacherDecision) -> tuple[bool, float | None]:
    state = decision.state
    if (
        state.get("phase") != "discard"
        or decision.chosen_action.kind != "discard"
        or state.get("tour") is not None
        or bool(state.get("gold_locked", False))
    ):
        return False, None
    scores = _teacher_discard_scores(decision)
    if len(scores) < 2 or decision.chosen_index not in scores:
        return False, None
    ranked = sorted(
        scores,
        key=lambda index: (
            -scores[index],
            int(decision.legal_actions[index].tile),
        ),
    )
    if ranked[0] != decision.chosen_index:
        return False, None
    return True, scores[ranked[0]] - scores[ranked[1]]


def build_review_queue(
    paths: Iterable[str | Path],
    *,
    maximum_items: int = 600,
    maximum_items_per_group: int = 2,
    maximum_teacher_score_margin: float = 2.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select low-margin Teacher discard states without exporting replay data."""

    if maximum_items <= 0 or maximum_items_per_group <= 0:
        raise ValueError("review queue 数量门槛必须为正数")
    if maximum_teacher_score_margin < 0:
        raise ValueError("maximum_teacher_score_margin 不能为负数")
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    scanned_trajectories = 0
    scanned_decisions = 0
    score_reproduction_mismatches = 0
    for path in (Path(value) for value in paths):
        for trajectory in read_trajectory_jsonl(path):
            if (
                trajectory.profile != "classic"
                or trajectory.source_metadata.get("collector") != "teacher_self_play"
            ):
                continue
            scanned_trajectories += 1
            source_group = trajectory.split_group_id or trajectory.trajectory_id
            review_group_id = _opaque_hash(_REVIEW_GROUP_SALT, source_group)
            for ordinal, decision in enumerate(trajectory.decisions):
                scanned_decisions += 1
                eligible, margin = _queue_source_eligible(decision)
                if not eligible:
                    # Count only a genuine score-reproduction mismatch: an
                    # otherwise ordinary Teacher discard with >=2 discards.
                    reproduced_scores = _teacher_discard_scores(decision)
                    if (
                        decision.state.get("phase") == "discard"
                        and decision.chosen_action.kind == "discard"
                        and sum(
                            action.kind == "discard"
                            for action in decision.legal_actions
                        )
                        >= 2
                        and reproduced_scores
                        and decision.chosen_index
                        != min(
                            reproduced_scores,
                            key=lambda index: (
                                -reproduced_scores[index],
                                int(decision.legal_actions[index].tile),
                            ),
                        )
                    ):
                        score_reproduction_mismatches += 1
                    continue
                assert margin is not None
                if margin > maximum_teacher_score_margin:
                    continue
                item_id = _opaque_hash(
                    _REVIEW_ITEM_SALT,
                    f"{trajectory.trajectory_id}|{ordinal}",
                )
                review_state = dict(decision.state)
                # The source trajectory may own an earlier complete public
                # prefix outside the decision payload.  A standalone review
                # item intentionally exports only its bounded actor-visible
                # history, so do not claim that the external prefix remains.
                review_state["public_history_complete"] = False
                review_state["public_history_encoding"] = (
                    "bounded_actor_visible_review_v1"
                )
                payload = {
                    "version": REVIEW_QUEUE_VERSION,
                    "item_id": item_id,
                    "review_group_id": review_group_id,
                    "review_group_version": REVIEW_GROUP_VERSION,
                    "profile": trajectory.profile,
                    "rules_version": trajectory.rules_version,
                    "state": review_state,
                    "legal_actions": [
                        _action_payload(action) for action in decision.legal_actions
                    ],
                    "reference_teacher_index": decision.chosen_index,
                    "teacher_score_margin": margin,
                    "source_metadata": {
                        "collector": "teacher_self_play_actor_visible_review_queue",
                        "source_scope": "ordinary_low_margin_discard",
                        "opponent_hand_reveal": "unavailable_by_construction",
                        "training_default": "excluded_until_human_review_audit",
                    },
                }
                order = _opaque_hash(
                    "xiamen-human-correction-review-order-v1", item_id
                )
                candidates.append((margin, order, payload))
    group_counts: Counter[str] = Counter()
    queue: list[dict[str, Any]] = []
    for _margin, _order, payload in sorted(candidates, key=lambda row: (row[0], row[1])):
        group_id = str(payload["review_group_id"])
        if group_counts[group_id] >= maximum_items_per_group:
            continue
        group_counts[group_id] += 1
        queue.append(payload)
        if len(queue) >= maximum_items:
            break
    return queue, {
        "status": "review_queue_ready" if queue else "review_queue_empty",
        "version": REVIEW_QUEUE_VERSION,
        "scanned_teacher_trajectories": scanned_trajectories,
        "scanned_teacher_decisions": scanned_decisions,
        "low_margin_candidates": len(candidates),
        "queue_items": len(queue),
        "queue_groups": len(group_counts),
        "maximum_items": maximum_items,
        "maximum_items_per_group": maximum_items_per_group,
        "maximum_teacher_score_margin": maximum_teacher_score_margin,
        "score_reproduction_mismatches": score_reproduction_mismatches,
        "privacy": (
            "actor_visible_only_no_seed_wall_opponent_hand_outcome_source_path_"
            "or_original_group_id_exported"
        ),
    }


def write_review_queue(path: str | Path, queue: Sequence[dict[str, Any]]) -> None:
    destination = Path(path)
    if destination.exists():
        raise ValueError("review queue 输出已存在，拒绝覆盖")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for item in queue:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def read_review_queue(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"review queue 第 {line_number} 行 JSON 无效") from error
            issues = validate_review_queue_item(item)
            if issues:
                raise ValueError(
                    f"review queue 第 {line_number} 行无效：" + ",".join(issues)
                )
            items.append(item)
    if len({item["item_id"] for item in items}) != len(items):
        raise ValueError("review queue 包含重复 item_id")
    return items


def validate_review_queue_item(item: Any) -> list[str]:
    issues: list[str] = []
    if not isinstance(item, dict) or item.get("version") != REVIEW_QUEUE_VERSION:
        return ["invalid_queue_version"]
    for key in ("item_id", "review_group_id"):
        value = item.get(key)
        if not (
            isinstance(value, str)
            and len(value) == 32
            and all(character in "0123456789abcdef" for character in value)
        ):
            issues.append(f"invalid_{key}")
    if item.get("review_group_version") != REVIEW_GROUP_VERSION:
        issues.append("invalid_review_group_version")
    if item.get("profile") != "classic":
        issues.append("non_classic_profile")
    state = item.get("state")
    if not isinstance(state, dict) or state.get("phase") != "discard":
        issues.append("invalid_actor_visible_state")
    private_paths = _private_key_paths(item)
    if private_paths:
        issues.append("private_key:" + ",".join(sorted(private_paths)))
    actions = item.get("legal_actions")
    reference = item.get("reference_teacher_index")
    if not (
        isinstance(actions, list)
        and len(actions) >= 2
        and all(isinstance(action, dict) for action in actions)
        and isinstance(reference, int)
        and not isinstance(reference, bool)
        and 0 <= reference < len(actions)
        and actions[reference].get("kind") == "discard"
        and sum(action.get("kind") == "discard" for action in actions) >= 2
    ):
        issues.append("invalid_discard_actions_or_reference")
    margin = item.get("teacher_score_margin")
    if not (
        isinstance(margin, (int, float))
        and not isinstance(margin, bool)
        and math.isfinite(float(margin))
        and float(margin) >= 0
    ):
        issues.append("invalid_teacher_score_margin")
    metadata = item.get("source_metadata")
    if not isinstance(metadata, dict) or metadata.get(
        "opponent_hand_reveal"
    ) != "unavailable_by_construction":
        issues.append("opponent_hand_not_unavailable_by_construction")
    return issues


def public_review_item(item: dict[str, Any], *, completed: int, total: int) -> dict[str, Any]:
    """Hide the Teacher reference while a human makes an independent choice."""

    return {
        "status": "reviewing",
        "item_id": item["item_id"],
        "profile": item["profile"],
        "rules_version": item["rules_version"],
        "state": item["state"],
        "legal_actions": item["legal_actions"],
        "progress": {"completed": completed, "total": total},
        "privacy": "actor_visible_only_teacher_choice_hidden_until_label",
    }


def make_review_label(
    item: dict[str, Any],
    *,
    chosen_index: int,
    confidence: str,
) -> dict[str, Any]:
    issues = validate_review_queue_item(item)
    if issues:
        raise ValueError("不能标注无效 review item：" + ",".join(issues))
    actions = item["legal_actions"]
    if (
        isinstance(chosen_index, bool)
        or not isinstance(chosen_index, int)
        or not 0 <= chosen_index < len(actions)
        or actions[chosen_index].get("kind") != "discard"
    ):
        raise ValueError("review chosen_index 必须对应一张合法弃牌")
    if confidence not in _CONFIDENCE_VALUES:
        raise ValueError("review confidence 必须为 confirmed 或 uncertain")
    decision = {
        "version": DATASET_VERSION,
        "profile": item["profile"],
        "seat": 0,
        "state": item["state"],
        "legal_actions": actions,
        "chosen_index": chosen_index,
        "reference_teacher_index": item["reference_teacher_index"],
    }
    return {
        "version": REVIEW_LABEL_VERSION,
        "review_id": uuid4().hex,
        "item_id": item["item_id"],
        "review_group_id": item["review_group_id"],
        "review_group_version": item["review_group_version"],
        "queue_item_digest": _queue_item_digest(item),
        "profile": item["profile"],
        "rules_version": item["rules_version"],
        "decision": decision,
        "reviewer_confidence": confidence,
        "source_metadata": {
            "collector": "local_human_review_opt_in",
            "recording_purpose": "training",
            "opponent_hand_reveal": "unavailable_by_construction",
            "teacher_reference_hidden_before_choice": True,
            "training_default": "excluded_until_separate_quality_review",
            "outcome_target": "unavailable_review_behavior_only",
        },
    }


def append_review_label(path: str | Path, label: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")


def read_review_labels(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"review label 第 {line_number} 行 JSON 无效") from error
            records.append(record)
    return records


def read_confirmed_review_decisions(
    path: str | Path,
) -> list[tuple[str, TeacherDecision]]:
    """Load one audited split as behavioral labels, rejecting uncertainty.

    The returned group ID is used only to enforce train/validation isolation;
    it is never an inference feature.
    """

    rows: list[tuple[str, TeacherDecision]] = []
    seen_items: set[str] = set()
    for line_number, record in enumerate(read_review_labels(path), start=1):
        issues, decision = _validate_review_label(record)
        if issues:
            raise ValueError(
                f"review training 第 {line_number} 行无效：" + ",".join(issues)
            )
        if record.get("reviewer_confidence") != "confirmed":
            raise ValueError("review training split 只能包含 confirmed 标签")
        item_id = str(record["item_id"])
        if item_id in seen_items:
            raise ValueError("review training split 包含重复 item_id")
        seen_items.add(item_id)
        assert decision is not None
        rows.append((str(record["review_group_id"]), decision))
    if not rows:
        raise ValueError("review training split 不能为空")
    return rows


def _validate_review_label(record: Any) -> tuple[list[str], TeacherDecision | None]:
    issues: list[str] = []
    if not isinstance(record, dict) or record.get("version") != REVIEW_LABEL_VERSION:
        return ["invalid_label_version"], None
    for key in ("review_id", "item_id", "review_group_id"):
        value = record.get(key)
        if not (
            isinstance(value, str)
            and len(value) == 32
            and all(character in "0123456789abcdef" for character in value)
        ):
            issues.append(f"invalid_{key}")
    if record.get("review_group_version") != REVIEW_GROUP_VERSION:
        issues.append("invalid_review_group_version")
    digest = record.get("queue_item_digest")
    if not (
        isinstance(digest, str)
        and len(digest) == 32
        and all(character in "0123456789abcdef" for character in digest)
    ):
        issues.append("invalid_queue_item_digest")
    if record.get("profile") != "classic":
        issues.append("non_classic_profile")
    if record.get("reviewer_confidence") not in _CONFIDENCE_VALUES:
        issues.append("invalid_reviewer_confidence")
    metadata = record.get("source_metadata")
    if not isinstance(metadata, dict):
        issues.append("missing_source_metadata")
    else:
        if metadata.get("collector") != "local_human_review_opt_in":
            issues.append("invalid_collector")
        if metadata.get("recording_purpose") != "training":
            issues.append("invalid_recording_purpose")
        if metadata.get("opponent_hand_reveal") != "unavailable_by_construction":
            issues.append("opponent_hand_not_unavailable_by_construction")
        if metadata.get("teacher_reference_hidden_before_choice") is not True:
            issues.append("teacher_reference_not_hidden_before_choice")
        if metadata.get("outcome_target") != "unavailable_review_behavior_only":
            issues.append("invalid_outcome_target")
    private_paths = _private_key_paths(record)
    if private_paths:
        issues.append("private_key:" + ",".join(sorted(private_paths)))
    decision = None
    try:
        decision = TeacherDecision.from_payload(dict(record.get("decision", {})))
    except (TypeError, ValueError, KeyError):
        issues.append("invalid_review_decision")
    if decision is not None and not is_eligible_human_teacher_discard_correction(decision):
        issues.append("decision_outside_discard_gate_scope")
    return issues, decision


def audit_review_labels(
    records: Sequence[dict[str, Any]],
    *,
    minimum_confirmed_labels: int = 500,
    minimum_confirmed_disagreements: int = 50,
    minimum_groups: int = 100,
) -> dict[str, Any]:
    if min(
        minimum_confirmed_labels,
        minimum_confirmed_disagreements,
        minimum_groups,
    ) <= 0:
        raise ValueError("review audit 门槛必须为正数")
    issues: Counter[str] = Counter()
    item_ids: Counter[str] = Counter()
    fingerprints: Counter[str] = Counter()
    confirmed = 0
    uncertain = 0
    disagreements = 0
    confirmed_groups: set[str] = set()
    for record in records:
        record_issues, decision = _validate_review_label(record)
        if record_issues:
            issues.update(record_issues)
            continue
        item_id = str(record["item_id"])
        item_ids[item_id] += 1
        fingerprint_payload = dict(record)
        fingerprint_payload.pop("review_id", None)
        fingerprint = hashlib.blake2b(
            json.dumps(
                fingerprint_payload, ensure_ascii=False, sort_keys=True
            ).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
        fingerprints[fingerprint] += 1
        if record["reviewer_confidence"] == "uncertain":
            uncertain += 1
            continue
        confirmed += 1
        confirmed_groups.add(str(record["review_group_id"]))
        assert decision is not None
        disagreements += decision.chosen_index != decision.reference_teacher_index
    duplicate_items = sum(count - 1 for count in item_ids.values() if count > 1)
    duplicate_labels = sum(count - 1 for count in fingerprints.values() if count > 1)
    gate_reasons: list[str] = []
    if issues:
        gate_reasons.append("invalid_records")
    if duplicate_items or duplicate_labels:
        gate_reasons.append("duplicate_reviews")
    if confirmed < minimum_confirmed_labels:
        gate_reasons.append("insufficient_confirmed_labels")
    if disagreements < minimum_confirmed_disagreements:
        gate_reasons.append("insufficient_confirmed_disagreements")
    if len(confirmed_groups) < minimum_groups:
        gate_reasons.append("insufficient_review_groups")
    return {
        "status": "local_human_correction_review_audit",
        "records": len(records),
        "confirmed_labels": confirmed,
        "uncertain_labels": uncertain,
        "confirmed_teacher_disagreements": disagreements,
        "confirmed_review_groups": len(confirmed_groups),
        "minimum_confirmed_labels": minimum_confirmed_labels,
        "minimum_confirmed_disagreements": minimum_confirmed_disagreements,
        "minimum_groups": minimum_groups,
        "duplicate_items": duplicate_items,
        "duplicate_labels": duplicate_labels,
        "issues": dict(sorted(issues.items())),
        "ready_for_manual_training_review": not gate_reasons,
        "gate_reasons": gate_reasons,
        "privacy": "aggregate_only_no_state_hand_action_face_item_or_group_id_exported",
        "warning": (
            "Review labels are behavioral corrections without terminal outcomes. "
            "Passing this gate does not prove optimal actions or human-level strength."
        ),
    }


def audit_review_labels_against_queue(
    records: Sequence[dict[str, Any]],
    queue: Sequence[dict[str, Any]],
    *,
    minimum_confirmed_labels: int = 500,
    minimum_confirmed_disagreements: int = 50,
    minimum_groups: int = 100,
) -> dict[str, Any]:
    """Add immutable queue binding checks to the aggregate label audit."""

    report = audit_review_labels(
        records,
        minimum_confirmed_labels=minimum_confirmed_labels,
        minimum_confirmed_disagreements=minimum_confirmed_disagreements,
        minimum_groups=minimum_groups,
    )
    by_id = {str(item["item_id"]): item for item in queue}
    mismatch_issues: Counter[str] = Counter()
    for record in records:
        if not isinstance(record, dict):
            mismatch_issues["invalid_record"] += 1
            continue
        item = by_id.get(str(record.get("item_id", "")))
        if item is None:
            mismatch_issues["item_not_in_queue"] += 1
            continue
        mismatch_issues.update(validate_review_label_against_item(record, item))
    gate_reasons = list(report["gate_reasons"])
    if mismatch_issues and "queue_binding_mismatch" not in gate_reasons:
        gate_reasons.append("queue_binding_mismatch")
    return {
        **report,
        "queue_items": len(queue),
        "queue_binding_issues": dict(sorted(mismatch_issues.items())),
        "ready_for_manual_training_review": not gate_reasons,
        "gate_reasons": gate_reasons,
    }


def validate_review_label_against_item(
    record: dict[str, Any], item: dict[str, Any]
) -> list[str]:
    """Verify that one persisted label came from the immutable queue item."""

    issues, decision = _validate_review_label(record)
    if issues:
        return issues
    if record.get("item_id") != item.get("item_id"):
        return ["item_id_queue_mismatch"]
    if record.get("review_group_id") != item.get("review_group_id"):
        issues.append("review_group_queue_mismatch")
    if record.get("profile") != item.get("profile"):
        issues.append("profile_queue_mismatch")
    if record.get("rules_version") != item.get("rules_version"):
        issues.append("rules_version_queue_mismatch")
    if record.get("queue_item_digest") != _queue_item_digest(item):
        issues.append("queue_item_digest_mismatch")
    assert decision is not None
    if decision.state != item.get("state"):
        issues.append("state_queue_mismatch")
    expected_actions = item.get("legal_actions")
    actual_actions = [_action_payload(action) for action in decision.legal_actions]
    if actual_actions != expected_actions:
        issues.append("legal_actions_queue_mismatch")
    if decision.reference_teacher_index != item.get("reference_teacher_index"):
        issues.append("reference_teacher_queue_mismatch")
    return issues


def split_confirmed_review_labels(
    records: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Split by original physical-hand group, never by review item."""

    partitions: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    seen_items: set[str] = set()
    group_partitions: dict[str, str] = {}
    for record in records:
        issues, _decision = _validate_review_label(record)
        if issues or record.get("reviewer_confidence") != "confirmed":
            continue
        item_id = str(record["item_id"])
        if item_id in seen_items:
            raise ValueError("review labels 包含重复 item_id")
        seen_items.add(item_id)
        group_id = str(record["review_group_id"])
        digest = hashlib.blake2b(
            f"{_REVIEW_SPLIT_SALT}|{group_id}".encode("utf-8"), digest_size=8
        ).digest()
        bucket = int.from_bytes(digest, "big") % 10_000
        split = "train" if bucket < 8000 else "validation" if bucket < 9000 else "test"
        existing = group_partitions.setdefault(group_id, split)
        if existing != split:
            raise RuntimeError("review group 被分到多个 split")
        partitions[split].append(record)
    return partitions

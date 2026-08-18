"""Blind pairwise human review for exact Teacher-score tiebreaks."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from .agents import HeuristicTeacherAgent, PublicProgressTieBreakTeacherAgent
from .human_data import _private_key_paths
from .human_review import (
    REVIEW_GROUP_VERSION,
    _ActorVisibleReviewGame,
    _action_payload,
    _opaque_hash,
    _queue_item_digest,
    validate_review_queue_item,
)
from .training import DATASET_VERSION, TeacherDecision


PAIRWISE_QUEUE_VERSION = "xiamen-exact-tie-pairwise-review-queue-v1"
PAIRWISE_LABEL_VERSION = "xiamen-exact-tie-pairwise-review-label-v1"
PAIRWISE_SPLIT_VERSION = "xiamen-exact-tie-pairwise-review-split-v1"
PAIRWISE_CANDIDATE_VERSION = "public-progress-tiebreak-v1-rejected"
_PAIRWISE_ITEM_SALT = "xiamen-exact-tie-pairwise-item-v1"
_PAIRWISE_ORDER_SALT = "xiamen-exact-tie-pairwise-order-v1"
_PAIRWISE_DISPLAY_SALT = "xiamen-exact-tie-pairwise-display-v1"
_PAIRWISE_SPLIT_SALT = "xiamen-exact-tie-pairwise-split-v1"
_CONFIDENCE_VALUES = frozenset({"confirmed", "uncertain"})


def _pairwise_queue_item_digest(item: dict[str, Any]) -> str:
    encoded = json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def build_pairwise_review_queue(
    source_queue: Sequence[dict[str, Any]],
    *,
    maximum_items: int = 240,
    maximum_items_per_group: int = 1,
    candidate: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract exact-score divergences without exporting outcome or identity."""

    if maximum_items <= 0 or maximum_items_per_group <= 0:
        raise ValueError("pairwise queue 数量门槛必须为正数")
    frozen = HeuristicTeacherAgent()
    alternative = candidate or PublicProgressTieBreakTeacherAgent()
    provisional: list[tuple[str, dict[str, Any]]] = []
    source_issues: Counter[str] = Counter()
    exact_tie_items = 0
    divergences = 0
    for item in source_queue:
        issues = validate_review_queue_item(item)
        if issues:
            source_issues.update(issues)
            continue
        if abs(float(item["teacher_score_margin"])) > 1e-9:
            continue
        exact_tie_items += 1
        game = _ActorVisibleReviewGame(item)
        frozen_ranked = frozen.explain_discard(game, 0)
        reference_index = int(item["reference_teacher_index"])
        reference_tile = int(item["legal_actions"][reference_index]["tile"])
        if not frozen_ranked or int(frozen_ranked[0]["tile"]) != reference_tile:
            source_issues["unreproducible_teacher"] += 1
            continue
        candidate_ranked = alternative.explain_discard(game, 0)
        if not candidate_ranked:
            source_issues["candidate_without_discard"] += 1
            continue
        candidate_tile = int(candidate_ranked[0]["tile"])
        candidate_indices = [
            index
            for index, action in enumerate(item["legal_actions"])
            if action.get("kind") == "discard"
            and int(action.get("tile")) == candidate_tile
        ]
        if len(candidate_indices) != 1:
            source_issues["candidate_action_not_unique"] += 1
            continue
        candidate_index = candidate_indices[0]
        if candidate_index == reference_index:
            continue
        divergences += 1
        pair_indices = [reference_index, candidate_index]
        display_hash = _opaque_hash(_PAIRWISE_DISPLAY_SALT, str(item["item_id"]))
        if int(display_hash[-1], 16) % 2:
            pair_indices.reverse()
        pair_item_id = _opaque_hash(_PAIRWISE_ITEM_SALT, str(item["item_id"]))
        pair_item = {
            "version": PAIRWISE_QUEUE_VERSION,
            "item_id": pair_item_id,
            "review_group_id": item["review_group_id"],
            "review_group_version": REVIEW_GROUP_VERSION,
            "profile": item["profile"],
            "rules_version": item["rules_version"],
            "state": item["state"],
            "legal_actions": item["legal_actions"],
            "pair_action_indices": pair_indices,
            "reference_teacher_index": reference_index,
            "candidate_index": candidate_index,
            "source_queue_item_digest": _queue_item_digest(item),
            "source_metadata": {
                "collector": "actor_visible_exact_tie_pairwise_review_queue",
                "source_scope": "exact_teacher_score_tie_divergence",
                "candidate_version": PAIRWISE_CANDIDATE_VERSION,
                "opponent_hand_reveal": "unavailable_by_construction",
                "outcome_target": "unavailable_pairwise_behavior_only",
                "training_default": "excluded_until_pairwise_review_audit",
            },
        }
        order = _opaque_hash(_PAIRWISE_ORDER_SALT, pair_item_id)
        provisional.append((order, pair_item))

    group_counts: Counter[str] = Counter()
    queue: list[dict[str, Any]] = []
    for _order, item in sorted(provisional, key=lambda row: row[0]):
        group_id = str(item["review_group_id"])
        if group_counts[group_id] >= maximum_items_per_group:
            continue
        group_counts[group_id] += 1
        queue.append(item)
        if len(queue) >= maximum_items:
            break
    return queue, {
        "status": "pairwise_review_queue_ready" if queue else "pairwise_review_queue_empty",
        "version": PAIRWISE_QUEUE_VERSION,
        "source_queue_items": len(source_queue),
        "source_exact_top_tie_items": exact_tie_items,
        "candidate_teacher_divergences": divergences,
        "queue_items": len(queue),
        "queue_groups": len(group_counts),
        "maximum_items": maximum_items,
        "maximum_items_per_group": maximum_items_per_group,
        "source_issues": dict(sorted(source_issues.items())),
        "display_order": "deterministic_blinded_balanced_hash",
        "target_semantics": "pairwise_only_not_full_action_classification",
        "privacy": (
            "actor_visible_only_no_seed_wall_opponent_hand_outcome_source_path_"
            "or_original_item_id_exported"
        ),
    }


def validate_pairwise_queue_item(item: Any) -> list[str]:
    issues: list[str] = []
    if not isinstance(item, dict) or item.get("version") != PAIRWISE_QUEUE_VERSION:
        return ["invalid_pairwise_queue_version"]
    for key in ("item_id", "review_group_id", "source_queue_item_digest"):
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
    pair = item.get("pair_action_indices")
    reference = item.get("reference_teacher_index")
    candidate = item.get("candidate_index")
    if not (
        isinstance(actions, list)
        and len(actions) >= 2
        and all(isinstance(action, dict) for action in actions)
        and isinstance(pair, list)
        and len(pair) == 2
        and all(isinstance(index, int) and not isinstance(index, bool) for index in pair)
        and len(set(pair)) == 2
        and all(0 <= index < len(actions) for index in pair)
        and all(actions[index].get("kind") == "discard" for index in pair)
        and isinstance(reference, int)
        and not isinstance(reference, bool)
        and isinstance(candidate, int)
        and not isinstance(candidate, bool)
        and reference != candidate
        and set(pair) == {reference, candidate}
    ):
        issues.append("invalid_pair_actions_or_hidden_identities")
    metadata = item.get("source_metadata")
    if not isinstance(metadata, dict):
        issues.append("missing_source_metadata")
    else:
        if metadata.get("candidate_version") != PAIRWISE_CANDIDATE_VERSION:
            issues.append("invalid_candidate_version")
        if metadata.get("opponent_hand_reveal") != "unavailable_by_construction":
            issues.append("opponent_hand_not_unavailable_by_construction")
        if metadata.get("outcome_target") != "unavailable_pairwise_behavior_only":
            issues.append("invalid_outcome_target")
    return sorted(set(issues))


def write_pairwise_queue(path: str | Path, queue: Sequence[dict[str, Any]]) -> None:
    destination = Path(path)
    if destination.exists():
        raise ValueError("pairwise queue 输出已存在，拒绝覆盖")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for item in queue:
            issues = validate_pairwise_queue_item(item)
            if issues:
                raise ValueError("不能写入无效 pairwise item：" + ",".join(issues))
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def read_pairwise_queue(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"pairwise queue 第 {line_number} 行 JSON 无效") from error
            issues = validate_pairwise_queue_item(item)
            if issues:
                raise ValueError(
                    f"pairwise queue 第 {line_number} 行无效：" + ",".join(issues)
                )
            items.append(item)
    if not items:
        raise ValueError("pairwise queue 不能为空")
    if len({item["item_id"] for item in items}) != len(items):
        raise ValueError("pairwise queue 包含重复 item_id")
    return items


def public_pairwise_item(
    item: dict[str, Any], *, completed: int, total: int
) -> dict[str, Any]:
    """Expose only two anonymous actions before the reviewer chooses."""

    pair = item["pair_action_indices"]
    return {
        "status": "reviewing",
        "item_id": item["item_id"],
        "profile": item["profile"],
        "rules_version": item["rules_version"],
        "state": item["state"],
        "legal_actions": [item["legal_actions"][index] for index in pair],
        "progress": {"completed": completed, "total": total},
        "review_scope": "anonymous_exact_tie_pairwise_discard",
        "privacy": "actor_visible_only_both_action_identities_hidden_until_label",
        "target_semantics": "pairwise_only",
    }


def make_pairwise_label(
    item: dict[str, Any], *, chosen_position: int, confidence: str
) -> dict[str, Any]:
    issues = validate_pairwise_queue_item(item)
    if issues:
        raise ValueError("不能标注无效 pairwise item：" + ",".join(issues))
    if (
        isinstance(chosen_position, bool)
        or not isinstance(chosen_position, int)
        or chosen_position not in {0, 1}
    ):
        raise ValueError("pairwise chosen_position 必须是 0 或 1")
    if confidence not in _CONFIDENCE_VALUES:
        raise ValueError("pairwise confidence 必须为 confirmed 或 uncertain")
    chosen_index = int(item["pair_action_indices"][chosen_position])
    decision = {
        "version": DATASET_VERSION,
        "profile": item["profile"],
        "seat": 0,
        "state": item["state"],
        "legal_actions": item["legal_actions"],
        "chosen_index": chosen_index,
        "reference_teacher_index": item["reference_teacher_index"],
    }
    return {
        "version": PAIRWISE_LABEL_VERSION,
        "review_id": uuid4().hex,
        "item_id": item["item_id"],
        "review_group_id": item["review_group_id"],
        "review_group_version": item["review_group_version"],
        "queue_item_digest": _pairwise_queue_item_digest(item),
        "profile": item["profile"],
        "rules_version": item["rules_version"],
        "pair_action_indices": item["pair_action_indices"],
        "chosen_position": chosen_position,
        "candidate_index": item["candidate_index"],
        "decision": decision,
        "reviewer_confidence": confidence,
        "source_metadata": {
            "collector": "local_human_exact_tie_pairwise_review_opt_in",
            "recording_purpose": "training",
            "opponent_hand_reveal": "unavailable_by_construction",
            "both_action_identities_hidden_before_choice": True,
            "training_default": "excluded_until_pairwise_review_audit",
            "outcome_target": "unavailable_pairwise_behavior_only",
            "target_semantics": "pairwise_only_not_full_action_classification",
        },
    }


def append_pairwise_label(path: str | Path, label: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")


def read_pairwise_labels(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not Path(path).exists():
        return records
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"pairwise label 第 {line_number} 行 JSON 无效") from error
    return records


def _validate_pairwise_label(
    record: Any,
) -> tuple[list[str], TeacherDecision | None]:
    issues: list[str] = []
    if not isinstance(record, dict) or record.get("version") != PAIRWISE_LABEL_VERSION:
        return ["invalid_pairwise_label_version"], None
    for key in ("review_id", "item_id", "review_group_id", "queue_item_digest"):
        value = record.get(key)
        if not (
            isinstance(value, str)
            and len(value) == 32
            and all(character in "0123456789abcdef" for character in value)
        ):
            issues.append(f"invalid_{key}")
    if record.get("review_group_version") != REVIEW_GROUP_VERSION:
        issues.append("invalid_review_group_version")
    if record.get("profile") != "classic":
        issues.append("non_classic_profile")
    if record.get("reviewer_confidence") not in _CONFIDENCE_VALUES:
        issues.append("invalid_reviewer_confidence")
    pair = record.get("pair_action_indices")
    position = record.get("chosen_position")
    candidate_index = record.get("candidate_index")
    if not (
        isinstance(pair, list)
        and len(pair) == 2
        and all(isinstance(index, int) and not isinstance(index, bool) for index in pair)
        and len(set(pair)) == 2
        and isinstance(position, int)
        and not isinstance(position, bool)
        and position in {0, 1}
        and isinstance(candidate_index, int)
        and not isinstance(candidate_index, bool)
        and candidate_index in pair
    ):
        issues.append("invalid_pairwise_choice")
    metadata = record.get("source_metadata")
    if not isinstance(metadata, dict):
        issues.append("missing_source_metadata")
    else:
        if metadata.get("collector") != "local_human_exact_tie_pairwise_review_opt_in":
            issues.append("invalid_collector")
        if metadata.get("recording_purpose") != "training":
            issues.append("invalid_recording_purpose")
        if metadata.get("both_action_identities_hidden_before_choice") is not True:
            issues.append("pair_identities_not_hidden_before_choice")
        if metadata.get("target_semantics") != "pairwise_only_not_full_action_classification":
            issues.append("invalid_target_semantics")
    private_paths = _private_key_paths(record)
    if private_paths:
        issues.append("private_key:" + ",".join(sorted(private_paths)))
    decision = None
    try:
        decision = TeacherDecision.from_payload(dict(record.get("decision", {})))
    except (TypeError, ValueError, KeyError):
        issues.append("invalid_pairwise_decision")
    if decision is not None and isinstance(pair, list) and isinstance(position, int):
        if decision.chosen_index != pair[position]:
            issues.append("chosen_position_decision_mismatch")
        if decision.reference_teacher_index not in pair:
            issues.append("teacher_not_in_pair")
    return sorted(set(issues)), decision


def validate_pairwise_label_against_item(
    record: dict[str, Any], item: dict[str, Any]
) -> list[str]:
    issues, decision = _validate_pairwise_label(record)
    if issues:
        return issues
    if record.get("item_id") != item.get("item_id"):
        return ["item_id_queue_mismatch"]
    if record.get("review_group_id") != item.get("review_group_id"):
        issues.append("review_group_queue_mismatch")
    if record.get("queue_item_digest") != _pairwise_queue_item_digest(item):
        issues.append("queue_item_digest_mismatch")
    if record.get("pair_action_indices") != item.get("pair_action_indices"):
        issues.append("pair_actions_queue_mismatch")
    if record.get("candidate_index") != item.get("candidate_index"):
        issues.append("candidate_queue_mismatch")
    assert decision is not None
    if decision.state != item.get("state"):
        issues.append("state_queue_mismatch")
    if [_action_payload(action) for action in decision.legal_actions] != item.get(
        "legal_actions"
    ):
        issues.append("legal_actions_queue_mismatch")
    if decision.reference_teacher_index != item.get("reference_teacher_index"):
        issues.append("reference_teacher_queue_mismatch")
    return sorted(set(issues))


def audit_pairwise_labels_against_queue(
    records: Sequence[dict[str, Any]],
    queue: Sequence[dict[str, Any]],
    *,
    minimum_confirmed_labels: int = 100,
    minimum_candidate_preferences: int = 20,
    minimum_groups: int = 75,
) -> dict[str, Any]:
    if min(minimum_confirmed_labels, minimum_candidate_preferences, minimum_groups) <= 0:
        raise ValueError("pairwise audit 门槛必须为正数")
    by_id = {str(item["item_id"]): item for item in queue}
    issues: Counter[str] = Counter()
    seen: Counter[str] = Counter()
    confirmed = 0
    uncertain = 0
    teacher_preferences = 0
    candidate_preferences = 0
    confirmed_groups: set[str] = set()
    display_first = 0
    for record in records:
        if not isinstance(record, dict):
            issues["invalid_record"] += 1
            continue
        item_id = str(record.get("item_id", ""))
        item = by_id.get(item_id)
        if item is None:
            issues["item_not_in_queue"] += 1
            continue
        record_issues = validate_pairwise_label_against_item(record, item)
        if record_issues:
            issues.update(record_issues)
            continue
        seen[item_id] += 1
        if record["reviewer_confidence"] == "uncertain":
            uncertain += 1
            continue
        confirmed += 1
        confirmed_groups.add(str(record["review_group_id"]))
        decision = TeacherDecision.from_payload(dict(record["decision"]))
        candidate_preferences += int(decision.chosen_index == int(item["candidate_index"]))
        teacher_preferences += int(
            decision.chosen_index == int(item["reference_teacher_index"])
        )
        display_first += int(record["chosen_position"] == 0)
    duplicate_items = sum(count - 1 for count in seen.values() if count > 1)
    gate_reasons: list[str] = []
    if issues:
        gate_reasons.append("invalid_or_queue_mismatched_records")
    if duplicate_items:
        gate_reasons.append("duplicate_reviews")
    if confirmed < minimum_confirmed_labels:
        gate_reasons.append("insufficient_confirmed_labels")
    if candidate_preferences < minimum_candidate_preferences:
        gate_reasons.append("insufficient_teacher_corrections")
    if len(confirmed_groups) < minimum_groups:
        gate_reasons.append("insufficient_review_groups")
    return {
        "status": "local_human_exact_tie_pairwise_review_audit",
        "queue_items": len(queue),
        "records": len(records),
        "confirmed_labels": confirmed,
        "uncertain_labels": uncertain,
        "confirmed_teacher_preferences": teacher_preferences,
        "confirmed_candidate_preferences": candidate_preferences,
        "confirmed_review_groups": len(confirmed_groups),
        "confirmed_display_first_choices": display_first,
        "minimum_confirmed_labels": minimum_confirmed_labels,
        "minimum_candidate_preferences": minimum_candidate_preferences,
        "minimum_groups": minimum_groups,
        "duplicate_items": duplicate_items,
        "issues": dict(sorted(issues.items())),
        "ready_for_pairwise_split": not gate_reasons,
        "gate_reasons": gate_reasons,
        "target_semantics": "pairwise_only_not_full_action_classification",
        "privacy": "aggregate_only_no_hand_action_face_item_or_group_id_exported",
        "warning": (
            "Pairwise preference has no terminal value target and does not prove "
            "either action is globally optimal or human-level strength."
        ),
    }


def split_confirmed_pairwise_labels(
    records: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    partitions: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    seen_items: set[str] = set()
    group_partitions: dict[str, str] = {}
    for record in records:
        issues, _decision = _validate_pairwise_label(record)
        if issues or record.get("reviewer_confidence") != "confirmed":
            continue
        item_id = str(record["item_id"])
        if item_id in seen_items:
            raise ValueError("pairwise labels 包含重复 item_id")
        seen_items.add(item_id)
        group_id = str(record["review_group_id"])
        digest = hashlib.blake2b(
            f"{_PAIRWISE_SPLIT_SALT}|{group_id}".encode("utf-8"), digest_size=8
        ).digest()
        bucket = int.from_bytes(digest, "big") % 10_000
        split = "train" if bucket < 8000 else "validation" if bucket < 9000 else "test"
        existing = group_partitions.setdefault(group_id, split)
        if existing != split:
            raise RuntimeError("pairwise review group 被分到多个 split")
        partitions[split].append(record)
    return partitions


def pairwise_training_comparisons(
    records: Sequence[dict[str, Any]],
) -> list[tuple[str, TeacherDecision, tuple[int, int]]]:
    """Return only the displayed action pair; never imply full-action targets."""

    rows: list[tuple[str, TeacherDecision, tuple[int, int]]] = []
    seen: set[str] = set()
    for record in records:
        issues, decision = _validate_pairwise_label(record)
        if issues:
            raise ValueError("无效 pairwise training label：" + ",".join(issues))
        if record.get("reviewer_confidence") != "confirmed":
            raise ValueError("pairwise training 只能读取 confirmed 标签")
        item_id = str(record["item_id"])
        if item_id in seen:
            raise ValueError("pairwise training 包含重复 item_id")
        seen.add(item_id)
        assert decision is not None
        pair = tuple(int(index) for index in record["pair_action_indices"])
        rows.append((str(record["review_group_id"]), decision, pair))
    if not rows:
        raise ValueError("pairwise training split 不能为空")
    return rows

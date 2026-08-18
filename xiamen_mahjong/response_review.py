"""Actor-visible human review data for pass/chi/pong response corrections."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
from uuid import uuid4

from .agents import DeficiencyMeldTeacherAgent, HeuristicTeacherAgent
from .human_data import _private_key_paths
from .human_review import _action_payload, _opaque_hash, _queue_item_digest
from .rules import XiamenRules
from .tiles import WHITE_DRAGON
from .training import DATASET_VERSION, TeacherDecision, read_trajectory_jsonl


RESPONSE_REVIEW_QUEUE_VERSION = "xiamen-human-response-review-queue-v1"
RESPONSE_REVIEW_LABEL_VERSION = "xiamen-human-response-review-label-v1"
RESPONSE_REVIEW_GROUP_VERSION = "xiamen-human-response-review-group-v1"
RESPONSE_SLOW_EXPERT_VERSION = "deficiency-meld-teacher-v1-rejected"
_GROUP_SALT = "xiamen-human-response-review-group-v1"
_ITEM_SALT = "xiamen-human-response-review-item-v1"
_ORDER_SALT = "xiamen-human-response-review-order-v1"
_SPLIT_SALT = "xiamen-human-response-review-split-v1"
_CONFIDENCE_VALUES = frozenset({"confirmed", "uncertain"})
_REVIEW_ACTION_KINDS = frozenset({"pass", "chi", "pong"})


class _ActorVisibleResponseGame:
    """Minimal game facade rebuilt solely from one safe response snapshot."""

    def __init__(self, item_or_state: dict[str, Any]) -> None:
        state = item_or_state.get("state", item_or_state)
        self.rules = XiamenRules.classic()
        self.gold_tile = int(state["gold_tile"])
        self.gold_indicator = int(state["gold_indicator"])
        self.gold_proxy_tile = (
            WHITE_DRAGON if self.gold_tile != WHITE_DRAGON else None
        )
        self.wildcard_tiles = (self.gold_tile,)
        self.last_discard = int(state["last_discard"])
        self.discarder = int(state["discarder_relative"])
        self.tour_state = state.get("tour")
        self.gold_discard_lock_seat = 0 if state.get("gold_locked") else None
        self.opening_wait_seats = set(
            int(seat) for seat in state.get("opening_wait_relative_seats", [])
        )
        self.wall = [None] * int(state.get("wall_remaining", 0))
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


def _response_source_eligible(decision: TeacherDecision) -> bool:
    state = decision.state
    actions = decision.legal_actions
    kinds = [action.kind for action in actions]
    return bool(
        decision.profile == "classic"
        and state.get("phase") == "response"
        and state.get("tour") is None
        and not state.get("gold_locked")
        and 0 not in state.get("opening_wait_relative_seats", [])
        and set(kinds).issubset(_REVIEW_ACTION_KINDS)
        and kinds.count("pass") == 1
        and any(kind in {"chi", "pong"} for kind in kinds)
        and decision.chosen_action.kind in _REVIEW_ACTION_KINDS
        and state.get("last_discard") is not None
        and state.get("discarder_relative") is not None
    )


def response_slow_expert_index(item: dict[str, Any]) -> int:
    """Reproduce the rejected acquisition policy for post-label feedback."""

    issues = validate_response_review_queue_item(item)
    if issues:
        raise ValueError("不能重建无效 response review item：" + ",".join(issues))
    game = _ActorVisibleResponseGame(item)
    actions = tuple(
        TeacherDecision.from_payload(
            {
                "version": DATASET_VERSION,
                "profile": "classic",
                "seat": 0,
                "state": item["state"],
                "legal_actions": item["legal_actions"],
                "chosen_index": item["reference_teacher_index"],
            }
        ).legal_actions
    )
    frozen = HeuristicTeacherAgent().choose_response(game, 0, list(actions))
    reference = actions[int(item["reference_teacher_index"])]
    if frozen != reference:
        raise ValueError("response review item 的 Teacher 参考无法复现")
    selected = DeficiencyMeldTeacherAgent().choose_response(
        game, 0, list(actions)
    )
    return actions.index(selected)


def build_response_review_queue(
    paths: Sequence[str | Path],
    *,
    maximum_items: int = 400,
    maximum_items_per_group: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select safe response states, prioritizing opaque candidate disagreements."""

    if maximum_items <= 0 or maximum_items_per_group <= 0:
        raise ValueError("response review 数量门槛必须为正数")
    frozen_agent = HeuristicTeacherAgent()
    slow_agent = DeficiencyMeldTeacherAgent()
    candidates: list[tuple[bool, str, dict[str, Any]]] = []
    scanned_trajectories = 0
    scanned_decisions = 0
    eligible_decisions = 0
    reproduction_mismatches = 0
    acquisition_disagreements = 0
    for path in (Path(value) for value in paths):
        for trajectory in read_trajectory_jsonl(path):
            if (
                trajectory.profile != "classic"
                or trajectory.source_metadata.get("collector")
                != "teacher_self_play"
            ):
                continue
            scanned_trajectories += 1
            source_group = trajectory.split_group_id or trajectory.trajectory_id
            group_id = _opaque_hash(_GROUP_SALT, source_group)
            for ordinal, decision in enumerate(trajectory.decisions):
                scanned_decisions += 1
                if not _response_source_eligible(decision):
                    continue
                eligible_decisions += 1
                state = dict(decision.state)
                state["public_history_complete"] = False
                state["public_history_encoding"] = (
                    "bounded_actor_visible_response_review_v1"
                )
                item_id = _opaque_hash(
                    _ITEM_SALT, f"{trajectory.trajectory_id}|{ordinal}"
                )
                item = {
                    "version": RESPONSE_REVIEW_QUEUE_VERSION,
                    "item_id": item_id,
                    "review_group_id": group_id,
                    "review_group_version": RESPONSE_REVIEW_GROUP_VERSION,
                    "profile": "classic",
                    "rules_version": trajectory.rules_version,
                    "state": state,
                    "legal_actions": [
                        _action_payload(action) for action in decision.legal_actions
                    ],
                    "reference_teacher_index": decision.chosen_index,
                    "source_metadata": {
                        "collector": "teacher_self_play_actor_visible_response_review",
                        "source_scope": "ordinary_pass_chi_pong_response",
                        "opponent_hand_reveal": "unavailable_by_construction",
                        "training_default": "excluded_until_human_review_audit",
                    },
                }
                game = _ActorVisibleResponseGame(item)
                frozen = frozen_agent.choose_response(
                    game, 0, list(decision.legal_actions)
                )
                if frozen != decision.chosen_action:
                    reproduction_mismatches += 1
                    continue
                slow = slow_agent.choose_response(
                    game, 0, list(decision.legal_actions)
                )
                differs = slow != frozen
                acquisition_disagreements += differs
                order = _opaque_hash(_ORDER_SALT, item_id)
                candidates.append((not differs, order, item))

    queue: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    selected_acquisition_disagreements = 0
    for agreement_first, _order, item in sorted(
        candidates, key=lambda row: (row[0], row[1])
    ):
        group_id = str(item["review_group_id"])
        if group_counts[group_id] >= maximum_items_per_group:
            continue
        group_counts[group_id] += 1
        queue.append(item)
        selected_acquisition_disagreements += not agreement_first
        if len(queue) >= maximum_items:
            break
    return queue, {
        "status": "response_review_queue_ready" if queue else "response_review_queue_empty",
        "version": RESPONSE_REVIEW_QUEUE_VERSION,
        "scanned_teacher_trajectories": scanned_trajectories,
        "scanned_teacher_decisions": scanned_decisions,
        "eligible_response_decisions": eligible_decisions,
        "teacher_reproduction_mismatches": reproduction_mismatches,
        "acquisition_slow_expert": RESPONSE_SLOW_EXPERT_VERSION,
        "acquisition_disagreements": acquisition_disagreements,
        "queue_acquisition_disagreements": selected_acquisition_disagreements,
        "queue_items": len(queue),
        "queue_groups": len(group_counts),
        "maximum_items": maximum_items,
        "maximum_items_per_group": maximum_items_per_group,
        "blindness": "teacher_and_rejected_slow_expert_hidden_until_label",
        "privacy": (
            "actor_visible_only_no_seed_wall_opponent_hand_outcome_source_path_"
            "or_original_group_id_exported"
        ),
    }


def validate_response_review_queue_item(item: Any) -> list[str]:
    issues: list[str] = []
    if (
        not isinstance(item, dict)
        or item.get("version") != RESPONSE_REVIEW_QUEUE_VERSION
    ):
        return ["invalid_response_queue_version"]
    for key in ("item_id", "review_group_id"):
        value = item.get(key)
        if not (
            isinstance(value, str)
            and len(value) == 32
            and all(character in "0123456789abcdef" for character in value)
        ):
            issues.append(f"invalid_{key}")
    if item.get("review_group_version") != RESPONSE_REVIEW_GROUP_VERSION:
        issues.append("invalid_review_group_version")
    if item.get("profile") != "classic":
        issues.append("non_classic_profile")
    state = item.get("state")
    if not isinstance(state, dict) or state.get("phase") != "response":
        issues.append("invalid_actor_visible_response_state")
    elif (
        state.get("tour") is not None
        or state.get("gold_locked")
        or 0 in state.get("opening_wait_relative_seats", [])
    ):
        issues.append("special_response_state_outside_scope")
    private_paths = _private_key_paths(item)
    if private_paths:
        issues.append("private_key:" + ",".join(sorted(private_paths)))
    actions = item.get("legal_actions")
    reference = item.get("reference_teacher_index")
    if not (
        isinstance(actions, list)
        and len(actions) >= 2
        and all(isinstance(action, dict) for action in actions)
        and {action.get("kind") for action in actions}.issubset(
            _REVIEW_ACTION_KINDS
        )
        and sum(action.get("kind") == "pass" for action in actions) == 1
        and any(action.get("kind") in {"chi", "pong"} for action in actions)
        and isinstance(reference, int)
        and not isinstance(reference, bool)
        and 0 <= reference < len(actions)
    ):
        issues.append("invalid_response_actions_or_reference")
    elif isinstance(state, dict):
        last_discard = state.get("last_discard")
        if state.get("discarder_relative") is None or last_discard is None:
            issues.append("missing_response_discard_context")
        for action in actions:
            kind = action.get("kind")
            if kind == "pass" and set(action) != {"kind"}:
                issues.append("invalid_pass_payload")
            if kind in {"chi", "pong"} and not (
                action.get("tile") == last_discard
                and isinstance(action.get("tiles"), list)
                and len(action["tiles"]) == 2
                and all(isinstance(tile, int) for tile in action["tiles"])
            ):
                issues.append("invalid_claim_payload")
    metadata = item.get("source_metadata")
    if not isinstance(metadata, dict) or metadata.get(
        "opponent_hand_reveal"
    ) != "unavailable_by_construction":
        issues.append("opponent_hand_not_unavailable_by_construction")
    return issues


def write_response_review_queue(
    path: str | Path, queue: Sequence[dict[str, Any]]
) -> None:
    destination = Path(path)
    if destination.exists():
        raise ValueError("response review queue 输出已存在，拒绝覆盖")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for item in queue:
            issues = validate_response_review_queue_item(item)
            if issues:
                raise ValueError("不能写入无效 response review item：" + ",".join(issues))
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def read_response_review_queue(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"response review queue 第 {line_number} 行 JSON 无效"
                ) from error
            issues = validate_response_review_queue_item(item)
            if issues:
                raise ValueError(
                    f"response review queue 第 {line_number} 行无效："
                    + ",".join(issues)
                )
            items.append(item)
    if len({item["item_id"] for item in items}) != len(items):
        raise ValueError("response review queue 包含重复 item_id")
    return items


def public_response_review_item(
    item: dict[str, Any], *, completed: int, total: int
) -> dict[str, Any]:
    return {
        "status": "reviewing",
        "item_id": item["item_id"],
        "profile": item["profile"],
        "rules_version": item["rules_version"],
        "review_scope": "pass_chi_pong_response",
        "state": item["state"],
        "legal_actions": item["legal_actions"],
        "progress": {"completed": completed, "total": total},
        "privacy": "actor_visible_only_all_automatic_choices_hidden_until_label",
    }


def make_response_review_label(
    item: dict[str, Any], *, chosen_index: int, confidence: str
) -> dict[str, Any]:
    issues = validate_response_review_queue_item(item)
    if issues:
        raise ValueError("不能标注无效 response review item：" + ",".join(issues))
    actions = item["legal_actions"]
    if (
        isinstance(chosen_index, bool)
        or not isinstance(chosen_index, int)
        or not 0 <= chosen_index < len(actions)
    ):
        raise ValueError("response review chosen_index 必须对应合法响应")
    if confidence not in _CONFIDENCE_VALUES:
        raise ValueError("response review confidence 必须为 confirmed 或 uncertain")
    return {
        "version": RESPONSE_REVIEW_LABEL_VERSION,
        "review_id": uuid4().hex,
        "item_id": item["item_id"],
        "review_group_id": item["review_group_id"],
        "review_group_version": item["review_group_version"],
        "queue_item_digest": _queue_item_digest(item),
        "profile": item["profile"],
        "rules_version": item["rules_version"],
        "decision": {
            "version": DATASET_VERSION,
            "profile": item["profile"],
            "seat": 0,
            "state": item["state"],
            "legal_actions": actions,
            "chosen_index": chosen_index,
            "reference_teacher_index": item["reference_teacher_index"],
        },
        "reviewer_confidence": confidence,
        "source_metadata": {
            "collector": "local_human_response_review_opt_in",
            "recording_purpose": "training",
            "opponent_hand_reveal": "unavailable_by_construction",
            "automatic_references_hidden_before_choice": True,
            "training_default": "excluded_until_separate_quality_review",
            "outcome_target": "unavailable_review_behavior_only",
        },
    }


def append_response_review_label(path: str | Path, label: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")


def read_response_review_labels(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"response review label 第 {line_number} 行 JSON 无效"
                ) from error
    return records


def _validate_response_review_label(
    record: Any,
) -> tuple[list[str], TeacherDecision | None]:
    """Validate one standalone split row without reopening its source queue."""

    issues: list[str] = []
    if (
        not isinstance(record, dict)
        or record.get("version") != RESPONSE_REVIEW_LABEL_VERSION
    ):
        return ["invalid_response_label_version"], None
    for key in ("review_id", "item_id", "review_group_id"):
        value = record.get(key)
        if not (
            isinstance(value, str)
            and len(value) == 32
            and all(character in "0123456789abcdef" for character in value)
        ):
            issues.append(f"invalid_{key}")
    digest = record.get("queue_item_digest")
    if not (
        isinstance(digest, str)
        and len(digest) == 32
        and all(character in "0123456789abcdef" for character in digest)
    ):
        issues.append("invalid_queue_item_digest")
    if record.get("review_group_version") != RESPONSE_REVIEW_GROUP_VERSION:
        issues.append("invalid_review_group_version")
    if record.get("profile") != "classic":
        issues.append("non_classic_profile")
    rules_version = record.get("rules_version")
    if not isinstance(rules_version, str) or not rules_version:
        issues.append("invalid_rules_version")
    if record.get("reviewer_confidence") not in _CONFIDENCE_VALUES:
        issues.append("invalid_reviewer_confidence")
    metadata = record.get("source_metadata")
    if not isinstance(metadata, dict):
        issues.append("missing_source_metadata")
    else:
        if metadata.get("collector") != "local_human_response_review_opt_in":
            issues.append("invalid_collector")
        if metadata.get("recording_purpose") != "training":
            issues.append("invalid_recording_purpose")
        if metadata.get("opponent_hand_reveal") != "unavailable_by_construction":
            issues.append("opponent_hand_not_unavailable_by_construction")
        if metadata.get("automatic_references_hidden_before_choice") is not True:
            issues.append("automatic_references_not_hidden_before_choice")
        if metadata.get("outcome_target") != "unavailable_review_behavior_only":
            issues.append("invalid_outcome_target")
    private_paths = _private_key_paths(record)
    if private_paths:
        issues.append("private_key:" + ",".join(sorted(private_paths)))
    decision = None
    try:
        decision = TeacherDecision.from_payload(dict(record.get("decision", {})))
    except (TypeError, ValueError, KeyError):
        issues.append("invalid_response_review_decision")
    if decision is not None:
        if not _response_source_eligible(decision):
            issues.append("decision_outside_response_gate_scope")
        if decision.seat != 0 or decision.profile != record.get("profile"):
            issues.append("decision_record_identity_mismatch")
        if (
            decision.seed is not None
            or decision.executed_index is not None
            or decision.executed_probability is not None
            or decision.action_values is not None
            or decision.action_value_stderrs is not None
            or decision.action_value_gap_stderrs is not None
        ):
            issues.append("response_review_contains_nonbehavior_target")
    return sorted(set(issues)), decision


def read_confirmed_response_review_decisions(
    path: str | Path,
) -> list[tuple[str, TeacherDecision]]:
    """Load one audited response split while rejecting uncertainty/duplicates."""

    rows: list[tuple[str, TeacherDecision]] = []
    seen_items: set[str] = set()
    for line_number, record in enumerate(
        read_response_review_labels(path), start=1
    ):
        issues, decision = _validate_response_review_label(record)
        if issues:
            raise ValueError(
                f"response review training 第 {line_number} 行无效："
                + ",".join(issues)
            )
        if record.get("reviewer_confidence") != "confirmed":
            raise ValueError("response review training split 只能包含 confirmed 标签")
        item_id = str(record["item_id"])
        if item_id in seen_items:
            raise ValueError("response review training split 包含重复 item_id")
        seen_items.add(item_id)
        assert decision is not None
        rows.append((str(record["review_group_id"]), decision))
    if not rows:
        raise ValueError("response review training split 不能为空")
    return rows


def validate_response_review_label_against_item(
    record: Any, item: dict[str, Any]
) -> list[str]:
    issues, decision = _validate_response_review_label(record)
    if not isinstance(record, dict):
        return issues
    if record.get("profile") != item.get("profile"):
        issues.append("profile_queue_mismatch")
    if record.get("rules_version") != item.get("rules_version"):
        issues.append("rules_version_queue_mismatch")
    if record.get("item_id") != item.get("item_id"):
        issues.append("item_id_queue_mismatch")
    if record.get("review_group_id") != item.get("review_group_id"):
        issues.append("review_group_queue_mismatch")
    if record.get("queue_item_digest") != _queue_item_digest(item):
        issues.append("queue_item_digest_mismatch")
    if decision is not None:
        if decision.state != item.get("state"):
            issues.append("state_queue_mismatch")
        if [
            _action_payload(action) for action in decision.legal_actions
        ] != item.get("legal_actions"):
            issues.append("legal_actions_queue_mismatch")
        if decision.reference_teacher_index != item.get(
            "reference_teacher_index"
        ):
            issues.append("reference_teacher_queue_mismatch")
    return sorted(set(issues))


def audit_response_review_labels(
    records: Sequence[dict[str, Any]],
    *,
    minimum_confirmed_labels: int = 300,
    minimum_confirmed_disagreements: int = 50,
    minimum_groups: int = 100,
) -> dict[str, Any]:
    """Audit already queue-bound split rows without retaining private states."""

    if min(
        minimum_confirmed_labels,
        minimum_confirmed_disagreements,
        minimum_groups,
    ) <= 0:
        raise ValueError("response review audit 门槛必须为正数")
    issues: Counter[str] = Counter()
    item_ids: Counter[str] = Counter()
    fingerprints: Counter[str] = Counter()
    confirmed = 0
    uncertain = 0
    disagreements = 0
    groups: set[str] = set()
    action_counts: Counter[str] = Counter()
    for record in records:
        record_issues, decision = _validate_response_review_label(record)
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
        assert decision is not None
        confirmed += 1
        groups.add(str(record["review_group_id"]))
        action_counts[decision.chosen_action.kind] += 1
        disagreements += decision.chosen_index != decision.reference_teacher_index
    duplicate_items = sum(count - 1 for count in item_ids.values() if count > 1)
    duplicate_labels = sum(
        count - 1 for count in fingerprints.values() if count > 1
    )
    gate_reasons: list[str] = []
    if issues:
        gate_reasons.append("invalid_records")
    if duplicate_items or duplicate_labels:
        gate_reasons.append("duplicate_reviews")
    if confirmed < minimum_confirmed_labels:
        gate_reasons.append("insufficient_confirmed_labels")
    if disagreements < minimum_confirmed_disagreements:
        gate_reasons.append("insufficient_confirmed_disagreements")
    if len(groups) < minimum_groups:
        gate_reasons.append("insufficient_review_groups")
    return {
        "status": "local_human_response_review_audit",
        "records": len(records),
        "confirmed_labels": confirmed,
        "uncertain_labels": uncertain,
        "confirmed_teacher_disagreements": disagreements,
        "confirmed_review_groups": len(groups),
        "confirmed_action_counts": dict(sorted(action_counts.items())),
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
            "Response review labels are behavioral corrections without terminal "
            "outcomes and do not by themselves prove optimal play."
        ),
    }


def split_confirmed_response_review_labels(
    records: Sequence[dict[str, Any]], queue: Sequence[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically split confirmed response labels by physical group."""

    partitions: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    by_id = {str(item["item_id"]): item for item in queue}
    seen_items: set[str] = set()
    group_splits: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("reviewer_confidence") != "confirmed":
            continue
        item_id = str(record.get("item_id", ""))
        item = by_id.get(item_id)
        if item is None:
            raise ValueError("response review label 不属于 queue")
        issues = validate_response_review_label_against_item(record, item)
        if issues:
            raise ValueError("response review label 无效：" + ",".join(issues))
        if item_id in seen_items:
            raise ValueError("response review labels 包含重复 item_id")
        seen_items.add(item_id)
        group_id = str(record["review_group_id"])
        bucket = int(_opaque_hash(_SPLIT_SALT, group_id)[:8], 16) % 10_000
        split = "train" if bucket < 8_000 else "validation" if bucket < 9_000 else "test"
        previous = group_splits.setdefault(group_id, split)
        if previous != split:
            raise AssertionError("同一 response review group 跨越 split")
        partitions[split].append(record)
    return partitions


def audit_response_review_labels_against_queue(
    records: Sequence[dict[str, Any]],
    queue: Sequence[dict[str, Any]],
    *,
    minimum_confirmed_labels: int = 300,
    minimum_confirmed_disagreements: int = 50,
    minimum_groups: int = 100,
) -> dict[str, Any]:
    if min(
        minimum_confirmed_labels,
        minimum_confirmed_disagreements,
        minimum_groups,
    ) <= 0:
        raise ValueError("response review audit 门槛必须为正数")
    by_id = {str(item["item_id"]): item for item in queue}
    issues: Counter[str] = Counter()
    item_ids: Counter[str] = Counter()
    confirmed = 0
    uncertain = 0
    disagreements = 0
    groups: set[str] = set()
    action_counts: Counter[str] = Counter()
    for record in records:
        item = by_id.get(str(record.get("item_id", ""))) if isinstance(record, dict) else None
        if item is None:
            issues["item_not_in_queue"] += 1
            continue
        record_issues = validate_response_review_label_against_item(record, item)
        if record_issues:
            issues.update(record_issues)
            continue
        item_ids[str(record["item_id"])] += 1
        if record["reviewer_confidence"] == "uncertain":
            uncertain += 1
            continue
        decision = TeacherDecision.from_payload(dict(record["decision"]))
        confirmed += 1
        groups.add(str(record["review_group_id"]))
        action_counts[decision.chosen_action.kind] += 1
        disagreements += decision.chosen_index != decision.reference_teacher_index
    duplicate_items = sum(count - 1 for count in item_ids.values() if count > 1)
    gate_reasons = []
    if issues:
        gate_reasons.append("invalid_or_unbound_records")
    if duplicate_items:
        gate_reasons.append("duplicate_reviews")
    if confirmed < minimum_confirmed_labels:
        gate_reasons.append("insufficient_confirmed_labels")
    if disagreements < minimum_confirmed_disagreements:
        gate_reasons.append("insufficient_confirmed_disagreements")
    if len(groups) < minimum_groups:
        gate_reasons.append("insufficient_review_groups")
    return {
        "status": "local_human_response_review_audit",
        "records": len(records),
        "queue_items": len(queue),
        "confirmed_labels": confirmed,
        "uncertain_labels": uncertain,
        "confirmed_teacher_disagreements": disagreements,
        "confirmed_review_groups": len(groups),
        "confirmed_action_counts": dict(sorted(action_counts.items())),
        "minimum_confirmed_labels": minimum_confirmed_labels,
        "minimum_confirmed_disagreements": minimum_confirmed_disagreements,
        "minimum_groups": minimum_groups,
        "duplicate_items": duplicate_items,
        "issues": dict(sorted(issues.items())),
        "ready_for_manual_training_review": not gate_reasons,
        "gate_reasons": gate_reasons,
        "privacy": "aggregate_only_no_state_hand_action_face_item_or_group_id_exported",
        "warning": (
            "Response review labels are human behavior judgments without terminal "
            "outcomes and do not by themselves prove optimal play."
        ),
    }

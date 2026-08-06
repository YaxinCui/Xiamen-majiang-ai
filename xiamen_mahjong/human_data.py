"""Read-only quality gates for explicitly recorded local human-play data."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .training import TrainingTrajectory, read_trajectory_jsonl


_FORBIDDEN_PRIVATE_KEYS = {
    "seed",
    "behavior_seed",
    "wall",
    "wall_order",
    "opponent_hands",
}


def _private_key_paths(value: Any, *, prefix: str = "") -> list[str]:
    """Find prohibited private replay fields without banning public counts."""

    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            rendered_key = str(key)
            path = f"{prefix}.{rendered_key}" if prefix else rendered_key
            if rendered_key.lower() in _FORBIDDEN_PRIVATE_KEYS:
                paths.append(path)
            paths.extend(_private_key_paths(item, prefix=path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            paths.extend(_private_key_paths(item, prefix=f"{prefix}[{index}]"))
    return paths


def _trajectory_fingerprint(trajectory: TrainingTrajectory) -> str:
    """Detect exact duplicate human hands without writing their content out."""

    payload = trajectory.payload()
    payload.pop("trajectory_id", None)
    payload.pop("split_group_id", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _validate_human_trajectory(trajectory: TrainingTrajectory) -> list[str]:
    issues: list[str] = []
    metadata = trajectory.source_metadata
    if metadata.get("collector") != "local_human_opt_in":
        issues.append("collector_not_local_human_opt_in")
    if metadata.get("training_default") != "excluded_until_separate_quality_review":
        issues.append("missing_training_isolation_marker")
    if not isinstance(metadata.get("opponent_policy"), str):
        issues.append("missing_opponent_policy_identity")
    if trajectory.seed is not None or any(
        decision.seed is not None for decision in trajectory.decisions
    ):
        issues.append("replay_seed_present")
    if trajectory.outcome.get("score_semantics") != "single_hand_delta":
        issues.append("outcome_is_not_single_hand_delta")
    if bool(trajectory.outcome.get("synthetic")):
        issues.append("synthetic_outcome")
    if not trajectory.decisions:
        issues.append("no_human_decisions")
    for decision in trajectory.decisions:
        if decision.seat != 0:
            issues.append("non_human_seat_decision")
        if decision.executed_index != decision.chosen_index:
            issues.append("human_choice_not_executed_action")
        if decision.executed_probability is not None:
            issues.append("human_propensity_must_be_unknown")
        private_paths = _private_key_paths(decision.state)
        if private_paths:
            issues.append("private_state_key:" + ",".join(sorted(private_paths)))
    metadata_private_paths = _private_key_paths(metadata)
    if metadata_private_paths:
        issues.append("private_metadata_key:" + ",".join(sorted(metadata_private_paths)))
    return issues


def audit_local_human_trajectories(
    paths: Iterable[str | Path],
    *,
    minimum_hands: int = 100,
) -> dict[str, Any]:
    """Summarize opt-in human records and enforce their pre-training boundary.

    The result contains only aggregate counts, rule/profile distributions, and
    opaque duplicate fingerprints counts. It deliberately never returns a
    decision state, a hand, a public action sequence, or an opponent identity
    beyond its already-recorded checkpoint/Teacher label.
    """

    if minimum_hands <= 0:
        raise ValueError("minimum_hands 必须为正数")
    files = [Path(path) for path in paths]
    if not files:
        raise ValueError("至少需要一个人类轨迹文件")
    trajectories: list[TrainingTrajectory] = []
    for path in files:
        trajectories.extend(read_trajectory_jsonl(path))

    issue_counts: Counter[str] = Counter()
    valid: list[TrainingTrajectory] = []
    fingerprints: Counter[str] = Counter()
    profiles: Counter[str] = Counter()
    rules_versions: Counter[str] = Counter()
    opponent_policies: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for trajectory in trajectories:
        fingerprint = _trajectory_fingerprint(trajectory)
        fingerprints[fingerprint] += 1
        issues = _validate_human_trajectory(trajectory)
        if issues:
            issue_counts.update(issues)
            continue
        valid.append(trajectory)
        profiles[trajectory.profile] += 1
        rules_versions[trajectory.rules_version] += 1
        opponent_policies[str(trajectory.source_metadata["opponent_policy"])] += 1
        action_counts.update(decision.chosen_action.kind for decision in trajectory.decisions)
    duplicate_hands = sum(count - 1 for count in fingerprints.values() if count > 1)
    gate_reasons: list[str] = []
    if issue_counts:
        gate_reasons.append("invalid_records")
    if duplicate_hands:
        gate_reasons.append("duplicate_hands")
    if len(valid) < minimum_hands:
        gate_reasons.append("insufficient_completed_hands")
    if len(profiles) != 1:
        gate_reasons.append("mixed_rule_profiles")
    if len(opponent_policies) != 1:
        gate_reasons.append("mixed_opponent_policies")
    return {
        "source": "local_human_opt_in",
        "files": len(files),
        "records": len(trajectories),
        "valid_hands": len(valid),
        "invalid_hands": len(trajectories) - len(valid),
        "minimum_hands": minimum_hands,
        "duplicate_hands": duplicate_hands,
        "rule_profiles": dict(sorted(profiles.items())),
        "rules_versions": dict(sorted(rules_versions.items())),
        "opponent_policies": dict(sorted(opponent_policies.items())),
        "human_action_counts": dict(sorted(action_counts.items())),
        "issues": dict(sorted(issue_counts.items())),
        "ready_for_manual_review": not gate_reasons,
        "gate_reasons": gate_reasons,
        "warning": (
            "Passing this structural gate does not establish human skill or "
            "authorize training/promotion; use independent held-out human "
            "matches before any strength claim."
        ),
    }

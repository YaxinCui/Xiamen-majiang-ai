"""Read-only quality gates for explicitly recorded local human-play data."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .training import (
    TrainingTrajectory,
    read_trajectory_jsonl,
    split_trajectories_by_hand,
)


_FORBIDDEN_PRIVATE_KEYS = {
    "seed",
    "behavior_seed",
    "wall",
    "wall_order",
    "opponent_hands",
}
HUMAN_SPLIT_VERSION = "xiamen-local-human-hand-split-v1"
_HUMAN_SPLIT_SALT = "xiamen-local-human-hand-split-v1"


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
    scores = trajectory.outcome.get("scores")
    if not (
        isinstance(scores, list)
        and len(scores) == 4
        and all(isinstance(score, (int, float)) and not isinstance(score, bool) for score in scores)
    ):
        issues.append("non_numeric_score_delta")
    if not trajectory.decisions:
        issues.append("no_human_decisions")
    for decision in trajectory.decisions:
        if decision.seat != 0:
            issues.append("non_human_seat_decision")
        if decision.executed_index != decision.chosen_index:
            issues.append("human_choice_not_executed_action")
        if decision.executed_probability is not None:
            issues.append("human_propensity_must_be_unknown")
        if decision.action_values is not None:
            issues.append("human_record_must_not_have_action_value_targets")
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
    human_scores: list[float] = []
    human_wins = 0
    draws = 0
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
        scores = trajectory.outcome["scores"]
        human_scores.append(float(scores[0]))
        human_wins += trajectory.outcome.get("winner") == 0
        draws += trajectory.outcome.get("win_type") == "draw"
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
    human_score_mean = sum(human_scores) / len(human_scores) if human_scores else None
    human_score_stderr = (
        math.sqrt(
            sum((score - human_score_mean) ** 2 for score in human_scores)
            / (len(human_scores) * (len(human_scores) - 1))
        )
        if len(human_scores) > 1 and human_score_mean is not None
        else None
    )
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
        "human_match_summary": {
            "scope": "structurally_valid_completed_hands_only",
            "hands": len(human_scores),
            "human_score_delta_mean": human_score_mean,
            "human_score_delta_stderr": human_score_stderr,
            "human_score_delta_95pct_low": (
                human_score_mean - 1.96 * human_score_stderr
                if human_score_mean is not None and human_score_stderr is not None
                else None
            ),
            "human_score_delta_95pct_high": (
                human_score_mean + 1.96 * human_score_stderr
                if human_score_mean is not None and human_score_stderr is not None
                else None
            ),
            "human_win_rate": human_wins / len(human_scores) if human_scores else None,
            "draw_rate": draws / len(human_scores) if human_scores else None,
        },
        "issues": dict(sorted(issue_counts.items())),
        "ready_for_manual_review": not gate_reasons,
        "gate_reasons": gate_reasons,
        "warning": (
            "The match summary is a descriptive, normal-approximation aggregate "
            "over structurally valid records only. Passing this structural gate "
            "does not establish human skill or authorize training/promotion; use "
            "independent held-out human matches before any strength claim."
        ),
    }


def require_local_human_training_approval(
    paths: Iterable[str | Path],
    *,
    manually_approved: bool,
    minimum_hands: int = 100,
) -> dict[str, Any]:
    """Return a safe audit only after an explicit human-data training gate.

    Local records are intentionally excluded from ordinary model training:
    structural validation cannot establish a player's skill, consent beyond
    the local recorder, or that a held-out human benchmark remains untouched.
    A trainer must therefore opt in *and* the source must pass the existing
    privacy, duplication, rule-profile, and opponent-identity checks.

    The returned aggregate contains no local paths or decision state, so it
    may safely be embedded in a checkpoint training report as provenance.
    """

    if not manually_approved:
        raise ValueError(
            "本地人类轨迹默认禁止训练；完成审计和人工质量复核后，"
            "请显式传入 --allow-local-human-data"
        )
    audit = audit_local_human_trajectories(paths, minimum_hands=minimum_hands)
    if not audit["ready_for_manual_review"]:
        reasons = ", ".join(str(item) for item in audit["gate_reasons"])
        raise ValueError("本地人类轨迹未通过训练前结构门槛：" + reasons)
    return {
        "source": "local_human_opt_in",
        "manual_training_approval": True,
        "training_scope": "behavioral_imitation_only; not_human_strength_evidence",
        "audit": audit,
    }


def split_local_human_trajectories(
    paths: Iterable[str | Path],
    *,
    minimum_hands: int = 100,
) -> tuple[dict[str, list[TrainingTrajectory]], dict[str, Any]]:
    """Create a deterministic full-hand split only from audited local logs.

    This is intentionally a data-preparation boundary, not a training
    authorization.  It first applies the same opt-in, privacy, duplicate,
    rule-profile and opponent-identity checks used by the trainer.  Only then
    are complete hands assigned by an opaque group identity to train,
    validation and test.  No state, hand, seed or local pathname appears in
    the returned audit; callers must keep written files under the ignored
    ``local_human_data/`` area.
    """

    if minimum_hands <= 0:
        raise ValueError("minimum_hands 必须为正数")
    files = [Path(path) for path in paths]
    if not files:
        raise ValueError("至少需要一个人类轨迹文件")
    audit = audit_local_human_trajectories(files, minimum_hands=minimum_hands)
    if not audit["ready_for_manual_review"]:
        reasons = ", ".join(str(item) for item in audit["gate_reasons"])
        raise ValueError("人类轨迹未通过切分前结构门槛：" + reasons)

    trajectories: list[TrainingTrajectory] = []
    for path in files:
        trajectories.extend(read_trajectory_jsonl(path))
    partitions = split_trajectories_by_hand(
        trajectories,
        split_salt=_HUMAN_SPLIT_SALT,
    )
    if any(not records for records in partitions.values()):
        raise ValueError(
            "按完整牌局切分后 train、validation、test 都必须至少有一局；"
            "请收集更多独立人类对局"
        )

    group_sets = {
        split: {
            trajectory.split_group_id or trajectory.trajectory_id
            for trajectory in records
        }
        for split, records in partitions.items()
    }
    split_names = tuple(group_sets)
    if any(
        group_sets[left] & group_sets[right]
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    ):
        raise RuntimeError("人类训练切分出现跨集合完整牌局重叠")
    return partitions, {
        **audit,
        "split_version": HUMAN_SPLIT_VERSION,
        "split_salt": _HUMAN_SPLIT_SALT,
        "split_hand_counts": {
            split: len(records) for split, records in partitions.items()
        },
        "split_group_counts": {
            split: len(group_sets[split]) for split in partitions
        },
        "split_group_overlap": 0,
    }

"""Read-only quality gates for explicitly recorded local human-play data."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID

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
_RECORDING_PURPOSES = frozenset({"training", "evaluation"})


def _is_opaque_session_id(value: Any) -> bool:
    """Accept only a UUID-shaped random recorder session identity."""

    if not isinstance(value, str):
        return False
    try:
        UUID(hex=value)
    except (TypeError, ValueError, AttributeError):
        return False
    return len(value.replace("-", "")) == 32


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
    metadata = payload.get("source_metadata")
    if isinstance(metadata, dict):
        # A recorder session is an audit block, not part of the physical hand.
        # Retaining it here would let the same hand evade duplicate detection
        # simply by being copied into a newly started browser session.
        metadata.pop("recording_session_id", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _validate_human_trajectory(trajectory: TrainingTrajectory) -> list[str]:
    issues: list[str] = []
    metadata = trajectory.source_metadata
    if metadata.get("collector") != "local_human_opt_in":
        issues.append("collector_not_local_human_opt_in")
    if metadata.get("training_default") != "excluded_until_separate_quality_review":
        issues.append("missing_training_isolation_marker")
    purpose = metadata.get("recording_purpose")
    if purpose not in _RECORDING_PURPOSES:
        issues.append("missing_or_invalid_recording_purpose")
    elif purpose == "evaluation" and not _is_opaque_session_id(
        metadata.get("recording_session_id")
    ):
        issues.append("missing_or_invalid_evaluation_session_id")
    if not isinstance(metadata.get("opponent_policy"), str):
        issues.append("missing_opponent_policy_identity")
    if metadata.get("opponent_hand_reveal") != "server_forced_disabled":
        issues.append("opponent_hand_reveal_not_forced_disabled")
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
    elif not math.isclose(sum(float(score) for score in scores), 0.0, abs_tol=1e-9):
        issues.append("non_zero_sum_score_delta")
    if not trajectory.decisions:
        issues.append("no_human_decisions")
    reference_count = sum(
        decision.reference_teacher_index is not None
        for decision in trajectory.decisions
    )
    if reference_count:
        if reference_count != len(trajectory.decisions):
            issues.append("partial_reference_teacher_labels")
        if not isinstance(metadata.get("reference_policy"), str):
            issues.append("missing_reference_policy_identity")
        if metadata.get("reference_label") != "same_state_frozen_teacher_action_index":
            issues.append("missing_reference_label_contract")
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


def is_eligible_human_teacher_discard_correction(
    decision: Any,
) -> bool:
    """Match the exact discard-only scope used by the deployed v1 gate."""

    reference = getattr(decision, "reference_teacher_index", None)
    legal_actions = getattr(decision, "legal_actions", ())
    if (
        isinstance(reference, bool)
        or not isinstance(reference, int)
        or not 0 <= reference < len(legal_actions)
    ):
        return False
    state = getattr(decision, "state", {})
    chosen_action = getattr(decision, "chosen_action", None)
    if not isinstance(state, dict) or chosen_action is None:
        return False
    return bool(
        state.get("phase") == "discard"
        and legal_actions[reference].kind == "discard"
        and chosen_action.kind == "discard"
        and state.get("tour") is None
        and not bool(state.get("gold_locked", False))
        and sum(action.kind == "discard" for action in legal_actions) >= 2
    )


def audit_local_human_teacher_corrections(
    paths: Iterable[str | Path],
    *,
    minimum_hands: int = 100,
    minimum_reference_decisions: int = 500,
    minimum_disagreements: int = 50,
) -> dict[str, Any]:
    """Gate human demonstrations for Teacher-residual model selection.

    A human action is an independent behavioral target; the frozen Teacher
    index merely identifies the subset on which that target actually corrects
    the rule policy.  The audit is aggregate-only and requires training-only
    records, complete reference coverage, one reference identity and enough
    disagreements to justify a separate residual experiment.
    """

    if minimum_reference_decisions <= 0:
        raise ValueError("minimum_reference_decisions 必须为正数")
    if minimum_disagreements <= 0:
        raise ValueError("minimum_disagreements 必须为正数")
    audit = audit_local_human_trajectories(paths, minimum_hands=minimum_hands)
    gate_reasons = list(audit["gate_reasons"])
    if set(audit["recording_purposes"]) != {"training"}:
        gate_reasons.append("records_are_not_training_only")
    summary = audit["reference_teacher_summary"]
    total_decisions = sum(audit["human_action_counts"].values())
    all_reference_decisions = int(summary["decisions_with_reference"])
    reference_decisions = int(
        summary["eligible_ordinary_discard_reference_decisions"]
    )
    disagreements = int(summary["eligible_ordinary_discard_disagreements"])
    if all_reference_decisions != total_decisions:
        gate_reasons.append("incomplete_reference_teacher_coverage")
    if len(summary["reference_policies"]) != 1:
        gate_reasons.append("mixed_or_missing_reference_policy")
    if reference_decisions < minimum_reference_decisions:
        gate_reasons.append("insufficient_reference_decisions")
    if disagreements < minimum_disagreements:
        gate_reasons.append("insufficient_teacher_disagreements")
    return {
        "status": "local_human_teacher_correction_audit",
        "source": "local_human_opt_in",
        "audit": audit,
        "correction_summary": {
            "total_human_decisions": total_decisions,
            "all_reference_decisions": all_reference_decisions,
            "reference_decisions": reference_decisions,
            "minimum_reference_decisions": minimum_reference_decisions,
            "teacher_disagreements": disagreements,
            "minimum_teacher_disagreements": minimum_disagreements,
            "teacher_disagreement_rate": (
                disagreements / reference_decisions
                if reference_decisions
                else None
            ),
            "reference_policies": summary["reference_policies"],
            "disagreement_action_pairs": summary["disagreement_action_pairs"],
            "scope": "aggregate_only_eligible_ordinary_discard_training_records",
        },
        "ready_for_teacher_residual_experiment": not gate_reasons,
        "gate_reasons": list(dict.fromkeys(gate_reasons)),
        "warning": (
            "Passing this gate only makes a lightweight Teacher-residual experiment "
            "statistically worthwhile. It does not prove that each human correction "
            "is optimal; use a full-hand held-out human split and independent match "
            "evaluation before promotion."
        ),
    }


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
    recording_purposes: Counter[str] = Counter()
    recording_sessions: set[str] = set()
    action_counts: Counter[str] = Counter()
    reference_policy_counts: Counter[str] = Counter()
    reference_decisions = 0
    reference_agreements = 0
    eligible_discard_reference_decisions = 0
    eligible_discard_reference_agreements = 0
    reference_disagreement_pairs: Counter[str] = Counter()
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
        recording_purposes[str(trajectory.source_metadata["recording_purpose"])] += 1
        session_id = trajectory.source_metadata.get("recording_session_id")
        if isinstance(session_id, str):
            recording_sessions.add(session_id)
        action_counts.update(decision.chosen_action.kind for decision in trajectory.decisions)
        reference_policy = trajectory.source_metadata.get("reference_policy")
        if isinstance(reference_policy, str):
            reference_policy_counts[reference_policy] += 1
        for decision in trajectory.decisions:
            reference_index = decision.reference_teacher_index
            if reference_index is None:
                continue
            reference_decisions += 1
            if is_eligible_human_teacher_discard_correction(decision):
                eligible_discard_reference_decisions += 1
                if reference_index == decision.chosen_index:
                    eligible_discard_reference_agreements += 1
            if reference_index == decision.chosen_index:
                reference_agreements += 1
                continue
            reference_kind = decision.legal_actions[reference_index].kind
            human_kind = decision.chosen_action.kind
            reference_disagreement_pairs[
                f"teacher:{reference_kind}->human:{human_kind}"
            ] += 1
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
    if len(recording_purposes) != 1:
        gate_reasons.append("mixed_recording_purposes")
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
        "recording_purposes": dict(sorted(recording_purposes.items())),
        "recording_session_count": len(recording_sessions),
        "human_action_counts": dict(sorted(action_counts.items())),
        "reference_teacher_summary": {
            "reference_policies": dict(sorted(reference_policy_counts.items())),
            "decisions_with_reference": reference_decisions,
            "agreements": reference_agreements,
            "disagreements": reference_decisions - reference_agreements,
            "agreement_rate": (
                reference_agreements / reference_decisions
                if reference_decisions
                else None
            ),
            "disagreement_action_pairs": dict(
                sorted(reference_disagreement_pairs.items())
            ),
            "eligible_ordinary_discard_reference_decisions": (
                eligible_discard_reference_decisions
            ),
            "eligible_ordinary_discard_agreements": (
                eligible_discard_reference_agreements
            ),
            "eligible_ordinary_discard_disagreements": (
                eligible_discard_reference_decisions
                - eligible_discard_reference_agreements
            ),
            "scope": "aggregate_only_no_state_or_action_face_export",
        },
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
    if set(audit["recording_purposes"]) != {"training"}:
        raise ValueError("独立真人评测记录不得用于训练；训练输入必须全部标记为 training")
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
    if set(audit["recording_purposes"]) != {"training"}:
        raise ValueError("独立真人评测记录不得切分为训练数据")

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


def audit_local_human_evaluation(
    paths: Iterable[str | Path],
    *,
    minimum_hands: int = 200,
    minimum_sessions: int = 10,
) -> dict[str, Any]:
    """Audit an evaluation-only human-versus-fixed-AI match corpus.

    The browser records one human at seat zero and three copies of one frozen
    AI identity.  The AI-side score is therefore the negation of the human
    hand score.  This routine reports a conservative normal-approximation
    lower bound for that *specific local match corpus*, with each local
    recording session weighted equally. It deliberately does not claim
    population-level human strength or authorize a deployment: the recording
    purpose only makes leakage into this project's trainers fail closed;
    participant quality and recruitment still require manual review.
    """

    if minimum_sessions < 2:
        raise ValueError("minimum_sessions 至少为 2")
    files = [Path(path) for path in paths]
    audit = audit_local_human_trajectories(files, minimum_hands=minimum_hands)
    gate_reasons = list(audit["gate_reasons"])
    if set(audit["recording_purposes"]) != {"evaluation"}:
        gate_reasons.append("records_are_not_evaluation_only")
    session_scores: dict[str, list[float]] = {}
    for path in files:
        for trajectory in read_trajectory_jsonl(path):
            if _validate_human_trajectory(trajectory):
                continue
            metadata = trajectory.source_metadata
            if metadata.get("recording_purpose") != "evaluation":
                continue
            session_id = metadata.get("recording_session_id")
            if not _is_opaque_session_id(session_id):
                continue
            scores = trajectory.outcome["scores"]
            session_scores.setdefault(str(session_id), []).append(float(scores[0]))
    session_means = [
        sum(scores) / len(scores)
        for scores in session_scores.values()
        if scores
    ]
    if len(session_means) < minimum_sessions:
        gate_reasons.append("insufficient_evaluation_sessions")
    human_mean = sum(session_means) / len(session_means) if session_means else None
    human_stderr = (
        math.sqrt(
            sum((score - human_mean) ** 2 for score in session_means)
            / (len(session_means) * (len(session_means) - 1))
        )
        if len(session_means) > 1 and human_mean is not None
        else None
    )
    human_high = (
        human_mean + 1.96 * human_stderr
        if human_mean is not None and human_stderr is not None
        else None
    )
    ai_mean = -float(human_mean) if human_mean is not None else None
    ai_low = -float(human_high) if human_high is not None else None
    ai_roster_size = 3
    ai_per_seat_mean = (
        ai_mean / ai_roster_size if ai_mean is not None else None
    )
    ai_per_seat_low = (
        ai_low / ai_roster_size if ai_low is not None else None
    )
    return {
        "status": "local_human_evaluation_audit_only",
        "source": "local_human_opt_in",
        "audit": audit,
        "comparison": {
            "scope": (
                "one_human_seat_vs_three_copies_of_one_fixed_ai_identity; "
                "session_blocked_equal_weight"
            ),
            "recording_sessions": len(session_means),
            "minimum_sessions": minimum_sessions,
            "hands_per_session_min": (
                min(len(scores) for scores in session_scores.values())
                if session_scores
                else 0
            ),
            "hands_per_session_max": (
                max(len(scores) for scores in session_scores.values())
                if session_scores
                else 0
            ),
            "ai_roster_size": ai_roster_size,
            "ai_team_score_delta_mean": ai_mean,
            "ai_team_score_delta_95pct_low": ai_low,
            "ai_per_seat_score_delta_mean": ai_per_seat_mean,
            "ai_per_seat_score_delta_95pct_low": ai_per_seat_low,
            "positive_ai_per_seat_lcb": (
                ai_per_seat_low is not None and ai_per_seat_low > 0.0
            ),
            # Backward-compatible aliases.  Their historical meaning is the
            # aggregate score of all three identical AI seats, not one seat.
            "ai_side_score_delta_mean": ai_mean,
            "ai_side_score_delta_95pct_low": ai_low,
            "positive_ai_side_lcb": ai_low is not None and ai_low > 0.0,
        },
        "ready_for_manual_human_strength_review": not gate_reasons,
        "gate_reasons": gate_reasons,
        "warning": (
            "This is an evaluation-only local match audit, not a claim that an AI "
            "beats humans generally. Before any such claim, manually verify consent, "
            "that each session corresponds to the intended independent participant or "
            "pre-registered block, participant recruitment/skill, frozen AI identity, "
            "and that these hands were never read for training or model selection. "
            "The per-seat AI metric divides the three-seat AI team's zero-sum score "
            "by three; the legacy ai_side fields are team aggregates."
        ),
    }

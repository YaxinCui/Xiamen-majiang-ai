#!/usr/bin/env python3
"""Select a discard-only human-correction gate on full-hand held-out data.

The validation split selects one of a fixed set of low override-rate margins.
The test split is not opened unless a validation candidate has a positive
hand-grouped 95% lower bound and a Wilson precision lower bound above chance.
All reports are aggregate-only; human states, actions, histories and paths are
never written to the output.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.human_data import (
    audit_local_human_trajectories,
    is_eligible_human_teacher_discard_correction,
)
from xiamen_mahjong.human_review import (
    audit_review_labels,
    read_confirmed_review_decisions,
    read_review_labels,
)
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import TeacherDecision, read_trajectory_jsonl


DEFAULT_TARGET_OVERRIDE_RATES = (0.01, 0.02, 0.05)


@dataclass(frozen=True)
class HumanGateObservation:
    """One aggregate-safe unit retained in memory during gate selection."""

    group_id: str
    policy_gap: float
    teacher_matches_human: bool
    alternative_matches_human: bool


@dataclass(frozen=True)
class HumanGateScan:
    observations: tuple[HumanGateObservation, ...]
    trajectory_groups: frozenset[str]
    structural_audit: dict[str, Any]


def choose_strict_margin(
    positive_gaps: Sequence[float],
    *,
    eligible_decisions: int,
    target_override_rate: float,
) -> float | None:
    """Choose ``gap > margin`` without exceeding the requested rate."""

    if eligible_decisions <= 0:
        return None
    if not 0.0 < target_override_rate < 1.0:
        raise ValueError("target_override_rate 必须在 0 和 1 之间")
    gaps = sorted(
        (
            float(gap)
            for gap in positive_gaps
            if math.isfinite(float(gap)) and float(gap) > 0.0
        ),
        reverse=True,
    )
    if not gaps:
        return None
    maximum_overrides = max(1, math.floor(eligible_decisions * target_override_rate))
    if len(gaps) <= maximum_overrides:
        return 0.0
    upper = gaps[maximum_overrides - 1]
    lower = gaps[maximum_overrides]
    if upper > lower:
        return (upper + lower) / 2.0
    # Strict comparison drops a tied boundary as one indivisible group.
    return upper


def wilson_lower_bound(successes: int, total: int, *, z: float = 1.96) -> float | None:
    """Two-sided normal-score Wilson lower bound for a Bernoulli rate."""

    if total <= 0:
        return None
    if not 0 <= successes <= total:
        raise ValueError("successes 必须在 0 和 total 之间")
    if z <= 0:
        raise ValueError("z 必须为正数")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return (centre - radius) / denominator


def _mean_stderr(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def summarize_gate(
    observations: Sequence[HumanGateObservation],
    *,
    margin: float | None,
    scope: str = "ordinary_discard_alternative_only_full_hand_grouped",
) -> dict[str, Any]:
    """Compare a gated policy with Teacher using complete-hand groups."""

    selected = [
        observation
        for observation in observations
        if margin is not None and observation.policy_gap > margin
    ]
    corrections = sum(observation.alternative_matches_human for observation in selected)
    harms = sum(observation.teacher_matches_human for observation in selected)
    unresolved = len(selected) - corrections - harms
    teacher_correct = sum(observation.teacher_matches_human for observation in observations)
    gated_correct = teacher_correct + corrections - harms
    grouped_counts: dict[str, int] = defaultdict(int)
    grouped_net: dict[str, int] = defaultdict(int)
    for observation in observations:
        grouped_counts[observation.group_id] += 1
    for observation in selected:
        grouped_net[observation.group_id] += (
            int(observation.alternative_matches_human)
            - int(observation.teacher_matches_human)
        )
    group_gains = [
        grouped_net[group_id] / count
        for group_id, count in sorted(grouped_counts.items())
    ]
    gain_mean, gain_stderr = _mean_stderr(group_gains)
    gain_low = (
        gain_mean - 1.96 * gain_stderr
        if gain_mean is not None and gain_stderr is not None
        else None
    )
    precision = corrections / len(selected) if selected else None
    return {
        "eligible_decisions": len(observations),
        "eligible_hand_groups": len(grouped_counts),
        "strict_margin": margin,
        "overrides": len(selected),
        "override_rate": len(selected) / len(observations) if observations else None,
        "human_corrections": corrections,
        "teacher_to_human_harms": harms,
        "both_actions_miss_human": unresolved,
        "override_human_precision": precision,
        "override_human_precision_wilson_95pct_low": wilson_lower_bound(
            corrections, len(selected)
        ),
        "teacher_human_action_accuracy": (
            teacher_correct / len(observations) if observations else None
        ),
        "gated_human_action_accuracy": (
            gated_correct / len(observations) if observations else None
        ),
        "gated_minus_teacher_accuracy": (
            (gated_correct - teacher_correct) / len(observations)
            if observations
            else None
        ),
        "hand_grouped_accuracy_gain_mean": gain_mean,
        "hand_grouped_accuracy_gain_stderr": gain_stderr,
        "hand_grouped_accuracy_gain_95pct_low": gain_low,
        "scope": scope,
    }


def gate_passes(
    summary: dict[str, Any],
    *,
    minimum_overrides: int,
    minimum_groups: int,
    maximum_override_rate: float,
) -> bool:
    precision_low = summary["override_human_precision_wilson_95pct_low"]
    gain_low = summary["hand_grouped_accuracy_gain_95pct_low"]
    rate = summary["override_rate"]
    return bool(
        summary["overrides"] >= minimum_overrides
        and summary["eligible_hand_groups"] >= minimum_groups
        and isinstance(rate, float)
        and rate <= maximum_override_rate
        and isinstance(precision_low, float)
        and precision_low > 0.5
        and isinstance(gain_low, float)
        and gain_low > 0.0
    )


def select_validation_gate(
    observations: Sequence[HumanGateObservation],
    *,
    target_override_rates: Sequence[float],
    minimum_overrides: int,
    minimum_groups: int,
    scope: str = "ordinary_discard_alternative_only_full_hand_grouped",
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Select one predeclared coverage target; never search raw margins."""

    if minimum_overrides <= 0 or minimum_groups < 2:
        raise ValueError("minimum_overrides 必须为正且 minimum_groups 至少为 2")
    if not target_override_rates:
        raise ValueError("至少需要一个 target_override_rate")
    rates = tuple(float(rate) for rate in target_override_rates)
    if any(not 0.0 < rate < 1.0 for rate in rates) or len(set(rates)) != len(rates):
        raise ValueError("target_override_rates 必须唯一且位于 (0,1)")
    positive_gaps = [
        observation.policy_gap
        for observation in observations
        if observation.policy_gap > 0.0
    ]
    candidates: list[dict[str, Any]] = []
    for target_rate in rates:
        margin = choose_strict_margin(
            positive_gaps,
            eligible_decisions=len(observations),
            target_override_rate=target_rate,
        )
        summary = summarize_gate(observations, margin=margin, scope=scope)
        summary["target_override_rate"] = target_rate
        summary["passes"] = gate_passes(
            summary,
            minimum_overrides=minimum_overrides,
            minimum_groups=minimum_groups,
            maximum_override_rate=target_rate,
        )
        candidates.append(summary)
    passing = [candidate for candidate in candidates if candidate["passes"]]
    winner = (
        max(
            passing,
            key=lambda candidate: (
                float(candidate["hand_grouped_accuracy_gain_95pct_low"]),
                float(candidate["override_human_precision_wilson_95pct_low"]),
                -float(candidate["override_rate"]),
            ),
        )
        if passing
        else None
    )
    return winner, candidates


def _eligible_discard_alternatives(decision: TeacherDecision) -> tuple[int, ...]:
    if not is_eligible_human_teacher_discard_correction(decision):
        return ()
    reference = decision.reference_teacher_index
    assert reference is not None
    return tuple(
        index
        for index, action in enumerate(decision.legal_actions)
        if index != reference and action.kind == "discard"
    )


def scan_human_inputs(
    paths: Iterable[Path],
    *,
    policy: Any,
    batch_size: int,
) -> HumanGateScan:
    """Read an audited split and retain no state beyond in-memory scoring."""

    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    files = tuple(Path(path) for path in paths)
    audit = audit_local_human_trajectories(files, minimum_hands=1)
    if not audit["ready_for_manual_review"]:
        raise ValueError("人类 gate 输入未通过结构审计")
    if set(audit["recording_purposes"]) != {"training"}:
        raise ValueError("人类 gate 只允许 training 记录，evaluation 永远不得用于选模")
    if (
        audit["reference_teacher_summary"]["decisions_with_reference"]
        != sum(audit["human_action_counts"].values())
    ):
        raise ValueError("人类 gate 输入的 Teacher 参考标签覆盖不完整")

    observations: list[HumanGateObservation] = []
    groups: set[str] = set()
    pending: list[tuple[str, TeacherDecision, tuple[int, ...]]] = []

    def flush() -> None:
        if not pending:
            return
        decisions = [item[1] for item in pending]
        results = policy.policy_values_batch(decisions)
        if len(results) != len(pending):
            raise ValueError("模型批量输出数量与人类决策不匹配")
        for (group_id, decision, alternative_indices), (logits, _value) in zip(
            pending, results
        ):
            if len(logits) != len(decision.legal_actions):
                raise ValueError("模型 logits 与合法动作数量不匹配")
            reference = decision.reference_teacher_index
            assert reference is not None
            alternative = max(
                alternative_indices,
                key=lambda index: (float(logits[index]), -index),
            )
            gap = float(logits[alternative]) - float(logits[reference])
            if not math.isfinite(gap):
                raise ValueError("模型产生非有限人类 gate logit gap")
            observations.append(
                HumanGateObservation(
                    group_id=group_id,
                    policy_gap=gap,
                    teacher_matches_human=reference == decision.chosen_index,
                    alternative_matches_human=alternative == decision.chosen_index,
                )
            )
        pending.clear()

    for path in files:
        for trajectory in read_trajectory_jsonl(path):
            group_id = trajectory.split_group_id or trajectory.trajectory_id
            groups.add(group_id)
            for decision in trajectory.decisions:
                alternatives = _eligible_discard_alternatives(decision)
                if not alternatives:
                    continue
                pending.append((group_id, decision, alternatives))
                if len(pending) >= batch_size:
                    flush()
    flush()
    return HumanGateScan(
        observations=tuple(observations),
        trajectory_groups=frozenset(groups),
        structural_audit=audit,
    )


def scan_review_inputs(
    paths: Iterable[Path],
    *,
    policy: Any,
    batch_size: int,
) -> HumanGateScan:
    """Score confirmed review labels without requiring terminal outcomes."""

    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    files = tuple(Path(path) for path in paths)
    records = [record for path in files for record in read_review_labels(path)]
    structural_audit = audit_review_labels(
        records,
        minimum_confirmed_labels=1,
        minimum_confirmed_disagreements=1,
        minimum_groups=1,
    )
    # Disagreement coverage is a model-selection concern below, not a format
    # error.  Invalid or duplicate rows, uncertainty, and empty inputs fail
    # closed here.
    if (
        structural_audit["issues"]
        or structural_audit["duplicate_items"]
        or structural_audit["duplicate_labels"]
        or structural_audit["uncertain_labels"]
        or structural_audit["confirmed_labels"] != len(records)
        or not records
    ):
        raise ValueError("review gate 输入未通过 confirmed 结构审计")

    rows = [
        row
        for path in files
        for row in read_confirmed_review_decisions(path)
    ]
    observations: list[HumanGateObservation] = []
    groups = {group_id for group_id, _decision in rows}
    pending: list[tuple[str, TeacherDecision, tuple[int, ...]]] = []

    def flush() -> None:
        if not pending:
            return
        decisions = [item[1] for item in pending]
        results = policy.policy_values_batch(decisions)
        if len(results) != len(pending):
            raise ValueError("模型批量输出数量与 review 决策不匹配")
        for (group_id, decision, alternatives), (logits, _value) in zip(
            pending, results
        ):
            if len(logits) != len(decision.legal_actions):
                raise ValueError("模型 logits 与 review 合法动作数量不匹配")
            reference = decision.reference_teacher_index
            assert reference is not None
            alternative = max(
                alternatives,
                key=lambda index: (float(logits[index]), -index),
            )
            gap = float(logits[alternative]) - float(logits[reference])
            if not math.isfinite(gap):
                raise ValueError("模型产生非有限 review gate logit gap")
            observations.append(
                HumanGateObservation(
                    group_id=group_id,
                    policy_gap=gap,
                    teacher_matches_human=reference == decision.chosen_index,
                    alternative_matches_human=alternative == decision.chosen_index,
                )
            )
        pending.clear()

    for group_id, decision in rows:
        alternatives = _eligible_discard_alternatives(decision)
        if not alternatives:
            raise ValueError("review gate 出现 scope 外决策")
        pending.append((group_id, decision, alternatives))
        if len(pending) >= batch_size:
            flush()
    flush()
    return HumanGateScan(
        observations=tuple(observations),
        trajectory_groups=frozenset(groups),
        structural_audit=structural_audit,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--validation-input", type=Path, action="append", required=True)
    parser.add_argument("--test-input", type=Path, action="append", required=True)
    parser.add_argument("--target-override-rate", type=float, action="append")
    parser.add_argument("--minimum-validation-overrides", type=int, default=20)
    parser.add_argument("--minimum-validation-groups", type=int, default=20)
    parser.add_argument("--minimum-test-overrides", type=int, default=20)
    parser.add_argument("--minimum-test-groups", type=int, default=20)
    parser.add_argument("--maximum-test-override-rate", type=float, default=0.075)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rates = tuple(args.target_override_rate or DEFAULT_TARGET_OVERRIDE_RATES)
    if args.minimum_test_overrides <= 0 or args.minimum_test_groups < 2:
        raise ValueError("test 最少覆盖数必须为正且 groups 至少为 2")
    if not 0.0 < args.maximum_test_override_rate < 1.0:
        raise ValueError("maximum-test-override-rate 必须位于 (0,1)")
    actual_sha = _sha256(args.checkpoint)
    if actual_sha != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 与冻结身份不匹配")
    policy = TorchPolicyValueAgent.load(args.checkpoint, device=args.device)

    validation = scan_human_inputs(
        args.validation_input,
        policy=policy,
        batch_size=args.batch_size,
    )
    winner, candidates = select_validation_gate(
        validation.observations,
        target_override_rates=rates,
        minimum_overrides=args.minimum_validation_overrides,
        minimum_groups=args.minimum_validation_groups,
    )
    payload: dict[str, Any] = {
        "status": "validation_gate_failed_test_unread",
        "checkpoint": {
            "sha256": actual_sha,
            "architecture": policy.architecture,
            "feature_version": policy.feature_version,
            "hidden_size": policy.hidden_size,
            "bytes": args.checkpoint.stat().st_size,
        },
        "protocol": {
            "target_override_rates": list(rates),
            "minimum_validation_overrides": args.minimum_validation_overrides,
            "minimum_validation_groups": args.minimum_validation_groups,
            "minimum_test_overrides": args.minimum_test_overrides,
            "minimum_test_groups": args.minimum_test_groups,
            "maximum_test_override_rate": args.maximum_test_override_rate,
            "allowed_teacher_action_kinds": ["discard"],
            "allowed_alternative_action_kinds": ["discard"],
            "test_read_only_after_validation_pass": True,
        },
        "validation": {
            "structural_hands": validation.structural_audit["valid_hands"],
            "trajectory_groups": len(validation.trajectory_groups),
            "candidates": candidates,
            "winner": winner,
        },
        "test": {"status": "unread"},
        "privacy": (
            "aggregate_only_no_human_path_state_hand_history_action_face_wall_"
            "opponent_hand_seed_session_id_or_rng_exported"
        ),
    }
    if winner is not None:
        test = scan_human_inputs(
            args.test_input,
            policy=policy,
            batch_size=args.batch_size,
        )
        overlap = validation.trajectory_groups & test.trajectory_groups
        if overlap:
            raise ValueError("validation/test 存在完整牌局 group 重叠")
        test_summary = summarize_gate(
            test.observations,
            margin=float(winner["strict_margin"]),
        )
        test_passes = gate_passes(
            test_summary,
            minimum_overrides=args.minimum_test_overrides,
            minimum_groups=args.minimum_test_groups,
            maximum_override_rate=args.maximum_test_override_rate,
        )
        payload["test"] = {
            "status": "audited",
            "structural_hands": test.structural_audit["valid_hands"],
            "trajectory_groups": len(test.trajectory_groups),
            "validation_group_overlap": 0,
            "summary": test_summary,
            "passes": test_passes,
        }
        payload["status"] = (
            "test_gate_passed_ready_for_100_wall_teacher_screen"
            if test_passes
            else "test_gate_rejected"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

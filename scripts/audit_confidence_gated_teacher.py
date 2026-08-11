#!/usr/bin/env python3
"""Calibrate a small-policy Teacher override rate on safe v4 trajectories.

The calibration target is behavioral coverage, not strength and not Teacher
accuracy.  On ordinary discard decisions only, it measures the compact
policy's best non-Teacher logit advantage.  A strict margin is selected from
the calibration split so no more than the requested fraction of eligible
decisions would override Teacher.  The untouched test split only checks that
coverage remains in a prespecified band.

The report is aggregate-only: no hand, history, action trace, wall, opponent
concealed tile or trajectory seed is written.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import TeacherDecision, TrainingTrajectory


def choose_strict_margin(
    positive_gaps: Sequence[float],
    *,
    eligible_decisions: int,
    target_override_rate: float,
) -> float | None:
    """Choose a strict ``gap > margin`` cutoff without exceeding coverage."""

    if eligible_decisions <= 0:
        return None
    if not 0.0 < target_override_rate < 1.0:
        raise ValueError("target_override_rate 必须在 0 和 1 之间")
    gaps = sorted(
        (float(gap) for gap in positive_gaps if math.isfinite(gap) and gap > 0.0),
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
    # A tied boundary cannot be split without another feature.  Strict
    # comparison drops the whole tie group and remains below the cap.
    return upper


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _eligible(decision: TeacherDecision) -> bool:
    state = decision.state
    return bool(
        state.get("phase") == "discard"
        and decision.chosen_action.kind == "discard"
        and len(decision.legal_actions) >= 2
        and state.get("tour") is None
        and not state.get("gold_locked", False)
    )


@dataclass
class CoverageScan:
    trajectories: int = 0
    decisions: int = 0
    eligible_decisions: int = 0
    positive_gaps: list[float] = field(default_factory=list)
    alternative_kind_counts: Counter[str] = field(default_factory=Counter)

    def add_batch(
        self,
        decisions: Sequence[TeacherDecision],
        policy_results: Sequence[tuple[list[float], float]],
    ) -> None:
        if len(decisions) != len(policy_results):
            raise ValueError("批量决策与模型输出数量不匹配")
        for decision, (logits, _value) in zip(decisions, policy_results):
            if len(logits) != len(decision.legal_actions):
                raise ValueError("模型 logits 与合法动作数量不匹配")
            self.eligible_decisions += 1
            teacher_index = decision.chosen_index
            alternative_index = max(
                (
                    index
                    for index in range(len(decision.legal_actions))
                    if index != teacher_index
                ),
                key=lambda index: (float(logits[index]), -index),
            )
            gap = float(logits[alternative_index]) - float(logits[teacher_index])
            if gap <= 0.0:
                continue
            self.positive_gaps.append(gap)
            self.alternative_kind_counts[
                decision.legal_actions[alternative_index].kind
            ] += 1

    def payload(self, *, margin: float | None) -> dict[str, Any]:
        override_count = (
            sum(gap > margin for gap in self.positive_gaps)
            if margin is not None
            else 0
        )
        return {
            "teacher_trajectories": self.trajectories,
            "decisions": self.decisions,
            "eligible_ordinary_discard_decisions": self.eligible_decisions,
            "positive_policy_disagreement_count": len(self.positive_gaps),
            "positive_policy_disagreement_rate": (
                len(self.positive_gaps) / self.eligible_decisions
                if self.eligible_decisions
                else None
            ),
            "strict_margin": margin,
            "override_count": override_count,
            "override_rate": (
                override_count / self.eligible_decisions
                if self.eligible_decisions
                else None
            ),
            "positive_gap_quantiles": {
                "p50": _quantile(self.positive_gaps, 0.50),
                "p75": _quantile(self.positive_gaps, 0.75),
                "p90": _quantile(self.positive_gaps, 0.90),
                "p95": _quantile(self.positive_gaps, 0.95),
                "p98": _quantile(self.positive_gaps, 0.98),
                "p99": _quantile(self.positive_gaps, 0.99),
                "max": max(self.positive_gaps) if self.positive_gaps else None,
            },
            "positive_alternative_kind_counts": dict(
                sorted(self.alternative_kind_counts.items())
            ),
        }


def scan_inputs(
    paths: Iterable[Path],
    *,
    policy: TorchPolicyValueAgent,
    profile: str,
    maximum_teacher_trajectories: int,
    batch_size: int,
) -> CoverageScan:
    scan = CoverageScan()
    pending: list[TeacherDecision] = []

    def flush() -> None:
        if not pending:
            return
        scan.add_batch(pending, policy.policy_values_batch(pending))
        pending.clear()

    stop = False
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                trajectory = TrainingTrajectory.from_payload(json.loads(line))
                if trajectory.profile != profile:
                    continue
                if trajectory.source_metadata.get("collector") != "teacher_self_play":
                    continue
                scan.trajectories += 1
                scan.decisions += len(trajectory.decisions)
                for decision in trajectory.decisions:
                    if not _eligible(decision):
                        continue
                    pending.append(decision)
                    if len(pending) >= batch_size:
                        flush()
                if scan.trajectories >= maximum_teacher_trajectories:
                    stop = True
                    break
        if stop:
            break
    flush()
    return scan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-input", action="append", type=Path, required=True)
    parser.add_argument("--test-input", action="append", type=Path, required=True)
    parser.add_argument("--profile", choices=("classic",), default="classic")
    parser.add_argument("--maximum-teacher-trajectories", type=int, default=1_000)
    parser.add_argument("--target-override-rate", type=float, default=0.02)
    parser.add_argument("--minimum-test-override-rate", type=float, default=0.005)
    parser.add_argument("--maximum-test-override-rate", type=float, default=0.035)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.maximum_teacher_trajectories <= 0 or args.batch_size <= 0:
        raise ValueError("maximum-teacher-trajectories 与 batch-size 必须为正数")
    if not 0.0 < args.target_override_rate < 1.0:
        raise ValueError("target-override-rate 必须在 0 和 1 之间")
    if not (
        0.0
        <= args.minimum_test_override_rate
        <= args.maximum_test_override_rate
        < 1.0
    ):
        raise ValueError("test override rate 范围无效")

    policy = TorchPolicyValueAgent.load(args.checkpoint, device=args.device)
    calibration = scan_inputs(
        args.calibration_input,
        policy=policy,
        profile=args.profile,
        maximum_teacher_trajectories=args.maximum_teacher_trajectories,
        batch_size=args.batch_size,
    )
    margin = choose_strict_margin(
        calibration.positive_gaps,
        eligible_decisions=calibration.eligible_decisions,
        target_override_rate=args.target_override_rate,
    )
    test = scan_inputs(
        args.test_input,
        policy=policy,
        profile=args.profile,
        maximum_teacher_trajectories=args.maximum_teacher_trajectories,
        batch_size=args.batch_size,
    )
    calibration_payload = calibration.payload(margin=margin)
    test_payload = test.payload(margin=margin)
    test_rate = test_payload["override_rate"]
    passes = bool(
        margin is not None
        and isinstance(test_rate, float)
        and args.minimum_test_override_rate
        <= test_rate
        <= args.maximum_test_override_rate
    )
    payload = {
        "status": (
            "coverage_calibrated_ready_for_preregistered_strength_screen"
            if passes
            else "coverage_stability_gate_failed"
        ),
        "profile": args.profile,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _sha256(args.checkpoint),
            "architecture": policy.architecture,
            "feature_version": policy.feature_version,
            "hidden_size": policy.hidden_size,
            "bytes": args.checkpoint.stat().st_size,
        },
        "scope": {
            "allowed_teacher_action_kinds": ["discard"],
            "ordinary_non_tour_non_gold_lock_only": True,
            "response_win_kong_and_tour_fallback": "heuristic_teacher",
        },
        "calibration_rule": {
            "kind": "strict_policy_logit_advantage_quantile",
            "target_override_rate": args.target_override_rate,
            "selected_margin": margin,
            "uses_strength_outcomes": False,
        },
        "test_coverage_gate": {
            "minimum": args.minimum_test_override_rate,
            "maximum": args.maximum_test_override_rate,
            "passes": passes,
        },
        "calibration": calibration_payload,
        "test": test_payload,
        "maximum_teacher_trajectories_per_split": args.maximum_teacher_trajectories,
        "device": args.device,
        "privacy": "aggregate_only_no_hand_history_action_trace_wall_opponent_hand_or_seed_exported",
        "interpretation": "coverage_only_not_strength_or_human_level_evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

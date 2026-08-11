#!/usr/bin/env python3
"""Select/test the fixed pass/chi/pong review residual without test peeking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.select_human_teacher_residual_gate import (
    HumanGateObservation,
    HumanGateScan,
    gate_passes,
    select_validation_gate,
    summarize_gate,
)
from xiamen_mahjong.response_review import (
    audit_response_review_labels,
    read_confirmed_response_review_decisions,
    read_response_review_labels,
)
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import TeacherDecision


TARGET_OVERRIDE_RATES = (0.10, 0.20, 0.30)
MINIMUM_VALIDATION_OVERRIDES = 5
MINIMUM_VALIDATION_GROUPS = 15
MINIMUM_TEST_OVERRIDES = 5
MINIMUM_TEST_GROUPS = 15
MAXIMUM_TEST_OVERRIDE_RATE = 0.35
RESPONSE_SCOPE = "ordinary_pass_chi_pong_alternative_only_grouped"
_RESPONSE_KINDS = frozenset({"pass", "chi", "pong"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _response_alternatives(decision: TeacherDecision) -> tuple[int, ...]:
    reference = decision.reference_teacher_index
    if (
        decision.profile != "classic"
        or decision.state.get("phase") != "response"
        or reference is None
        or not {action.kind for action in decision.legal_actions}.issubset(
            _RESPONSE_KINDS
        )
    ):
        return ()
    return tuple(
        index
        for index, action in enumerate(decision.legal_actions)
        if index != reference and action.kind in _RESPONSE_KINDS
    )


def scan_response_review_inputs(
    paths: Iterable[Path],
    *,
    policy: Any,
    batch_size: int,
) -> HumanGateScan:
    """Score confirmed response labels, retaining aggregate-safe observations."""

    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    files = tuple(Path(path) for path in paths)
    records = [
        record for path in files for record in read_response_review_labels(path)
    ]
    structural_audit = audit_response_review_labels(
        records,
        minimum_confirmed_labels=1,
        minimum_confirmed_disagreements=1,
        minimum_groups=1,
    )
    if (
        structural_audit["issues"]
        or structural_audit["duplicate_items"]
        or structural_audit["duplicate_labels"]
        or structural_audit["uncertain_labels"]
        or structural_audit["confirmed_labels"] != len(records)
        or not records
    ):
        raise ValueError("response review gate 输入未通过 confirmed 结构审计")

    rows = [
        row
        for path in files
        for row in read_confirmed_response_review_decisions(path)
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
            raise ValueError("模型批量输出数量与 response review 决策不匹配")
        for (group_id, decision, alternatives), (logits, _value) in zip(
            pending, results
        ):
            if len(logits) != len(decision.legal_actions):
                raise ValueError("模型 logits 与 response review 合法动作数量不匹配")
            reference = decision.reference_teacher_index
            assert reference is not None
            alternative = max(
                alternatives,
                key=lambda index: (float(logits[index]), -index),
            )
            gap = float(logits[alternative]) - float(logits[reference])
            if not math.isfinite(gap):
                raise ValueError("模型产生非有限 response review logit gap")
            observations.append(
                HumanGateObservation(
                    group_id=group_id,
                    policy_gap=gap,
                    teacher_matches_human=reference == decision.chosen_index,
                    alternative_matches_human=(
                        alternative == decision.chosen_index
                    ),
                )
            )
        pending.clear()

    for group_id, decision in rows:
        alternatives = _response_alternatives(decision)
        if not alternatives:
            raise ValueError("response review gate 出现 scope 外决策")
        pending.append((group_id, decision, alternatives))
        if len(pending) >= batch_size:
            flush()
    flush()
    return HumanGateScan(
        observations=tuple(observations),
        trajectory_groups=frozenset(groups),
        structural_audit=structural_audit,
    )


def run_gate(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    validation_input: Path,
    test_input: Path,
    output: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    actual_sha = _sha256(checkpoint)
    if actual_sha != checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 与冻结身份不匹配")
    policy = TorchPolicyValueAgent.load(checkpoint, device=device)
    validation = scan_response_review_inputs(
        [validation_input], policy=policy, batch_size=batch_size
    )
    winner, candidates = select_validation_gate(
        validation.observations,
        target_override_rates=TARGET_OVERRIDE_RATES,
        minimum_overrides=MINIMUM_VALIDATION_OVERRIDES,
        minimum_groups=MINIMUM_VALIDATION_GROUPS,
        scope=RESPONSE_SCOPE,
    )
    payload: dict[str, Any] = {
        "status": "response_review_validation_gate_failed_test_unread",
        "checkpoint": {
            "sha256": actual_sha,
            "architecture": policy.architecture,
            "feature_version": policy.feature_version,
            "hidden_size": policy.hidden_size,
            "bytes": checkpoint.stat().st_size,
        },
        "protocol": {
            "source": "local_human_response_review_opt_in",
            "target_override_rates": list(TARGET_OVERRIDE_RATES),
            "minimum_validation_overrides": MINIMUM_VALIDATION_OVERRIDES,
            "minimum_validation_groups": MINIMUM_VALIDATION_GROUPS,
            "minimum_test_overrides": MINIMUM_TEST_OVERRIDES,
            "minimum_test_groups": MINIMUM_TEST_GROUPS,
            "maximum_test_override_rate": MAXIMUM_TEST_OVERRIDE_RATE,
            "test_read_only_after_validation_pass": True,
            "allowed_teacher_action_kinds": sorted(_RESPONSE_KINDS),
            "allowed_alternative_action_kinds": sorted(_RESPONSE_KINDS),
            "scope": RESPONSE_SCOPE,
        },
        "validation": {
            "confirmed_labels": validation.structural_audit["confirmed_labels"],
            "review_groups": len(validation.trajectory_groups),
            "candidates": candidates,
            "winner": winner,
        },
        "test": {"status": "unread"},
        "privacy": "aggregate_only_no_review_path_state_hand_history_action_face_or_group_id",
    }
    if winner is not None:
        test = scan_response_review_inputs(
            [test_input], policy=policy, batch_size=batch_size
        )
        overlap = validation.trajectory_groups & test.trajectory_groups
        if overlap:
            raise ValueError("response review validation/test 存在物理牌局 group 重叠")
        summary = summarize_gate(
            test.observations,
            margin=float(winner["strict_margin"]),
            scope=RESPONSE_SCOPE,
        )
        passed = gate_passes(
            summary,
            minimum_overrides=MINIMUM_TEST_OVERRIDES,
            minimum_groups=MINIMUM_TEST_GROUPS,
            maximum_override_rate=MAXIMUM_TEST_OVERRIDE_RATE,
        )
        payload["test"] = {
            "status": "audited",
            "confirmed_labels": test.structural_audit["confirmed_labels"],
            "review_groups": len(test.trajectory_groups),
            "validation_group_overlap": 0,
            "summary": summary,
            "passes": passed,
        }
        payload["status"] = (
            "response_review_test_gate_passed_ready_for_100_wall_teacher_screen"
            if passed
            else "response_review_test_gate_rejected"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--validation-input", type=Path, required=True)
    parser.add_argument("--test-input", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_gate(
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        validation_input=args.validation_input,
        test_input=args.test_input,
        output=args.output,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

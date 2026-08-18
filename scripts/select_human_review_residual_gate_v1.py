#!/usr/bin/env python3
"""Select and test the fixed review-trained discard gate without test peeking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.select_human_teacher_residual_gate import (
    gate_passes,
    scan_review_inputs,
    select_validation_gate,
    summarize_gate,
)
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent


TARGET_OVERRIDE_RATES = (0.10, 0.20, 0.30)
MINIMUM_VALIDATION_OVERRIDES = 8
MINIMUM_VALIDATION_GROUPS = 20
MINIMUM_TEST_OVERRIDES = 8
MINIMUM_TEST_GROUPS = 20
MAXIMUM_TEST_OVERRIDE_RATE = 0.35


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    validation = scan_review_inputs(
        [validation_input], policy=policy, batch_size=batch_size
    )
    winner, candidates = select_validation_gate(
        validation.observations,
        target_override_rates=TARGET_OVERRIDE_RATES,
        minimum_overrides=MINIMUM_VALIDATION_OVERRIDES,
        minimum_groups=MINIMUM_VALIDATION_GROUPS,
    )
    payload: dict[str, Any] = {
        "status": "validation_gate_failed_review_test_unread",
        "checkpoint": {
            "sha256": actual_sha,
            "architecture": policy.architecture,
            "feature_version": policy.feature_version,
            "hidden_size": policy.hidden_size,
            "bytes": checkpoint.stat().st_size,
        },
        "protocol": {
            "source": "local_human_review_opt_in",
            "target_override_rates": list(TARGET_OVERRIDE_RATES),
            "minimum_validation_overrides": MINIMUM_VALIDATION_OVERRIDES,
            "minimum_validation_groups": MINIMUM_VALIDATION_GROUPS,
            "minimum_test_overrides": MINIMUM_TEST_OVERRIDES,
            "minimum_test_groups": MINIMUM_TEST_GROUPS,
            "maximum_test_override_rate": MAXIMUM_TEST_OVERRIDE_RATE,
            "test_read_only_after_validation_pass": True,
            "allowed_teacher_action_kinds": ["discard"],
            "allowed_alternative_action_kinds": ["discard"],
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
        test = scan_review_inputs([test_input], policy=policy, batch_size=batch_size)
        overlap = validation.trajectory_groups & test.trajectory_groups
        if overlap:
            raise ValueError("review validation/test 存在物理牌局 group 重叠")
        summary = summarize_gate(
            test.observations, margin=float(winner["strict_margin"])
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
            "review_test_gate_passed_ready_for_100_wall_teacher_screen"
            if passed
            else "review_test_gate_rejected"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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

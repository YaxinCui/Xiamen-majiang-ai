#!/usr/bin/env python3
"""Train the fixed CPU-small-MLP residual from confirmed review labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.human_review import (
    audit_review_labels,
    read_confirmed_review_decisions,
    read_review_labels,
)


TEACHER_DATA_DIR = Path("artifacts/trajectory-contract-audit-classic-v4")
FRESH_SEED = 202623200
HIDDEN_SIZE = 128
EPOCHS = 12
HUMAN_REVIEW_WEIGHT = 1.0
HUMAN_DISAGREEMENT_WEIGHT = 2.0
MINIMUM_TRAIN_VALIDATION_LABELS = 350
MINIMUM_TRAIN_VALIDATION_DISAGREEMENTS = 30
MINIMUM_TRAIN_VALIDATION_GROUPS = 80
MINIMUM_SELECTION_DECISIONS = 35


def _review_path(directory: Path, split: str) -> Path:
    return directory / f"{split}.review.jsonl"


def _teacher_path(split: str) -> Path:
    return TEACHER_DATA_DIR / f"{split}.trajectories.jsonl"


def validate_inputs(review_split_dir: Path) -> dict[str, object]:
    teacher_paths = {split: _teacher_path(split) for split in ("train", "validation", "test")}
    review_paths = {split: _review_path(review_split_dir, split) for split in ("train", "validation", "test")}
    missing = [
        str(path)
        for path in (*teacher_paths.values(), *review_paths.values())
        if not path.is_file()
    ]
    if missing:
        raise ValueError("固定 review 训练输入缺失：" + ", ".join(missing))

    # Only check test existence.  Its bytes are deliberately unopened until a
    # validation-selected gate is frozen.
    train_records = read_review_labels(review_paths["train"])
    validation_records = read_review_labels(review_paths["validation"])
    audit = audit_review_labels(
        [*train_records, *validation_records],
        minimum_confirmed_labels=MINIMUM_TRAIN_VALIDATION_LABELS,
        minimum_confirmed_disagreements=MINIMUM_TRAIN_VALIDATION_DISAGREEMENTS,
        minimum_groups=MINIMUM_TRAIN_VALIDATION_GROUPS,
    )
    if not audit["ready_for_manual_training_review"]:
        raise ValueError(
            "review train/validation 未通过 fixed v1 门槛："
            + ", ".join(audit["gate_reasons"])
        )
    train_rows = read_confirmed_review_decisions(review_paths["train"])
    validation_rows = read_confirmed_review_decisions(review_paths["validation"])
    train_groups = {group for group, _decision in train_rows}
    validation_groups = {group for group, _decision in validation_rows}
    if train_groups & validation_groups:
        raise ValueError("review train/validation 存在物理牌局 group 重叠")
    return {
        "teacher_paths": teacher_paths,
        "review_paths": review_paths,
        "aggregate_audit": audit,
        "train_validation_group_overlap": 0,
        "review_test_contract": "exists_but_bytes_unopened_reserved_for_gate",
    }


def build_training_command(
    validated: dict[str, object], *, output_dir: Path, device: str
) -> list[str]:
    teacher_paths = validated["teacher_paths"]
    review_paths = validated["review_paths"]
    assert isinstance(teacher_paths, dict)
    assert isinstance(review_paths, dict)
    return [
        sys.executable,
        str(ROOT / "scripts/train_policy_value.py"),
        "--train", str(teacher_paths["train"]),
        "--validation", str(teacher_paths["validation"]),
        "--test", str(teacher_paths["test"]),
        "--human-review-train", str(review_paths["train"]),
        "--human-review-validation", str(review_paths["validation"]),
        "--allow-local-human-review-data",
        "--human-review-weight", str(HUMAN_REVIEW_WEIGHT),
        "--human-teacher-disagreement-weight", str(HUMAN_DISAGREEMENT_WEIGHT),
        "--architecture", "candidate_mlp",
        "--feature-version", "3",
        "--hidden-size", str(HIDDEN_SIZE),
        "--epochs", str(EPOCHS),
        "--batch-size", "256",
        "--learning-rate", "0.001",
        "--weight-decay", "0.0001",
        "--value-weight", "0",
        "--stream-train-shards",
        "--stream-shuffle-buffer", "4096",
        "--checkpoint-selection-source", "local_human_review_opt_in",
        "--checkpoint-selection-metric", "policy_loss",
        "--minimum-selection-decisions", str(MINIMUM_SELECTION_DECISIONS),
        "--device", device,
        "--seed", str(FRESH_SEED),
        "--output-dir", str(output_dir),
    ]


def _safe_summary(validated: dict[str, object], output_dir: Path) -> dict[str, object]:
    audit = validated["aggregate_audit"]
    assert isinstance(audit, dict)
    return {
        "status": "fixed_human_review_residual_v1_inputs_ready",
        "protocol": {
            "architecture": "candidate_mlp",
            "feature_version": 3,
            "hidden_size": HIDDEN_SIZE,
            "epochs": EPOCHS,
            "fresh_seed": FRESH_SEED,
            "human_review_weight": HUMAN_REVIEW_WEIGHT,
            "human_teacher_disagreement_weight": HUMAN_DISAGREEMENT_WEIGHT,
            "checkpoint_selection_source": "local_human_review_opt_in",
            "review_test": validated["review_test_contract"],
        },
        "review_train_validation": {
            "confirmed_labels": audit["confirmed_labels"],
            "teacher_disagreements": audit["confirmed_teacher_disagreements"],
            "groups": audit["confirmed_review_groups"],
            "group_overlap": validated["train_validation_group_overlap"],
        },
        "output_dir": str(output_dir),
        "privacy": "aggregate_only_no_review_path_state_hand_history_or_action_face",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-split-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/human-review-residual-v1")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validated = validate_inputs(args.review_split_dir)
    summary = _safe_summary(validated, args.output_dir)
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise ValueError("fixed review v1 输出已存在且非空，拒绝覆盖或续训")
    subprocess.run(
        build_training_command(validated, output_dir=args.output_dir, device=args.device),
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()

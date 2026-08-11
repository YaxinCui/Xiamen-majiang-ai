#!/usr/bin/env python3
"""Run the fixed fresh-MLP human-correction v1 training protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.human_data import audit_local_human_teacher_corrections
from xiamen_mahjong.training import read_trajectory_jsonl


TEACHER_DATA_DIR = Path("artifacts/trajectory-contract-audit-classic-v4")
FRESH_SEED = 202623100
HIDDEN_SIZE = 128
EPOCHS = 12
HUMAN_WEIGHT = 1.0
HUMAN_DISAGREEMENT_WEIGHT = 2.0
MINIMUM_TRAIN_VALIDATION_HANDS = 80
MINIMUM_TRAIN_VALIDATION_REFERENCE_DECISIONS = 400
MINIMUM_TRAIN_VALIDATION_DISAGREEMENTS = 40
MINIMUM_SELECTION_DECISIONS = 40


def _split_path(directory: Path, split: str) -> Path:
    return directory / f"{split}.trajectories.jsonl"


def _group_ids(path: Path) -> set[str]:
    return {
        trajectory.split_group_id or trajectory.trajectory_id
        for trajectory in read_trajectory_jsonl(path)
    }


def validate_inputs(human_split_dir: Path) -> dict[str, object]:
    """Audit train/validation while leaving human test content unopened."""

    teacher_paths = {
        split: _split_path(TEACHER_DATA_DIR, split)
        for split in ("train", "validation", "test")
    }
    human_paths = {
        split: _split_path(human_split_dir, split)
        for split in ("train", "validation", "test")
    }
    missing = [
        str(path)
        for path in (*teacher_paths.values(), *human_paths.values())
        if not path.is_file()
    ]
    if missing:
        raise ValueError("固定训练输入缺失：" + ", ".join(missing))

    # Deliberately do not pass human test here. Existence was checked above,
    # but its bytes remain unopened until the validation gate has a winner.
    train_validation = (human_paths["train"], human_paths["validation"])
    audit = audit_local_human_teacher_corrections(
        train_validation,
        minimum_hands=MINIMUM_TRAIN_VALIDATION_HANDS,
        minimum_reference_decisions=MINIMUM_TRAIN_VALIDATION_REFERENCE_DECISIONS,
        minimum_disagreements=MINIMUM_TRAIN_VALIDATION_DISAGREEMENTS,
    )
    if not audit["ready_for_teacher_residual_experiment"]:
        raise ValueError(
            "真人 train/validation 未通过 fixed v1 门槛："
            + ", ".join(audit["gate_reasons"])
        )
    train_groups = _group_ids(human_paths["train"])
    validation_groups = _group_ids(human_paths["validation"])
    if train_groups & validation_groups:
        raise ValueError("真人 train/validation 存在完整牌局 group 重叠")
    return {
        "teacher_paths": teacher_paths,
        "human_paths": human_paths,
        "aggregate_audit": audit,
        "train_validation_group_overlap": 0,
        "human_test_contract": "exists_but_bytes_unopened_reserved_for_gate",
    }


def build_training_command(
    validated: dict[str, object],
    *,
    output_dir: Path,
    device: str,
) -> list[str]:
    teacher_paths = validated["teacher_paths"]
    human_paths = validated["human_paths"]
    assert isinstance(teacher_paths, dict)
    assert isinstance(human_paths, dict)
    return [
        sys.executable,
        str(ROOT / "scripts/train_policy_value.py"),
        "--train",
        str(teacher_paths["train"]),
        "--additional-train",
        str(human_paths["train"]),
        "--validation",
        str(teacher_paths["validation"]),
        "--additional-validation",
        str(human_paths["validation"]),
        "--test",
        str(teacher_paths["test"]),
        "--architecture",
        "candidate_mlp",
        "--feature-version",
        "3",
        "--hidden-size",
        str(HIDDEN_SIZE),
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        "256",
        "--learning-rate",
        "0.001",
        "--weight-decay",
        "0.0001",
        "--value-weight",
        "0",
        "--stream-train-shards",
        "--stream-shuffle-buffer",
        "4096",
        "--allow-local-human-data",
        "--reserve-local-human-test-for-gate",
        "--human-minimum-hands",
        str(MINIMUM_TRAIN_VALIDATION_HANDS),
        "--human-weight",
        str(HUMAN_WEIGHT),
        "--human-teacher-disagreement-weight",
        str(HUMAN_DISAGREEMENT_WEIGHT),
        "--human-discard-corrections-only",
        "--checkpoint-selection-source",
        "local_human_opt_in",
        "--checkpoint-selection-metric",
        "policy_loss",
        "--minimum-selection-decisions",
        str(MINIMUM_SELECTION_DECISIONS),
        "--device",
        device,
        "--seed",
        str(FRESH_SEED),
        "--output-dir",
        str(output_dir),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-split-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/human-teacher-residual-v1"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只执行聚合审计并打印脱敏协议摘要，不启动训练",
    )
    return parser.parse_args()


def _safe_summary(validated: dict[str, object], *, output_dir: Path) -> dict[str, object]:
    audit = validated["aggregate_audit"]
    assert isinstance(audit, dict)
    return {
        "status": "fixed_human_teacher_residual_v1_inputs_ready",
        "protocol": {
            "fresh_seed": FRESH_SEED,
            "architecture": "candidate_mlp",
            "feature_version": 3,
            "hidden_size": HIDDEN_SIZE,
            "epochs": EPOCHS,
            "human_weight": HUMAN_WEIGHT,
            "human_teacher_disagreement_weight": HUMAN_DISAGREEMENT_WEIGHT,
            "checkpoint_selection_source": "local_human_opt_in",
            "human_test": validated["human_test_contract"],
        },
        "human_train_validation": {
            "hands": audit["audit"]["valid_hands"],
            "reference_decisions": audit["correction_summary"]["reference_decisions"],
            "teacher_disagreements": audit["correction_summary"]["teacher_disagreements"],
            "group_overlap": validated["train_validation_group_overlap"],
        },
        "output_dir": str(output_dir),
        "privacy": "no_human_input_path_state_hand_history_action_face_or_session_exported",
    }


def main() -> None:
    args = parse_args()
    validated = validate_inputs(args.human_split_dir)
    summary = _safe_summary(validated, output_dir=args.output_dir)
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise ValueError("输出路径已存在且不是空目录；fixed v1 不允许覆盖或续训")
    command = build_training_command(
        validated,
        output_dir=args.output_dir,
        device=args.device,
    )
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

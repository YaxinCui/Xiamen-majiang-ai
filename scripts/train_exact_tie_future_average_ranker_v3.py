#!/usr/bin/env python3
"""Train the fixed structured ranker on future-wall-averaged tie labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_exact_tie_rollout_ranker_v1 import (
    _train_member,
    evaluate_unanimous_future_average,
)
from xiamen_mahjong.exact_tie_rollout import (
    exact_tie_distinct_decision_count,
    exact_tie_future_averaged_pairwise_examples,
    read_exact_tie_rollout_records,
    split_exact_tie_rollout_records_by_group,
)


FEATURE_VERSION = 4
HIDDEN_SIZE = 64
ENSEMBLE_SEEDS = (202643100, 202643101, 202643102, 202643103, 202643104)
EPOCHS = 40
INNER_SPLIT_SALT = "exact-tie-future-average-ranker-v3-inner"
MINIMUM_TRAIN_GROUPS = 300
MINIMUM_VALIDATION_GROUPS = 75
MINIMUM_VALIDATION_DECISIONS = 275
MINIMUM_VALIDATION_OVERRIDES = 20
REQUIRED_FUTURE_PERMUTATIONS = 4


def _path(directory: Path, split: str) -> Path:
    return directory / f"{split}.exact-tie-rollout.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise ValueError("exact-tie future-average ranker 输出目录已存在且非空")
    paths = {split: _path(args.split_dir, split) for split in ("train", "validation", "test")}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("exact-tie future-average split 不完整")
    test_sha256 = _sha256(paths["test"])
    train_records = read_exact_tie_rollout_records(paths["train"])
    train_groups = {record.split_group_id for record in train_records}
    if len(train_groups) < MINIMUM_TRAIN_GROUPS:
        raise ValueError("exact-tie v3 train group 数不足")
    inner = split_exact_tie_rollout_records_by_group(
        train_records,
        split_salt=INNER_SPLIT_SALT,
        train_fraction=0.9,
        validation_fraction=0.05,
    )
    fit_records = inner["train"]
    epoch_records = [*inner["validation"], *inner["test"]]
    fit_examples = exact_tie_future_averaged_pairwise_examples(fit_records)
    epoch_examples = exact_tie_future_averaged_pairwise_examples(epoch_records)
    policies = []
    members = []
    for seed in ENSEMBLE_SEEDS:
        policy, report = _train_member(
            fit_examples,
            epoch_examples,
            seed=seed,
            device=args.device,
            feature_version=FEATURE_VERSION,
            hidden_size=HIDDEN_SIZE,
            epochs=EPOCHS,
        )
        policies.append(policy)
        members.append(report)

    validation_records = read_exact_tie_rollout_records(paths["validation"])
    validation_groups = {record.split_group_id for record in validation_records}
    if train_groups & validation_groups:
        raise ValueError("exact-tie v3 train/validation group 泄漏")
    if len(validation_groups) < MINIMUM_VALIDATION_GROUPS:
        raise ValueError("exact-tie v3 validation group 数不足")
    if exact_tie_distinct_decision_count(validation_records) < MINIMUM_VALIDATION_DECISIONS:
        raise ValueError("exact-tie v3 validation decision 数不足")
    validation = evaluate_unanimous_future_average(policies, validation_records)
    reasons = []
    if (
        validation["minimum_future_permutations"] != REQUIRED_FUTURE_PERMUTATIONS
        or validation["maximum_future_permutations"] != REQUIRED_FUTURE_PERMUTATIONS
    ):
        reasons.append("future_permutation_count_mismatch")
    if validation["overrides"] < MINIMUM_VALIDATION_OVERRIDES:
        reasons.append("insufficient_validation_overrides")
    if validation["paired_wall_delta_95pct_low"] <= 0.0:
        reasons.append("validation_paired_lower_bound_not_positive")
    status = (
        "validation_passed_test_unread_ready_for_fixed_test"
        if not reasons
        else "validation_rejected_test_unread"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = []
    for index, (policy, member) in enumerate(zip(policies, members)):
        name = f"member-{index}.pt"
        policy.save(
            args.output_dir / name,
            metadata={
                "source": "exact_tie_future_wall_average_v1",
                "feature_version": FEATURE_VERSION,
                "future_wall_permutations": REQUIRED_FUTURE_PERMUTATIONS,
                "ensemble_rule": "unanimous_else_frozen_teacher",
                "selected_epoch": member["selected_epoch"],
                "final_test_read": False,
            },
        )
        checkpoints.append(name)
    report = {
        "status": status,
        "gate_reasons": reasons,
        "protocol": {
            "architecture": "five_member_hidden64_candidate_mlp",
            "feature_version": FEATURE_VERSION,
            "future_wall_permutations": REQUIRED_FUTURE_PERMUTATIONS,
            "hidden_hand_allocations_per_decision": 1,
            "ensemble_seeds": ENSEMBLE_SEEDS,
            "epochs": EPOCHS,
            "target": "paired_terminal_order_after_future_wall_average",
            "deployment": "exact_teacher_tie_unanimous_override_else_teacher",
        },
        "input": {
            "train_records": len(train_records),
            "train_decisions": exact_tie_distinct_decision_count(train_records),
            "train_groups": len(train_groups),
            "fit_pairwise_examples": len(fit_examples),
            "epoch_pairwise_examples": len(epoch_examples),
            "validation_sha256": _sha256(paths["validation"]),
            "test_sha256": test_sha256,
            "test_contract": "file_hashed_but_records_unparsed",
        },
        "members": members,
        "outer_validation": validation,
        "checkpoints": checkpoints,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    printable = {
        **report,
        "members": [
            {key: value for key, value in member.items() if key != "history"}
            for member in members
        ],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

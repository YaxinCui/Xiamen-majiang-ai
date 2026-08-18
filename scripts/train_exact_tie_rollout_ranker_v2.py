#!/usr/bin/env python3
"""Train structured-feature exact-tie ranker with a fresh outer validation."""

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
    evaluate_unanimous_ensemble,
)
from xiamen_mahjong.exact_tie_rollout import (
    exact_tie_pairwise_examples,
    read_exact_tie_rollout_records,
    split_exact_tie_rollout_records_by_group,
)


FEATURE_VERSION = 4
HIDDEN_SIZE = 64
ENSEMBLE_SEEDS = (202641100, 202641101, 202641102, 202641103, 202641104)
EPOCHS = 40
INNER_SPLIT_SALT = "exact-tie-rollout-ranker-v2-structured-inner"
MINIMUM_TRAIN_GROUPS = 750
MINIMUM_VALIDATION_GROUPS = 75
MINIMUM_VALIDATION_RECORDS = 275
MINIMUM_VALIDATION_OVERRIDES = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, action="append", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise ValueError("exact-tie structured ranker 输出目录已存在且非空")
    for path in [*args.train_input, args.validation, args.test]:
        if not path.is_file():
            raise ValueError(f"exact-tie structured ranker 输入不存在：{path}")

    # The final test bytes are hashed for identity but never parsed here.
    test_sha256 = _sha256(args.test)
    train_records = [
        record
        for path in args.train_input
        for record in read_exact_tie_rollout_records(path)
    ]
    if len({record.item_id for record in train_records}) != len(train_records):
        raise ValueError("exact-tie v2 train item_id 重复")
    train_groups = {record.split_group_id for record in train_records}
    if len(train_groups) < MINIMUM_TRAIN_GROUPS:
        raise ValueError("exact-tie v2 train group 数不足")
    inner = split_exact_tie_rollout_records_by_group(
        train_records,
        split_salt=INNER_SPLIT_SALT,
        train_fraction=0.9,
        validation_fraction=0.05,
    )
    fit_records = inner["train"]
    epoch_records = [*inner["validation"], *inner["test"]]
    fit_examples = exact_tie_pairwise_examples(fit_records)
    epoch_examples = exact_tie_pairwise_examples(epoch_records)
    policies = []
    member_reports = []
    for seed in ENSEMBLE_SEEDS:
        policy, member = _train_member(
            fit_examples,
            epoch_examples,
            seed=seed,
            device=args.device,
            feature_version=FEATURE_VERSION,
            hidden_size=HIDDEN_SIZE,
            epochs=EPOCHS,
        )
        policies.append(policy)
        member_reports.append(member)

    validation_records = read_exact_tie_rollout_records(args.validation)
    validation_groups = {record.split_group_id for record in validation_records}
    if train_groups & validation_groups:
        raise ValueError("exact-tie v2 train/validation group 泄漏")
    if len(validation_groups) < MINIMUM_VALIDATION_GROUPS:
        raise ValueError("exact-tie v2 validation group 数不足")
    if len(validation_records) < MINIMUM_VALIDATION_RECORDS:
        raise ValueError("exact-tie v2 validation record 数不足")
    validation = evaluate_unanimous_ensemble(policies, validation_records)
    reasons = []
    if validation["overrides"] < MINIMUM_VALIDATION_OVERRIDES:
        reasons.append("insufficient_fresh_validation_overrides")
    if validation["paired_wall_delta_95pct_low"] <= 0.0:
        reasons.append("fresh_validation_paired_lower_bound_not_positive")
    status = (
        "validation_passed_test_unread_ready_for_fixed_test"
        if not reasons
        else "validation_rejected_test_unread"
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = []
    for index, (policy, member) in enumerate(zip(policies, member_reports)):
        name = f"member-{index}.pt"
        policy.save(
            args.output_dir / name,
            metadata={
                "source": "exact_tie_source_world_paired_rollout_v2_structured",
                "feature_version": FEATURE_VERSION,
                "deployment_scope": "exact_teacher_top_score_tie_only",
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
            "structured_features": (
                "teacher_shape_components_exact_shanten_public_live_counts"
            ),
            "ensemble_seeds": ENSEMBLE_SEEDS,
            "epochs": EPOCHS,
            "target": "unequal_source_world_paired_terminal_order",
            "deployment": "exact_teacher_tie_unanimous_override_else_teacher",
            "maximum_intended_interventions_per_hand": 1,
        },
        "input": {
            "train_files": [str(path) for path in args.train_input],
            "train_records": len(train_records),
            "train_groups": len(train_groups),
            "fit_records": len(fit_records),
            "epoch_selection_records": len(epoch_records),
            "fit_pairwise_examples": len(fit_examples),
            "epoch_pairwise_examples": len(epoch_examples),
            "fresh_validation_file": str(args.validation),
            "fresh_validation_sha256": _sha256(args.validation),
            "final_test_sha256": test_sha256,
            "final_test_contract": "file_hashed_but_records_unparsed",
        },
        "members": member_reports,
        "fresh_outer_validation": validation,
        "checkpoints": checkpoints,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    printable = {
        **report,
        "members": [
            {key: value for key, value in member.items() if key != "history"}
            for member in member_reports
        ],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

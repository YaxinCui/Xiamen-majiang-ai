#!/usr/bin/env python3
"""Merge and wall-group split exact-tie rollout shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.exact_tie_rollout import (
    exact_tie_pairwise_examples,
    exact_tie_future_averaged_pairwise_examples,
    read_exact_tie_rollout_records,
    split_exact_tie_rollout_records_by_group,
    write_exact_tie_rollout_records,
)


SPLIT_SALT = "exact-tie-source-world-rollout-v1-formal-split"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--future-average", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise ValueError("exact-tie split 输出目录已存在且非空")
    records = [
        record
        for path in args.input
        for record in read_exact_tie_rollout_records(path)
    ]
    item_ids = [record.item_id for record in records]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("exact-tie merge 后 item_id 重复")
    partitions = split_exact_tie_rollout_records_by_group(
        records,
        split_salt=SPLIT_SALT,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    if any(not rows for rows in partitions.values()):
        raise ValueError("exact-tie split 产生空分区")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "exact_tie_rollout_wall_group_split_ready",
        "split_salt": SPLIT_SALT,
        "source_files": [str(path) for path in args.input],
        "records": len(records),
        "groups": len({record.split_group_id for record in records}),
        "future_averaged_pairwise_targets": args.future_average,
        "partitions": {},
    }
    memberships: dict[str, str] = {}
    for split, rows in partitions.items():
        path = args.output_dir / f"{split}.exact-tie-rollout.jsonl"
        write_exact_tie_rollout_records(rows, path)
        groups = {record.split_group_id for record in rows}
        for group in groups:
            if group in memberships:
                raise AssertionError("物理墙 group 跨 split")
            memberships[group] = split
        report["partitions"][split] = {
            "file": path.name,
            "records": len(rows),
            "groups": len(groups),
            "pairwise_examples": len(
                exact_tie_future_averaged_pairwise_examples(rows)
                if args.future_average
                else exact_tie_pairwise_examples(rows)
            ),
            "sha256": _sha256(path),
        }
    report_path = args.output_dir / "manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

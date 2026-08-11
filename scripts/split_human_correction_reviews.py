#!/usr/bin/env python3
"""Split confirmed human correction reviews by physical-hand group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.human_review import (
    REVIEW_SPLIT_VERSION,
    audit_review_labels_against_queue,
    read_review_labels,
    read_review_queue,
    split_confirmed_review_labels,
)


def _require_local_human_directory(path: Path) -> None:
    root = (ROOT / "local_human_data").resolve()
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("review split 必须写入 Git 忽略的 local_human_data/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--minimum-confirmed-labels", type=int, default=500)
    parser.add_argument("--minimum-confirmed-disagreements", type=int, default=50)
    parser.add_argument("--minimum-groups", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_local_human_directory(args.output_dir)
    if args.output_dir.exists():
        raise ValueError("review split 输出目录已存在，拒绝覆盖")
    records = read_review_labels(args.input)
    audit = audit_review_labels_against_queue(
        records,
        read_review_queue(args.queue),
        minimum_confirmed_labels=args.minimum_confirmed_labels,
        minimum_confirmed_disagreements=args.minimum_confirmed_disagreements,
        minimum_groups=args.minimum_groups,
    )
    if not audit["ready_for_manual_training_review"]:
        raise ValueError("review labels 未通过切分门槛：" + ",".join(audit["gate_reasons"]))
    partitions = split_confirmed_review_labels(records)
    if any(not rows for rows in partitions.values()):
        raise ValueError("review train/validation/test 必须都非空")
    args.output_dir.mkdir(parents=True)
    for split, rows in partitions.items():
        destination = args.output_dir / f"{split}.review.jsonl"
        with destination.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "version": REVIEW_SPLIT_VERSION,
        "source": "local_human_review_opt_in",
        "split_record_counts": {
            split: len(rows) for split, rows in partitions.items()
        },
        "group_overlap": 0,
        "audit": audit,
        "privacy": "aggregate_manifest_no_local_input_path_or_review_content",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

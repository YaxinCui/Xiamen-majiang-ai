#!/usr/bin/env python3
"""Split confirmed response reviews by opaque physical-hand group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.response_review import (
    audit_response_review_labels_against_queue,
    read_response_review_labels,
    read_response_review_queue,
    split_confirmed_response_review_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    local_root = (ROOT / "local_human_data").resolve()
    output = args.output_dir.resolve()
    if output != local_root and local_root not in output.parents:
        raise ValueError("response review split 必须写入 local_human_data/")
    if args.output_dir.exists():
        raise ValueError("response review split 输出目录已存在，拒绝覆盖")
    queue = read_response_review_queue(args.queue)
    records = read_response_review_labels(args.input)
    report = audit_response_review_labels_against_queue(records, queue)
    if not report["ready_for_manual_training_review"]:
        raise ValueError("response review 数据未达到预注册训练门槛")
    partitions = split_confirmed_response_review_labels(records, queue)
    if any(not rows for rows in partitions.values()):
        raise ValueError("response review group split 产生空分区")
    args.output_dir.mkdir(parents=True)
    for split, rows in partitions.items():
        path = args.output_dir / f"{split}.response-review.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "status": "response_review_split_ready",
        "counts": {split: len(rows) for split, rows in partitions.items()},
        "group_isolation": True,
        "test_training_access": "forbidden_reserved_for_gate",
    }
    (args.output_dir / "split-report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

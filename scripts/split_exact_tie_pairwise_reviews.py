#!/usr/bin/env python3
"""Group-split audited exact-tie pairwise labels into 80/10/10 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.pairwise_review import (
    PAIRWISE_SPLIT_VERSION,
    audit_pairwise_labels_against_queue,
    read_pairwise_labels,
    read_pairwise_queue,
    split_confirmed_pairwise_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("pairwise split 输出目录已存在且非空，拒绝覆盖")
    records = read_pairwise_labels(args.input)
    queue = read_pairwise_queue(args.queue)
    audit = audit_pairwise_labels_against_queue(records, queue)
    if not audit["ready_for_pairwise_split"]:
        raise ValueError("pairwise review 未通过固定审计门槛")
    partitions = split_confirmed_pairwise_labels(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in partitions.items():
        path = args.output_dir / f"{split}.pairwise-review.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "version": PAIRWISE_SPLIT_VERSION,
        "audit": audit,
        "split_counts": {split: len(rows) for split, rows in partitions.items()},
        "split_policy": "opaque_original_hand_group_80_10_10",
        "target_semantics": "pairwise_only_not_full_action_classification",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

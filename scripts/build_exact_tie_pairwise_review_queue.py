#!/usr/bin/env python3
"""Build a blind exact-tie pairwise queue from the immutable discard queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.human_review import read_review_queue
from xiamen_mahjong.pairwise_review import (
    build_pairwise_review_queue,
    write_pairwise_queue,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--maximum-items", type=int, default=240)
    parser.add_argument("--maximum-items-per-group", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise ValueError("pairwise queue 或 report 已存在，拒绝覆盖")
    source = read_review_queue(args.input_queue)
    queue, report = build_pairwise_review_queue(
        source,
        maximum_items=args.maximum_items,
        maximum_items_per_group=args.maximum_items_per_group,
    )
    write_pairwise_queue(args.output, queue)
    payload = {
        **report,
        "input_queue_sha256": _sha256(args.input_queue),
        "output_queue_sha256": _sha256(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

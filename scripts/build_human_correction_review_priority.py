#!/usr/bin/env python3
"""Build a blind active-learning order for the immutable review queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.human_review import (
    build_review_slow_expert_priority,
    read_review_queue,
    validate_review_priority_against_queue,
    write_review_priority,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local_root = (ROOT / "local_human_data").resolve()
    for path in (args.queue, args.output):
        resolved = path.resolve()
        if resolved != local_root and local_root not in resolved.parents:
            raise ValueError("queue 与 priority 必须位于 local_human_data/")
    if args.queue.resolve() == args.output.resolve():
        parser.error("--queue 与 --output 不能是同一个文件")
    queue = read_review_queue(args.queue)
    records, report = build_review_slow_expert_priority(queue)
    issues = validate_review_priority_against_queue(records, queue)
    if issues:
        raise ValueError("生成的 priority 无效：" + ",".join(issues))
    write_review_priority(args.output, records)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

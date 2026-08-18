#!/usr/bin/env python3
"""Build the immutable actor-visible pass/chi/pong human review queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.response_review import (
    build_response_review_queue,
    write_response_review_queue,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--maximum-items", type=int, default=400)
    parser.add_argument("--maximum-items-per-group", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local_root = (ROOT / "local_human_data").resolve()
    output = args.output.resolve()
    if output != local_root and local_root not in output.parents:
        raise ValueError("response review queue 必须写入 local_human_data/")
    queue, report = build_response_review_queue(
        args.input,
        maximum_items=args.maximum_items,
        maximum_items_per_group=args.maximum_items_per_group,
    )
    write_response_review_queue(args.output, queue)
    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a local actor-visible low-margin Teacher correction queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.human_review import build_review_queue, write_review_queue


def _require_local_human_path(path: Path) -> None:
    root = (ROOT / "local_human_data").resolve()
    resolved = path.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError("review queue 必须写入 Git 忽略的 local_human_data/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--maximum-items", type=int, default=600)
    parser.add_argument("--maximum-items-per-group", type=int, default=2)
    parser.add_argument("--maximum-teacher-score-margin", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_local_human_path(args.output)
    queue, report = build_review_queue(
        args.input,
        maximum_items=args.maximum_items,
        maximum_items_per_group=args.maximum_items_per_group,
        maximum_teacher_score_margin=args.maximum_teacher_score_margin,
    )
    if not queue:
        raise ValueError("没有找到符合固定范围的 review item")
    write_review_queue(args.output, queue)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

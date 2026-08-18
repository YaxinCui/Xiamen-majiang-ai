#!/usr/bin/env python3
"""Aggregate-only audit for local actor-visible human correction reviews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.human_review import (
    audit_review_labels_against_queue,
    read_review_labels,
    read_review_queue,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--minimum-confirmed-labels", type=int, default=500)
    parser.add_argument("--minimum-confirmed-disagreements", type=int, default=50)
    parser.add_argument("--minimum-groups", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_review_labels_against_queue(
        read_review_labels(args.input),
        read_review_queue(args.queue),
        minimum_confirmed_labels=args.minimum_confirmed_labels,
        minimum_confirmed_disagreements=args.minimum_confirmed_disagreements,
        minimum_groups=args.minimum_groups,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

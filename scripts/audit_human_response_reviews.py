#!/usr/bin/env python3
"""Audit local response review labels against their immutable queue."""

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
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--minimum-confirmed-labels", type=int, default=300)
    parser.add_argument("--minimum-disagreements", type=int, default=50)
    parser.add_argument("--minimum-groups", type=int, default=100)
    args = parser.parse_args()
    report = audit_response_review_labels_against_queue(
        read_response_review_labels(args.input),
        read_response_review_queue(args.queue),
        minimum_confirmed_labels=args.minimum_confirmed_labels,
        minimum_confirmed_disagreements=args.minimum_disagreements,
        minimum_groups=args.minimum_groups,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

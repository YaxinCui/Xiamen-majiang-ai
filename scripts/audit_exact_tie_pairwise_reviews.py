#!/usr/bin/env python3
"""Audit blind exact-tie pairwise human labels against their fixed queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.pairwise_review import (
    audit_pairwise_labels_against_queue,
    read_pairwise_labels,
    read_pairwise_queue,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--minimum-confirmed-labels", type=int, default=100)
    parser.add_argument("--minimum-candidate-preferences", type=int, default=20)
    parser.add_argument("--minimum-groups", type=int, default=75)
    args = parser.parse_args()
    report = audit_pairwise_labels_against_queue(
        read_pairwise_labels(args.input),
        read_pairwise_queue(args.queue),
        minimum_confirmed_labels=args.minimum_confirmed_labels,
        minimum_candidate_preferences=args.minimum_candidate_preferences,
        minimum_groups=args.minimum_groups,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

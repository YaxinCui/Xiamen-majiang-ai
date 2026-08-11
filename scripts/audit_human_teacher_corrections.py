#!/usr/bin/env python3
"""Audit opt-in human/Teacher disagreements before residual training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.human_data import audit_local_human_teacher_corrections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--minimum-hands", type=int, default=100)
    parser.add_argument("--minimum-reference-decisions", type=int, default=500)
    parser.add_argument("--minimum-disagreements", type=int, default=50)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_local_human_teacher_corrections(
        args.input,
        minimum_hands=args.minimum_hands,
        minimum_reference_decisions=args.minimum_reference_decisions,
        minimum_disagreements=args.minimum_disagreements,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

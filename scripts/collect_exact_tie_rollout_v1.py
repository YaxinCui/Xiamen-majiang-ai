#!/usr/bin/env python3
"""Collect fixed paired terminal labels for exact Teacher discard ties."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.exact_tie_rollout import (
    collect_exact_tie_rollout_records,
    write_exact_tie_rollout_records,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=202638700)
    parser.add_argument("--samples-per-rotation", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=202638701)
    parser.add_argument("--future-wall-permutations", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.report):
        if path.exists():
            raise ValueError(f"拒绝覆盖已有 exact-tie rollout 产物：{path}")
    started = time.perf_counter()
    records, report = collect_exact_tie_rollout_records(
        seed_count=args.seed_count,
        seed=args.seed,
        samples_per_rotation=args.samples_per_rotation,
        selection_seed=args.selection_seed,
        future_wall_permutations=args.future_wall_permutations,
    )
    count = write_exact_tie_rollout_records(records, args.output)
    payload = {
        **report,
        "records": count,
        "elapsed_seconds": time.perf_counter() - started,
        "data_file": args.output.name,
        "data_sha256": _sha256(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

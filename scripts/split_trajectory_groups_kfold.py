#!/usr/bin/env python3
"""Split a train-only trajectory corpus into wall-group-disjoint K folds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.training import (
    read_trajectory_jsonl,
    split_trajectories_by_group_folds,
    trajectory_manifest,
    write_trajectory_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, required=True)
    parser.add_argument("--split-salt", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    trajectories = read_trajectory_jsonl(args.input)
    folds = split_trajectories_by_group_folds(
        trajectories, fold_count=args.fold_count, split_salt=args.split_salt
    )
    group_sets = [
        {trajectory.split_group_id for trajectory in fold} for fold in folds
    ]
    if any(left & right for index, left in enumerate(group_sets) for right in group_sets[index + 1 :]):
        raise RuntimeError("cross-fitting folds 的墙组重叠")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, fold in enumerate(folds):
        filename = f"fold-{index}.trajectories.jsonl"
        output = args.output_dir / filename
        reports.append(
            {
                "fold": index,
                "file": filename,
                "hands": write_trajectory_jsonl(fold, output),
                "wall_groups": len(group_sets[index]),
                "sha256": sha256(output),
                "manifest": trajectory_manifest(fold),
            }
        )
    payload = {
        "status": "cross_fitting_folds_created",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "fold_count": args.fold_count,
        "split_salt": args.split_salt,
        "group_overlap": 0,
        "folds": reports,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

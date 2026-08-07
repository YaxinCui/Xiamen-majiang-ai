#!/usr/bin/env python3
"""Split an unopened held-out trajectory pool into selector and terminal sets.

The input must contain complete physical-wall ``split_group_id`` groups.  The
``selection`` output is the only set allowed for choosing a member from a
pre-registered candidate grid.  Keep ``terminal`` unread until the choice is
recorded, then use it exactly once for final OPE.
"""

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
    split_heldout_trajectories_by_group,
    trajectory_manifest,
    write_trajectory_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-fraction", type=float, default=0.5)
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
    partitions = split_heldout_trajectories_by_group(
        trajectories,
        selection_fraction=args.selection_fraction,
        split_salt=args.split_salt,
    )
    group_sets = {
        name: {trajectory.split_group_id for trajectory in records}
        for name, records in partitions.items()
    }
    if group_sets["selection"] & group_sets["terminal"]:
        raise RuntimeError("selector 与 terminal 墙组重叠")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_partitions = {}
    for name, records in partitions.items():
        filename = f"{name}.trajectories.jsonl"
        output = args.output_dir / filename
        partition_report = {
            "file": filename,
            "hands": write_trajectory_jsonl(records, output),
            "wall_groups": len(group_sets[name]),
            "sha256": sha256(output),
        }
        if name == "selection":
            partition_report["manifest"] = trajectory_manifest(records)
        else:
            partition_report["manifest_omitted"] = (
                "terminal outcome/action summaries stay unread until final OPE"
            )
        report_partitions[name] = partition_report
    report = {
        "status": "selection_and_terminal_groups_created",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "selection_fraction": args.selection_fraction,
        "split_salt": args.split_salt,
        "group_overlap": 0,
        "protocol": (
            "selection may choose only a pre-registered candidate; terminal "
            "must remain unopened until that choice is recorded and is final-OPE only"
        ),
        "partitions": report_partitions,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

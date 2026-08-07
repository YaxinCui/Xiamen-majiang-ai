#!/usr/bin/env python3
"""Safely split reviewed opt-in local human Mahjong hands by whole hand."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.human_data import split_local_human_trajectories
from xiamen_mahjong.training import write_trajectory_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="必须位于 local_human_data/ 下；输出 train/validation/test JSONL 和 manifest",
    )
    parser.add_argument("--minimum-hands", type=int, default=100)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许替换该输出目录中的同名切分文件",
    )
    return parser.parse_args()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    local_root = ROOT / "local_human_data"
    if not _is_within(args.output_dir, local_root):
        raise ValueError("人类切分输出必须位于项目 local_human_data/ 目录内")
    filenames = {
        "train": "train.trajectories.jsonl",
        "validation": "validation.trajectories.jsonl",
        "test": "test.trajectories.jsonl",
    }
    outputs = {
        split: args.output_dir / filename for split, filename in filenames.items()
    }
    manifest_path = args.output_dir / "manifest.json"
    existing = [path for path in (*outputs.values(), manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "人类切分输出已存在；请使用新的 local_human_data 子目录，"
            "或在确认目标后显式传入 --overwrite"
        )

    partitions, audit = split_local_human_trajectories(
        args.input,
        minimum_hands=args.minimum_hands,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_reports = {}
    for split, records in partitions.items():
        output = outputs[split]
        split_reports[split] = {
            "file": output.name,
            "hands": write_trajectory_jsonl(records, output),
            "sha256": sha256(output),
        }
    report = {
        "status": "audited_local_human_hand_splits_created",
        "input_file_count": len(args.input),
        "minimum_hands": args.minimum_hands,
        "split_version": audit["split_version"],
        "split_salt": audit["split_salt"],
        "split_group_overlap": audit["split_group_overlap"],
        "splits": split_reports,
        "audit": audit,
        "warning": (
            "This only prepares structurally audited behavioral-imitation inputs. "
            "It does not establish human skill, authorize training by itself, "
            "or prove a model beats humans."
        ),
    }
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

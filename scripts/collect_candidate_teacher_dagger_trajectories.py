#!/usr/bin/env python3
"""Collect deployment-matched DAgger data: one candidate versus three Teachers.

The saved train/validation/test JSONL files are safe trajectory v3 exports:
they contain no deal seed or random-behaviour seed.  A replay index is opt-in
and must remain private.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.training import (
    collect_candidate_teacher_dagger_trajectories,
    split_trajectories_by_hand,
    trajectory_manifest,
    write_trajectory_replay_index,
    write_trajectory_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=("classic", "core"), default="classic")
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20265004)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="候选 .pt 的推理设备；实验比较应固定同一设备",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-salt", default="candidate-teacher-dagger-v1")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/candidate-teacher-dagger")
    )
    parser.add_argument(
        "--replay-index",
        type=Path,
        help="可选的本地私有 replay 索引；不得用于训练或提交",
    )
    return parser.parse_args()


def load_policy(path: Path, *, device: str = "cpu"):
    if path.suffix not in {".pt", ".pth"}:
        raise ValueError("在线 policy-value DAgger 当前需要 .pt checkpoint")
    from xiamen_mahjong.torch_policy import TorchPolicyValueAgent

    return TorchPolicyValueAgent.load(path, device=device)


def main() -> None:
    args = parse_args()
    if args.seed_count <= 0:
        raise ValueError("seed-count 必须为正数")
    policy = load_policy(args.checkpoint, device=args.device)
    trajectories, summary = collect_candidate_teacher_dagger_trajectories(
        policy,
        seed_count=args.seed_count,
        profile=args.profile,
        seed=args.seed,
    )
    partitions = split_trajectories_by_hand(
        trajectories,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        split_salt=args.split_salt,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partition_reports = {}
    for name, records in partitions.items():
        filename = f"{name}.trajectories.jsonl"
        partition_reports[name] = {
            "file": filename,
            "hands": write_trajectory_jsonl(records, args.output_dir / filename),
            "manifest": trajectory_manifest(records),
        }
    if args.replay_index:
        write_trajectory_replay_index(trajectories, args.replay_index)
    report = {
        "source": "candidate_vs_teacher_dagger",
        "profile": args.profile,
        "behavior_checkpoint": str(args.checkpoint),
        "inference_device": args.device,
        "seed_count": args.seed_count,
        "seat_rotations_per_seed": 4,
        "summary": summary.payload(),
        "replay_metadata": {
            "included_in_training_jsonl": False,
            "private_replay_index": str(args.replay_index) if args.replay_index else None,
        },
        "split": {
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "split_salt": args.split_salt,
            "rotation_grouping": "all four candidate seats share one partition",
        },
        "dataset": trajectory_manifest(trajectories),
        "partitions": partition_reports,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

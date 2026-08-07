#!/usr/bin/env python3
"""Build a split, versioned, no-hidden-state Xiamen Mahjong trajectory set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.training import (
    collect_exploration_trajectories,
    collect_gold_lock_trajectories,
    collect_response_pass_curriculum,
    collect_teacher_trajectories,
    collect_tour_trajectories,
    split_trajectories_by_hand,
    trajectory_manifest,
    write_trajectory_replay_index,
    write_trajectory_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("classic", "core"), default="classic")
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument(
        "--exploration-hands",
        type=int,
        default=0,
        help="额外随机合法行为局数；仍由冻结 Teacher 标注",
    )
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--exploration-seed", type=int)
    parser.add_argument(
        "--response-pass-curriculum",
        type=int,
        default=0,
        help="追加 Teacher 明确选择 pass 的物理响应课程数",
    )
    parser.add_argument(
        "--tour-curriculum",
        type=int,
        default=136,
        help="追加规则引擎验证、Teacher 标注的游金/双游课程数；设 0 关闭",
    )
    parser.add_argument(
        "--gold-lock-curriculum",
        type=int,
        default=64,
        help="追加规则引擎验证的金牌弃置后仅自摸响应课程数；设 0 关闭",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-salt", default="xiamen-trajectory-v2")
    parser.add_argument(
        "--replay-index",
        type=Path,
        help="可选的本地私有 replay 索引；包含牌局/行为种子，不能交给训练器或提交",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/trajectory-teacher-classic")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.hands <= 0
        or args.exploration_hands < 0
        or args.response_pass_curriculum < 0
        or args.tour_curriculum < 0
        or args.gold_lock_curriculum < 0
    ):
        raise ValueError("hands 必须为正数，课程局数不能为负数")
    trajectories, summary = collect_teacher_trajectories(
        hands=args.hands, profile=args.profile, seed=args.seed
    )
    exploration_summary = None
    if args.exploration_hands:
        exploration, exploration_summary = collect_exploration_trajectories(
            hands=args.exploration_hands,
            profile=args.profile,
            seed=(args.exploration_seed or args.seed + 1_000_000),
        )
        trajectories.extend(exploration)
    response_pass = []
    if args.response_pass_curriculum:
        response_pass = collect_response_pass_curriculum(
            examples=args.response_pass_curriculum,
            profile=args.profile,
            seed=args.seed + 2_000_000,
        )
        trajectories.extend(response_pass)
    tour = []
    if args.profile == "classic" and args.tour_curriculum:
        tour = collect_tour_trajectories(
            examples=args.tour_curriculum, seed=args.seed + 3_000_000
        )
        trajectories.extend(tour)
    gold_lock = []
    if args.profile == "classic" and args.gold_lock_curriculum:
        gold_lock = collect_gold_lock_trajectories(
            examples=args.gold_lock_curriculum, seed=args.seed + 4_000_000
        )
        trajectories.extend(gold_lock)
    partitions = split_trajectories_by_hand(
        trajectories,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        split_salt=args.split_salt,
    )
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    partition_reports = {}
    for name, records in partitions.items():
        filename = f"{name}.trajectories.jsonl"
        partition_reports[name] = {
            "file": filename,
            "hands": write_trajectory_jsonl(records, output_dir / filename),
            "manifest": trajectory_manifest(records),
        }
    report = {
        "source": "heuristic_teacher_self_play",
        "profile": args.profile,
        "hands_requested": args.hands,
        "exploration_hands_requested": args.exploration_hands,
        "response_pass_curriculum_requested": args.response_pass_curriculum,
        "tour_curriculum_requested": args.tour_curriculum,
        "gold_lock_curriculum_requested": args.gold_lock_curriculum,
        "teacher_summary": summary.payload(),
        "exploration_summary": exploration_summary.payload()
        if exploration_summary
        else None,
        "response_pass_curriculum_hands": len(response_pass),
        "tour_curriculum_hands": len(tour),
        "gold_lock_curriculum_hands": len(gold_lock),
        "split": {
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "split_salt": args.split_salt,
        },
        "replay_metadata": {
            "included_in_training_jsonl": False,
            "private_replay_index": str(args.replay_index) if args.replay_index else None,
        },
        "dataset": trajectory_manifest(trajectories),
        "partitions": partition_reports,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.replay_index:
        write_trajectory_replay_index(trajectories, args.replay_index)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

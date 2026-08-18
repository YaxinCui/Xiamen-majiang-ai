#!/usr/bin/env python3
"""Improve a legal-action MLP with DAgger trajectories from its own play."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.training import (
    NeuralRulePolicyModel,
    collect_dagger_decisions,
    collect_teacher_decisions,
    collect_tour_curriculum,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/rule-policy-classic-mlp/rule-policy.json"),
        help="要继续训练的 MLP 检查点",
    )
    parser.add_argument("--profile", choices=("classic", "core"), default="classic")
    parser.add_argument("--teacher-hands", type=int, default=40)
    parser.add_argument("--validation-hands", type=int, default=12)
    parser.add_argument("--rollout-hands", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs-per-round", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.006)
    parser.add_argument("--tour-curriculum", type=int, default=136)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dagger-classic"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rounds <= 0 or args.epochs_per_round <= 0:
        raise ValueError("rounds 和 epochs-per-round 必须为正数")
    model = NeuralRulePolicyModel.load(args.checkpoint)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_data, teacher_summary = collect_teacher_decisions(
        hands=args.teacher_hands, profile=args.profile, seed=args.seed
    )
    validation, validation_summary = collect_teacher_decisions(
        hands=args.validation_hands,
        profile=args.profile,
        seed=args.seed + 10_000,
    )
    curriculum = []
    if args.profile == "classic" and args.tour_curriculum:
        curriculum = collect_tour_curriculum(
            examples=args.tour_curriculum, seed=args.seed + 20_000
        )
    base_data = [*teacher_data, *curriculum]
    cumulative_dagger = []
    report: dict[str, object] = {
        "profile": args.profile,
        "seed": args.seed,
        "checkpoint_source": str(args.checkpoint),
        "teacher": teacher_summary.payload(),
        "validation": validation_summary.payload(),
        "tour_curriculum_decisions": len(curriculum),
        "validation_before": model.evaluate(validation),
        "rounds": [],
    }
    for round_index in range(1, args.rounds + 1):
        dagger_data, summary = collect_dagger_decisions(
            model,
            hands=args.rollout_hands,
            profile=args.profile,
            seed=args.seed + round_index * 100_000,
        )
        cumulative_dagger.extend(dagger_data)
        write_jsonl(dagger_data, output_dir / f"dagger-round-{round_index}.jsonl")
        history = model.fit(
            [*base_data, *cumulative_dagger],
            epochs=args.epochs_per_round,
            learning_rate=args.learning_rate,
            seed=args.seed + round_index,
        )
        report["rounds"].append(
            {
                "round": round_index,
                "dagger": summary.payload(),
                "cumulative_dagger_decisions": len(cumulative_dagger),
                "training_history": history,
                "validation_after": model.evaluate(validation),
            }
        )
    report["validation_after"] = model.evaluate(validation)
    model.save(output_dir / "rule-policy-dagger.json", metadata=report)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

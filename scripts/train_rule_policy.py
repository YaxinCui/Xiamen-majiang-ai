#!/usr/bin/env python3
"""Generate rule-Teacher trajectories and train the legal-action baseline."""

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
    RulePolicyModel,
    collect_teacher_decisions,
    collect_tour_curriculum,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("classic", "core"), default="classic")
    parser.add_argument("--hands", type=int, default=80, help="Teacher 训练局数")
    parser.add_argument("--validation-hands", type=int, default=20, help="保留评测局数")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--model", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--hidden-size", type=int, default=12, help="MLP 隐藏层宽度")
    parser.add_argument(
        "--tour-curriculum",
        type=int,
        default=136,
        help="经典规则中追加的游金/双游 Teacher 课程样本数；设为 0 可关闭",
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/rule-policy-classic")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train, train_summary = collect_teacher_decisions(
        hands=args.hands, profile=args.profile, seed=args.seed
    )
    validation, validation_summary = collect_teacher_decisions(
        hands=args.validation_hands,
        profile=args.profile,
        seed=args.seed + args.hands + 10_000,
    )
    tour_train = []
    tour_validation = []
    if args.profile == "classic" and args.tour_curriculum:
        tour_train = collect_tour_curriculum(
            examples=args.tour_curriculum, seed=args.seed + 20_000
        )
        tour_validation = collect_tour_curriculum(
            examples=max(32, args.tour_curriculum // 4), seed=args.seed + 30_000
        )
        train.extend(tour_train)
    train_jsonl = output_dir / "teacher-train.jsonl"
    validation_jsonl = output_dir / "teacher-validation.jsonl"
    write_jsonl(train, train_jsonl)
    write_jsonl(validation, validation_jsonl)

    model = (
        RulePolicyModel()
        if args.model == "linear"
        else NeuralRulePolicyModel(hidden_size=args.hidden_size, seed=args.seed)
    )
    before = model.evaluate(validation)
    tour_before = model.evaluate(tour_validation)
    learning_rate = args.learning_rate or (0.035 if args.model == "linear" else 0.012)
    history = model.fit(
        train,
        epochs=args.epochs,
        learning_rate=learning_rate,
        seed=args.seed,
    )
    after = model.evaluate(validation)
    tour_after = model.evaluate(tour_validation)
    report = {
        "profile": args.profile,
        "model": args.model,
        "hidden_size": args.hidden_size if args.model == "mlp" else None,
        "learning_rate": learning_rate,
        "seed": args.seed,
        "train": train_summary.payload(),
        "validation": validation_summary.payload(),
        "tour_curriculum_decisions": len(tour_train),
        "tour_validation_decisions": len(tour_validation),
        "validation_before": before,
        "validation_after": after,
        "tour_validation_before": tour_before,
        "tour_validation_after": tour_after,
        "history": history,
        "checkpoint": "rule-policy.json",
    }
    model.save(output_dir / "rule-policy.json", metadata=report)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

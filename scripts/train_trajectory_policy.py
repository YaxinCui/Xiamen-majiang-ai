#!/usr/bin/env python3
"""Train a public-state v3 legal-action policy from split trajectory JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.training import (
    NeuralRulePolicyModel,
    TeacherDecision,
    read_trajectory_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("artifacts/trajectory-teacher-classic-v2-run1/train.trajectories.jsonl"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("artifacts/trajectory-teacher-classic-v2-run1/validation.trajectories.jsonl"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("artifacts/trajectory-teacher-classic-v2-run1/test.trajectories.jsonl"),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.006)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--feature-version", type=int, choices=(2, 3), default=3)
    parser.add_argument("--balance", choices=("none", "sqrt"), default="sqrt")
    parser.add_argument("--max-class-weight", type=float, default=6.0)
    parser.add_argument("--exploration-weight", type=float, default=0.35)
    parser.add_argument("--synthetic-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/trajectory-policy-classic-v3")
    )
    return parser.parse_args()


def flatten(path: Path) -> list[tuple[TeacherDecision, str]]:
    return [
        (decision, str(trajectory.source_metadata.get("collector", "legacy")))
        for trajectory in read_trajectory_jsonl(path)
        for decision in trajectory.decisions
    ]


def class_weights(
    decisions: list[TeacherDecision], *, mode: str, maximum: float
) -> dict[str, float]:
    if maximum <= 0:
        raise ValueError("max-class-weight 必须为正数")
    counts = Counter(decision.chosen_action.kind for decision in decisions)
    if mode == "none":
        return {kind: 1.0 for kind in sorted(counts)}
    largest = max(counts.values())
    return {
        kind: min(maximum, math.sqrt(largest / count))
        for kind, count in sorted(counts.items())
    }


def by_action(model: NeuralRulePolicyModel, decisions: list[TeacherDecision]) -> dict[str, dict[str, float]]:
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for decision in decisions:
        kind = decision.chosen_action.kind
        counts[kind] += 1
        correct[kind] += model.predict_index(decision) == decision.chosen_index
    return {
        kind: {"decisions": float(count), "accuracy": correct[kind] / count}
        for kind, count in sorted(counts.items())
    }


def by_source(
    model: NeuralRulePolicyModel, labeled: list[tuple[TeacherDecision, str]]
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[TeacherDecision]] = {}
    for decision, source in labeled:
        groups.setdefault(source, []).append(decision)
    return {
        source: model.evaluate(decisions) for source, decisions in sorted(groups.items())
    }


def report_metrics(
    model: NeuralRulePolicyModel, labeled: list[tuple[TeacherDecision, str]]
) -> dict[str, object]:
    decisions = [decision for decision, _source in labeled]
    return {
        "overall": model.evaluate(decisions),
        "by_action": by_action(model, decisions),
        "by_source": by_source(model, labeled),
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.learning_rate <= 0 or args.hidden_size <= 0:
        raise ValueError("epochs、learning-rate 和 hidden-size 必须为正数")
    train_labeled = flatten(args.train)
    validation_labeled = flatten(args.validation)
    test_labeled = flatten(args.test)
    if not train_labeled or not validation_labeled or not test_labeled:
        raise ValueError("train、validation 和 test 都必须含有决策")
    if args.exploration_weight <= 0 or args.synthetic_weight <= 0:
        raise ValueError("exploration-weight 和 synthetic-weight 必须为正数")
    train = [decision for decision, _source in train_labeled]
    sample_weights = [
        args.exploration_weight
        if source == "random_legal_teacher_labeled"
        else args.synthetic_weight
        if source == "physical_response_pass_search"
        else 1.0
        for _decision, source in train_labeled
    ]
    weights = class_weights(train, mode=args.balance, maximum=args.max_class_weight)
    model = NeuralRulePolicyModel(
        hidden_size=args.hidden_size,
        feature_version=args.feature_version,
        seed=args.seed,
    )
    history = model.fit(
        train,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        class_weights=weights if args.balance != "none" else None,
        sample_weights=sample_weights,
    )
    report = {
        "model": "trajectory_public_state_policy",
        "feature_version": args.feature_version,
        "hidden_size": args.hidden_size,
        "seed": args.seed,
        "inputs": {
            "train": str(args.train),
            "validation": str(args.validation),
            "test": str(args.test),
        },
        "class_balance": {"mode": args.balance, "weights": weights},
        "source_weights": {
            "heuristic_teacher_self_play": 1.0,
            "random_legal_teacher_labeled": args.exploration_weight,
            "physical_response_pass_search": args.synthetic_weight,
        },
        "history": history,
        "validation": report_metrics(model, validation_labeled),
        "test": report_metrics(model, test_labeled),
    }
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "rule-policy.json", metadata=report)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

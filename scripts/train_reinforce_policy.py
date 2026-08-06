#!/usr/bin/env python3
"""Fine-tune a legal-action MLP by on-policy REINFORCE versus rule Teacher."""

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
    StateValueBaseline,
    collect_policy_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/dagger-classic-v2-run1/rule-policy-dagger.json"),
    )
    parser.add_argument("--profile", choices=("classic", "core"), default="classic")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--episodes-per-iteration", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--reward-scale", type=float, default=40.0)
    parser.add_argument(
        "--value-checkpoint",
        type=Path,
        help="可选：继续训练已保存的 actor-critic 价值基线",
    )
    parser.add_argument("--value-epochs", type=int, default=4)
    parser.add_argument("--value-learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/reinforce-classic")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations <= 0 or args.episodes_per_iteration <= 0:
        raise ValueError("iterations 和 episodes-per-iteration 必须为正数")
    if args.value_epochs <= 0 or args.value_learning_rate <= 0:
        raise ValueError("value-epochs 和 value-learning-rate 必须为正数")
    model = NeuralRulePolicyModel.load(args.checkpoint)
    critic = (
        StateValueBaseline.load(args.value_checkpoint)
        if args.value_checkpoint
        else StateValueBaseline(model.feature_dim)
    )
    if critic.feature_dim != model.feature_dim:
        raise ValueError("value-checkpoint 与策略检查点特征维度不匹配")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "profile": args.profile,
        "checkpoint_source": str(args.checkpoint),
        "seed": args.seed,
        "iterations": [],
    }
    for iteration in range(1, args.iterations + 1):
        episodes = collect_policy_episodes(
            model,
            episodes=args.episodes_per_iteration,
            profile=args.profile,
            seed=args.seed + iteration * 100_000,
        )
        advantages = critic.advantages(model, episodes, reward_scale=args.reward_scale)
        metrics = model.reinforce(
            episodes,
            learning_rate=args.learning_rate,
            reward_scale=args.reward_scale,
            advantages=advantages,
        )
        value_metrics = critic.fit(
            model,
            episodes,
            reward_scale=args.reward_scale,
            epochs=args.value_epochs,
            learning_rate=args.value_learning_rate,
        )
        report["iterations"].append({"iteration": iteration, **metrics, **value_metrics})
    model.save(output_dir / "rule-policy-reinforce.json", metadata=report)
    critic.save(output_dir / "state-value-baseline.json")
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

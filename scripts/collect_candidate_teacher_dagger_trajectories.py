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
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.agents import GameAction, HeuristicTeacherAgent
from xiamen_mahjong.training import (
    _turn_actions,
    collect_candidate_teacher_dagger_trajectories,
    split_trajectories_by_hand,
    trajectory_manifest,
    write_trajectory_replay_index,
    write_trajectory_jsonl,
)


class SingleInterventionEpsilonBehavior:
    """One randomized candidate action per hand, with target-policy suffix.

    A target candidate-decision index is sampled before each hand.  Before and
    after that index this wrapper exactly follows the frozen base policy; only
    the selected decision uses a uniform epsilon mixture.  Its terminal score
    is consequently a valid outcome for the selected action followed by the
    target policy, unlike persistent epsilon self-play.
    """

    def __init__(
        self,
        policy,
        *,
        epsilon: float,
        seed: int,
        max_decisions: int,
        intervention_phase: str,
    ):
        if not 0.0 <= epsilon < 1.0:
            raise ValueError("uniform-exploration-probability 必须在 [0, 1) 内")
        if max_decisions <= 0:
            raise ValueError("intervention-max-decisions 必须为正数")
        if intervention_phase not in {"all", "discard", "response"}:
            raise ValueError("intervention-phase 必须是 all、discard 或 response")
        self.policy = policy
        self.epsilon = float(epsilon)
        self.rng = random.Random(seed)
        self.max_decisions = max_decisions
        self.intervention_phase = intervention_phase
        self.target_decision_index = 0
        self.eligible_decision_index = 0
        self._last_action: GameAction | None = None
        self._last_probability: float | None = None

    def reset_episode(self) -> None:
        self.target_decision_index = self.rng.randrange(self.max_decisions)
        self.eligible_decision_index = 0
        self._last_action = None
        self._last_probability = None

    def _sample(
        self,
        legal: tuple[GameAction, ...],
        base_action: GameAction,
        *,
        is_response: bool,
    ) -> GameAction:
        if base_action not in legal:
            raise RuntimeError("基础策略选择了规则引擎未列出的动作")
        selected = base_action
        probability = 1.0
        phase = "response" if is_response else "discard"
        eligible = self.intervention_phase in {"all", phase}
        if eligible and self.eligible_decision_index == self.target_decision_index:
            if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
                selected = legal[self.rng.randrange(len(legal))]
            probability = self.epsilon / len(legal)
            if selected == base_action:
                probability += 1.0 - self.epsilon
        if eligible:
            self.eligible_decision_index += 1
        self._last_action = selected
        self._last_probability = probability
        return selected

    def choose_turn_action(self, game, player_id: int) -> GameAction:
        legal = tuple(_turn_actions(game, player_id))
        return self._sample(
            legal, self.policy.choose_turn_action(game, player_id), is_response=False
        )

    def choose_response(self, game, player_id: int, options) -> GameAction:
        legal = tuple(options)
        return self._sample(
            legal, self.policy.choose_response(game, player_id, legal), is_response=True
        )

    def action_probability(
        self,
        game,
        player_id: int,
        legal: tuple[GameAction, ...],
        action: GameAction,
        *,
        is_response: bool,
    ) -> float:
        del game, player_id, legal, is_response
        if action != self._last_action or self._last_probability is None:
            raise ValueError("action_probability 必须紧随同一行为策略动作调用")
        return self._last_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base_group = parser.add_mutually_exclusive_group(required=True)
    base_group.add_argument(
        "--checkpoint",
        type=Path,
        help="冻结的 policy-value 行为基线；只用于收集，不会在本脚本中更新",
    )
    base_group.add_argument(
        "--teacher-base",
        action="store_true",
        help=(
            "以网页默认 HeuristicTeacher 作为单点随机干预的行为与后缀基线；"
            "用于收集直接对 Teacher 的因果 response 数据。"
        ),
    )
    parser.add_argument("--profile", choices=("classic", "core"), default="classic")
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20265004)
    parser.add_argument(
        "--uniform-exploration-probability",
        type=float,
        default=0.0,
        help="候选座位以该概率均匀随机合法动作，并写入 executed_probability",
    )
    parser.add_argument(
        "--behavior-seed",
        type=int,
        help="探索随机数种子；省略时稳定使用 --seed 的派生值",
    )
    parser.add_argument(
        "--intervention-max-decisions",
        type=int,
        default=32,
        help="每局从 [0, N) 随机选择一个候选决策作为唯一 epsilon 干预点",
    )
    parser.add_argument(
        "--intervention-phase",
        choices=("all", "discard", "response"),
        default="all",
        help="仅在该类候选决策中计数并选择单点干预；response 用于补齐吃/碰/过覆盖。",
    )
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


def load_base_policy(args: argparse.Namespace):
    """Resolve an explicitly named frozen checkpoint or the rule Teacher."""

    if args.teacher_base:
        return HeuristicTeacherAgent(), {
            "kind": "heuristic_teacher",
            "checkpoint": None,
            "inference_device": None,
        }
    if args.checkpoint is None:  # pragma: no cover - argparse enforces one choice.
        raise ValueError("必须指定 --checkpoint 或 --teacher-base")
    return load_policy(args.checkpoint, device=args.device), {
        "kind": "policy_value_checkpoint",
        "checkpoint": str(args.checkpoint),
        "inference_device": args.device,
    }


def main() -> None:
    args = parse_args()
    if args.seed_count <= 0:
        raise ValueError("seed-count 必须为正数")
    if not 0.0 <= args.uniform_exploration_probability < 1.0:
        raise ValueError("uniform-exploration-probability 必须在 [0, 1) 内")
    if args.intervention_max_decisions <= 0:
        raise ValueError("intervention-max-decisions 必须为正数")
    base_policy, base_identity = load_base_policy(args)
    behavior_seed = (
        args.behavior_seed if args.behavior_seed is not None else args.seed + 40_000_000
    )
    policy = SingleInterventionEpsilonBehavior(
        base_policy,
        epsilon=args.uniform_exploration_probability,
        seed=behavior_seed,
        max_decisions=args.intervention_max_decisions,
        intervention_phase=args.intervention_phase,
    )
    trajectories, summary = collect_candidate_teacher_dagger_trajectories(
        policy,
        seed_count=args.seed_count,
        profile=args.profile,
        seed=args.seed,
        behavior_metadata={
            "behavior_policy": "single_intervention_epsilon_uniform",
            "base_policy": base_identity["kind"],
            "uniform_exploration_probability": args.uniform_exploration_probability,
            "intervention_max_decisions": args.intervention_max_decisions,
            "intervention_phase": args.intervention_phase,
            "behavior_seed": behavior_seed,
        },
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
        "behavior_base": base_identity,
        "behavior_policy": {
            "type": "single_intervention_epsilon_uniform",
            "base_policy": base_identity["kind"],
            "uniform_exploration_probability": args.uniform_exploration_probability,
            "intervention_max_decisions": args.intervention_max_decisions,
            "intervention_phase": args.intervention_phase,
            "seed_in_training_jsonl": False,
        },
        "inference_device": base_identity["inference_device"],
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

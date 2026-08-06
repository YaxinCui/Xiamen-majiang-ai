#!/usr/bin/env python3
"""Audit a public afterstate ensemble without authorizing action selection.

The script reads safe trajectory exports and reports only aggregate calibration,
ensemble spread and *hypothetical* policy-prior drift.  It intentionally does
not expose a selector or evaluate a replacement action: logged terminal labels
belong only to the action actually executed in each trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import TeacherDecision, read_trajectory_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="可重复指定；至少两个独立 seed 的 afterstate checkpoint",
    )
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        required=True,
        help="可重复指定；只可追加与其他输入墙组独立的同分区数据。",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--decision-phase", choices=("all", "discard", "response"), default="response")
    parser.add_argument("--only-randomized-actions", action="store_true")
    parser.add_argument("--value-scale", type=float, default=80.0)
    parser.add_argument(
        "--policy-logit-margin",
        type=float,
        default=0.25,
        help="仅作审计：与 policy 最优 logit 相差不超过此值的动作视为 policy-prior 集合",
    )
    parser.add_argument(
        "--policy-top-k",
        type=int,
        help="仅作审计：改为只在 policy 概率最高的 k 个合法动作中比较 outcome LCB。",
    )
    parser.add_argument(
        "--score-lcb-z",
        type=float,
        default=1.0,
        help="仅作审计：mean(score) - z × ensemble_std(score)",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("均值需要至少一个值")
    return sum(values) / len(values)


def population_std(values: Sequence[float]) -> float:
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def best_index(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def selected_decisions(
    paths: Iterable[Path], *, phase: str, only_randomized_actions: bool
) -> Iterable[tuple[TeacherDecision, dict[str, object]]]:
    for path in paths:
        for trajectory in read_trajectory_jsonl(path):
            scores = trajectory.outcome.get("scores")
            winner = trajectory.outcome.get("winner")
            if not isinstance(scores, list) or len(scores) != 4:
                raise ValueError("训练轨迹缺少四家终局得分")
            for decision in trajectory.decisions:
                if phase != "all" and decision.state.get("phase") != phase:
                    continue
                if only_randomized_actions and (
                    decision.executed_probability is None
                    or decision.executed_probability >= 1.0
                ):
                    continue
                if decision.executed_index is None:
                    continue
                yield decision, {
                    "score": float(scores[decision.seat]),
                    "own_win": 1.0 if winner == decision.seat else 0.0,
                    "opponent_win": (
                        1.0 if isinstance(winner, int) and winner != decision.seat else 0.0
                    ),
                }


def main() -> None:
    args = parse_args()
    if len(args.checkpoint) < 2:
        raise ValueError("ensemble audit 至少需要两个 checkpoint")
    if args.value_scale <= 0:
        raise ValueError("value-scale 必须为正数")
    if args.policy_logit_margin < 0 or args.score_lcb_z < 0:
        raise ValueError("policy-logit-margin 和 score-lcb-z 不能为负数")
    if args.policy_top_k is not None and args.policy_top_k <= 0:
        raise ValueError("policy-top-k 必须为正数")
    agents = [TorchPolicyValueAgent.load(path, device=args.device) for path in args.checkpoint]
    if any(agent.architecture != "candidate_mlp" for agent in agents):
        raise ValueError("afterstate ensemble 当前只支持 candidate_mlp")
    score_errors: list[float] = []
    own_brier: list[float] = []
    opponent_brier: list[float] = []
    score_spreads: list[float] = []
    own_spreads: list[float] = []
    opponent_spreads: list[float] = []
    policy_spreads: list[float] = []
    raw_changed = 0
    lcb_changed = 0
    prior_lcb_changed = 0
    unanimous_raw = 0
    decisions = 0
    action_count = 0
    kinds: dict[str, int] = {}
    for decision, target in selected_decisions(
        args.data,
        phase=args.decision_phase,
        only_randomized_actions=args.only_randomized_actions,
    ):
        outputs = [agent.afterstate_outcomes(decision) for agent in agents]
        if any(output is None for output in outputs):
            raise ValueError("checkpoint 缺少 afterstate outcome head")
        safe_outputs = [output for output in outputs if output is not None]
        policy_logits, _value = agents[0].policy_value(decision)
        if any(
            any(abs(left - right) > 1e-6 for left, right in zip(policy_logits, agent.policy_value(decision)[0]))
            for agent in agents[1:]
        ):
            raise ValueError("ensemble policy logits 不一致；必须来自同一冻结 policy")
        score_by_action = [
            [output[0][index] for output in safe_outputs]
            for index in range(len(decision.legal_actions))
        ]
        own_by_action = [
            [output[1][index] for output in safe_outputs]
            for index in range(len(decision.legal_actions))
        ]
        opponent_by_action = [
            [output[2][index] for output in safe_outputs]
            for index in range(len(decision.legal_actions))
        ]
        score_means = [mean(values) for values in score_by_action]
        score_stds = [population_std(values) for values in score_by_action]
        score_lcbs = [
            value - args.score_lcb_z * spread
            for value, spread in zip(score_means, score_stds)
        ]
        own_means = [mean(values) for values in own_by_action]
        opponent_means = [mean(values) for values in opponent_by_action]
        policy_index = best_index(policy_logits)
        raw_indices = [best_index(output[0]) for output in safe_outputs]
        raw_index = best_index(score_means)
        lcb_index = best_index(score_lcbs)
        if args.policy_top_k is None:
            prior_eligible = [
                index
                for index, logit in enumerate(policy_logits)
                if logit >= policy_logits[policy_index] - args.policy_logit_margin
            ]
        else:
            prior_eligible = sorted(
                range(len(policy_logits)),
                key=lambda index: (policy_logits[index], -index),
                reverse=True,
            )[: args.policy_top_k]
        prior_lcb_index = max(
            prior_eligible,
            key=lambda index: (score_lcbs[index], -index),
        )
        raw_changed += raw_index != policy_index
        lcb_changed += lcb_index != policy_index
        prior_lcb_changed += prior_lcb_index != policy_index
        unanimous_raw += len(set(raw_indices)) == 1
        executed = decision.executed_index
        score_errors.append(abs(score_means[executed] * args.value_scale - float(target["score"])))
        own_brier.append((own_means[executed] - float(target["own_win"])) ** 2)
        opponent_brier.append(
            (opponent_means[executed] - float(target["opponent_win"])) ** 2
        )
        score_spreads.extend(spread * args.value_scale for spread in score_stds)
        own_spreads.extend(population_std(values) for values in own_by_action)
        opponent_spreads.extend(population_std(values) for values in opponent_by_action)
        policy_spreads.append(score_stds[policy_index] * args.value_scale)
        decisions += 1
        action_count += len(decision.legal_actions)
        for action in decision.legal_actions:
            kinds[action.kind] = kinds.get(action.kind, 0) + 1
    if decisions <= 0:
        raise ValueError("没有符合筛选条件的审计决策")
    payload = {
        "status": "audit_only_not_authorized_for_action_selection",
        "checkpoints": [str(path) for path in args.checkpoint],
        "data": [str(path) for path in args.data],
        "members": len(agents),
        "decision_phase": args.decision_phase,
        "only_randomized_actions": args.only_randomized_actions,
        "policy_prior_audit": {
            "policy_logit_margin": args.policy_logit_margin,
            "policy_top_k": args.policy_top_k,
            "score_lcb_z": args.score_lcb_z,
            "raw_outcome_argmax_differs_from_policy_rate": raw_changed / decisions,
            "lcb_outcome_argmax_differs_from_policy_rate": lcb_changed / decisions,
            "policy_prior_lcb_argmax_differs_from_policy_rate": prior_lcb_changed / decisions,
            "raw_outcome_argmax_unanimous_rate": unanimous_raw / decisions,
        },
        "calibration_on_logged_actions": {
            "decisions": decisions,
            "score_mae_points": mean(score_errors),
            "own_win_brier": mean(own_brier),
            "opponent_win_brier": mean(opponent_brier),
        },
        "ensemble_spread": {
            "actions": action_count,
            "score_std_points_mean": mean(score_spreads),
            "score_std_points_policy_action_mean": mean(policy_spreads),
            "own_win_probability_std_mean": mean(own_spreads),
            "opponent_win_probability_std_mean": mean(opponent_spreads),
        },
        "legal_action_kinds": dict(sorted(kinds.items())),
        "warning": (
            "Drift rates are hypothetical. This data logs one executed action per state; "
            "it does not identify the terminal value of the proposed replacement action."
        ),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

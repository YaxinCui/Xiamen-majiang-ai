#!/usr/bin/env python3
"""Screen a response-only policy-prior afterstate ensemble against run4.

This is an explicit research evaluator, not a browser deployment path.  Its
candidate policy retains the frozen policy for every turn/discard and considers
only the policy top-k legal response actions.  Within that set it chooses the
maximum ensemble lower-confidence expected terminal score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.agents import GameAction
from xiamen_mahjong.evaluation import evaluate_against_teacher, paired_score_comparison
from xiamen_mahjong.game import XiamenMahjongGame
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import _decision, _turn_actions


class ResponsePolicyPriorEnsemble:
    """Use LCB outcome ranking only inside a frozen policy's response top-k."""

    def __init__(self, members: Sequence[TorchPolicyValueAgent], *, top_k: int, score_lcb_z: float):
        if len(members) < 2:
            raise ValueError("response ensemble 至少需要两个成员")
        if top_k <= 0 or score_lcb_z < 0:
            raise ValueError("top-k 必须为正数，score-lcb-z 不能为负数")
        self.members = tuple(members)
        self.top_k = top_k
        self.score_lcb_z = score_lcb_z

    @staticmethod
    def _best_index(scores: Sequence[float]) -> int:
        return max(range(len(scores)), key=lambda index: (scores[index], -index))

    def choose_turn_action(self, game: XiamenMahjongGame, player_id: int) -> GameAction:
        # The response model has no turn/discard intervention data.  Keeping
        # this exact base path prevents accidental extrapolation to discard.
        return self.members[0].choose_turn_action(game, player_id)

    def choose_response(
        self, game: XiamenMahjongGame, player_id: int, options: Sequence[GameAction]
    ) -> GameAction:
        legal = tuple(options)
        decision = _decision(game, game.seed or 0, player_id, legal, legal[0])
        policy_logits, _value = self.members[0].policy_value(decision)
        outcomes = [member.afterstate_outcomes(decision) for member in self.members]
        if any(output is None for output in outcomes):
            raise ValueError("afterstate ensemble 成员缺少 outcome head")
        safe_outcomes = [output for output in outcomes if output is not None]
        for member in self.members[1:]:
            logits, _member_value = member.policy_value(decision)
            if len(logits) != len(policy_logits) or any(
                abs(left - right) > 1e-6 for left, right in zip(logits, policy_logits)
            ):
                raise ValueError("ensemble 成员的冻结 policy 不一致")
        ranked_policy = sorted(
            range(len(legal)),
            key=lambda index: (policy_logits[index], -index),
            reverse=True,
        )
        eligible = ranked_policy[: self.top_k]
        def lcb(index: int) -> float:
            values = [output[0][index] for output in safe_outcomes]
            average = sum(values) / len(values)
            variance = sum((value - average) ** 2 for value in values) / len(values)
            return average - self.score_lcb_z * variance**0.5
        chosen_index = max(eligible, key=lambda index: (lcb(index), -index))
        return legal[chosen_index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--profile", choices=("classic", "core"), default="classic")
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--score-lcb-z", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hands <= 0:
        raise ValueError("hands 必须为正数")
    members = [TorchPolicyValueAgent.load(path, device=args.device) for path in args.checkpoint]
    reference = TorchPolicyValueAgent.load(args.reference, device=args.device)
    policy = ResponsePolicyPriorEnsemble(
        members, top_k=args.top_k, score_lcb_z=args.score_lcb_z
    )
    result = evaluate_against_teacher(
        policy, hands=args.hands, profile=args.profile, seed=args.seed
    )
    reference_result = evaluate_against_teacher(
        reference, hands=args.hands, profile=args.profile, seed=args.seed
    )
    payload = {
        "status": "experimental_not_authorized_for_browser_or_promotion",
        "candidate": {
            "type": "response_policy_prior_afterstate_ensemble_lcb",
            "members": [str(path) for path in args.checkpoint],
            "top_k": args.top_k,
            "score_lcb_z": args.score_lcb_z,
            "turn_selection": "frozen_policy_only",
            "response_selection": "max_score_lcb_within_policy_top_k",
        },
        "reference_checkpoint": str(args.reference),
        "profile": args.profile,
        "hands": args.hands,
        "seed": args.seed,
        "inference_device": args.device,
        "candidate_result": result.payload(),
        "reference_result": reference_result.payload(),
    }
    payload.update(paired_score_comparison(result, reference_result))
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

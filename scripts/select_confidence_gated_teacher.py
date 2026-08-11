#!/usr/bin/env python3
"""Evaluate the fixed run3 confidence-gated Teacher on fresh walls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.agents import HeuristicTeacherAgent
from xiamen_mahjong.evaluation import (
    PolicyEvaluation,
    evaluate_against_teacher,
    paired_score_comparison,
)
from xiamen_mahjong.teacher_anchored import ConfidenceGatedTeacherAgent
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent


CHECKPOINT = Path("artifacts/policy-value-classic-v1-run3/policy-value.pt")
CHECKPOINT_SHA256 = "01616a38ff34af71f1774995d5d3f9e08c4408fd85757ad563d9dfa339b19f70"
MINIMUM_POLICY_ADVANTAGE = 2.4152190685272217
SELECTION_HANDS = 100
SELECTION_SEED = 202619000
TERMINAL_HANDS = 400
TERMINAL_SEED = 202619200


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-hands", type=int, default=SELECTION_HANDS)
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--terminal-hands", type=int, default=TERMINAL_HANDS)
    parser.add_argument("--terminal-seed", type=int, default=TERMINAL_SEED)
    parser.add_argument("--profile", choices=("classic",), default="classic")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fixed_candidate(*, device: str) -> ConfidenceGatedTeacherAgent:
    actual_sha = _sha256(CHECKPOINT)
    if actual_sha != CHECKPOINT_SHA256:
        raise ValueError(
            f"run3 checkpoint SHA-256 不匹配：{actual_sha} != {CHECKPOINT_SHA256}"
        )
    policy = TorchPolicyValueAgent.load(CHECKPOINT, device=device)
    return ConfidenceGatedTeacherAgent(
        policy,
        minimum_policy_advantage=MINIMUM_POLICY_ADVANTAGE,
        allowed_teacher_kinds=("discard",),
    )


def baseline_evaluation(*, hands: int, seed: int, profile: str) -> PolicyEvaluation:
    return evaluate_against_teacher(
        HeuristicTeacherAgent(), hands=hands, seed=seed, profile=profile
    )


def audit_candidate(
    *,
    baseline: PolicyEvaluation,
    hands: int,
    seed: int,
    profile: str,
    device: str,
) -> dict[str, Any]:
    agent = fixed_candidate(device=device)
    result = evaluate_against_teacher(agent, hands=hands, seed=seed, profile=profile)
    comparison = paired_score_comparison(result, baseline)
    return {
        "candidate": result.payload(),
        "behavior": {
            "eligible_ordinary_discard_decisions": agent.eligible_decisions,
            "overrides": agent.override_count,
            "override_rate": (
                agent.override_count / agent.eligible_decisions
                if agent.eligible_decisions
                else None
            ),
        },
        "paired_against_heuristic_teacher": comparison,
        "passes": comparison["paired_seed_score_delta_95pct_low"] > 0.0,
    }


def main() -> None:
    args = parse_args()
    if (
        args.selection_hands != SELECTION_HANDS
        or args.selection_seed != SELECTION_SEED
        or args.terminal_hands != TERMINAL_HANDS
        or args.terminal_seed != TERMINAL_SEED
    ):
        raise ValueError("只接受 v1 预注册的墙数与 seed 范围")
    actual_sha = _sha256(CHECKPOINT)
    if actual_sha != CHECKPOINT_SHA256:
        raise ValueError("预注册 checkpoint 已发生变化")

    baseline = baseline_evaluation(
        hands=args.selection_hands,
        seed=args.selection_seed,
        profile=args.profile,
    )
    selection = audit_candidate(
        baseline=baseline,
        hands=args.selection_hands,
        seed=args.selection_seed,
        profile=args.profile,
        device=args.device,
    )
    payload: dict[str, Any] = {
        "status": "selection_rejected_terminal_unread",
        "protocol": {
            "profile": args.profile,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "minimum_policy_advantage": MINIMUM_POLICY_ADVANTAGE,
            "allowed_teacher_action_kinds": ["discard"],
            "selection_hands": args.selection_hands,
            "selection_seed": args.selection_seed,
            "terminal_hands": args.terminal_hands,
            "terminal_seed": args.terminal_seed,
            "terminal_read": False,
            "promotion_gate": "paired_seed_score_delta_95pct_low > 0",
            "device": args.device,
        },
        "selection_baseline": baseline.payload(),
        "selection_audit": selection,
    }
    if selection["passes"]:
        terminal_baseline = baseline_evaluation(
            hands=args.terminal_hands,
            seed=args.terminal_seed,
            profile=args.profile,
        )
        terminal = audit_candidate(
            baseline=terminal_baseline,
            hands=args.terminal_hands,
            seed=args.terminal_seed,
            profile=args.profile,
            device=args.device,
        )
        payload["status"] = "terminal_audited_not_authorized_for_deployment"
        payload["terminal_baseline"] = terminal_baseline.payload()
        payload["terminal_audit"] = terminal
        payload["protocol"]["terminal_read"] = True
        payload["ready_for_human_evaluation_review"] = terminal["passes"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

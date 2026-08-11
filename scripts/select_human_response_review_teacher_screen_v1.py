#!/usr/bin/env python3
"""Run the frozen response-review gate against Teacher on fresh physical walls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.select_human_response_review_residual_gate_v1 import RESPONSE_SCOPE
from xiamen_mahjong.agents import HeuristicTeacherAgent
from xiamen_mahjong.evaluation import (
    PolicyEvaluation,
    evaluate_against_teacher,
    paired_score_comparison,
)
from xiamen_mahjong.teacher_anchored import ConfidenceGatedTeacherAgent
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent


SELECTION_HANDS = 100
SELECTION_SEED = 202634000
TERMINAL_HANDS = 400
TERMINAL_SEED = 202634200


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gate_identity(
    gate_report: Path,
    *,
    gate_report_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Accept only an independently passed response test and frozen identities."""

    actual_gate_sha = _sha256(gate_report)
    if actual_gate_sha != gate_report_sha256:
        raise ValueError("response gate report SHA-256 与冻结身份不匹配")
    payload = json.loads(gate_report.read_text(encoding="utf-8"))
    winner = payload.get("validation", {}).get("winner")
    test = payload.get("test", {})
    protocol = payload.get("protocol", {})
    if (
        payload.get("status")
        != "response_review_test_gate_passed_ready_for_100_wall_teacher_screen"
        or payload.get("checkpoint", {}).get("sha256") != checkpoint_sha256
        or protocol.get("source") != "local_human_response_review_opt_in"
        or protocol.get("scope") != RESPONSE_SCOPE
        or not isinstance(winner, dict)
        or test.get("passes") is not True
        or test.get("summary", {}).get("scope") != RESPONSE_SCOPE
    ):
        raise ValueError("response gate report 未通过固定 test 契约")
    margin = winner.get("strict_margin")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
        or float(margin) < 0.0
    ):
        raise ValueError("response gate report 的 strict margin 无效")
    return {
        "gate_report_sha256": actual_gate_sha,
        "checkpoint_sha256": checkpoint_sha256,
        "minimum_policy_advantage": float(margin),
        "scope": RESPONSE_SCOPE,
    }


def fixed_candidate(
    checkpoint: Path,
    *,
    identity: dict[str, Any],
    device: str,
) -> ConfidenceGatedTeacherAgent:
    actual_checkpoint_sha = _sha256(checkpoint)
    if actual_checkpoint_sha != identity["checkpoint_sha256"]:
        raise ValueError("response checkpoint SHA-256 与 gate 身份不匹配")
    policy = TorchPolicyValueAgent.load(checkpoint, device=device)
    return ConfidenceGatedTeacherAgent(
        policy,
        minimum_policy_advantage=float(identity["minimum_policy_advantage"]),
        allowed_teacher_kinds=("pass", "chi", "pong"),
        allowed_alternative_kinds=("pass", "chi", "pong"),
        ordinary_response_only=True,
    )


def baseline_evaluation(*, hands: int, seed: int) -> PolicyEvaluation:
    return evaluate_against_teacher(
        HeuristicTeacherAgent(), hands=hands, seed=seed, profile="classic"
    )


def audit_candidate(
    *,
    checkpoint: Path,
    identity: dict[str, Any],
    baseline: PolicyEvaluation,
    hands: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    agent = fixed_candidate(checkpoint, identity=identity, device=device)
    result = evaluate_against_teacher(
        agent, hands=hands, seed=seed, profile="classic"
    )
    comparison = paired_score_comparison(result, baseline)
    return {
        "candidate": result.payload(),
        "behavior": {
            "eligible_ordinary_response_decisions": agent.eligible_decisions,
            "overrides": agent.override_count,
            "override_rate": (
                agent.override_count / agent.eligible_decisions
                if agent.eligible_decisions
                else None
            ),
            "special_response_override_scope": "forbidden",
        },
        "paired_against_heuristic_teacher": comparison,
        "passes": comparison["paired_seed_score_delta_95pct_low"] > 0.0,
    }


def run_screen(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    gate_report: Path,
    gate_report_sha256: str,
    output: Path,
    device: str,
    selection_hands: int = SELECTION_HANDS,
    selection_seed: int = SELECTION_SEED,
    terminal_hands: int = TERMINAL_HANDS,
    terminal_seed: int = TERMINAL_SEED,
) -> dict[str, Any]:
    if (
        selection_hands != SELECTION_HANDS
        or selection_seed != SELECTION_SEED
        or terminal_hands != TERMINAL_HANDS
        or terminal_seed != TERMINAL_SEED
    ):
        raise ValueError("只接受 response review v1 预注册的墙数与 seed 范围")
    identity = load_gate_identity(
        gate_report,
        gate_report_sha256=gate_report_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    selection_baseline = baseline_evaluation(
        hands=selection_hands, seed=selection_seed
    )
    selection = audit_candidate(
        checkpoint=checkpoint,
        identity=identity,
        baseline=selection_baseline,
        hands=selection_hands,
        seed=selection_seed,
        device=device,
    )
    payload: dict[str, Any] = {
        "status": "selection_rejected_terminal_unread",
        "protocol": {
            **identity,
            "profile": "classic",
            "selection_hands": selection_hands,
            "selection_seed": selection_seed,
            "terminal_hands": terminal_hands,
            "terminal_seed": terminal_seed,
            "terminal_read": False,
            "promotion_gate": "paired_seed_score_delta_95pct_low > 0",
            "device": device,
        },
        "selection_baseline": selection_baseline.payload(),
        "selection_audit": selection,
    }
    if selection["passes"]:
        terminal_baseline = baseline_evaluation(
            hands=terminal_hands, seed=terminal_seed
        )
        terminal = audit_candidate(
            checkpoint=checkpoint,
            identity=identity,
            baseline=terminal_baseline,
            hands=terminal_hands,
            seed=terminal_seed,
            device=device,
        )
        payload["status"] = "terminal_audited_not_authorized_for_deployment"
        payload["terminal_baseline"] = terminal_baseline.payload()
        payload["terminal_audit"] = terminal
        payload["protocol"]["terminal_read"] = True
        payload["ready_for_human_evaluation_review"] = terminal["passes"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--gate-report-sha256", required=True)
    parser.add_argument("--selection-hands", type=int, default=SELECTION_HANDS)
    parser.add_argument("--selection-seed", type=int, default=SELECTION_SEED)
    parser.add_argument("--terminal-hands", type=int, default=TERMINAL_HANDS)
    parser.add_argument("--terminal-seed", type=int, default=TERMINAL_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_screen(
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        gate_report=args.gate_report,
        gate_report_sha256=args.gate_report_sha256,
        output=args.output,
        device=args.device,
        selection_hands=args.selection_hands,
        selection_seed=args.selection_seed,
        terminal_hands=args.terminal_hands,
        terminal_seed=args.terminal_seed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

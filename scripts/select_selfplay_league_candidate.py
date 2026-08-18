#!/usr/bin/env python3
"""Gate one fresh self-play checkpoint against the frozen rule Teacher.

Selection and terminal walls are deliberately separate.  A failed selection
does not load or summarize terminal results, preventing an underpowered PPO
run from repeatedly tuning itself on its final benchmark.
"""

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
from xiamen_mahjong.evaluation import evaluate_against_teacher, paired_score_comparison
from xiamen_mahjong.teacher_anchored import TeacherAnchoredPolicyAgent
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_lower_bound(audit: dict[str, float]) -> bool:
    """The sole promotion predicate for selection and terminal."""

    return audit["paired_seed_score_delta_95pct_low"] > 0.0


def audit_candidate(
    candidate: Any,
    *,
    profile: str,
    hands: int,
    seed: int,
) -> dict[str, Any]:
    result = evaluate_against_teacher(
        candidate, hands=hands, profile=profile, seed=seed
    )
    baseline = evaluate_against_teacher(
        HeuristicTeacherAgent(), hands=hands, profile=profile, seed=seed
    )
    paired = paired_score_comparison(result, baseline)
    return {
        "candidate": result.payload(),
        "teacher_baseline": baseline.payload(),
        "paired_against_heuristic_teacher": paired,
        "passes": positive_lower_bound(paired),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=("classic", "core"), default="classic")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--teacher-prior-margin",
        type=float,
        default=0.0,
        help="与训练相同的 Teacher-anchored residual prior；0 表示普通 policy",
    )
    parser.add_argument("--selection-hands", type=int, required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--terminal-hands", type=int, required=True)
    parser.add_argument("--terminal-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError("self-play checkpoint 不存在")
    if args.selection_hands <= 1 or args.terminal_hands <= 1:
        raise ValueError("selection 与 terminal 都至少需要两副物理牌墙")
    if args.selection_seed == args.terminal_seed:
        raise ValueError("selection 与 terminal 必须使用不同起始种子")
    if args.teacher_prior_margin < 0.0:
        raise ValueError("Teacher prior margin 不能为负数")
    residual = TorchPolicyValueAgent.load(args.checkpoint, device=args.device)
    candidate: Any = (
        TeacherAnchoredPolicyAgent(residual, margin=args.teacher_prior_margin)
        if args.teacher_prior_margin > 0.0
        else residual
    )
    selection = audit_candidate(
        candidate,
        profile=args.profile,
        hands=args.selection_hands,
        seed=args.selection_seed,
    )
    payload: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "profile": args.profile,
        "inference_device": args.device,
        "teacher_prior_margin": args.teacher_prior_margin,
        "promotion_gate": "paired_seed_score_delta_95pct_low > 0",
        "selection": {
            "hands": args.selection_hands,
            "seed": args.selection_seed,
            "audit": selection,
        },
        "terminal": {
            "hands": args.terminal_hands,
            "seed": args.terminal_seed,
            "read": False,
        },
    }
    if not selection["passes"]:
        payload["status"] = "selection_rejected_terminal_unread"
    else:
        terminal = audit_candidate(
            candidate,
            profile=args.profile,
            hands=args.terminal_hands,
            seed=args.terminal_seed,
        )
        payload["terminal"] = {
            "hands": args.terminal_hands,
            "seed": args.terminal_seed,
            "read": True,
            "audit": terminal,
        }
        payload["status"] = (
            "terminal_passed_not_authorized_for_deployment"
            if terminal["passes"]
            else "terminal_rejected_not_authorized_for_deployment"
        )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

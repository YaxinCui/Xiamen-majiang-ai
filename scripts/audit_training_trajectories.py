#!/usr/bin/env python3
"""Audit safe trajectory-v4 exports before model training.

The audit checks that every decision points to a prefix of the enclosing
trajectory's public action stream, that the stored short history is exactly the
corresponding actor-relative suffix, and that replay-only fields are absent.
It also emits coverage counts for response and special-rule decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.training import (
    read_trajectory_jsonl,
    trajectory_contract_audit,
    trajectory_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="一个或多个安全导出的 *.trajectories.jsonl；可重复指定",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="可选 JSON 报告路径；报告不含单局种子或暗牌",
    )
    parser.add_argument("--minimum-hands", type=int, default=0)
    parser.add_argument(
        "--minimum-teacher-selfplay-hands",
        type=int,
        default=0,
        help="要求完整的 teacher_self_play 物理牌墙数，不以课程样本凑数",
    )
    parser.add_argument("--minimum-decisions", type=int, default=0)
    parser.add_argument("--minimum-response-decisions", type=int, default=0)
    parser.add_argument("--minimum-tour-decisions", type=int, default=0)
    parser.add_argument("--minimum-gold-locked-decisions", type=int, default=0)
    parser.add_argument(
        "--require-complete-history",
        action="store_true",
        help="将任何声明为 recent_window_only 的合成课程记录视为不合格",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.minimum_hands,
        args.minimum_teacher_selfplay_hands,
        args.minimum_decisions,
        args.minimum_response_decisions,
        args.minimum_tour_decisions,
        args.minimum_gold_locked_decisions,
    ) < 0:
        raise ValueError("最小覆盖门槛不能为负数")

    trajectories = []
    for path in args.input:
        trajectories.extend(read_trajectory_jsonl(path))
    manifest = trajectory_manifest(trajectories)
    contract = trajectory_contract_audit(trajectories, require_safe_export=True)
    phase_action_counts = manifest["phase_action_counts"]
    response_decisions = sum(
        count
        for key, count in phase_action_counts.items()
        if key.startswith("response:")
    )
    tour_decisions = sum(
        count
        for level, count in manifest["special_rule_context_counts"]["tour_levels"].items()
        if level != "0"
    )
    gates = {
        "minimum_hands": {
            "required": args.minimum_hands,
            "observed": manifest["hands"],
            "passed": manifest["hands"] >= args.minimum_hands,
        },
        "minimum_decisions": {
            "required": args.minimum_decisions,
            "observed": manifest["decisions"],
            "passed": manifest["decisions"] >= args.minimum_decisions,
        },
        "minimum_teacher_selfplay_hands": {
            "required": args.minimum_teacher_selfplay_hands,
            "observed": manifest["collectors"].get("teacher_self_play", 0),
            "passed": (
                manifest["collectors"].get("teacher_self_play", 0)
                >= args.minimum_teacher_selfplay_hands
            ),
        },
        "minimum_response_decisions": {
            "required": args.minimum_response_decisions,
            "observed": response_decisions,
            "passed": response_decisions >= args.minimum_response_decisions,
        },
        "minimum_tour_decisions": {
            "required": args.minimum_tour_decisions,
            "observed": tour_decisions,
            "passed": tour_decisions >= args.minimum_tour_decisions,
        },
        "minimum_gold_locked_decisions": {
            "required": args.minimum_gold_locked_decisions,
            "observed": manifest["special_rule_context_counts"]["gold_locked_decisions"],
            "passed": (
                manifest["special_rule_context_counts"]["gold_locked_decisions"]
                >= args.minimum_gold_locked_decisions
            ),
        },
        "complete_history": {
            "required": args.require_complete_history,
            "windowed_history_decisions": contract["windowed_history_decisions"],
            "passed": (
                not args.require_complete_history
                or contract["windowed_history_decisions"] == 0
            ),
        },
    }
    report = {
        "audit": "xiamen-training-trajectory-contract-v1",
        "inputs": [str(path) for path in args.input],
        "contract": contract,
        "coverage_gates": gates,
        "manifest": manifest,
        "passed": contract["valid"] and all(
            gate["passed"] for gate in gates.values()
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

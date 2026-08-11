#!/usr/bin/env python3
"""Build an aggregate causal error map for one Teacher response intervention.

The source data must come from the frozen-Teacher, single-intervention
epsilon-uniform collector.  Discovery is restricted to the physical-wall
``train`` partition.  Validation evaluates exactly one frozen candidate:
replace a Teacher ``chi`` by ``pass`` at the logged intervention position and
then resume the Teacher.  No per-hand state, action or reward is exported.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_teacher_response_intervention_ope import (
    teacher_epsilon_propensities,
)
from xiamen_mahjong.off_policy import LoggedIntervention, intervention_estimates


REPORT_VERSION = "xiamen-teacher-response-causal-error-map-v1"
CANDIDATE = "one_selected_teacher_chi_to_pass_then_teacher"
_ALLOWED_PARTITIONS = {"train", "validation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--mode", choices=("discovery", "validate-chi-pass"), required=True
    )
    parser.add_argument(
        "--partition", choices=tuple(sorted(_ALLOWED_PARTITIONS)), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage(wall_remaining: object) -> str:
    if isinstance(wall_remaining, bool) or not isinstance(wall_remaining, int):
        raise ValueError("响应状态缺少整数 wall_remaining")
    if wall_remaining >= 54:
        return "early"
    if wall_remaining >= 36:
        return "middle"
    return "late"


def _tile_family(tile: object) -> str:
    if isinstance(tile, bool) or not isinstance(tile, int) or not 0 <= tile < 34:
        raise ValueError("响应状态缺少合法 last_discard")
    return "honor" if tile >= 27 else "suit"


def _claim_category(teacher_kind: str, state: dict[str, object]) -> str:
    stage = _stage(state.get("wall_remaining"))
    if teacher_kind == "chi":
        # A legal chi can only claim a suited tile, so no redundant tile split.
        return f"chi_{stage}"
    if teacher_kind == "pong":
        return f"pong_{stage}_{_tile_family(state.get('last_discard'))}"
    raise ValueError("因果 claim 地图只接受 chi 或 pong")


def _aggregate(rows: Iterable[LoggedIntervention]) -> dict[str, object]:
    return intervention_estimates(list(rows))


def analyze_paths(
    paths: Iterable[Path], *, mode: str, partition: str
) -> dict[str, object]:
    paths = tuple(paths)
    if not paths:
        raise ValueError("至少需要一个输入文件")
    if partition not in _ALLOWED_PARTITIONS:
        raise ValueError("只允许 train 或 validation")
    if mode == "discovery" and partition != "train":
        raise ValueError("错误地图 discovery 只能读取 train 分区")
    if mode == "validate-chi-pass" and partition != "validation":
        raise ValueError("冻结候选只能在 validation 分区筛选")
    expected_name = f"{partition}.trajectories.jsonl"
    if any(path.name != expected_name for path in paths):
        raise ValueError(f"输入文件名必须为 {expected_name}")

    all_rows: list[LoggedIntervention] = []
    override_rows: list[LoggedIntervention] = []
    discovery_rows: dict[str, list[LoggedIntervention]] = defaultdict(list)
    teacher_kinds: Counter[str] = Counter()
    target_kinds: Counter[str] = Counter()
    scanned_decisions = 0
    randomized_responses = 0
    wall_groups: set[str] = set()

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    trajectory = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number} 不是合法 JSON") from error
                metadata = trajectory.get("source_metadata")
                if not isinstance(metadata, dict):
                    raise ValueError("轨迹缺少 source_metadata")
                if metadata.get("behavior_policy") != "single_intervention_epsilon_uniform":
                    raise ValueError("输入不是单点 epsilon 干预轨迹")
                if metadata.get("base_policy") != "heuristic_teacher":
                    raise ValueError("输入基线不是冻结 Teacher")
                if metadata.get("intervention_phase") != "response":
                    raise ValueError("输入不是 response-only 干预轨迹")
                epsilon = metadata.get("uniform_exploration_probability")
                if (
                    isinstance(epsilon, bool)
                    or not isinstance(epsilon, (int, float))
                    or not 0.0 < float(epsilon) < 1.0
                ):
                    raise ValueError("轨迹缺少合法 epsilon")
                group_id = trajectory.get("split_group_id")
                if not isinstance(group_id, str) or not group_id:
                    raise ValueError("轨迹缺少物理墙 split_group_id")
                scores = trajectory.get("outcome", {}).get("scores")
                if not isinstance(scores, list) or len(scores) != 4:
                    raise ValueError("轨迹缺少四家终局分数")
                decisions = trajectory.get("decisions")
                if not isinstance(decisions, list):
                    raise ValueError("轨迹缺少 decisions")
                wall_groups.add(group_id)

                for decision in decisions:
                    scanned_decisions += 1
                    if not isinstance(decision, dict):
                        raise ValueError("decision 必须为对象")
                    state = decision.get("state")
                    if not isinstance(state, dict) or state.get("phase") != "response":
                        continue
                    executed_probability = decision.get("executed_probability")
                    if (
                        isinstance(executed_probability, bool)
                        or not isinstance(executed_probability, (int, float))
                        or float(executed_probability) >= 1.0
                    ):
                        continue
                    randomized_responses += 1
                    legal_actions = decision.get("legal_actions")
                    teacher_index = decision.get("chosen_index")
                    logged_index = decision.get("executed_index")
                    seat = decision.get("seat")
                    if (
                        not isinstance(legal_actions, list)
                        or not legal_actions
                        or isinstance(teacher_index, bool)
                        or not isinstance(teacher_index, int)
                        or isinstance(logged_index, bool)
                        or not isinstance(logged_index, int)
                        or isinstance(seat, bool)
                        or not isinstance(seat, int)
                        or not 0 <= teacher_index < len(legal_actions)
                        or not 0 <= logged_index < len(legal_actions)
                        or not 0 <= seat < 4
                    ):
                        raise ValueError("随机 response decision 索引不合法")
                    kinds = [action.get("kind") for action in legal_actions]
                    if not all(isinstance(kind, str) for kind in kinds):
                        raise ValueError("legal_actions 缺少 kind")
                    propensities = teacher_epsilon_propensities(
                        action_count=len(legal_actions),
                        teacher_index=teacher_index,
                        epsilon=float(epsilon),
                    )
                    if not math.isclose(
                        propensities[logged_index],
                        float(executed_probability),
                        abs_tol=1e-8,
                    ):
                        raise ValueError("executed_probability 与 epsilon 契约不一致")
                    teacher_kind = str(kinds[teacher_index])
                    teacher_kinds[teacher_kind] += 1

                    target_index = teacher_index
                    if teacher_kind in {"chi", "pong"}:
                        try:
                            pass_index = kinds.index("pass")
                        except ValueError as error:
                            raise ValueError("吃碰响应缺少 pass 支持") from error
                        map_row = LoggedIntervention(
                            group_id=group_id,
                            logged_index=logged_index,
                            propensities=propensities,
                            reward=float(scores[seat]),
                            baseline_index=teacher_index,
                            target_index=pass_index,
                        )
                        discovery_rows[_claim_category(teacher_kind, state)].append(
                            map_row
                        )
                        if mode == "validate-chi-pass" and teacher_kind == "chi":
                            target_index = pass_index

                    target_kind = str(kinds[target_index])
                    target_kinds[target_kind] += 1
                    row = LoggedIntervention(
                        group_id=group_id,
                        logged_index=logged_index,
                        propensities=propensities,
                        reward=float(scores[seat]),
                        baseline_index=teacher_index,
                        target_index=target_index,
                    )
                    all_rows.append(row)
                    if target_index != teacher_index:
                        override_rows.append(row)

    if not all_rows:
        raise ValueError("没有随机 response 干预")

    common: dict[str, object] = {
        "version": REPORT_VERSION,
        "mode": mode,
        "partition": partition,
        "inputs": [
            {"path": str(path), "sha256": sha256(path)} for path in paths
        ],
        "scanned_decisions": scanned_decisions,
        "randomized_response_interventions": randomized_responses,
        "wall_groups": len(wall_groups),
        "teacher_action_kinds": dict(sorted(teacher_kinds.items())),
        "per_decision_state_exported": False,
        "terminal_partition_read": False,
        "causal_scope": "one selected response intervention followed by frozen Teacher",
    }
    if mode == "discovery":
        return {
            **common,
            "status": "train_only_aggregate_causal_error_map",
            "taxonomy": {
                "mutually_exclusive": True,
                "stage_boundaries": {"early_min_wall": 54, "middle_min_wall": 36},
                "categories": sorted(discovery_rows),
                "target": "pass_minus_teacher_claim",
            },
            "categories": {
                category: _aggregate(rows)
                for category, rows in sorted(discovery_rows.items())
            },
            "warning": (
                "Discovery estimates are exploratory and multiplicity-unadjusted. "
                "They cannot authorize a policy or strength claim."
            ),
        }

    if not override_rows:
        raise ValueError("validation 没有 Teacher chi 干预支持")
    policy = _aggregate(all_rows)
    conditional = _aggregate(override_rows)
    policy_ips = policy["ips"]
    conditional_ips = conditional["ips"]
    conditional_support = conditional["support"]
    assert isinstance(policy_ips, dict)
    assert isinstance(conditional_ips, dict)
    assert isinstance(conditional_support, dict)
    gate = {
        "minimum_override_observations": 100,
        "minimum_override_wall_groups": 75,
        "minimum_target_effective_sample_size": 30.0,
        "minimum_baseline_effective_sample_size": 30.0,
        "requires_policy_ips_95pct_low_positive": True,
        "requires_conditional_ips_95pct_low_positive": True,
    }
    passes = (
        len(override_rows) >= int(gate["minimum_override_observations"])
        and int(conditional_ips["groups"]) >= int(gate["minimum_override_wall_groups"])
        and float(conditional_support["target_effective_sample_size"])
        >= float(gate["minimum_target_effective_sample_size"])
        and float(conditional_support["baseline_effective_sample_size"])
        >= float(gate["minimum_baseline_effective_sample_size"])
        and float(policy_ips["95pct_low"]) > 0.0
        and float(conditional_ips["95pct_low"]) > 0.0
    )
    return {
        **common,
        "status": (
            "validation_passed_new_targeted_randomization_required"
            if passes
            else "validation_rejected_no_targeted_collection"
        ),
        "candidate": {
            "name": CANDIDATE,
            "target_action_kinds": dict(sorted(target_kinds.items())),
            "override_observations": len(override_rows),
            "override_rate": len(override_rows) / len(all_rows),
        },
        "estimates": {
            "policy_over_all_randomized_response_positions": policy,
            "conditional_teacher_chi_to_pass": conditional,
        },
        "gate": {**gate, "passes": passes},
        "next_step": (
            "collect fresh binary first-eligible-chi interventions"
            if passes
            else "freeze candidate; keep terminal unread"
        ),
        "warning": (
            "Even a pass identifies only one selected chi replacement under a "
            "Teacher suffix. It is not evidence for never calling chi."
        ),
    }


def main() -> None:
    args = parse_args()
    payload = analyze_paths(args.input, mode=args.mode, partition=args.partition)
    if args.output.exists():
        raise ValueError("输出已存在，拒绝覆盖")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

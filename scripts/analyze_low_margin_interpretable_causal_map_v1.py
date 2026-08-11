#!/usr/bin/env python3
"""Explore a frozen interpretable taxonomy on already-consumed random trials."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.select_low_margin_top2_causal_residual_v1 import _load_split
from scripts.select_low_margin_top2_wall_control_v2 import _load_validation
from xiamen_mahjong.low_margin_intervention import LowMarginCausalRecord
from xiamen_mahjong.wall_control import adjusted_wall_values, policy_wall_components


CONTROL_COEFFICIENT = -2.5
MINIMUM_OVERRIDES = 60
MINIMUM_OVERRIDE_GROUPS = 60
MINIMUM_LOGGED_PER_ARM = 20
FAMILY_ALPHA = 0.05


Predicate = Callable[[LowMarginCausalRecord], bool]


def _tile_class(tile: int) -> str:
    return "honor" if tile >= 27 else "suit"


def _stage(record: LowMarginCausalRecord) -> str:
    remaining = int(record.state["wall_remaining"])
    return "early" if remaining >= 54 else "middle" if remaining >= 36 else "late"


def _multiplicity(record: LowMarginCausalRecord, tile: int) -> int:
    return list(record.state["hand"]).count(tile)


def _connectivity(record: LowMarginCausalRecord, tile: int) -> int:
    if tile >= 27:
        return 0
    suit = tile // 9
    rank = tile % 9
    hand = list(record.state["hand"])
    return sum(
        hand.count(suit * 9 + neighbor)
        for neighbor in range(max(0, rank - 2), min(8, rank + 2) + 1)
        if neighbor != rank
    )


def _public_remaining(record: LowMarginCausalRecord, tile: int) -> int:
    state = record.state
    known = (
        list(state["hand"]).count(tile)
        + int(state["river_counts"][tile])
        + int(state["meld_counts"][tile])
        + (1 if state.get("gold_indicator") == tile else 0)
    )
    return max(0, 4 - known)


def _edge(tile: int) -> bool:
    return tile < 27 and tile % 9 in {0, 8}


def interpretable_category_flags(
    record: LowMarginCausalRecord,
) -> dict[str, bool]:
    if (
        record.executed_arm == "none"
        or record.state is None
        or record.teacher_action is None
        or record.alternative_action is None
        or record.teacher_action.tile is None
        or record.alternative_action.tile is None
    ):
        return {}
    teacher = int(record.teacher_action.tile)
    alternative = int(record.alternative_action.tile)
    teacher_class = _tile_class(teacher)
    alternative_class = _tile_class(alternative)
    multiplicity_delta = _multiplicity(record, alternative) - _multiplicity(
        record, teacher
    )
    connectivity_delta = _connectivity(record, alternative) - _connectivity(
        record, teacher
    )
    remaining_delta = _public_remaining(
        record, alternative
    ) - _public_remaining(record, teacher)
    drawn = record.state.get("drawn_tile")
    teacher_edge = _edge(teacher)
    alternative_edge = _edge(alternative)
    stage = _stage(record)
    exact_tie = float(record.teacher_margin) <= 1e-12

    flags = {
        f"stage_{stage}": True,
        "margin_exact_tie": exact_tie,
        "margin_positive": not exact_tie,
        "actor_dealer": bool(record.state.get("is_dealer")),
        "actor_nondealer": not bool(record.state.get("is_dealer")),
        f"transition_{teacher_class}_to_{alternative_class}": True,
        "alternative_lower_multiplicity": multiplicity_delta < 0,
        "alternative_equal_multiplicity": multiplicity_delta == 0,
        "alternative_higher_multiplicity": multiplicity_delta > 0,
        "alternative_lower_connectivity": connectivity_delta < 0,
        "alternative_equal_connectivity": connectivity_delta == 0,
        "alternative_higher_connectivity": connectivity_delta > 0,
        "alternative_is_drawn_teacher_is_not": alternative == drawn and teacher != drawn,
        "teacher_is_drawn_alternative_is_not": teacher == drawn and alternative != drawn,
        "neither_action_is_drawn": teacher != drawn and alternative != drawn,
        "alternative_more_public_remaining": remaining_delta > 0,
        "alternative_equal_public_remaining": remaining_delta == 0,
        "alternative_less_public_remaining": remaining_delta < 0,
        "alternative_edge_teacher_not": alternative_edge and not teacher_edge,
        "teacher_edge_alternative_not": teacher_edge and not alternative_edge,
        "both_suited_edges": teacher_edge and alternative_edge,
        "both_suited_non_edges": (
            teacher < 27
            and alternative < 27
            and not teacher_edge
            and not alternative_edge
        ),
        "same_suit_alternative_lower_face": (
            teacher < 27
            and alternative < 27
            and teacher // 9 == alternative // 9
            and alternative < teacher
        ),
        "same_suit_alternative_higher_face": (
            teacher < 27
            and alternative < 27
            and teacher // 9 == alternative // 9
            and alternative > teacher
        ),
        "suited_different_suits": (
            teacher < 27
            and alternative < 27
            and teacher // 9 != alternative // 9
        ),
    }
    intersections = (
        "stage_early",
        "stage_middle",
        "stage_late",
        "transition_suit_to_suit",
        "transition_suit_to_honor",
        "transition_honor_to_suit",
        "transition_honor_to_honor",
        "alternative_lower_multiplicity",
        "alternative_equal_multiplicity",
        "alternative_higher_multiplicity",
        "alternative_lower_connectivity",
        "alternative_equal_connectivity",
        "alternative_higher_connectivity",
        "alternative_is_drawn_teacher_is_not",
        "teacher_is_drawn_alternative_is_not",
        "neither_action_is_drawn",
    )
    for name in intersections:
        flags[f"exact_tie__{name}"] = exact_tie and flags.get(name, False)
    return flags


def _category_names(records: Sequence[LowMarginCausalRecord]) -> list[str]:
    names: set[str] = set()
    for record in records:
        names.update(interpretable_category_flags(record))
    return sorted(names)


def _interval(values: Sequence[float], *, z: float = 1.96) -> dict[str, float | int | None]:
    if not values:
        return {"groups": 0, "mean": None, "stderr": None, "low": None, "high": None}
    mean = sum(values) / len(values)
    if len(values) < 2:
        return {"groups": len(values), "mean": mean, "stderr": None, "low": None, "high": None}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    stderr = math.sqrt(variance / len(values))
    return {
        "groups": len(values),
        "mean": mean,
        "stderr": stderr,
        "low": mean - z * stderr,
        "high": mean + z * stderr,
    }


def _evaluate_category(
    records: Sequence[LowMarginCausalRecord], name: str
) -> dict[str, Any]:
    effects = [
        None
        if record.executed_arm == "none"
        else 1.0
        if interpretable_category_flags(record).get(name, False)
        else -1.0
        for record in records
    ]
    components = policy_wall_components(records, effects, threshold=0.0)
    adjusted = adjusted_wall_values(
        components["raw_wall_values"],
        components["control_wall_values"],
        control_coefficient=CONTROL_COEFFICIENT,
    )
    interval = _interval(adjusted)
    support_reasons = []
    if components["overrides"] < MINIMUM_OVERRIDES:
        support_reasons.append("insufficient_overrides")
    if components["override_groups"] < MINIMUM_OVERRIDE_GROUPS:
        support_reasons.append("insufficient_override_groups")
    if components["logged_alternative_assignments"] < MINIMUM_LOGGED_PER_ARM:
        support_reasons.append("insufficient_logged_alternative_support")
    if components["logged_teacher_assignments"] < MINIMUM_LOGGED_PER_ARM:
        support_reasons.append("insufficient_logged_teacher_support")
    return {
        "records": len(records),
        "wall_groups": len(components["group_ids"]),
        "overrides": components["overrides"],
        "override_groups": components["override_groups"],
        "logged_alternative_assignments": components[
            "logged_alternative_assignments"
        ],
        "logged_teacher_assignments": components["logged_teacher_assignments"],
        "control_adjusted_nominal_95pct": interval,
        "support_ready": not support_reasons,
        "support_reasons": support_reasons,
        "_values": adjusted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1-root",
        type=Path,
        default=Path("artifacts/low-margin-top2-causal-residual-v1"),
    )
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=Path("artifacts/low-margin-top2-wall-control-v2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/low-margin-interpretable-causal-map-v1/report.json"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("可解释因果错误地图已存在，拒绝覆盖")

    v1_train, v1_train_files = _load_split(args.v1_root, "train")
    v1_validation, v1_validation_files = _load_split(args.v1_root, "validation")
    v2_validation, v2_validation_files = _load_validation(args.v2_root)
    partitions = {
        "v1_train_consumed_for_exploration": v1_train,
        "v1_validation_consumed_for_exploration": v1_validation,
        "v2_validation_consumed_for_exploration": v2_validation,
    }
    group_sets = [
        {record.group_id for record in records} for records in partitions.values()
    ]
    if any(
        group_sets[left] & group_sets[right]
        for left in range(len(group_sets))
        for right in range(left + 1, len(group_sets))
    ):
        raise ValueError("错误地图三个历史分区 group 重合")
    all_records = [record for records in partitions.values() for record in records]
    names = _category_names(all_records)
    family_z = NormalDist().inv_cdf(1.0 - FAMILY_ALPHA / (2 * len(names)))
    category_reports = []
    candidates = []
    for name in names:
        split_reports = {
            split: _evaluate_category(records, name)
            for split, records in partitions.items()
        }
        pooled = _evaluate_category(all_records, name)
        pooled_family = _interval(pooled["_values"], z=family_z)
        split_means = [
            report["control_adjusted_nominal_95pct"]["mean"]
            for report in split_reports.values()
        ]
        split_z = [
            report["control_adjusted_nominal_95pct"]["mean"]
            / report["control_adjusted_nominal_95pct"]["stderr"]
            if report["control_adjusted_nominal_95pct"]["stderr"]
            else -math.inf
            for report in split_reports.values()
        ]
        eligible = (
            all(report["support_ready"] for report in split_reports.values())
            and all(mean is not None and mean > 0.0 for mean in split_means)
            and pooled_family["low"] is not None
            and pooled_family["low"] > 0.0
        )
        for report in split_reports.values():
            report.pop("_values")
        pooled.pop("_values")
        row = {
            "category": name,
            "partitions": split_reports,
            "pooled": {
                **pooled,
                "family_bonferroni_interval": pooled_family,
            },
            "minimum_partition_z": min(split_z),
            "eligible_for_one_fresh_confirmation": eligible,
        }
        category_reports.append(row)
        if eligible:
            candidates.append(row)
    selected = None
    if candidates:
        selected = max(
            candidates,
            key=lambda row: (
                row["minimum_partition_z"],
                row["pooled"]["family_bonferroni_interval"]["low"],
                row["category"],
            ),
        )["category"]

    payload = {
        "status": (
            "interpretable_category_selected_pending_fresh_protocol"
            if selected is not None
            else "no_stable_interpretable_category_top2_family_frozen"
        ),
        "selected_category": selected,
        "protocol": {
            "control_coefficient": CONTROL_COEFFICIENT,
            "minimum_overrides_per_partition": MINIMUM_OVERRIDES,
            "minimum_override_groups_per_partition": MINIMUM_OVERRIDE_GROUPS,
            "minimum_logged_per_arm_per_partition": MINIMUM_LOGGED_PER_ARM,
            "family_alpha": FAMILY_ALPHA,
            "family_tests": len(names),
            "family_bonferroni_z": family_z,
            "selection": (
                "positive_each_partition_and_pooled_family_lower_bound_positive; "
                "maximize_minimum_partition_z"
            ),
            "strength_claim": False,
            "terminal_read": False,
        },
        "inputs": {
            "v1_train": v1_train_files,
            "v1_consumed_validation": v1_validation_files,
            "v2_consumed_validation": v2_validation_files,
        },
        "partition_counts": {
            name: {
                "records": len(records),
                "wall_groups": len({record.group_id for record in records}),
            }
            for name, records in partitions.items()
        },
        "categories": category_reports,
        "next_decision": (
            "Write a separate fresh randomized confirmation protocol for the one "
            "selected category. Do not deploy or train from this exploratory map."
            if selected is not None
            else "Do not add categories or collect more low-margin top-2 data."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_category": selected,
                "family_tests": len(names),
                "eligible_categories": [
                    row["category"]
                    for row in category_reports
                    if row["eligible_for_one_fresh_confirmation"]
                ],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

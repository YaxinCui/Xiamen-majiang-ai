#!/usr/bin/env python3
"""Collect the preregistered low-margin top-two intervention mechanics pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.low_margin_intervention import (
    LOW_MARGIN_TOP2_BEHAVIOR_VERSION,
    LOW_MARGIN_TOP2_POLICY_VERSION,
    LowMarginTop2InterventionBehavior,
    audit_low_margin_top2_interventions,
)
from xiamen_mahjong.training import (
    collect_candidate_teacher_dagger_trajectories,
    read_trajectory_jsonl,
    trajectory_manifest,
    write_trajectory_jsonl,
)


PHYSICAL_WALLS = 25
FIRST_SEED = 202649000
BEHAVIOR_SEED = 202649025
MAXIMUM_TEACHER_SCORE_MARGIN = 2.0
ALTERNATIVE_ACTION_PROBABILITY = 0.5
MINIMUM_INTERVENTIONS = 80
MINIMUM_WALL_GROUPS = 25


def run_pilot(*, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise ValueError("low-margin pilot 输出已存在且非空，拒绝覆盖")
    output_dir.mkdir(parents=True, exist_ok=True)
    behavior = LowMarginTop2InterventionBehavior(
        seed=BEHAVIOR_SEED,
        maximum_margin=MAXIMUM_TEACHER_SCORE_MARGIN,
        alternative_probability=ALTERNATIVE_ACTION_PROBABILITY,
    )
    started = time.monotonic()
    trajectories, summary = collect_candidate_teacher_dagger_trajectories(
        behavior,
        seed_count=PHYSICAL_WALLS,
        profile="classic",
        seed=FIRST_SEED,
        behavior_metadata={
            "behavior_policy": LOW_MARGIN_TOP2_BEHAVIOR_VERSION,
            "base_policy": "heuristic_teacher",
            "candidate_policy": LOW_MARGIN_TOP2_POLICY_VERSION,
            "alternative_action_probability": ALTERNATIVE_ACTION_PROBABILITY,
            "maximum_teacher_score_margin": MAXIMUM_TEACHER_SCORE_MARGIN,
            "maximum_interventions_per_trajectory": 1,
            "decision_scope": "first_eligible_ordinary_discard",
            "behavior_seed_in_training_jsonl": False,
        },
    )
    data_path = output_dir / "pilot.trajectories.jsonl"
    write_trajectory_jsonl(trajectories, data_path)
    safe_rows = read_trajectory_jsonl(data_path)
    audit = audit_low_margin_top2_interventions(
        safe_rows,
        maximum_margin=MAXIMUM_TEACHER_SCORE_MARGIN,
        minimum_interventions=MINIMUM_INTERVENTIONS,
        minimum_wall_groups=MINIMUM_WALL_GROUPS,
    )
    payload: dict[str, object] = {
        "status": (
            "pilot_ready_for_formal_data_design"
            if audit["structurally_ready"]
            else "pilot_rejected_before_formal_data_design"
        ),
        "protocol": {
            "profile": "classic",
            "physical_walls": PHYSICAL_WALLS,
            "first_seed": FIRST_SEED,
            "seat_rotations": 4,
            "behavior_seed": BEHAVIOR_SEED,
            "behavior_version": LOW_MARGIN_TOP2_BEHAVIOR_VERSION,
            "maximum_teacher_score_margin": MAXIMUM_TEACHER_SCORE_MARGIN,
            "alternative_action_probability": ALTERNATIVE_ACTION_PROBABILITY,
            "maximum_interventions_per_trajectory": 1,
            "minimum_interventions": MINIMUM_INTERVENTIONS,
            "minimum_wall_groups": MINIMUM_WALL_GROUPS,
            "strength_selection": False,
        },
        "runtime_seconds": time.monotonic() - started,
        "collector_summary": summary.payload(),
        "safe_dataset_manifest": trajectory_manifest(safe_rows),
        "behavior_counters": {
            "eligible_trajectories": behavior.eligible_trajectories,
            "alternative_assignments": behavior.alternative_assignments,
            "teacher_assignments": behavior.teacher_assignments,
        },
        "audit": audit,
        "next_decision": (
            "Use only coverage, assignment balance and grouped standard deviation "
            "to size a separately seeded train/validation/terminal collection. "
            "Do not use the pilot mean or confidence interval to select a policy."
        ),
    }
    (output_dir / "pilot-report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/low-margin-top2-intervention-pilot-v1"),
    )
    args = parser.parse_args()
    print(json.dumps(run_pilot(output_dir=args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect the fixed small high-support TwoDraw/Teacher intervention pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xiamen_mahjong.targeted_intervention import (
    TARGETED_TWO_DRAW_BEHAVIOR_VERSION,
    TargetedTwoDrawInterventionBehavior,
    audit_targeted_two_draw_interventions,
)
from xiamen_mahjong.training import (
    collect_candidate_teacher_dagger_trajectories,
    read_trajectory_jsonl,
    trajectory_manifest,
    write_trajectory_jsonl,
)


PHYSICAL_WALLS = 20
FIRST_SEED = 202635000
BEHAVIOR_SEED = 202635020
CANDIDATE_ACTION_PROBABILITY = 0.5
MINIMUM_INTERVENTIONS = 10
MINIMUM_WALL_GROUPS = 20


def run_pilot(*, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise ValueError("targeted pilot 输出已存在且非空，拒绝覆盖")
    output_dir.mkdir(parents=True, exist_ok=True)
    behavior = TargetedTwoDrawInterventionBehavior(
        seed=BEHAVIOR_SEED,
        candidate_probability=CANDIDATE_ACTION_PROBABILITY,
    )
    started = time.monotonic()
    trajectories, summary = collect_candidate_teacher_dagger_trajectories(
        behavior,
        seed_count=PHYSICAL_WALLS,
        profile="classic",
        seed=FIRST_SEED,
        behavior_metadata={
            "behavior_policy": TARGETED_TWO_DRAW_BEHAVIOR_VERSION,
            "base_policy": "heuristic_teacher",
            "candidate_policy": "two_draw_tenpai_reach_v1",
            "candidate_action_probability": CANDIDATE_ACTION_PROBABILITY,
            "maximum_interventions_per_trajectory": 1,
            "decision_scope": "first_ordinary_discard_disagreement",
            "behavior_seed_in_training_jsonl": False,
        },
    )
    data_path = output_dir / "pilot.trajectories.jsonl"
    write_trajectory_jsonl(trajectories, data_path)
    # Re-open the safe export rather than auditing private in-memory records.
    safe_rows = read_trajectory_jsonl(data_path)
    audit = audit_targeted_two_draw_interventions(
        safe_rows,
        minimum_interventions=MINIMUM_INTERVENTIONS,
        minimum_wall_groups=MINIMUM_WALL_GROUPS,
    )
    payload: dict[str, object] = {
        "status": (
            "pilot_ready_for_scale_design"
            if audit["structurally_ready"]
            else "pilot_rejected_before_scale_design"
        ),
        "protocol": {
            "profile": "classic",
            "physical_walls": PHYSICAL_WALLS,
            "first_seed": FIRST_SEED,
            "seat_rotations": 4,
            "behavior_seed": BEHAVIOR_SEED,
            "behavior_version": TARGETED_TWO_DRAW_BEHAVIOR_VERSION,
            "candidate_action_probability": CANDIDATE_ACTION_PROBABILITY,
            "maximum_interventions_per_trajectory": 1,
            "minimum_interventions": MINIMUM_INTERVENTIONS,
            "minimum_wall_groups": MINIMUM_WALL_GROUPS,
            "strength_selection": False,
        },
        "runtime_seconds": time.monotonic() - started,
        "collector_summary": summary.payload(),
        "safe_dataset_manifest": trajectory_manifest(safe_rows),
        "behavior_counters": {
            "disagreement_hands": behavior.disagreement_hands,
            "candidate_assignments": behavior.candidate_assignments,
            "teacher_assignments": behavior.teacher_assignments,
        },
        "audit": audit,
        "next_decision": (
            "Use coverage and grouped variance only to pre-register a new, larger "
            "wall count; this 20-wall pilot cannot select or deploy a policy."
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
        default=Path("artifacts/two-draw-targeted-intervention-pilot-v1"),
    )
    args = parser.parse_args()
    print(json.dumps(run_pilot(output_dir=args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

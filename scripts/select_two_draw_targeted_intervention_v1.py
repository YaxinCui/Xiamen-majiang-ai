#!/usr/bin/env python3
"""Run the fixed 100-wall high-support TwoDraw intervention selection."""

from __future__ import annotations

import argparse
import hashlib
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


PHYSICAL_WALLS = 100
FIRST_SEED = 202635100
BEHAVIOR_SEED = 202635300
CANDIDATE_ACTION_PROBABILITY = 0.5
MINIMUM_INTERVENTIONS = 100
MINIMUM_WALL_GROUPS = 100
PILOT_REPORT_SHA256 = (
    "2f193e813fc9447320502cc0740dbff94b00eec8dc9c1a970eb8125c896c8b28"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_selection(*, output_dir: Path) -> dict[str, object]:
    pilot_report = Path(
        "artifacts/two-draw-targeted-intervention-pilot-v1/pilot-report.json"
    )
    if not pilot_report.is_file() or _sha256(pilot_report) != PILOT_REPORT_SHA256:
        raise ValueError("targeted intervention pilot 身份与预注册协议不匹配")
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise ValueError("targeted selection 输出已存在且非空，拒绝覆盖")
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
    data_path = output_dir / "selection.trajectories.jsonl"
    write_trajectory_jsonl(trajectories, data_path)
    safe_rows = read_trajectory_jsonl(data_path)
    audit = audit_targeted_two_draw_interventions(
        safe_rows,
        minimum_interventions=MINIMUM_INTERVENTIONS,
        minimum_wall_groups=MINIMUM_WALL_GROUPS,
    )
    estimate = audit["horvitz_thompson_candidate_minus_teacher_per_wall"]
    assert isinstance(estimate, dict)
    lower = estimate.get("95pct_low")
    assignment_fraction = audit.get("candidate_assignment_fraction")
    passes = bool(
        audit["structurally_ready"]
        and isinstance(assignment_fraction, (int, float))
        and 0.4 <= float(assignment_fraction) <= 0.6
        and isinstance(lower, (int, float))
        and float(lower) > 0.0
    )
    payload: dict[str, object] = {
        "status": (
            "selection_passed_ready_for_honest_gate_data_design"
            if passes
            else "selection_rejected_no_targeted_training_labels"
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
            "promotion_gate": (
                "structurally_ready and grouped Horvitz-Thompson "
                "candidate assignment fraction in [0.4,0.6] and "
                "candidate-minus-Teacher 95pct_low > 0"
            ),
            "pilot_report_sha256": PILOT_REPORT_SHA256,
            "additional_terminal_wall_stage": False,
        },
        "runtime_seconds": time.monotonic() - started,
        "collector_summary": summary.payload(),
        "safe_dataset_manifest": trajectory_manifest(safe_rows),
        "safe_dataset_sha256": _sha256(data_path),
        "audit": audit,
        "passes": passes,
        "authorization": (
            "design a new honest train/validation gate from fresh walls only"
            if passes
            else "none"
        ),
        "warning": (
            "Even a pass estimates one first-disagreement intervention followed "
            "by Teacher; it does not authorize deployment or prove human strength."
        ),
    }
    (output_dir / "selection-report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/two-draw-targeted-intervention-v1"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_selection(output_dir=args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

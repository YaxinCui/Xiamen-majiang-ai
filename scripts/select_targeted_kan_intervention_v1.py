#!/usr/bin/env python3
"""Run the fixed 100-wall first-Teacher-kan binary intervention."""

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

from xiamen_mahjong.kan_intervention import (
    KAN_INTERVENTION_BEHAVIOR_VERSION,
    KAN_KINDS,
    TargetedKanInterventionBehavior,
    audit_targeted_kan_interventions,
)
from xiamen_mahjong.training import (
    collect_candidate_teacher_dagger_trajectories,
    read_trajectory_jsonl,
    trajectory_manifest,
    write_trajectory_jsonl,
)


PHYSICAL_WALLS = 100
FIRST_SEED = 202637000
BEHAVIOR_SEED = 202637200
FALLBACK_PROBABILITY = 0.5
MINIMUM_INTERVENTIONS = 100
MINIMUM_WALL_GROUPS = 100
MINIMUM_KIND_INTERVENTIONS = 30


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_selection(*, output_dir: Path) -> dict[str, object]:
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise ValueError("kan intervention 输出已存在且非空，拒绝覆盖")
    output_dir.mkdir(parents=True, exist_ok=True)
    behavior = TargetedKanInterventionBehavior(
        seed=BEHAVIOR_SEED,
        fallback_probability=FALLBACK_PROBABILITY,
    )
    started = time.monotonic()
    trajectories, summary = collect_candidate_teacher_dagger_trajectories(
        behavior,
        seed_count=PHYSICAL_WALLS,
        profile="classic",
        seed=FIRST_SEED,
        behavior_metadata={
            "behavior_policy": KAN_INTERVENTION_BEHAVIOR_VERSION,
            "base_policy": "heuristic_teacher",
            "fallback_policy": "teacher_non_kan_fallback_v1",
            "fallback_probability": FALLBACK_PROBABILITY,
            "maximum_interventions_per_trajectory": 1,
            "decision_scope": "first_teacher_kan_opportunity",
            "behavior_seed_in_training_jsonl": False,
        },
    )
    data_path = output_dir / "selection.trajectories.jsonl"
    write_trajectory_jsonl(trajectories, data_path)
    safe_rows = read_trajectory_jsonl(data_path)
    audit = audit_targeted_kan_interventions(
        safe_rows,
        minimum_interventions=MINIMUM_INTERVENTIONS,
        minimum_wall_groups=MINIMUM_WALL_GROUPS,
    )
    passing_kinds = []
    kind_reports = audit["by_teacher_kan_kind"]
    assert isinstance(kind_reports, dict)
    for kind in KAN_KINDS:
        report = kind_reports[kind]
        assert isinstance(report, dict)
        interval = report["bonferroni_98_333pct_interval_per_wall"]
        assert isinstance(interval, dict)
        count = report["interventions"]
        fraction = report["fallback_assignment_fraction"]
        low = interval["low"]
        if (
            isinstance(count, int)
            and count >= MINIMUM_KIND_INTERVENTIONS
            and isinstance(fraction, (int, float))
            and 0.3 <= float(fraction) <= 0.7
            and isinstance(low, (int, float))
            and float(low) > 0.0
        ):
            passing_kinds.append(kind)
    overall_fraction = audit["fallback_assignment_fraction"]
    passes = bool(
        audit["structurally_ready"]
        and isinstance(overall_fraction, (int, float))
        and 0.4 <= float(overall_fraction) <= 0.6
        and passing_kinds
    )
    payload: dict[str, object] = {
        "status": (
            "selection_passed_ready_for_kind_scoped_rule_candidate"
            if passes
            else "selection_rejected_no_kan_training_labels"
        ),
        "protocol": {
            "profile": "classic",
            "physical_walls": PHYSICAL_WALLS,
            "first_seed": FIRST_SEED,
            "seat_rotations": 4,
            "behavior_seed": BEHAVIOR_SEED,
            "behavior_version": KAN_INTERVENTION_BEHAVIOR_VERSION,
            "fallback_probability": FALLBACK_PROBABILITY,
            "maximum_interventions_per_trajectory": 1,
            "minimum_interventions": MINIMUM_INTERVENTIONS,
            "minimum_wall_groups": MINIMUM_WALL_GROUPS,
            "minimum_kind_interventions": MINIMUM_KIND_INTERVENTIONS,
            "kind_hypotheses": list(KAN_KINDS),
            "promotion_gate": (
                "structurally_ready; overall assignment in [0.4,0.6]; "
                "kind support >=30 and assignment in [0.3,0.7]; Bonferroni "
                "three-kind 98.333% skip-kan-minus-Teacher interval low > 0"
            ),
            "additional_terminal_wall_stage": False,
        },
        "runtime_seconds": time.monotonic() - started,
        "collector_summary": summary.payload(),
        "safe_dataset_manifest": trajectory_manifest(safe_rows),
        "safe_dataset_sha256": _sha256(data_path),
        "audit": audit,
        "passing_kinds": passing_kinds,
        "passes": passes,
        "authorization": (
            "pre_register_fresh_kind_scoped_100_wall_rule_candidate"
            if passes
            else "none"
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
        default=Path("artifacts/targeted-kan-intervention-v1"),
    )
    args = parser.parse_args()
    print(json.dumps(run_selection(output_dir=args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

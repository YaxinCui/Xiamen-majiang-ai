#!/usr/bin/env python3
"""Run fixed 100-wall binary causal selection for exact-tie ensemble v2."""

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

from xiamen_mahjong.exact_tie_intervention import (
    TARGETED_EXACT_TIE_ENSEMBLE_BEHAVIOR_VERSION,
    TargetedExactTieEnsembleInterventionBehavior,
    audit_targeted_exact_tie_interventions,
)
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import (
    collect_candidate_teacher_dagger_trajectories,
    read_trajectory_jsonl,
    trajectory_manifest,
    write_trajectory_jsonl,
)


PHYSICAL_WALLS = 100
FIRST_SEED = 202644000
BEHAVIOR_SEED = 202644100
MINIMUM_INTERVENTIONS = 250
MINIMUM_WALL_GROUPS = 100
CHECKPOINTS = tuple(
    Path(f"artifacts/exact-tie-source-world-rollout-ranker-v2/member-{index}.pt")
    for index in range(5)
)
CHECKPOINT_SHA256S = (
    "0c6dd7733a5e9499bc817b5c6889fffd5cceedf8b7a40eae1828ab2eb60f89cc",
    "36cf985febaa85c25bef6be7a1f5cb16b2c5a12dfca64a4f4b8f6e834b75f903",
    "89769baa6335275992c8a61998ad10b6fa7929817629d60fa4bf3afc660d6d60",
    "e33b7efd7b5261484e5da2ec095658c9bc8d778f7850e15b5b5dcee0eda3ef2e",
    "001cfd1f2c95c9a0c89c75042d56e1ad5d72667985243c28b1f9ae4822e0bc12",
)
OPE_DIAGNOSTIC_SHA256 = (
    "0373cf69db34a731bf0fb446e5c3e7ad0ab5de0d4aa9633d8500a82f8255a524"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(output_dir: Path) -> dict:
    diagnostic = Path(
        "artifacts/exact-tie-source-world-rollout-ranker-v2/randomized-ope-diagnostic.json"
    )
    if not diagnostic.is_file() or _sha256(diagnostic) != OPE_DIAGNOSTIC_SHA256:
        raise ValueError("exact-tie randomized OPE 身份不匹配")
    if tuple(_sha256(path) for path in CHECKPOINTS) != CHECKPOINT_SHA256S:
        raise ValueError("exact-tie v2 checkpoint 身份不匹配")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("targeted exact-tie selection 输出已存在且非空")
    policies = [TorchPolicyValueAgent.load(path, device="cpu") for path in CHECKPOINTS]
    behavior = TargetedExactTieEnsembleInterventionBehavior(
        policies, seed=BEHAVIOR_SEED, candidate_probability=0.5
    )
    started = time.monotonic()
    trajectories, summary = collect_candidate_teacher_dagger_trajectories(
        behavior,
        seed_count=PHYSICAL_WALLS,
        profile="classic",
        seed=FIRST_SEED,
        behavior_metadata={
            "behavior_policy": TARGETED_EXACT_TIE_ENSEMBLE_BEHAVIOR_VERSION,
            "base_policy": "heuristic_teacher",
            "candidate_policy": "exact_tie_structured_ensemble_v2",
            "candidate_action_probability": 0.5,
            "maximum_interventions_per_trajectory": 1,
            "decision_scope": "first_unanimous_exact_teacher_tie_disagreement",
            "checkpoint_sha256s": CHECKPOINT_SHA256S,
            "behavior_seed_in_training_jsonl": False,
        },
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    data_path = output_dir / "selection.trajectories.jsonl"
    write_trajectory_jsonl(trajectories, data_path)
    safe_rows = read_trajectory_jsonl(data_path)
    audit = audit_targeted_exact_tie_interventions(
        safe_rows,
        policies=policies,
        minimum_interventions=MINIMUM_INTERVENTIONS,
        minimum_wall_groups=MINIMUM_WALL_GROUPS,
    )
    estimate = audit["horvitz_thompson_candidate_minus_teacher_per_wall"]
    lower = estimate["95pct_low"]
    passes = bool(
        audit["structurally_ready"]
        and isinstance(lower, (int, float))
        and lower > 0.0
    )
    report = {
        "status": (
            "selection_passed_ready_for_live_one_override_candidate"
            if passes
            else "selection_rejected_no_deployment"
        ),
        "protocol": {
            "physical_walls": PHYSICAL_WALLS,
            "first_seed": FIRST_SEED,
            "seat_rotations": 4,
            "behavior_seed": BEHAVIOR_SEED,
            "candidate_probability": 0.5,
            "minimum_interventions": MINIMUM_INTERVENTIONS,
            "minimum_wall_groups": MINIMUM_WALL_GROUPS,
            "promotion_gate": "structural checks and grouped HT 95pct_low > 0",
            "checkpoint_sha256s": CHECKPOINT_SHA256S,
            "ope_diagnostic_sha256": OPE_DIAGNOSTIC_SHA256,
        },
        "runtime_seconds": time.monotonic() - started,
        "collector_summary": summary.payload(),
        "safe_dataset_manifest": trajectory_manifest(safe_rows),
        "safe_dataset_sha256": _sha256(data_path),
        "behavior_counters": {
            "disagreement_hands": behavior.disagreement_hands,
            "candidate_assignments": behavior.candidate_assignments,
            "teacher_assignments": behavior.teacher_assignments,
        },
        "audit": audit,
        "passes": passes,
        "authorization": (
            "implement one-override live wrapper and run fresh 100-wall paired game gate"
            if passes
            else "none"
        ),
        "warning": "This is one intervention, not repeated-policy or human-strength evidence.",
    }
    (output_dir / "selection-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/targeted-exact-tie-ensemble-v1"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

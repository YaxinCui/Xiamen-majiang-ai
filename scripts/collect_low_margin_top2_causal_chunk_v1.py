#!/usr/bin/env python3
"""Collect one immutable 100-wall chunk of low-margin randomized data."""

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

from xiamen_mahjong.low_margin_intervention import (
    LOW_MARGIN_TOP2_BEHAVIOR_VERSION,
    LOW_MARGIN_TOP2_POLICY_VERSION,
    LowMarginTop2InterventionBehavior,
    audit_low_margin_causal_records,
    audit_low_margin_top2_interventions,
    extract_low_margin_causal_records,
    read_low_margin_causal_jsonl,
    write_low_margin_causal_jsonl,
)
from xiamen_mahjong.training import collect_candidate_teacher_dagger_trajectories


CHUNK_WALLS = 100
MAXIMUM_TEACHER_SCORE_MARGIN = 2.0
ALTERNATIVE_ACTION_PROBABILITY = 0.5
RECORD_ORDER_CONTRACT = "lexicographic_opaque_record_id_not_wall_or_seat_order"
SPLITS = {
    "train": {
        "first_seed": 202650000,
        "chunks": 40,
        "behavior_seed": 202670000,
    },
    "validation": {
        "first_seed": 202654000,
        "chunks": 20,
        "behavior_seed": 202671000,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_chunk(
    *, split: str, chunk_index: int, output_root: Path
) -> dict[str, object]:
    if split not in SPLITS:
        raise ValueError("正式低分差 split 只能是 train 或 validation")
    config = SPLITS[split]
    chunk_count = int(config["chunks"])
    if not 0 <= chunk_index < chunk_count:
        raise ValueError(f"{split} chunk_index 必须位于 [0,{chunk_count})")
    directory = output_root / "data" / split
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"chunk-{chunk_index:02d}"
    data_path = directory / f"{prefix}.records.jsonl"
    report_path = directory / f"{prefix}.report.json"
    if data_path.exists() or report_path.exists():
        raise ValueError(f"{split}/{prefix} 已存在，拒绝覆盖")

    first_seed = int(config["first_seed"]) + chunk_index * CHUNK_WALLS
    behavior_seed = int(config["behavior_seed"]) + chunk_index
    behavior = LowMarginTop2InterventionBehavior(
        seed=behavior_seed,
        maximum_margin=MAXIMUM_TEACHER_SCORE_MARGIN,
        alternative_probability=ALTERNATIVE_ACTION_PROBABILITY,
    )
    started = time.monotonic()
    trajectories, summary = collect_candidate_teacher_dagger_trajectories(
        behavior,
        seed_count=CHUNK_WALLS,
        profile="classic",
        seed=first_seed,
        behavior_metadata={
            "behavior_policy": LOW_MARGIN_TOP2_BEHAVIOR_VERSION,
            "base_policy": "heuristic_teacher",
            "candidate_policy": LOW_MARGIN_TOP2_POLICY_VERSION,
            "alternative_action_probability": ALTERNATIVE_ACTION_PROBABILITY,
            "maximum_teacher_score_margin": MAXIMUM_TEACHER_SCORE_MARGIN,
            "maximum_interventions_per_trajectory": 1,
            "decision_scope": "first_eligible_ordinary_discard",
            "behavior_seed_in_training_jsonl": False,
            "formal_split": split,
            "formal_chunk_index": chunk_index,
        },
    )
    full_audit = audit_low_margin_top2_interventions(
        trajectories,
        maximum_margin=MAXIMUM_TEACHER_SCORE_MARGIN,
        minimum_interventions=int(CHUNK_WALLS * 4 * 0.95),
        minimum_wall_groups=CHUNK_WALLS,
    )
    if not full_audit["structurally_ready"]:
        raise RuntimeError(
            "正式低分差 full trajectory 审计失败："
            + ", ".join(full_audit["gate_reasons"])
        )
    records = extract_low_margin_causal_records(
        trajectories, maximum_margin=MAXIMUM_TEACHER_SCORE_MARGIN
    )
    # ``collect_candidate_teacher_dagger_trajectories`` returns wall/seat
    # order.  Even without an explicit seed field that row position would be
    # reversible from the public seed range.  UUID order is opaque and keeps
    # all four group identities while breaking that positional side channel.
    records.sort(key=lambda record: record.record_id)
    write_low_margin_causal_jsonl(records, data_path)
    # Re-open only the safe slim file and fail before finalizing its report.
    safe_records = read_low_margin_causal_jsonl(data_path)
    slim_audit = audit_low_margin_causal_records(
        safe_records,
        expected_wall_groups=CHUNK_WALLS,
        minimum_alternative_fraction=0.3,
        maximum_alternative_fraction=0.7,
        minimum_intervention_fraction=0.95,
    )
    if not slim_audit["ready"]:
        raise RuntimeError(
            "正式低分差 slim record 审计失败："
            + ", ".join(slim_audit["issues"])
        )
    payload: dict[str, object] = {
        "status": "formal_low_margin_top2_causal_chunk_ready",
        "split": split,
        "chunk_index": chunk_index,
        "physical_walls": CHUNK_WALLS,
        "first_seed": first_seed,
        "last_seed": first_seed + CHUNK_WALLS - 1,
        "seed_fields_in_safe_records": False,
        "behavior_seed_in_safe_records": False,
        "runtime_seconds": time.monotonic() - started,
        "collector_summary": summary.payload(),
        "full_trajectory_audit": full_audit,
        "slim_record_audit": slim_audit,
        "safe_records": {
            "path": str(data_path),
            "sha256": _sha256(data_path),
            "bytes": data_path.stat().st_size,
            "record_order": RECORD_ORDER_CONTRACT,
        },
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/low-margin-top2-causal-residual-v1"),
    )
    args = parser.parse_args()
    result = collect_chunk(
        split=args.split,
        chunk_index=args.chunk_index,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "split": result["split"],
                "chunk_index": result["chunk_index"],
                "runtime_seconds": result["runtime_seconds"],
                "records": result["slim_record_audit"]["records"],
                "alternative_fraction": result["slim_record_audit"][
                    "alternative_assignment_fraction"
                ],
                "sha256": result["safe_records"]["sha256"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

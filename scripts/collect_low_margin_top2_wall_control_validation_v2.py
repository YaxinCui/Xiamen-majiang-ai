#!/usr/bin/env python3
"""Collect/resume the new immutable wall-control v2 validation split."""

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

from scripts.collect_low_margin_top2_causal_chunk_v1 import (
    ALTERNATIVE_ACTION_PROBABILITY,
    CHUNK_WALLS,
    MAXIMUM_TEACHER_SCORE_MARGIN,
    RECORD_ORDER_CONTRACT,
)
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


VALIDATION_FIRST_SEED = 202658000
VALIDATION_CHUNKS = 20
VALIDATION_BEHAVIOR_SEED = 202673000
CHUNK_STATUS = "formal_low_margin_top2_wall_control_v2_validation_chunk_ready"
SPLIT_STATUS = "formal_low_margin_top2_wall_control_v2_validation_ready"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_chunk(*, chunk_index: int, output_root: Path) -> dict[str, object]:
    if not 0 <= chunk_index < VALIDATION_CHUNKS:
        raise ValueError(f"v2 validation chunk 必须位于 [0,{VALIDATION_CHUNKS})")
    directory = output_root / "data" / "validation"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"chunk-{chunk_index:02d}"
    data_path = directory / f"{prefix}.records.jsonl"
    report_path = directory / f"{prefix}.report.json"
    if data_path.exists() or report_path.exists():
        raise ValueError(f"v2 validation/{prefix} 已存在，拒绝覆盖")

    first_seed = VALIDATION_FIRST_SEED + chunk_index * CHUNK_WALLS
    behavior = LowMarginTop2InterventionBehavior(
        seed=VALIDATION_BEHAVIOR_SEED + chunk_index,
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
            "formal_split": "wall_control_v2_validation",
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
            "v2 validation full trajectory 审计失败："
            + ", ".join(full_audit["gate_reasons"])
        )
    records = extract_low_margin_causal_records(
        trajectories, maximum_margin=MAXIMUM_TEACHER_SCORE_MARGIN
    )
    records.sort(key=lambda record: record.record_id)
    write_low_margin_causal_jsonl(records, data_path)
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
            "v2 validation slim record 审计失败："
            + ", ".join(slim_audit["issues"])
        )
    report: dict[str, object] = {
        "status": CHUNK_STATUS,
        "split": "wall_control_v2_validation",
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
        "separation": {
            "v1_validation_reused": False,
            "v1_terminal_reused": False,
            "v2_terminal_collected": False,
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def audit_validation(output_root: Path) -> dict[str, object]:
    directory = output_root / "data" / "validation"
    records = []
    files = []
    for index in range(VALIDATION_CHUNKS):
        data_path = directory / f"chunk-{index:02d}.records.jsonl"
        report_path = directory / f"chunk-{index:02d}.report.json"
        if not data_path.is_file() or not report_path.is_file():
            raise ValueError(f"v2 validation 缺少 chunk-{index:02d}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        digest = _sha256(data_path)
        if (
            report.get("status") != CHUNK_STATUS
            or report.get("split") != "wall_control_v2_validation"
            or report.get("chunk_index") != index
            or report.get("safe_records", {}).get("sha256") != digest
            or report.get("safe_records", {}).get("record_order")
            != RECORD_ORDER_CONTRACT
            or report.get("separation", {}).get("v1_validation_reused") is not False
            or report.get("separation", {}).get("v1_terminal_reused") is not False
        ):
            raise ValueError(f"v2 validation chunk-{index:02d} 身份或 SHA 无效")
        records.extend(read_low_margin_causal_jsonl(data_path))
        files.append(
            {
                "chunk_index": index,
                "path": str(data_path),
                "sha256": digest,
                "bytes": data_path.stat().st_size,
            }
        )
    aggregate = audit_low_margin_causal_records(
        records,
        expected_wall_groups=VALIDATION_CHUNKS * CHUNK_WALLS,
        validate_actor_visible_pairs=False,
    )
    return {
        "status": SPLIT_STATUS if aggregate["ready"] else f"{SPLIT_STATUS}_rejected",
        "split": "wall_control_v2_validation",
        "chunks": VALIDATION_CHUNKS,
        "chunk_walls": CHUNK_WALLS,
        "expected_wall_groups": VALIDATION_CHUNKS * CHUNK_WALLS,
        "aggregate_audit": aggregate,
        "chunk_files": files,
        "separation": {
            "first_seed": VALIDATION_FIRST_SEED,
            "last_seed": (
                VALIDATION_FIRST_SEED + VALIDATION_CHUNKS * CHUNK_WALLS - 1
            ),
            "v1_validation_seed_range": "202654000..202655999_not_reused",
            "v1_terminal_seed_range": "202656000..202657999_not_reused",
            "v2_terminal_seed_range": "202660000..202661999_not_collected",
        },
        "warning": "Mechanics audit only; no policy strength is estimated here.",
    }


def collect_validation(output_root: Path) -> dict[str, object]:
    directory = output_root / "data" / "validation"
    for index in range(VALIDATION_CHUNKS):
        data_path = directory / f"chunk-{index:02d}.records.jsonl"
        report_path = directory / f"chunk-{index:02d}.report.json"
        if data_path.is_file() and report_path.is_file():
            print(
                json.dumps(
                    {"chunk_index": index, "status": "existing"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue
        if data_path.exists() or report_path.exists():
            raise ValueError(f"v2 validation chunk-{index:02d} 部分存在")
        report = collect_chunk(chunk_index=index, output_root=output_root)
        print(
            json.dumps(
                {
                    "chunk_index": index,
                    "status": report["status"],
                    "runtime_seconds": report["runtime_seconds"],
                    "records": report["slim_record_audit"]["records"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return audit_validation(output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/low-margin-top2-wall-control-v2"),
    )
    parser.add_argument("--chunk-index", type=int)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    split_report_path = args.output_root / "validation-split-audit.json"
    if args.chunk_index is not None:
        if args.audit_only:
            raise ValueError("--chunk-index 不能与 --audit-only 同时使用")
        report = collect_chunk(
            chunk_index=args.chunk_index, output_root=args.output_root
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if split_report_path.exists():
        raise ValueError("v2 validation 总审计已存在，拒绝覆盖")
    report = (
        audit_validation(args.output_root)
        if args.audit_only
        else collect_validation(args.output_root)
    )
    split_report_path.parent.mkdir(parents=True, exist_ok=True)
    split_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

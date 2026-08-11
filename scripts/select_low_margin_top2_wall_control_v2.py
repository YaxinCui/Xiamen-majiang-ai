#!/usr/bin/env python3
"""Apply the frozen wall-control v2 gates to its new validation split once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.collect_low_margin_top2_causal_chunk_v1 import RECORD_ORDER_CONTRACT
from scripts.collect_low_margin_top2_wall_control_validation_v2 import (
    CHUNK_STATUS,
    SPLIT_STATUS,
    VALIDATION_CHUNKS,
)
from scripts.train_low_margin_top2_causal_residual_v1 import load_formal_train
from xiamen_mahjong.causal_residual import CAUSAL_SCORE_SCALE, causal_pair_features
from xiamen_mahjong.low_margin_intervention import (
    LowMarginCausalRecord,
    audit_low_margin_causal_records,
    read_low_margin_causal_jsonl,
)
from xiamen_mahjong.wall_control import (
    CONTROL_COEFFICIENT_ABS_LIMIT,
    adjusted_wall_values,
    load_wall_control_checkpoint,
    policy_wall_components,
)


FAMILY_TESTS = 4
FAMILY_ALPHA = 0.05
BONFERRONI_Z = NormalDist().inv_cdf(1.0 - FAMILY_ALPHA / (2 * FAMILY_TESTS))
MINIMUM_OVERRIDES = 60
MINIMUM_OVERRIDE_GROUPS = 60
MINIMUM_LOGGED_ASSIGNMENTS_PER_ARM = 20
TRAINING_STATUS = "wall_control_v2_trained_train_only_new_validation_uncollected"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256() -> str:
    paths = [Path(__file__).resolve(), *sorted((ROOT / "xiamen_mahjong").glob("*.py"))]
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _interval(values: Sequence[float], *, z: float) -> dict[str, float | int | None]:
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


def evaluate_wall_control_gate(
    records: Sequence[LowMarginCausalRecord],
    effects: Sequence[float | None],
    *,
    target_coverage: float,
    threshold: float,
    control_coefficient: float,
) -> dict[str, Any]:
    if (
        not math.isfinite(control_coefficient)
        or abs(control_coefficient) > CONTROL_COEFFICIENT_ABS_LIMIT
    ):
        raise ValueError("v2 validation control coefficient 无效")
    components = policy_wall_components(records, effects, threshold=threshold)
    values = adjusted_wall_values(
        components["raw_wall_values"],
        components["control_wall_values"],
        control_coefficient=control_coefficient,
    )
    corrected = _interval(values, z=BONFERRONI_Z)
    nominal = _interval(values, z=1.96)
    raw = _interval(components["raw_wall_values"], z=1.96)
    reasons = []
    if components["overrides"] < MINIMUM_OVERRIDES:
        reasons.append("insufficient_overrides")
    if components["override_groups"] < MINIMUM_OVERRIDE_GROUPS:
        reasons.append("insufficient_override_groups")
    if (
        components["logged_alternative_assignments"]
        < MINIMUM_LOGGED_ASSIGNMENTS_PER_ARM
    ):
        reasons.append("insufficient_logged_alternative_support")
    if (
        components["logged_teacher_assignments"]
        < MINIMUM_LOGGED_ASSIGNMENTS_PER_ARM
    ):
        reasons.append("insufficient_logged_teacher_support")
    if corrected["low"] is None or corrected["low"] <= 0.0:
        reasons.append("bonferroni_lower_bound_not_positive")
    eligible = int(components["eligible_records"])
    return {
        "target_coverage": target_coverage,
        "strict_minimum_member_effect_threshold": threshold,
        "frozen_train_control_coefficient": control_coefficient,
        "eligible_records": eligible,
        "overrides": components["overrides"],
        "realized_coverage": components["overrides"] / eligible if eligible else None,
        "override_groups": components["override_groups"],
        "logged_alternative_assignments": components[
            "logged_alternative_assignments"
        ],
        "logged_teacher_assignments": components["logged_teacher_assignments"],
        "unadjusted_ht_nominal_95pct_diagnostic": raw,
        "wall_control_policy_minus_teacher_nominal_95pct": nominal,
        "wall_control_policy_minus_teacher_bonferroni": corrected,
        "passes": not reasons,
        "gate_reasons": reasons,
    }


def _load_validation(
    output_root: Path,
) -> tuple[list[LowMarginCausalRecord], list[dict[str, Any]]]:
    split_report_path = output_root / "validation-split-audit.json"
    if not split_report_path.is_file():
        raise ValueError("v2 validation 总审计报告不存在")
    frozen_report = json.loads(split_report_path.read_text(encoding="utf-8"))
    if frozen_report.get("status") != SPLIT_STATUS:
        raise ValueError("v2 validation 总审计未通过")
    records = []
    files = []
    directory = output_root / "data" / "validation"
    for index in range(VALIDATION_CHUNKS):
        path = directory / f"chunk-{index:02d}.records.jsonl"
        report_path = directory / f"chunk-{index:02d}.report.json"
        if not path.is_file() or not report_path.is_file():
            raise ValueError(f"v2 validation 缺少 chunk-{index:02d}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        digest = _sha256(path)
        if (
            report.get("status") != CHUNK_STATUS
            or report.get("safe_records", {}).get("sha256") != digest
            or report.get("safe_records", {}).get("record_order")
            != RECORD_ORDER_CONTRACT
        ):
            raise ValueError(f"v2 validation chunk-{index:02d} SHA 或顺序无效")
        records.extend(
            read_low_margin_causal_jsonl(path, validate_actor_visible_pairs=False)
        )
        files.append(
            {
                "chunk_index": index,
                "path": str(path),
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
        )
    audit = audit_low_margin_causal_records(
        records,
        expected_wall_groups=VALIDATION_CHUNKS * 100,
        validate_actor_visible_pairs=False,
    )
    if not audit["ready"] or audit != frozen_report.get("aggregate_audit"):
        raise ValueError("v2 selector validation 审计与冻结报告不一致")
    return records, files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-train-root",
        type=Path,
        default=Path("artifacts/low-margin-top2-causal-residual-v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/low-margin-top2-wall-control-v2"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    selection_path = args.output_root / "validation-selection-report.json"
    if selection_path.exists():
        raise ValueError("v2 validation selection 已存在，拒绝覆盖")
    if (args.output_root / "data" / "terminal").exists():
        raise ValueError("v2 validation 前 terminal 不应存在")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA selection，但 PyTorch 没有可用 GPU")

    training_path = args.output_root / "model-training-report.json"
    if not training_path.is_file():
        raise ValueError("v2 模型训练报告不存在")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if (
        training.get("status") != TRAINING_STATUS
        or training.get("provenance", {}).get("v1_validation_read_or_hashed")
        is not False
        or training.get("provenance", {}).get("v1_terminal_read_or_hashed")
        is not False
        or training.get("v2_validation_contract")
        != "not_collected_opened_or_hashed_by_trainer"
        or training.get("v2_terminal_contract") != "not_collected"
    ):
        raise ValueError("v2 training report 数据边界无效")
    gates = training.get("frozen_train_gates")
    if not isinstance(gates, list) or len(gates) != FAMILY_TESTS:
        raise ValueError("v2 training report 冻结 gate 无效")

    train_records, train_files = load_formal_train(args.source_train_root)
    expected_train_sha = [item["sha256"] for item in train_files]
    report_train_sha = [
        item["sha256"]
        for item in training.get("provenance", {}).get(
            "v1_train_input_files", []
        )
    ]
    if report_train_sha != expected_train_sha:
        raise ValueError("v2 training report 与 v1 train SHA 不一致")

    networks = []
    checkpoints = []
    for index, member in enumerate(training.get("members", [])):
        path = args.output_root / "models" / f"member-{index}.pt"
        digest = _sha256(path)
        if member.get("checkpoint_sha256") != digest:
            raise ValueError(f"v2 checkpoint member-{index} SHA 不一致")
        network, metadata = load_wall_control_checkpoint(
            path, device=args.device
        )
        if (
            metadata.get("train_input_sha256") != expected_train_sha
            or metadata.get("v1_validation_used") is not False
            or metadata.get("v1_terminal_used") is not False
            or metadata.get("v2_validation_read") is not False
            or metadata.get("v2_terminal_read") is not False
        ):
            raise ValueError(f"v2 checkpoint member-{index} 元数据边界无效")
        networks.append(network)
        checkpoints.append(
            {"path": str(path), "sha256": digest, "metadata": metadata}
        )
    if len(networks) != 3:
        raise ValueError("v2 必须严格三个 ensemble checkpoint")

    records, validation_files = _load_validation(args.output_root)
    train_groups = {record.group_id for record in train_records}
    validation_groups = {record.group_id for record in records}
    overlap = train_groups & validation_groups
    if overlap:
        raise ValueError("v2 train/validation opaque group 重合")

    eligible_indices = [
        index for index, record in enumerate(records)
        if record.executed_arm != "none"
    ]
    feature_tensor = torch.tensor(
        [causal_pair_features(records[index]) for index in eligible_indices],
        dtype=torch.float32,
        device=args.device,
    )
    with torch.no_grad():
        member_effects = [
            network(feature_tensor)[1] * CAUSAL_SCORE_SCALE
            for network in networks
        ]
        conservative = torch.stack(member_effects).min(dim=0).values.cpu().tolist()
    effects: list[float | None] = [None] * len(records)
    for index, value in zip(eligible_indices, conservative):
        effects[index] = float(value)

    results = []
    for gate in gates:
        results.append(
            evaluate_wall_control_gate(
                records,
                effects,
                target_coverage=float(gate["target_coverage"]),
                threshold=float(gate["strict_effect_threshold"]),
                control_coefficient=float(gate["train_control_coefficient"]),
            )
        )
    passing = [result for result in results if result["passes"]]
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda row: (
                row["wall_control_policy_minus_teacher_bonferroni"]["low"],
                -row["target_coverage"],
                row["strict_minimum_member_effect_threshold"],
            ),
        )
    status = (
        "wall_control_v2_validation_passed_terminal_locked_until_collection"
        if selected is not None
        else "wall_control_v2_validation_rejected_terminal_not_collected"
    )
    payload = {
        "status": status,
        "selected_gate": selected,
        "protocol": {
            "family_tests": FAMILY_TESTS,
            "family_alpha": FAMILY_ALPHA,
            "bonferroni_z": BONFERRONI_Z,
            "minimum_overrides": MINIMUM_OVERRIDES,
            "minimum_override_groups": MINIMUM_OVERRIDE_GROUPS,
            "minimum_logged_assignments_per_arm": (
                MINIMUM_LOGGED_ASSIGNMENTS_PER_ARM
            ),
            "gate_score": "minimum_effect_across_three_members",
            "control_coefficient_source": "frozen_v1_train_only",
            "v1_validation_used": False,
            "v1_terminal_used": False,
            "v2_terminal_collected": False,
        },
        "provenance": {
            "selector_source_tree_sha256": _source_tree_sha256(),
            "training_report": str(training_path),
            "training_report_sha256": _sha256(training_path),
            "checkpoints": checkpoints,
            "train_files": train_files,
            "validation_files": validation_files,
        },
        "validation": {
            "records": len(records),
            "eligible_records": len(eligible_indices),
            "wall_groups": len(validation_groups),
            "train_group_overlap": len(overlap),
            "gates": results,
        },
        "terminal_contract": (
            "eligible_for_new_collection_not_yet_collected"
            if selected is not None
            else "must_not_be_collected"
        ),
        "warning": (
            "A passing result concerns one first-opportunity intervention only; "
            "it is not a browser or human-strength claim."
        ),
    }
    selection_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

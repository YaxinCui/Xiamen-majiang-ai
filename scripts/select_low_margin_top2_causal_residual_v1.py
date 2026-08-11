#!/usr/bin/env python3
"""Select one frozen causal-residual coverage gate on formal validation."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from scripts.collect_low_margin_top2_causal_chunk_v1 import (
    CHUNK_WALLS,
    RECORD_ORDER_CONTRACT,
    SPLITS,
)
from xiamen_mahjong.causal_residual import (
    CAUSAL_SCORE_SCALE,
    causal_pair_features,
    load_causal_checkpoint,
)
from xiamen_mahjong.low_margin_intervention import (
    LowMarginCausalRecord,
    audit_low_margin_causal_records,
    read_low_margin_causal_jsonl,
)


FAMILY_TESTS = 4
FAMILY_ALPHA = 0.05
BONFERRONI_Z = NormalDist().inv_cdf(1.0 - FAMILY_ALPHA / (2 * FAMILY_TESTS))
MINIMUM_OVERRIDES = 60
MINIMUM_OVERRIDE_GROUPS = 60
MINIMUM_LOGGED_ASSIGNMENTS_PER_ARM = 20


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


def policy_difference_contribution(
    *,
    terminal_score: float,
    logged_arm: str,
    override: bool,
    propensity: float = 0.5,
) -> float:
    """Unbiased one-decision gated-policy minus Teacher contribution."""

    if not math.isfinite(terminal_score):
        raise ValueError("因果 gate terminal_score 必须有限")
    if logged_arm not in {"teacher", "alternative", "none"}:
        raise ValueError("因果 gate logged_arm 无效")
    if logged_arm == "none":
        if override:
            raise ValueError("no_intervention 行不能 override")
        return 0.0
    if not 0.0 < propensity < 1.0:
        raise ValueError("因果 gate propensity 无效")
    if not override:
        return 0.0
    if logged_arm == "alternative":
        return terminal_score / propensity
    return -terminal_score / (1.0 - propensity)


def _interval(values: Sequence[float], *, z: float) -> dict[str, float | int | None]:
    if not values:
        return {
            "groups": 0,
            "mean": None,
            "stderr": None,
            "low": None,
            "high": None,
        }
    mean = sum(values) / len(values)
    if len(values) < 2:
        return {
            "groups": len(values),
            "mean": mean,
            "stderr": None,
            "low": None,
            "high": None,
        }
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    stderr = math.sqrt(variance / len(values))
    return {
        "groups": len(values),
        "mean": mean,
        "stderr": stderr,
        "low": mean - z * stderr,
        "high": mean + z * stderr,
    }


def _load_split(
    output_root: Path, split: str
) -> tuple[list[LowMarginCausalRecord], list[dict[str, Any]]]:
    if split not in {"train", "validation"}:
        raise ValueError("因果 selector 只允许读取 train/validation")
    split_report_path = output_root / f"{split}-split-audit.json"
    if not split_report_path.is_file():
        raise ValueError(f"正式 {split} split 总审计报告不存在")
    split_report = json.loads(split_report_path.read_text(encoding="utf-8"))
    if (
        split_report.get("status")
        != "formal_low_margin_top2_causal_split_ready"
        or split_report.get("split") != split
        or not split_report.get("aggregate_audit", {}).get("ready")
    ):
        raise ValueError(f"正式 {split} split 总审计未通过")
    records = []
    files = []
    for index in range(int(SPLITS[split]["chunks"])):
        path = output_root / "data" / split / f"chunk-{index:02d}.records.jsonl"
        chunk_report_path = output_root / "data" / split / f"chunk-{index:02d}.report.json"
        if not path.is_file():
            raise ValueError(f"正式 {split} 缺少 chunk-{index:02d}")
        digest = _sha256(path)
        chunk_report = json.loads(chunk_report_path.read_text(encoding="utf-8"))
        if (
            chunk_report.get("safe_records", {}).get("sha256") != digest
            or chunk_report.get("safe_records", {}).get("record_order")
            != RECORD_ORDER_CONTRACT
        ):
            raise ValueError(f"正式 {split} chunk-{index:02d} 顺序契约或 SHA 无效")
        records.extend(
            read_low_margin_causal_jsonl(
                path, validate_actor_visible_pairs=False
            )
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
        expected_wall_groups=int(SPLITS[split]["chunks"]) * CHUNK_WALLS,
        validate_actor_visible_pairs=False,
    )
    if not audit["ready"] or audit != split_report["aggregate_audit"]:
        raise ValueError(f"selector 重算的 {split} 审计与冻结报告不一致")
    return records, files


def _evaluate_gate(
    records: Sequence[LowMarginCausalRecord],
    effects: Sequence[float | None],
    *,
    threshold: float,
    target_coverage: float,
) -> dict[str, Any]:
    if len(records) != len(effects) or not math.isfinite(threshold):
        raise ValueError("因果 gate 输入长度或阈值无效")
    group_values: dict[str, list[float]] = defaultdict(list)
    override_groups: set[str] = set()
    logged_arms: dict[str, int] = defaultdict(int)
    overrides = 0
    eligible = 0
    for record, effect in zip(records, effects):
        if record.executed_arm == "none":
            if effect is not None:
                raise ValueError("no_intervention 行出现模型 effect")
            override = False
        else:
            if effect is None or not math.isfinite(effect):
                raise ValueError("eligible 行缺少有限模型 effect")
            eligible += 1
            override = effect > threshold
        if override:
            overrides += 1
            override_groups.add(record.group_id)
            logged_arms[record.executed_arm] += 1
        contribution = policy_difference_contribution(
            terminal_score=record.terminal_candidate_score,
            logged_arm=record.executed_arm,
            override=override,
            propensity=record.propensity,
        )
        group_values[record.group_id].append(contribution)
    if any(len(values) != 4 for values in group_values.values()):
        raise ValueError("因果 validation 出现非四座墙组")
    wall_values = [
        sum(values) / 4.0 for _group, values in sorted(group_values.items())
    ]
    corrected = _interval(wall_values, z=BONFERRONI_Z)
    nominal = _interval(wall_values, z=1.96)
    reasons = []
    if overrides < MINIMUM_OVERRIDES:
        reasons.append("insufficient_overrides")
    if len(override_groups) < MINIMUM_OVERRIDE_GROUPS:
        reasons.append("insufficient_override_groups")
    if logged_arms["alternative"] < MINIMUM_LOGGED_ASSIGNMENTS_PER_ARM:
        reasons.append("insufficient_logged_alternative_support")
    if logged_arms["teacher"] < MINIMUM_LOGGED_ASSIGNMENTS_PER_ARM:
        reasons.append("insufficient_logged_teacher_support")
    if corrected["low"] is None or corrected["low"] <= 0.0:
        reasons.append("bonferroni_lower_bound_not_positive")
    return {
        "target_coverage": target_coverage,
        "strict_effect_threshold": threshold,
        "eligible_records": eligible,
        "overrides": overrides,
        "realized_coverage": overrides / eligible if eligible else None,
        "override_groups": len(override_groups),
        "logged_alternative_assignments": logged_arms["alternative"],
        "logged_teacher_assignments": logged_arms["teacher"],
        "paired_wall_policy_minus_teacher_nominal_95pct": nominal,
        "paired_wall_policy_minus_teacher_bonferroni": corrected,
        "passes": not reasons,
        "gate_reasons": reasons,
    }


def _ensemble_effects_once(
    networks: Sequence[Any],
    records: Sequence[LowMarginCausalRecord],
    *,
    device: str,
) -> list[float | None]:
    """Encode validation once, then reuse one device-resident tensor."""

    eligible_indices = [
        index for index, record in enumerate(records)
        if record.executed_arm != "none"
    ]
    if not eligible_indices:
        raise ValueError("因果 validation 没有 eligible 记录")
    features = torch.tensor(
        [causal_pair_features(records[index]) for index in eligible_indices],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        member_effects = [network(features)[1] for network in networks]
        ensemble = (
            torch.stack(member_effects).mean(dim=0) * CAUSAL_SCORE_SCALE
        ).cpu().tolist()
    effects: list[float | None] = [None] * len(records)
    for index, value in zip(eligible_indices, ensemble):
        effects[index] = float(value)
    return effects


def select(output_root: Path, *, device: str = "cpu") -> dict[str, Any]:
    training_path = output_root / "model-training-report.json"
    if not training_path.is_file():
        raise ValueError("因果 residual train-only 报告不存在")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if training.get("status") != "causal_residual_trained_train_only_validation_unread":
        raise ValueError("因果 residual train-only 报告状态无效")
    checkpoints = [output_root / "models" / f"member-{index}.pt" for index in range(3)]
    if any(not path.is_file() for path in checkpoints):
        raise ValueError("因果 residual ensemble checkpoint 不完整")
    networks = []
    checkpoint_files = []
    expected_input_sha = [
        item["sha256"] for item in training["provenance"]["input_files"]
    ]
    for path in checkpoints:
        network, metadata = load_causal_checkpoint(path, device=device)
        if (
            metadata.get("train_input_sha256") != expected_input_sha
            or metadata.get("validation_read") is not False
            or metadata.get("terminal_read") is not False
        ):
            raise ValueError("因果 residual checkpoint provenance 无效")
        networks.append(network)
        checkpoint_files.append(
            {"path": str(path), "sha256": _sha256(path), "metadata": metadata}
        )
    train_records, train_files = _load_split(output_root, "train")
    if [item["sha256"] for item in train_files] != expected_input_sha:
        raise ValueError("当前 train 文件与 checkpoint 输入摘要不一致")
    validation_records, validation_files = _load_split(output_root, "validation")
    train_groups = {record.group_id for record in train_records}
    validation_groups = {record.group_id for record in validation_records}
    if train_groups & validation_groups:
        raise ValueError("因果 train/validation 物理墙 group 泄漏")
    effects = _ensemble_effects_once(
        networks, validation_records, device=device
    )
    threshold_rows = training.get("frozen_train_coverage_thresholds")
    if not isinstance(threshold_rows, list) or len(threshold_rows) != FAMILY_TESTS:
        raise ValueError("冻结 train coverage thresholds 无效")
    gates = [
        _evaluate_gate(
            validation_records,
            effects,
            threshold=float(row["strict_effect_threshold"]),
            target_coverage=float(row["target_coverage"]),
        )
        for row in threshold_rows
    ]
    passed = [gate for gate in gates if gate["passes"]]
    selected = None
    if passed:
        selected = max(
            passed,
            key=lambda gate: (
                gate["paired_wall_policy_minus_teacher_bonferroni"]["low"],
                -gate["target_coverage"],
                gate["strict_effect_threshold"],
            ),
        )
    return {
        "status": (
            "validation_passed_terminal_not_collected"
            if selected is not None
            else "validation_rejected_terminal_not_collected"
        ),
        "protocol": {
            "family_tests": FAMILY_TESTS,
            "family_alpha": FAMILY_ALPHA,
            "bonferroni_z": BONFERRONI_Z,
            "minimum_overrides": MINIMUM_OVERRIDES,
            "minimum_override_groups": MINIMUM_OVERRIDE_GROUPS,
            "minimum_logged_assignments_per_arm": MINIMUM_LOGGED_ASSIGNMENTS_PER_ARM,
            "selection_rule": (
                "highest_bonferroni_low_then_lower_coverage_then_higher_threshold"
            ),
            "terminal": "not_collected_or_read",
        },
        "provenance": {
            "selector_source_tree_sha256": _source_tree_sha256(),
            "training_report": str(training_path),
            "training_report_sha256": _sha256(training_path),
            "checkpoints": checkpoint_files,
            "train_files": train_files,
            "validation_files": validation_files,
        },
        "validation": {
            "records": len(validation_records),
            "wall_groups": len(validation_groups),
            "train_group_overlap": 0,
            "gates": gates,
        },
        "selected_gate": selected,
        "warning": (
            "A passing result estimates one first-opportunity intervention. It "
            "does not authorize repeated overrides, browser deployment or a "
            "human-strength claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/low-margin-top2-causal-residual-v1"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA selector，但当前 PyTorch 无可用 GPU")
    output = args.output_root / "validation-selection-report.json"
    if output.exists():
        raise ValueError("因果 validation selection 报告已存在，拒绝覆盖")
    report = select(args.output_root, device=args.device)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train wall-controlled v2 on v1 train without reading any validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.train_low_margin_top2_causal_residual_v1 import (
    BATCH_SIZE,
    GRADIENT_CLIP,
    HUBER_BETA,
    LEARNING_RATE,
    MAX_EPOCHS,
    TARGET_COVERAGES,
    WEIGHT_DECAY,
    _coverage_thresholds,
    train_member_tensors,
    load_formal_train,
)
from xiamen_mahjong.causal_residual import (
    CAUSAL_HIDDEN_SIZE,
    CAUSAL_PAIR_FEATURE_DIM,
    CAUSAL_SCORE_SCALE,
)
from xiamen_mahjong.low_margin_intervention import LowMarginCausalRecord
from xiamen_mahjong.wall_control import (
    CONTROL_COEFFICIENT_ABS_LIMIT,
    TRAIN_CONTROL_COEFFICIENT,
    estimate_control_coefficient,
    leave_one_rotation_baselines,
    policy_wall_components,
    save_wall_control_checkpoint,
)


ENSEMBLE_SEEDS = (202674000, 202674001, 202674002)
INNER_FIT_FRACTION = 0.9
INNER_SPLIT_SALT = "low-margin-wall-control-v2-inner-epoch-selection"
SOURCE_FEATURE_CACHE_VERSION = "low-margin-causal-train-tensors-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _source_tree_sha256() -> str:
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts" / "train_low_margin_top2_causal_residual_v1.py",
        *sorted((ROOT / "xiamen_mahjong").glob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _inner_fit_mask(group_ids: Sequence[str]) -> torch.Tensor:
    fit_groups: set[str] = set()
    holdout_groups: set[str] = set()
    for group_id in set(group_ids):
        digest = hashlib.blake2b(
            f"{INNER_SPLIT_SALT}|{group_id}".encode("utf-8"), digest_size=8
        ).digest()
        bucket = int.from_bytes(digest, "big") % 10_000
        target = (
            fit_groups
            if bucket < int(INNER_FIT_FRACTION * 10_000)
            else holdout_groups
        )
        target.add(group_id)
    if not fit_groups or not holdout_groups or fit_groups & holdout_groups:
        raise ValueError("wall-control v2 内部分组切分无效")
    return torch.tensor(
        [group_id in fit_groups for group_id in group_ids], dtype=torch.bool
    )


def _load_source_features(
    source_root: Path,
    input_files: Sequence[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = source_root / "train-feature-cache-v1.pt"
    if not path.is_file():
        raise ValueError("v1 train feature cache 不存在")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    input_sha = [item["sha256"] for item in input_files]
    if (
        not isinstance(payload, dict)
        or payload.get("version") != SOURCE_FEATURE_CACHE_VERSION
        or payload.get("input_sha256") != input_sha
        or not isinstance(payload.get("features"), torch.Tensor)
        or not isinstance(payload.get("treatment"), torch.Tensor)
    ):
        raise ValueError("v1 train feature cache 身份或 tensor 无效")
    return payload["features"], payload["treatment"], {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "reused_features_only": True,
    }


def _controlled_outcome(
    records: Sequence[LowMarginCausalRecord],
) -> tuple[torch.Tensor, list[str]]:
    baselines = leave_one_rotation_baselines(records)
    values = []
    group_ids = []
    for record, baseline in zip(records, baselines):
        if record.executed_arm == "none":
            continue
        values.append(
            (
                record.terminal_candidate_score
                - TRAIN_CONTROL_COEFFICIENT * baseline
            )
            / CAUSAL_SCORE_SCALE
        )
        group_ids.append(record.group_id)
    if not values:
        raise ValueError("wall-control v2 没有 eligible train target")
    return torch.tensor(values, dtype=torch.float32), group_ids


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("wall-control SD 输入无效")
    mean = sum(values) / len(values)
    return math.sqrt(
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
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
    report_path = args.output_root / "model-training-report.json"
    model_dir = args.output_root / "models"
    if report_path.exists() or model_dir.exists():
        raise ValueError("wall-control v2 模型输出已存在，拒绝覆盖")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 训练，但 PyTorch 没有可用 GPU")

    records, input_files = load_formal_train(args.source_root)
    features, treatment, source_cache = _load_source_features(
        args.source_root, input_files
    )
    outcome, eligible_group_ids = _controlled_outcome(records)
    expected_treatment = torch.tensor(
        [
            0.5 if record.executed_arm == "alternative" else -0.5
            for record in records
            if record.executed_arm != "none"
        ],
        dtype=torch.float32,
    )
    if (
        features.ndim != 2
        or features.shape[1] != CAUSAL_PAIR_FEATURE_DIM
        or not (len(features) == len(treatment) == len(outcome) > 0)
        or not torch.equal(treatment, expected_treatment)
    ):
        raise ValueError("wall-control v2 source feature 对齐失败")
    fit_mask = _inner_fit_mask(eligible_group_ids)
    fit_group_count = len(
        {
            group_id
            for group_id, flag in zip(eligible_group_ids, fit_mask.tolist())
            if flag
        }
    )
    holdout_group_count = len(set(eligible_group_ids)) - fit_group_count

    resolved = torch.device(args.device)
    features = features.to(resolved)
    treatment = treatment.to(resolved)
    outcome = outcome.to(resolved)
    fit_mask = fit_mask.to(resolved)
    source_sha = _source_tree_sha256()
    revision = _source_revision()
    model_dir.mkdir(parents=True, exist_ok=False)
    networks = []
    member_reports = []
    for member_index, seed in enumerate(ENSEMBLE_SEEDS):
        network, member_report = train_member_tensors(
            features,
            treatment,
            outcome,
            fit_mask,
            seed=seed,
            device=args.device,
        )
        member_report["fit_groups"] = fit_group_count
        member_report["inner_holdout_groups"] = holdout_group_count
        checkpoint = model_dir / f"member-{member_index}.pt"
        save_wall_control_checkpoint(
            network,
            checkpoint,
            metadata={
                "seed": seed,
                "selected_epoch": member_report["selected_epoch"],
                "train_input_sha256": [item["sha256"] for item in input_files],
                "source_feature_cache_sha256": source_cache["sha256"],
                "source_tree_sha256": source_sha,
                "source_revision": revision,
                "v1_validation_used": False,
                "v1_terminal_used": False,
                "v2_validation_read": False,
                "v2_terminal_read": False,
            },
        )
        member_report["checkpoint"] = str(checkpoint)
        member_report["checkpoint_sha256"] = _sha256(checkpoint)
        networks.append(network)
        member_reports.append(member_report)

    with torch.no_grad():
        member_effects_tensor = torch.stack(
            [network(features)[1] * CAUSAL_SCORE_SCALE for network in networks]
        )
        conservative_effects = member_effects_tensor.min(dim=0).values.cpu().tolist()
        member_effect_summaries = [
            {
                "mean_score_points": float(values.mean().cpu()),
                "std_score_points": float(values.std(unbiased=True).cpu()),
            }
            for values in member_effects_tensor
        ]
    thresholds = _coverage_thresholds(conservative_effects)
    full_effects: list[float | None] = []
    effect_iter = iter(conservative_effects)
    for record in records:
        full_effects.append(
            None if record.executed_arm == "none" else float(next(effect_iter))
        )
    try:
        next(effect_iter)
    except StopIteration:
        pass
    else:
        raise RuntimeError("wall-control v2 train effect 对齐失败")

    frozen_gates = []
    for threshold_report in thresholds:
        components = policy_wall_components(
            records,
            full_effects,
            threshold=float(threshold_report["strict_effect_threshold"]),
        )
        coefficient = estimate_control_coefficient(
            components["raw_wall_values"],
            components["control_wall_values"],
        )
        adjusted = [
            raw - coefficient * control
            for raw, control in zip(
                components["raw_wall_values"],
                components["control_wall_values"],
            )
        ]
        frozen_gates.append(
            {
                **threshold_report,
                "train_control_coefficient": coefficient,
                "control_coefficient_clip": [
                    -CONTROL_COEFFICIENT_ABS_LIMIT,
                    CONTROL_COEFFICIENT_ABS_LIMIT,
                ],
                "train_raw_wall_sd": _sample_sd(
                    components["raw_wall_values"]
                ),
                "train_adjusted_wall_sd": _sample_sd(adjusted),
                "train_variance_sd_ratio": _sample_sd(adjusted)
                / _sample_sd(components["raw_wall_values"]),
            }
        )

    payload = {
        "status": "wall_control_v2_trained_train_only_new_validation_uncollected",
        "protocol": {
            "architecture": "three_seed_hidden64_wall_control_v2",
            "input_dim": CAUSAL_PAIR_FEATURE_DIM,
            "hidden_size": CAUSAL_HIDDEN_SIZE,
            "score_scale": CAUSAL_SCORE_SCALE,
            "train_control_coefficient": TRAIN_CONTROL_COEFFICIENT,
            "gate_score": "minimum_effect_across_three_members",
            "ensemble_seeds": list(ENSEMBLE_SEEDS),
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "device": args.device,
            "tensor_residency": "entire_train_tensor_on_training_device",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "huber_beta_normalized": HUBER_BETA,
            "inner_fit_fraction": INNER_FIT_FRACTION,
            "inner_split_salt": INNER_SPLIT_SALT,
            "target_coverages": list(TARGET_COVERAGES),
            "validation_control": "frozen_train_covariance_coefficient",
            "validation_control_coefficient_abs_limit": (
                CONTROL_COEFFICIENT_ABS_LIMIT
            ),
        },
        "provenance": {
            "source_revision": revision,
            "source_tree_sha256": source_sha,
            "v1_train_input_files": input_files,
            "source_feature_cache": source_cache,
            "v1_validation_read_or_hashed": False,
            "v1_terminal_read_or_hashed": False,
        },
        "train": {
            "records": len(records),
            "eligible_records": len(conservative_effects),
            "wall_groups": len({record.group_id for record in records}),
        },
        "members": member_reports,
        "member_effect_summaries": member_effect_summaries,
        "frozen_train_gates": frozen_gates,
        "v2_validation_contract": "not_collected_opened_or_hashed_by_trainer",
        "v2_terminal_contract": "not_collected",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    printable = {
        **payload,
        "members": [
            {key: value for key, value in row.items() if key != "history"}
            for row in member_reports
        ],
        "provenance": {
            **payload["provenance"],
            "v1_train_input_files": f"{len(input_files)} immutable chunks",
        },
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

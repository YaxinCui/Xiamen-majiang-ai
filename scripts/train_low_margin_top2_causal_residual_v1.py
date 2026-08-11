#!/usr/bin/env python3
"""Train the fixed three-seed causal residual on the formal train split."""

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
from torch.nn import functional as F

from scripts.collect_low_margin_top2_causal_chunk_v1 import (
    CHUNK_WALLS,
    RECORD_ORDER_CONTRACT,
    SPLITS,
)
from xiamen_mahjong.causal_residual import (
    CAUSAL_HIDDEN_SIZE,
    CAUSAL_PAIR_FEATURE_DIM,
    CAUSAL_SCORE_SCALE,
    LowMarginCausalNetwork,
    causal_training_tensors,
    save_causal_checkpoint,
)
from xiamen_mahjong.low_margin_intervention import (
    LowMarginCausalRecord,
    audit_low_margin_causal_records,
    read_low_margin_causal_jsonl,
)


ENSEMBLE_SEEDS = (202672000, 202672001, 202672002)
MAX_EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
GRADIENT_CLIP = 5.0
HUBER_BETA = 0.5
INNER_FIT_FRACTION = 0.9
INNER_SPLIT_SALT = "low-margin-causal-v1-inner-epoch-selection"
TARGET_COVERAGES = (0.01, 0.02, 0.05, 0.10)
FEATURE_CACHE_VERSION = "low-margin-causal-train-tensors-v1"


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


def load_formal_train(
    output_root: Path,
) -> tuple[list[LowMarginCausalRecord], list[dict[str, Any]]]:
    split = "train"
    chunk_count = int(SPLITS[split]["chunks"])
    audit_path = output_root / "train-split-audit.json"
    if not audit_path.is_file():
        raise ValueError("正式 train split 总审计报告不存在")
    aggregate_report = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        aggregate_report.get("status")
        != "formal_low_margin_top2_causal_split_ready"
        or aggregate_report.get("split") != split
        or not aggregate_report.get("aggregate_audit", {}).get("ready")
    ):
        raise ValueError("正式 train split 总审计未通过")
    records: list[LowMarginCausalRecord] = []
    files: list[dict[str, Any]] = []
    for index in range(chunk_count):
        path = output_root / "data" / split / f"chunk-{index:02d}.records.jsonl"
        chunk_report_path = output_root / "data" / split / f"chunk-{index:02d}.report.json"
        if not path.is_file():
            raise ValueError(f"正式 train 缺少 chunk-{index:02d}")
        digest = _sha256(path)
        chunk_report = json.loads(chunk_report_path.read_text(encoding="utf-8"))
        if (
            chunk_report.get("safe_records", {}).get("sha256") != digest
            or chunk_report.get("safe_records", {}).get("record_order")
            != RECORD_ORDER_CONTRACT
        ):
            raise ValueError(f"正式 train chunk-{index:02d} 顺序契约或 SHA 无效")
        # The immutable SHA is bound to a chunk report and the split-wide
        # actor-visible replay audit.  Parse the schema here without performing
        # a third identical exact-hand replay before every training run.
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
        expected_wall_groups=chunk_count * CHUNK_WALLS,
        validate_actor_visible_pairs=False,
    )
    if not audit["ready"] or audit != aggregate_report["aggregate_audit"]:
        raise ValueError("训练时重算的 train split 审计与冻结报告不一致")
    return records, files


def _inner_fit_mask(group_ids: Sequence[str]) -> torch.Tensor:
    fit_groups: set[str] = set()
    holdout_groups: set[str] = set()
    for group_id in set(group_ids):
        digest = hashlib.blake2b(
            f"{INNER_SPLIT_SALT}|{group_id}".encode("utf-8"), digest_size=8
        ).digest()
        bucket = int.from_bytes(digest, "big") % 10_000
        (fit_groups if bucket < int(INNER_FIT_FRACTION * 10_000) else holdout_groups).add(
            group_id
        )
    if not fit_groups or not holdout_groups or fit_groups & holdout_groups:
        raise ValueError("因果 residual 内部分组切分无效")
    return torch.tensor(
        [group_id in fit_groups for group_id in group_ids], dtype=torch.bool
    )


def _network(seed: int, device: torch.device) -> LowMarginCausalNetwork:
    torch.manual_seed(seed)
    network = LowMarginCausalNetwork().to(device)
    return network


def _metrics(
    network: LowMarginCausalNetwork,
    features: torch.Tensor,
    treatment: torch.Tensor,
    outcome: torch.Tensor,
) -> dict[str, float | int]:
    network.eval()
    with torch.no_grad():
        nuisance, effect = network(features)
        predicted = nuisance + treatment * effect
    error = predicted - outcome
    loss = F.smooth_l1_loss(predicted, outcome, beta=HUBER_BETA)
    return {
        "examples": len(outcome),
        "factual_huber_loss": float(loss),
        "factual_rmse_score_points": float(torch.sqrt((error * error).mean()) * CAUSAL_SCORE_SCALE),
        "factual_mae_score_points": float(error.abs().mean() * CAUSAL_SCORE_SCALE),
        "effect_mean_score_points": float(effect.mean() * CAUSAL_SCORE_SCALE),
        "effect_std_score_points": float(effect.std(unbiased=False) * CAUSAL_SCORE_SCALE),
    }


def _fit_epochs(
    network: LowMarginCausalNetwork,
    features: torch.Tensor,
    treatment: torch.Tensor,
    outcome: torch.Tensor,
    *,
    epochs: int,
    seed: int,
    epoch_holdout: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        network.train()
        order = torch.randperm(
            len(features), generator=generator, device=features.device
        )
        total_loss = torch.zeros((), device=features.device)
        observations = 0
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            x = features[indices]
            t = treatment[indices]
            y = outcome[indices]
            optimizer.zero_grad(set_to_none=True)
            nuisance, effect = network(x)
            prediction = nuisance + t * effect
            loss = F.smooth_l1_loss(prediction, y, beta=HUBER_BETA)
            if not torch.isfinite(loss):
                raise RuntimeError("因果 residual loss 非有限数")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), GRADIENT_CLIP)
            optimizer.step()
            total_loss += loss.detach() * len(indices)
            observations += len(indices)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_factual_huber_loss": float(total_loss.cpu()) / observations,
        }
        if epoch_holdout is not None:
            row["inner_holdout"] = _metrics(network, *epoch_holdout)
        history.append(row)
    return history


def train_member_tensors(
    features: torch.Tensor,
    treatment: torch.Tensor,
    outcome: torch.Tensor,
    fit_mask: torch.Tensor,
    *,
    seed: int,
    device: str,
) -> tuple[LowMarginCausalNetwork, dict[str, Any]]:
    resolved = torch.device(device)
    if not (
        len(features) == len(treatment) == len(outcome) == len(fit_mask)
        and fit_mask.dtype == torch.bool
        and bool(fit_mask.any())
        and bool((~fit_mask).any())
    ):
        raise ValueError("因果 residual tensor 或内部分组 mask 无效")
    features = features.to(resolved)
    treatment = treatment.to(resolved)
    outcome = outcome.to(resolved)
    fit_mask = fit_mask.to(resolved)
    fit = (features[fit_mask], treatment[fit_mask], outcome[fit_mask])
    holdout = (features[~fit_mask], treatment[~fit_mask], outcome[~fit_mask])
    selector = _network(seed, resolved)
    history = _fit_epochs(
        selector,
        *fit,
        epochs=MAX_EPOCHS,
        seed=seed,
        epoch_holdout=holdout,
    )
    selected = min(
        history,
        key=lambda row: (
            row["inner_holdout"]["factual_huber_loss"], row["epoch"]
        ),
    )
    selected_epoch = int(selected["epoch"])
    final = _network(seed, resolved)
    _fit_epochs(
        final,
        features,
        treatment,
        outcome,
        epochs=selected_epoch,
        seed=seed,
    )
    report = {
        "seed": seed,
        "fit_examples": int(fit_mask.sum().cpu()),
        "inner_holdout_examples": int((~fit_mask).sum().cpu()),
        "selected_epoch": selected_epoch,
        "selected_inner_holdout": selected["inner_holdout"],
        "final_all_train": _metrics(final, features, treatment, outcome),
        "history": history,
    }
    return final, report


def _build_or_load_feature_cache(
    output_root: Path,
    records: Sequence[LowMarginCausalRecord],
    input_files: Sequence[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    cache_path = output_root / "train-feature-cache-v1.pt"
    input_sha = [item["sha256"] for item in input_files]
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, dict)
            or payload.get("version") != FEATURE_CACHE_VERSION
            or payload.get("input_sha256") != input_sha
        ):
            raise ValueError("因果 train feature cache 身份无效")
        features = payload.get("features")
        treatment = payload.get("treatment")
        outcome = payload.get("outcome")
        fit_mask = payload.get("fit_mask")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (features, treatment, outcome, fit_mask)
        ):
            raise ValueError("因果 train feature cache tensor 缺失")
        return features, treatment, outcome, fit_mask, {
            "path": str(cache_path),
            "sha256": _sha256(cache_path),
            "reused": True,
            "bytes": cache_path.stat().st_size,
        }
    eligible = [record for record in records if record.executed_arm != "none"]
    features, treatment, outcome = causal_training_tensors(eligible)
    fit_mask = _inner_fit_mask([record.group_id for record in eligible])
    temporary = cache_path.with_suffix(".pt.building")
    if temporary.exists():
        raise ValueError("因果 train feature cache 临时文件已存在")
    torch.save(
        {
            "version": FEATURE_CACHE_VERSION,
            "input_sha256": input_sha,
            "features": features,
            "treatment": treatment,
            "outcome": outcome,
            "fit_mask": fit_mask,
        },
        temporary,
    )
    temporary.replace(cache_path)
    return features, treatment, outcome, fit_mask, {
        "path": str(cache_path),
        "sha256": _sha256(cache_path),
        "reused": False,
        "bytes": cache_path.stat().st_size,
    }


def _coverage_thresholds(effects: Sequence[float]) -> list[dict[str, float | int]]:
    if not effects or any(not math.isfinite(value) for value in effects):
        raise ValueError("train ensemble effect 为空或非有限")
    ordered = sorted(effects)
    thresholds = []
    for coverage in TARGET_COVERAGES:
        rank = max(0, min(len(ordered) - 1, math.ceil((1.0 - coverage) * len(ordered)) - 1))
        threshold = ordered[rank]
        overrides = sum(value > threshold for value in effects)
        thresholds.append(
            {
                "target_coverage": coverage,
                "strict_effect_threshold": threshold,
                "train_overrides": overrides,
                "train_eligible": len(effects),
                "realized_train_coverage": overrides / len(effects),
            }
        )
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/low-margin-top2-causal-residual-v1"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    model_dir = args.output_root / "models"
    report_path = args.output_root / "model-training-report.json"
    if report_path.exists() or model_dir.exists():
        raise ValueError("正式因果 residual 模型输出已存在，拒绝覆盖")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 训练，但当前 PyTorch 无可用 GPU")
    records, input_files = load_formal_train(args.output_root)
    features, treatment, outcome, fit_mask, feature_cache = (
        _build_or_load_feature_cache(args.output_root, records, input_files)
    )
    eligible_group_ids = [
        record.group_id for record in records if record.executed_arm != "none"
    ]
    fit_flags = fit_mask.tolist()
    fit_group_count = len(
        {
            group_id
            for group_id, is_fit in zip(eligible_group_ids, fit_flags)
            if is_fit
        }
    )
    inner_holdout_group_count = len(
        {
            group_id
            for group_id, is_fit in zip(eligible_group_ids, fit_flags)
            if not is_fit
        }
    )
    resolved = torch.device(args.device)
    # The whole formal tensor is only ~40 MiB.  Keeping it resident avoids
    # thousands of tiny CPU→GPU copies and uses CUDA for the dense work while
    # leaving rule/state validation on CPU where it belongs.
    features = features.to(resolved)
    treatment = treatment.to(resolved)
    outcome = outcome.to(resolved)
    fit_mask = fit_mask.to(resolved)
    source_sha = _source_tree_sha256()
    revision = _source_revision()
    networks = []
    member_reports = []
    model_dir.mkdir(parents=True, exist_ok=False)
    for member_index, seed in enumerate(ENSEMBLE_SEEDS):
        network, report = train_member_tensors(
            features,
            treatment,
            outcome,
            fit_mask,
            seed=seed,
            device=args.device,
        )
        report["fit_groups"] = fit_group_count
        report["inner_holdout_groups"] = inner_holdout_group_count
        checkpoint = model_dir / f"member-{member_index}.pt"
        save_causal_checkpoint(
            network,
            checkpoint,
            metadata={
                "seed": seed,
                "selected_epoch": report["selected_epoch"],
                "train_input_sha256": [item["sha256"] for item in input_files],
                "source_tree_sha256": source_sha,
                "source_revision": revision,
                "validation_read": False,
                "terminal_read": False,
            },
        )
        report["checkpoint"] = str(checkpoint)
        report["checkpoint_sha256"] = _sha256(checkpoint)
        networks.append(network)
        member_reports.append(report)
    with torch.no_grad():
        member_effects = [network(features)[1] for network in networks]
        train_effects = (
            torch.stack(member_effects).mean(dim=0) * CAUSAL_SCORE_SCALE
        ).cpu().tolist()
    payload = {
        "status": "causal_residual_trained_train_only_validation_unread",
        "protocol": {
            "architecture": "three_seed_shared_hidden64_nuisance_and_effect_heads",
            "input_dim": CAUSAL_PAIR_FEATURE_DIM,
            "hidden_size": CAUSAL_HIDDEN_SIZE,
            "score_scale": CAUSAL_SCORE_SCALE,
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
            "epoch_selection": "minimum_inner_group_holdout_factual_huber_loss",
            "final_refit": "all_train_groups_for_selected_epoch",
            "target_coverages": list(TARGET_COVERAGES),
            "treatment_encoding": "alternative=+0.5_teacher=-0.5",
            "inference_features_exclude": [
                "executed_arm",
                "propensity",
                "terminal_score",
                "seed",
                "wall",
                "opponent_hands",
            ],
        },
        "provenance": {
            "source_revision": revision,
            "source_tree_sha256": source_sha,
            "input_files": input_files,
            "feature_cache": feature_cache,
        },
        "train": {
            "records": len(records),
            "eligible_records": len(train_effects),
            "wall_groups": len({record.group_id for record in records}),
        },
        "members": member_reports,
        "frozen_train_coverage_thresholds": _coverage_thresholds(train_effects),
        "validation_contract": "not_opened_or_hashed_by_trainer",
        "terminal_contract": "not_collected",
    }
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    printable = {
        **payload,
        "members": [
            {key: value for key, value in report.items() if key != "history"}
            for report in member_reports
        ],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

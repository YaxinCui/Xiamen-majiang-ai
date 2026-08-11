#!/usr/bin/env python3
"""Train a fresh small MLP using pairwise-only exact-tie preferences."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import Tensor
from torch.nn import functional as F

from xiamen_mahjong.pairwise_review import (
    pairwise_training_comparisons,
    read_pairwise_labels,
)
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import NEURAL_FEATURE_DIMS, _dense_action_features


FRESH_SEED = 202638600
FEATURE_VERSION = 3
HIDDEN_SIZE = 64
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MINIMUM_TRAIN_VALIDATION_LABELS = 75
MINIMUM_TRAIN_VALIDATION_CORRECTIONS = 15
MINIMUM_TRAIN_VALIDATION_GROUPS = 60


def _split_path(directory: Path, split: str) -> Path:
    return directory / f"{split}.pairwise-review.jsonl"


def build_pairwise_tensors(
    comparisons: Sequence[tuple[str, Any, tuple[int, int]]],
    *,
    feature_version: int = FEATURE_VERSION,
) -> tuple[Tensor, Tensor]:
    """Encode only the displayed pair and its relative target."""

    if feature_version not in NEURAL_FEATURE_DIMS:
        raise ValueError("不支持的 pairwise feature version")
    features: list[list[list[float]]] = []
    targets: list[int] = []
    for _group, decision, pair in comparisons:
        if len(pair) != 2 or len(set(pair)) != 2:
            raise ValueError("pairwise comparison 必须包含两个不同动作")
        if decision.chosen_index not in pair:
            raise ValueError("pairwise chosen action 不在展示动作对中")
        features.append(
            [
                _dense_action_features(
                    decision.state,
                    decision.legal_actions[index],
                    feature_version=feature_version,
                )
                for index in pair
            ]
        )
        targets.append(pair.index(decision.chosen_index))
    if not features:
        raise ValueError("pairwise comparison 不能为空")
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
    )


def validate_inputs(split_dir: Path) -> dict[str, Any]:
    paths = {split: _split_path(split_dir, split) for split in ("train", "validation", "test")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError("固定 pairwise 训练输入缺失：" + ", ".join(missing))

    # Test existence is checked, but its bytes remain unopened until a
    # validation-selected gate is frozen in a separate command.
    train_records = read_pairwise_labels(paths["train"])
    validation_records = read_pairwise_labels(paths["validation"])
    train_rows = pairwise_training_comparisons(train_records)
    validation_rows = pairwise_training_comparisons(validation_records)
    rows = [*train_rows, *validation_rows]
    groups = {group for group, _decision, _pair in rows}
    corrections = sum(
        decision.chosen_index != decision.reference_teacher_index
        for _group, decision, _pair in rows
    )
    train_groups = {group for group, _decision, _pair in train_rows}
    validation_groups = {group for group, _decision, _pair in validation_rows}
    if train_groups & validation_groups:
        raise ValueError("pairwise train/validation 存在原始牌局 group 重叠")
    reasons = []
    if len(rows) < MINIMUM_TRAIN_VALIDATION_LABELS:
        reasons.append("insufficient_train_validation_labels")
    if corrections < MINIMUM_TRAIN_VALIDATION_CORRECTIONS:
        reasons.append("insufficient_train_validation_corrections")
    if len(groups) < MINIMUM_TRAIN_VALIDATION_GROUPS:
        reasons.append("insufficient_train_validation_groups")
    if reasons:
        raise ValueError("pairwise train/validation 未通过 fixed v1 门槛：" + ", ".join(reasons))
    return {
        "paths": paths,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "labels": len(rows),
        "corrections": corrections,
        "groups": len(groups),
        "group_overlap": 0,
        "test_contract": "exists_but_bytes_unopened_reserved_for_gate",
    }


def _metrics(network, features: Tensor, targets: Tensor, device: torch.device) -> dict[str, float]:
    network.eval()
    with torch.no_grad():
        x = features.to(device)
        y = targets.to(device)
        mask = torch.ones((len(x), 2), dtype=torch.bool, device=device)
        logits, _value = network(x, mask)
        loss = F.cross_entropy(logits, y)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y).float().mean()
        gaps = logits.gather(1, y.unsqueeze(1)).squeeze(1) - logits.gather(
            1, (1 - y).unsqueeze(1)
        ).squeeze(1)
    return {
        "pairwise_loss": float(loss.cpu()),
        "pairwise_accuracy": float(accuracy.cpu()),
        "mean_chosen_logit_gap": float(gaps.mean().cpu()),
    }


def train_pairwise_model(
    train_rows: Sequence[tuple[str, Any, tuple[int, int]]],
    validation_rows: Sequence[tuple[str, Any, tuple[int, int]]],
    *,
    device: str = "cpu",
    seed: int = FRESH_SEED,
    hidden_size: int = HIDDEN_SIZE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tuple[TorchPolicyValueAgent, dict[str, Any]]:
    if epochs <= 0 or batch_size <= 0 or hidden_size <= 0:
        raise ValueError("pairwise 训练超参数必须为正数")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved = torch.device(device)
    policy = TorchPolicyValueAgent(
        feature_version=FEATURE_VERSION,
        hidden_size=hidden_size,
        architecture="candidate_mlp",
        device=str(resolved),
    )
    network = policy.network
    train_features, train_targets = build_pairwise_tensors(train_rows)
    validation_features, validation_targets = build_pairwise_tensors(validation_rows)
    trainable = [
        *network.candidate_encoder.parameters(),
        *network.policy_head.parameters(),
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    best_key: tuple[float, float, int] | None = None
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        network.train()
        order = torch.randperm(len(train_features), generator=generator)
        running_loss = 0.0
        observations = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            x = train_features[indices].to(resolved)
            y = train_targets[indices].to(resolved)
            mask = torch.ones((len(x), 2), dtype=torch.bool, device=resolved)
            optimizer.zero_grad(set_to_none=True)
            logits, _value = network(x, mask)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("pairwise loss 非有限数")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * len(indices)
            observations += len(indices)
        validation = _metrics(
            network, validation_features, validation_targets, resolved
        )
        row = {
            "epoch": epoch,
            "train_pairwise_loss": running_loss / observations,
            "validation": validation,
        }
        history.append(row)
        key = (
            validation["pairwise_loss"],
            -validation["pairwise_accuracy"],
            epoch,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_state = copy.deepcopy(network.state_dict())
    assert best_state is not None and best_key is not None
    network.load_state_dict(best_state)
    network.eval()
    selected_epoch = min(
        history,
        key=lambda row: (
            row["validation"]["pairwise_loss"],
            -row["validation"]["pairwise_accuracy"],
            row["epoch"],
        ),
    )["epoch"]
    report = {
        "status": "pairwise_model_trained_validation_selected_test_unread",
        "protocol": {
            "architecture": "candidate_mlp",
            "feature_version": FEATURE_VERSION,
            "hidden_size": hidden_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": seed,
            "target_semantics": "pairwise_only_not_full_action_classification",
            "value_target": "none",
            "test": "unread",
        },
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "selected_epoch": selected_epoch,
        "selected_validation": history[selected_epoch - 1]["validation"],
        "history": history,
    }
    return policy, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-split-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/exact-tie-pairwise-residual-v1"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validated = validate_inputs(args.pairwise_split_dir)
    safe = {
        "status": "fixed_exact_tie_pairwise_residual_v1_inputs_ready",
        "labels": validated["labels"],
        "corrections": validated["corrections"],
        "groups": validated["groups"],
        "group_overlap": validated["group_overlap"],
        "test": validated["test_contract"],
        "protocol": {
            "architecture": "candidate_mlp",
            "feature_version": FEATURE_VERSION,
            "hidden_size": HIDDEN_SIZE,
            "epochs": EPOCHS,
            "target_semantics": "pairwise_only_not_full_action_classification",
        },
        "privacy": "aggregate_only_no_state_hand_history_action_face_or_group_id",
    }
    if args.dry_run:
        print(json.dumps(safe, ensure_ascii=False, indent=2))
        return
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise ValueError("pairwise residual 输出已存在且非空，拒绝覆盖或续训")
    policy, report = train_pairwise_model(
        validated["train_rows"],
        validated["validation_rows"],
        device=args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "policy-value.pt"
    policy.save(
        checkpoint,
        metadata={
            "source": "local_human_exact_tie_pairwise_review_opt_in",
            "target_semantics": "pairwise_only_not_full_action_classification",
            "selected_epoch": report["selected_epoch"],
            "test_read": False,
        },
    )
    payload = {**report, "input_audit": safe, "checkpoint": checkpoint.name}
    (args.output_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

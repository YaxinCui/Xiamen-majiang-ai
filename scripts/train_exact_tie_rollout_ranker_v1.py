#!/usr/bin/env python3
"""Train a conservative small ensemble on exact-tie paired rollout labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
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

from xiamen_mahjong.exact_tie_rollout import (
    ExactTieRolloutRecord,
    exact_tie_pairwise_examples,
    read_exact_tie_rollout_records,
    split_exact_tie_rollout_records_by_group,
)
from xiamen_mahjong.torch_policy import TorchPolicyValueAgent
from xiamen_mahjong.training import _dense_action_features


FEATURE_VERSION = 3
HIDDEN_SIZE = 64
ENSEMBLE_SEEDS = (202640100, 202640101, 202640102, 202640103, 202640104)
EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
INNER_SPLIT_SALT = "exact-tie-rollout-ranker-v1-inner-epoch-selection"
MINIMUM_TRAIN_GROUPS = 500
MINIMUM_OUTER_VALIDATION_GROUPS = 60
MINIMUM_OUTER_VALIDATION_RECORDS = 200
MINIMUM_OUTER_VALIDATION_OVERRIDES = 20


def _split_path(directory: Path, split: str) -> Path:
    return directory / f"{split}.exact-tie-rollout.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_tensors(
    examples: Sequence[tuple[str, dict[str, Any], tuple[Any, Any], int]],
    *,
    feature_version: int = FEATURE_VERSION,
) -> tuple[Tensor, Tensor]:
    if not examples:
        raise ValueError("exact-tie pairwise examples 不能为空")
    features = [
        [
            _dense_action_features(state, action, feature_version=feature_version)
            for action in actions
        ]
        for _group, state, actions, _target in examples
    ]
    targets = [target for _group, _state, _actions, target in examples]
    return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.long)


def _pair_metrics(network, features: Tensor, targets: Tensor, device: torch.device) -> dict[str, float]:
    network.eval()
    with torch.no_grad():
        x = features.to(device)
        y = targets.to(device)
        mask = torch.ones((len(x), 2), dtype=torch.bool, device=device)
        logits, _value = network(x, mask)
        loss = F.cross_entropy(logits, y)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
    return {"loss": float(loss.cpu()), "accuracy": float(accuracy.cpu())}


def _train_member(
    train_examples,
    epoch_examples,
    *,
    seed: int,
    device: str,
    feature_version: int = FEATURE_VERSION,
    hidden_size: int = HIDDEN_SIZE,
    epochs: int = EPOCHS,
) -> tuple[TorchPolicyValueAgent, dict[str, Any]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved = torch.device(device)
    policy = TorchPolicyValueAgent(
        feature_version=feature_version,
        hidden_size=hidden_size,
        architecture="candidate_mlp",
        device=str(resolved),
    )
    network = policy.network
    train_x, train_y = _pair_tensors(train_examples, feature_version=feature_version)
    epoch_x, epoch_y = _pair_tensors(epoch_examples, feature_version=feature_version)
    trainable = [*network.candidate_encoder.parameters(), *network.policy_head.parameters()]
    optimizer = torch.optim.AdamW(
        trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    best_key = None
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        network.train()
        order = torch.randperm(len(train_x), generator=generator)
        total_loss = 0.0
        observations = 0
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            x = train_x[indices].to(resolved)
            y = train_y[indices].to(resolved)
            mask = torch.ones((len(x), 2), dtype=torch.bool, device=resolved)
            optimizer.zero_grad(set_to_none=True)
            logits, _value = network(x, mask)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("exact-tie ranker loss 非有限数")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(indices)
            observations += len(indices)
        epoch_metrics = _pair_metrics(network, epoch_x, epoch_y, resolved)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / observations,
                "inner_epoch_validation": epoch_metrics,
            }
        )
        key = (epoch_metrics["loss"], -epoch_metrics["accuracy"], epoch)
        if best_key is None or key < best_key:
            best_key = key
            best_state = copy.deepcopy(network.state_dict())
    assert best_state is not None
    network.load_state_dict(best_state)
    network.eval()
    selected = min(
        history,
        key=lambda row: (
            row["inner_epoch_validation"]["loss"],
            -row["inner_epoch_validation"]["accuracy"],
            row["epoch"],
        ),
    )
    return policy, {
        "seed": seed,
        "selected_epoch": selected["epoch"],
        "selected_inner_epoch_validation": selected["inner_epoch_validation"],
        "history": history,
    }


def _record_scores(policy: TorchPolicyValueAgent, record: ExactTieRolloutRecord) -> list[float]:
    from xiamen_mahjong.training import TeacherDecision

    decision = TeacherDecision(
        profile=record.profile,
        seed=None,
        seat=0,
        state=record.state,
        legal_actions=record.actions,
        chosen_index=record.teacher_index,
        reference_teacher_index=record.teacher_index,
    )
    return policy.scores(decision)


def evaluate_unanimous_ensemble(
    policies: Sequence[TorchPolicyValueAgent],
    records: Sequence[ExactTieRolloutRecord],
) -> dict[str, Any]:
    """Evaluate one-intervention SPIBB behavior on held-out source worlds."""

    if not policies or not records:
        raise ValueError("exact-tie ensemble/records 不能为空")
    group_deltas: dict[str, list[float]] = defaultdict(list)
    overrides = 0
    disagreements = 0
    positive = 0
    negative = 0
    total_delta = 0.0
    for record in records:
        votes = []
        for policy in policies:
            scores = _record_scores(policy, record)
            votes.append(max(range(len(scores)), key=lambda index: (scores[index], -index)))
        chosen = votes[0] if len(set(votes)) == 1 else record.teacher_index
        if len(set(votes)) != 1:
            disagreements += 1
        if chosen != record.teacher_index:
            overrides += 1
        delta = float(
            record.terminal_scores[chosen]
            - record.terminal_scores[record.teacher_index]
        )
        positive += delta > 0
        negative += delta < 0
        total_delta += delta
        group_deltas[record.split_group_id].append(delta)
    wall_values = [sum(values) / len(values) for values in group_deltas.values()]
    mean = sum(wall_values) / len(wall_values)
    variance = (
        sum((value - mean) ** 2 for value in wall_values) / (len(wall_values) - 1)
        if len(wall_values) > 1
        else 0.0
    )
    stderr = math.sqrt(variance / len(wall_values))
    return {
        "records": len(records),
        "wall_groups": len(wall_values),
        "overrides": overrides,
        "override_rate": overrides / len(records),
        "ensemble_disagreements": disagreements,
        "positive_record_deltas": positive,
        "negative_record_deltas": negative,
        "record_delta_mean": total_delta / len(records),
        "paired_wall_delta_mean": mean,
        "paired_wall_delta_stderr": stderr,
        "paired_wall_delta_95pct_low": mean - 1.96 * stderr,
        "paired_wall_delta_95pct_high": mean + 1.96 * stderr,
    }


def evaluate_unanimous_future_average(
    policies: Sequence[TorchPolicyValueAgent],
    records: Sequence[ExactTieRolloutRecord],
) -> dict[str, Any]:
    """Evaluate one public decision after averaging its future permutations."""

    if not policies or not records:
        raise ValueError("exact-tie averaged ensemble/records 不能为空")
    buckets = {}
    for record in records:
        key = (
            record.split_group_id,
            json.dumps(record.state, ensure_ascii=False, sort_keys=True),
            record.actions,
            record.teacher_index,
        )
        if key not in buckets:
            buckets[key] = (record, [[] for _ in record.actions])
        _representative, samples = buckets[key]
        for index, score in enumerate(record.terminal_scores):
            samples[index].append(score)
    group_deltas: dict[str, list[float]] = defaultdict(list)
    overrides = 0
    disagreements = 0
    positive = 0
    negative = 0
    deltas = []
    permutation_counts = []
    for representative, samples in buckets.values():
        means = [sum(values) / len(values) for values in samples]
        permutation_counts.append(len(samples[0]))
        votes = []
        for policy in policies:
            scores = _record_scores(policy, representative)
            votes.append(
                max(range(len(scores)), key=lambda index: (scores[index], -index))
            )
        chosen = votes[0] if len(set(votes)) == 1 else representative.teacher_index
        disagreements += len(set(votes)) != 1
        overrides += chosen != representative.teacher_index
        delta = float(means[chosen] - means[representative.teacher_index])
        positive += delta > 0.0
        negative += delta < 0.0
        deltas.append(delta)
        group_deltas[representative.split_group_id].append(delta)
    wall_values = [sum(values) / len(values) for values in group_deltas.values()]
    mean = sum(wall_values) / len(wall_values)
    variance = (
        sum((value - mean) ** 2 for value in wall_values) / (len(wall_values) - 1)
        if len(wall_values) > 1
        else 0.0
    )
    stderr = math.sqrt(variance / len(wall_values))
    return {
        "records": len(records),
        "distinct_public_decisions": len(buckets),
        "wall_groups": len(wall_values),
        "minimum_future_permutations": min(permutation_counts),
        "maximum_future_permutations": max(permutation_counts),
        "overrides": overrides,
        "override_rate": overrides / len(buckets),
        "ensemble_disagreements": disagreements,
        "positive_decision_deltas": positive,
        "negative_decision_deltas": negative,
        "decision_delta_mean": sum(deltas) / len(deltas),
        "paired_wall_delta_mean": mean,
        "paired_wall_delta_stderr": stderr,
        "paired_wall_delta_95pct_low": mean - 1.96 * stderr,
        "paired_wall_delta_95pct_high": mean + 1.96 * stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise ValueError("exact-tie ranker 输出目录已存在且非空")
    paths = {split: _split_path(args.split_dir, split) for split in ("train", "validation", "test")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError("exact-tie ranker 缺少 split：" + ", ".join(missing))
    # The outer test file is checked for existence and hashed, but its bytes
    # are deliberately not parsed until a separate validation-passed command.
    test_sha256 = _sha256(paths["test"])
    train_records = read_exact_tie_rollout_records(paths["train"])
    train_groups = {record.split_group_id for record in train_records}
    if len(train_groups) < MINIMUM_TRAIN_GROUPS:
        raise ValueError("exact-tie ranker train group 数不足")
    inner = split_exact_tie_rollout_records_by_group(
        train_records,
        split_salt=INNER_SPLIT_SALT,
        train_fraction=0.9,
        validation_fraction=0.05,
    )
    fit_records = inner["train"]
    # Both small holdouts are epoch-selection-only and remain part of the
    # outer train partition; joining them avoids a fragile tiny validation.
    epoch_records = [*inner["validation"], *inner["test"]]
    fit_examples = exact_tie_pairwise_examples(fit_records)
    epoch_examples = exact_tie_pairwise_examples(epoch_records)
    policies = []
    member_reports = []
    for seed in ENSEMBLE_SEEDS:
        policy, member_report = _train_member(
            fit_examples, epoch_examples, seed=seed, device=args.device
        )
        policies.append(policy)
        member_reports.append(member_report)
    outer_records = read_exact_tie_rollout_records(paths["validation"])
    outer_groups = {record.split_group_id for record in outer_records}
    if len(outer_groups) < MINIMUM_OUTER_VALIDATION_GROUPS:
        raise ValueError("exact-tie outer validation group 数不足")
    if len(outer_records) < MINIMUM_OUTER_VALIDATION_RECORDS:
        raise ValueError("exact-tie outer validation record 数不足")
    outer = evaluate_unanimous_ensemble(policies, outer_records)
    reasons = []
    if outer["overrides"] < MINIMUM_OUTER_VALIDATION_OVERRIDES:
        reasons.append("insufficient_outer_validation_overrides")
    if outer["paired_wall_delta_95pct_low"] <= 0.0:
        reasons.append("outer_validation_paired_lower_bound_not_positive")
    status = (
        "validation_passed_test_unread_ready_for_fixed_test"
        if not reasons
        else "validation_rejected_test_unread"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_names = []
    for index, (policy, member_report) in enumerate(zip(policies, member_reports)):
        name = f"member-{index}.pt"
        policy.save(
            args.output_dir / name,
            metadata={
                "source": "exact_tie_source_world_paired_rollout_v1",
                "deployment_scope": "exact_teacher_top_score_tie_only",
                "ensemble_rule": "unanimous_else_frozen_teacher",
                "selected_epoch": member_report["selected_epoch"],
                "outer_test_read": False,
            },
        )
        checkpoint_names.append(name)
    report = {
        "status": status,
        "gate_reasons": reasons,
        "protocol": {
            "architecture": "five_member_candidate_mlp_ensemble",
            "feature_version": FEATURE_VERSION,
            "hidden_size": HIDDEN_SIZE,
            "ensemble_seeds": ENSEMBLE_SEEDS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "target": "unequal_paired_terminal_order_only",
            "deployment": "exact_teacher_tie_unanimous_override_else_teacher",
            "maximum_intended_interventions_per_hand": 1,
        },
        "input": {
            "train_records": len(train_records),
            "train_groups": len(train_groups),
            "fit_records": len(fit_records),
            "epoch_selection_records": len(epoch_records),
            "fit_pairwise_examples": len(fit_examples),
            "epoch_pairwise_examples": len(epoch_examples),
            "outer_validation_file_sha256": _sha256(paths["validation"]),
            "outer_test_file_sha256": test_sha256,
            "outer_test_contract": "file_hashed_but_records_unparsed",
        },
        "members": member_reports,
        "outer_validation": outer,
        "checkpoints": checkpoint_names,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**report, "members": [
        {key: value for key, value in row.items() if key != "history"}
        for row in member_reports
    ]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

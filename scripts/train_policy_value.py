#!/usr/bin/env python3
"""Train a shared GPU policy-value network from versioned trajectories."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as error:
    raise SystemExit("请使用 .venv/bin/python 运行；该脚本需要 PyTorch") from error

from xiamen_mahjong.torch_policy import (
    ARCHITECTURE_CANDIDATE_MLP,
    ARCHITECTURE_PUBLIC_SEQUENCE_RESIDUAL,
    ARCHITECTURE_PUBLIC_SEQUENCE_TRANSFORMER,
    CandidatePolicyValueNetwork,
    PublicSequencePolicyValueNetwork,
    ResidualPublicSequencePolicyValueNetwork,
    TorchPolicyValueAgent,
)
from xiamen_mahjong.training import (
    NEURAL_FEATURE_DIMS,
    PUBLIC_ACTION_SEQUENCE_DIM,
    PUBLIC_ACTION_SEQUENCE_LENGTH,
    TeacherDecision,
    _dense_action_features,
    public_action_sequence_features,
    read_trajectory_jsonl,
)


@dataclass(frozen=True)
class Example:
    candidates: tuple[tuple[float, ...], ...]
    public_events: tuple[tuple[float, ...], ...]
    chosen_index: int
    action_kind: str
    source: str
    action_values: tuple[float, ...] | None
    action_value_stderrs: tuple[float, ...] | None
    value_target: float | None
    sample_weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("artifacts/trajectory-balanced-classic-v2-run1")
    parser.add_argument("--train", type=Path, default=base / "train.trajectories.jsonl")
    parser.add_argument(
        "--additional-train",
        type=Path,
        action="append",
        default=[],
        help="可重复指定；追加在线 DAgger 等训练轨迹",
    )
    parser.add_argument(
        "--validation", type=Path, default=base / "validation.trajectories.jsonl"
    )
    parser.add_argument("--test", type=Path, default=base / "test.trajectories.jsonl")
    parser.add_argument(
        "--additional-test",
        type=Path,
        action="append",
        default=[],
        help="可重复指定；追加与在线数据同分布的最终测试轨迹",
    )
    parser.add_argument(
        "--additional-validation",
        type=Path,
        action="append",
        default=[],
        help="可重复指定；追加与在线数据同分布的验证轨迹",
    )
    parser.add_argument("--feature-version", type=int, choices=(3,), default=3)
    parser.add_argument(
        "--architecture",
        choices=(
            ARCHITECTURE_CANDIDATE_MLP,
            ARCHITECTURE_PUBLIC_SEQUENCE_TRANSFORMER,
            ARCHITECTURE_PUBLIC_SEQUENCE_RESIDUAL,
        ),
        default=ARCHITECTURE_CANDIDATE_MLP,
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--base-learning-rate-scale",
        type=float,
        default=1.0,
        help="残差序列模型中继承的候选 MLP 参数学习率倍率",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--value-scale", type=float, default=80.0)
    parser.add_argument("--exploration-weight", type=float, default=0.35)
    parser.add_argument("--synthetic-weight", type=float, default=1.0)
    parser.add_argument(
        "--action-value-weight",
        type=float,
        default=1.0,
        help="反事实动作价值轨迹的样本权重；0 可关闭该来源",
    )
    parser.add_argument(
        "--action-value-regression-weight",
        type=float,
        default=1.0,
        help="直接动作 Q 回归损失的权重；0 时仅使用软动作偏好",
    )
    parser.add_argument(
        "--action-value-target-scale",
        type=float,
        default=80.0,
        help="动作 Q 的终局净分归一化尺度；会写入 checkpoint 元数据",
    )
    parser.add_argument(
        "--action-value-temperature",
        type=float,
        default=16.0,
        help="将动作结算分转成软偏好分布的温度（分数单位）",
    )
    parser.add_argument(
        "--action-value-stderr-scale",
        type=float,
        default=0.0,
        help="大于 0 时按 1/(1+(平均标准误/scale)^2) 下调高方差动作价值样本；0 关闭",
    )
    parser.add_argument(
        "--action-value-margin-scale",
        type=float,
        default=16.0,
        help="按 min(1, 动作价值跨度/scale) 下调无差异或近乎无差异的 rollout 标签；0 关闭",
    )
    parser.add_argument("--max-class-weight", type=float, default=6.0)
    parser.add_argument(
        "--checkpoint-selection-source",
        choices=("overall", "counterfactual_action_value_rollout"),
        default="overall",
        help="best checkpoint 使用的验证来源；动作价值微调应选择专属留出集",
    )
    parser.add_argument(
        "--checkpoint-selection-metric",
        choices=("policy_loss", "action_value_huber_loss"),
        default="policy_loss",
        help="best checkpoint 的主指标；直接 Q 回归应使用 action_value_huber_loss",
    )
    parser.add_argument(
        "--minimum-selection-decisions",
        type=int,
        default=1,
        help="所选验证来源的最小决策数，避免用极小样本挑 checkpoint",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="可选：从已有 .pt policy-value checkpoint 增量微调",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/policy-value-classic-v1")
    )
    return parser.parse_args()


def source_weight(source: str, args: argparse.Namespace) -> float:
    if source == "random_legal_teacher_labeled":
        return args.exploration_weight
    if source in {"physical_response_pass_search", "engine_validated_tour_curriculum"}:
        return args.synthetic_weight
    if source == "counterfactual_action_value_rollout":
        return args.action_value_weight
    return 1.0


def load_examples(path: Path, args: argparse.Namespace) -> list[Example]:
    examples: list[Example] = []
    for trajectory in read_trajectory_jsonl(path):
        source = str(trajectory.source_metadata.get("collector", "legacy"))
        synthetic = bool(trajectory.outcome.get("synthetic"))
        scores = trajectory.outcome.get("scores", [])
        for decision in trajectory.decisions:
            vectors = tuple(
                tuple(
                    _dense_action_features(
                        decision.state, action, feature_version=args.feature_version
                    )
                )
                for action in decision.legal_actions
            )
            events = public_action_sequence_features(decision.state)
            value_target = None
            # Random legal rollouts are useful to expose policy decisions, but
            # their terminal score belongs to the random continuation, not to
            # the Teacher-like policy being learned.  Keep value regression on
            # the coherent Teacher continuation until on-policy trajectories
            # are available.
            if not synthetic and source in {
                "teacher_self_play",
                "candidate_vs_teacher_dagger",
            }:
                value_target = float(scores[decision.seat]) / args.value_scale
            sample_weight = source_weight(source, args)
            if decision.action_values is not None and args.action_value_margin_scale > 0:
                action_value_span = max(decision.action_values) - min(
                    decision.action_values
                )
                sample_weight *= min(
                    1.0, action_value_span / args.action_value_margin_scale
                )
            if (
                decision.action_value_stderrs is not None
                and args.action_value_stderr_scale > 0
            ):
                mean_stderr = sum(decision.action_value_stderrs) / len(
                    decision.action_value_stderrs
                )
                sample_weight /= 1.0 + (
                    mean_stderr / args.action_value_stderr_scale
                ) ** 2
            examples.append(
                Example(
                    candidates=vectors,
                    public_events=events,
                    chosen_index=decision.chosen_index,
                    action_kind=decision.chosen_action.kind,
                    source=source,
                    action_values=decision.action_values,
                    action_value_stderrs=decision.action_value_stderrs,
                    value_target=value_target,
                    sample_weight=sample_weight,
                )
            )
    return examples


def class_weights(examples: Iterable[Example], maximum: float) -> dict[str, float]:
    if maximum <= 0:
        raise ValueError("max-class-weight 必须为正数")
    counts = Counter(example.action_kind for example in examples)
    largest = max(counts.values())
    return {
        kind: min(maximum, math.sqrt(largest / count))
        for kind, count in sorted(counts.items())
    }


def tensors(
    examples: list[Example],
    *,
    feature_dim: int,
    device: torch.device,
    weights: dict[str, float],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch_size = len(examples)
    max_actions = max(len(example.candidates) for example in examples)
    candidates = torch.zeros(
        (batch_size, max_actions, feature_dim), dtype=torch.float32, device=device
    )
    mask = torch.zeros((batch_size, max_actions), dtype=torch.bool, device=device)
    events = torch.zeros(
        (batch_size, PUBLIC_ACTION_SEQUENCE_LENGTH, PUBLIC_ACTION_SEQUENCE_DIM),
        dtype=torch.float32,
        device=device,
    )
    event_mask = torch.zeros(
        (batch_size, PUBLIC_ACTION_SEQUENCE_LENGTH), dtype=torch.bool, device=device
    )
    chosen = torch.empty(batch_size, dtype=torch.long, device=device)
    action_values = torch.zeros(
        (batch_size, max_actions), dtype=torch.float32, device=device
    )
    action_value_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    policy_weights = torch.empty(batch_size, dtype=torch.float32, device=device)
    values = torch.zeros(batch_size, dtype=torch.float32, device=device)
    value_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    action_value_stderrs = torch.zeros(
        (batch_size, max_actions), dtype=torch.float32, device=device
    )
    for row, example in enumerate(examples):
        count = len(example.candidates)
        candidates[row, :count] = torch.tensor(example.candidates, dtype=torch.float32)
        mask[row, :count] = True
        event_count = len(example.public_events)
        if event_count:
            events[row, :event_count] = torch.tensor(
                example.public_events, dtype=torch.float32, device=device
            )
            event_mask[row, :event_count] = True
        chosen[row] = example.chosen_index
        if example.action_values is not None:
            if len(example.action_values) != count:
                raise ValueError("动作价值目标与候选动作数不匹配")
            action_values[row, :count] = torch.tensor(
                example.action_values, dtype=torch.float32, device=device
            )
            action_value_mask[row] = True
            if example.action_value_stderrs is not None:
                if len(example.action_value_stderrs) != count:
                    raise ValueError("动作价值标准误与候选动作数不匹配")
                action_value_stderrs[row, :count] = torch.tensor(
                    example.action_value_stderrs,
                    dtype=torch.float32,
                    device=device,
                )
        # A held-out online source may contain a rare action not present in a
        # deliberately small training split.  Evaluation still needs a valid
        # target; use neutral class weight instead of failing on that action.
        policy_weights[row] = example.sample_weight * weights.get(
            example.action_kind, 1.0
        )
        if example.value_target is not None:
            values[row] = example.value_target
            value_mask[row] = True
    return (
        candidates,
        mask,
        events,
        event_mask,
        chosen,
        action_values,
        action_value_mask,
        policy_weights,
        values,
        value_mask,
        action_value_stderrs,
    )


def policy_preference_loss(
    logits: torch.Tensor,
    *,
    chosen: torch.Tensor,
    action_values: torch.Tensor,
    action_value_mask: torch.Tensor,
    action_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Return per-decision cross entropy against hard or rollout-soft labels."""

    if temperature <= 0:
        raise ValueError("action-value-temperature 必须为正数")
    hard_targets = F.one_hot(chosen, num_classes=logits.shape[1]).to(logits.dtype)
    value_logits = (action_values / temperature).masked_fill(
        ~action_mask, torch.finfo(logits.dtype).min
    )
    soft_targets = F.softmax(value_logits, dim=1)
    targets = torch.where(action_value_mask.unsqueeze(1), soft_targets, hard_targets)
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)


def action_value_regression_loss(
    predicted_action_values: torch.Tensor,
    *,
    action_values: torch.Tensor,
    action_value_mask: torch.Tensor,
    action_mask: torch.Tensor,
    target_scale: float,
    action_value_stderrs: torch.Tensor | None = None,
    stderr_scale: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-decision robust Q loss, absolute error and valid action mask.

    Q supervision predicts normalized terminal net score for every legal
    action.  It remains separate from the soft policy objective: two actions
    can induce similar choice probabilities while still differing materially
    in expected score.  Optional per-action standard-error weighting avoids a
    noisy rollout branch dominating a decision merely because its siblings are
    well estimated.
    """

    if target_scale <= 0:
        raise ValueError("action-value-target-scale 必须为正数")
    if stderr_scale < 0:
        raise ValueError("action-value-stderr-scale 不能为负数")
    valid = action_mask & action_value_mask.unsqueeze(1)
    targets = action_values / target_scale
    per_action_loss = F.smooth_l1_loss(
        predicted_action_values, targets, reduction="none"
    )
    weights = valid.to(predicted_action_values.dtype)
    if action_value_stderrs is not None and stderr_scale > 0:
        uncertainty = 1.0 / (1.0 + (action_value_stderrs / stderr_scale) ** 2)
        weights = weights * uncertainty
    denominator = weights.sum(dim=1).clamp_min(1.0)
    per_decision_loss = (per_action_loss * weights).sum(dim=1) / denominator
    absolute_error = (predicted_action_values - targets).abs()
    return per_decision_loss, absolute_error, valid


def forward_network(
    network: (
        CandidatePolicyValueNetwork
        | PublicSequencePolicyValueNetwork
        | ResidualPublicSequencePolicyValueNetwork
    ),
    candidates: torch.Tensor,
    action_mask: torch.Tensor,
    public_events: torch.Tensor,
    event_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if isinstance(
        network,
        (PublicSequencePolicyValueNetwork, ResidualPublicSequencePolicyValueNetwork),
    ):
        logits, values = network(candidates, action_mask, public_events, event_mask)
        return logits, values, None
    if isinstance(network, CandidatePolicyValueNetwork):
        return network.forward_with_action_values(candidates, action_mask)
    logits, values = network(candidates, action_mask)
    return logits, values, None


def evaluate(
    network: (
        CandidatePolicyValueNetwork
        | PublicSequencePolicyValueNetwork
        | ResidualPublicSequencePolicyValueNetwork
    ),
    examples: list[Example],
    *,
    feature_dim: int,
    device: torch.device,
    batch_size: int,
    class_weight_map: dict[str, float],
    action_value_temperature: float,
    action_value_target_scale: float,
    action_value_stderr_scale: float,
) -> dict[str, Any]:
    network.eval()
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "correct": 0.0,
            "loss": 0.0,
            "value_count": 0.0,
            "action_value_count": 0.0,
            "action_value_action_count": 0.0,
            "action_value_huber": 0.0,
            "action_value_absolute_error": 0.0,
            "action_value_rank_correct": 0.0,
            "action_value_rank_count": 0.0,
            "mae": 0.0,
            "mse": 0.0,
        }
    )
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            rows = examples[start : start + batch_size]
            (
                candidates,
                mask,
                events,
                event_mask,
                chosen,
                action_values,
                action_value_mask,
                _weights,
                values,
                value_mask,
                action_value_stderrs,
            ) = tensors(
                rows, feature_dim=feature_dim, device=device, weights=class_weight_map
            )
            logits, predicted_values, predicted_action_values = forward_network(
                network, candidates, mask, events, event_mask
            )
            loss = policy_preference_loss(
                logits,
                chosen=chosen,
                action_values=action_values,
                action_value_mask=action_value_mask,
                action_mask=mask,
                temperature=action_value_temperature,
            ).detach().cpu().tolist()
            predicted = logits.argmax(dim=1).detach().cpu().tolist()
            values_cpu = predicted_values.detach().cpu().tolist()
            targets_cpu = values.detach().cpu().tolist()
            masks_cpu = value_mask.detach().cpu().tolist()
            action_value_masks_cpu = action_value_mask.detach().cpu().tolist()
            if predicted_action_values is not None:
                q_losses, q_absolute_errors, q_valid = action_value_regression_loss(
                    predicted_action_values,
                    action_values=action_values,
                    action_value_mask=action_value_mask,
                    action_mask=mask,
                    target_scale=action_value_target_scale,
                    action_value_stderrs=action_value_stderrs,
                    stderr_scale=action_value_stderr_scale,
                )
                q_losses_cpu = q_losses.detach().cpu().tolist()
                q_absolute_errors_cpu = q_absolute_errors.detach().cpu().tolist()
                q_valid_cpu = q_valid.detach().cpu().tolist()
                q_predictions_cpu = predicted_action_values.detach().cpu().tolist()
                q_targets_cpu = action_values.detach().cpu().tolist()
            else:
                q_losses_cpu = []
                q_absolute_errors_cpu = []
                q_valid_cpu = []
                q_predictions_cpu = []
                q_targets_cpu = []
            for index, example in enumerate(rows):
                bucket = totals[example.source]
                bucket["count"] += 1
                bucket["correct"] += predicted[index] == example.chosen_index
                bucket["loss"] += loss[index]
                bucket["action_value_count"] += action_value_masks_cpu[index]
                if action_value_masks_cpu[index] and predicted_action_values is not None:
                    valid_indices = [
                        action_index
                        for action_index, active in enumerate(q_valid_cpu[index])
                        if active
                    ]
                    bucket["action_value_huber"] += q_losses_cpu[index]
                    bucket["action_value_action_count"] += len(valid_indices)
                    bucket["action_value_absolute_error"] += sum(
                        q_absolute_errors_cpu[index][action_index]
                        for action_index in valid_indices
                    )
                    predicted_best = max(
                        valid_indices,
                        key=lambda action_index: (
                            q_predictions_cpu[index][action_index],
                            -action_index,
                        ),
                    )
                    target_best = max(
                        valid_indices,
                        key=lambda action_index: (
                            q_targets_cpu[index][action_index],
                            -action_index,
                        ),
                    )
                    bucket["action_value_rank_count"] += 1
                    bucket["action_value_rank_correct"] += predicted_best == target_best
                if masks_cpu[index]:
                    error = values_cpu[index] - targets_cpu[index]
                    bucket["value_count"] += 1
                    bucket["mae"] += abs(error)
                    bucket["mse"] += error * error
    def render(bucket: dict[str, float]) -> dict[str, float]:
        count = max(bucket["count"], 1.0)
        result = {
            "decisions": bucket["count"],
            "accuracy": bucket["correct"] / count,
            "policy_loss": bucket["loss"] / count,
            "value_decisions": bucket["value_count"],
            "action_value_decisions": bucket["action_value_count"],
            "action_value_actions": bucket["action_value_action_count"],
        }
        if bucket["value_count"]:
            result["value_mae_scaled"] = bucket["mae"] / bucket["value_count"]
            result["value_rmse_scaled"] = math.sqrt(bucket["mse"] / bucket["value_count"])
        if bucket["action_value_count"]:
            result["action_value_huber_loss"] = (
                bucket["action_value_huber"] / bucket["action_value_count"]
            )
        if bucket["action_value_action_count"]:
            result["action_value_mae_scaled"] = (
                bucket["action_value_absolute_error"]
                / bucket["action_value_action_count"]
            )
            result["action_value_mae_score"] = (
                result["action_value_mae_scaled"] * action_value_target_scale
            )
        if bucket["action_value_rank_count"]:
            result["action_value_rank_accuracy"] = (
                bucket["action_value_rank_correct"]
                / bucket["action_value_rank_count"]
            )
        return result
    overall = {
        "count": 0.0,
        "correct": 0.0,
        "loss": 0.0,
        "value_count": 0.0,
        "action_value_count": 0.0,
        "action_value_action_count": 0.0,
        "action_value_huber": 0.0,
        "action_value_absolute_error": 0.0,
        "action_value_rank_correct": 0.0,
        "action_value_rank_count": 0.0,
        "mae": 0.0,
        "mse": 0.0,
    }
    for bucket in totals.values():
        for key in overall:
            overall[key] += bucket[key]
    return {
        "overall": render(overall),
        "by_source": {key: render(value) for key, value in sorted(totals.items())},
    }


def better_validation_checkpoint(
    candidate: dict[str, float],
    best: dict[str, float] | None,
    *,
    metric: str = "policy_loss",
) -> bool:
    """Choose by matching held-out objective with a deterministic tie break."""

    if best is None:
        return True
    if metric == "action_value_huber_loss":
        return (
            float(candidate["action_value_huber_loss"]),
            -float(candidate["action_value_rank_accuracy"]),
        ) < (
            float(best["action_value_huber_loss"]),
            -float(best["action_value_rank_accuracy"]),
        )
    if metric != "policy_loss":
        raise ValueError("不支持的 checkpoint 选择指标")
    return (
        float(candidate["policy_loss"]),
        -float(candidate["accuracy"]),
    ) < (
        float(best["policy_loss"]),
        -float(best["accuracy"]),
    )


def checkpoint_selection_metrics(
    validation: dict[str, Any],
    *,
    source: str,
    minimum_decisions: int,
    metric: str = "policy_loss",
) -> dict[str, float]:
    """Extract a held-out metric without silently drowning online Q labels."""

    if minimum_decisions <= 0:
        raise ValueError("minimum-selection-decisions 必须为正数")
    if source == "overall":
        metrics = validation["overall"]
    else:
        metrics = validation["by_source"].get(source)
        if metrics is None:
            raise ValueError(f"验证集缺少 checkpoint 选择来源：{source}")
    decision_key = (
        "action_value_decisions"
        if metric == "action_value_huber_loss"
        else "decisions"
    )
    if int(metrics[decision_key]) < minimum_decisions:
        raise ValueError(
            f"checkpoint 选择来源 {source} 只有 {int(metrics[decision_key])} 条决策，"
            f"低于最小要求 {minimum_decisions}"
        )
    if metric == "action_value_huber_loss":
        required = {"action_value_huber_loss", "action_value_rank_accuracy"}
        missing = sorted(required.difference(metrics))
        if missing:
            raise ValueError(
                "checkpoint 选择来源缺少直接动作 Q 指标：" + ", ".join(missing)
            )
        return {
            "action_value_huber_loss": float(metrics["action_value_huber_loss"]),
            "action_value_rank_accuracy": float(
                metrics["action_value_rank_accuracy"]
            ),
        }
    if metric != "policy_loss":
        raise ValueError("不支持的 checkpoint 选择指标")
    return {"policy_loss": float(metrics["policy_loss"]), "accuracy": float(metrics["accuracy"])}


def main() -> None:
    args = parse_args()
    if (
        args.hidden_size <= 0
        or args.epochs <= 0
        or args.batch_size <= 0
        or args.learning_rate <= 0
        or args.base_learning_rate_scale <= 0
        or args.attention_heads <= 0
        or args.value_scale <= 0
        or args.exploration_weight <= 0
        or args.synthetic_weight <= 0
        or args.action_value_weight < 0
        or args.action_value_regression_weight < 0
        or args.action_value_target_scale <= 0
        or args.action_value_temperature <= 0
        or args.action_value_stderr_scale < 0
        or args.action_value_margin_scale < 0
        or args.minimum_selection_decisions <= 0
    ):
        raise ValueError("训练超参数必须为正数")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    resolved_device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    device = torch.device(resolved_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但当前 PyTorch 无可用 GPU")
    feature_dim = NEURAL_FEATURE_DIMS[args.feature_version]
    train = load_examples(args.train, args)
    for path in args.additional_train:
        train.extend(load_examples(path, args))
    validation = load_examples(args.validation, args)
    for path in args.additional_validation:
        validation.extend(load_examples(path, args))
    test = load_examples(args.test, args)
    for path in args.additional_test:
        test.extend(load_examples(path, args))
    if not train or not validation or not test:
        raise ValueError("train、validation 和 test 都必须含有决策")
    has_action_value_targets = any(
        example.action_values is not None
        for examples in (train, validation, test)
        for example in examples
    )
    weights = class_weights(train, args.max_class_weight)
    if args.init_checkpoint:
        initial_agent = TorchPolicyValueAgent.load(
            args.init_checkpoint, device=str(device)
        )
        matching_sequence_heads = (
            args.architecture
            in {
                ARCHITECTURE_PUBLIC_SEQUENCE_TRANSFORMER,
                ARCHITECTURE_PUBLIC_SEQUENCE_RESIDUAL,
            }
            and initial_agent.attention_heads != args.attention_heads
        )
        residual_from_candidate = (
            args.architecture == ARCHITECTURE_PUBLIC_SEQUENCE_RESIDUAL
            and initial_agent.architecture == ARCHITECTURE_CANDIDATE_MLP
        )
        if (
            initial_agent.feature_version != args.feature_version
            or initial_agent.hidden_size != args.hidden_size
            or matching_sequence_heads
            or (
                initial_agent.architecture != args.architecture
                and not residual_from_candidate
            )
        ):
            raise ValueError("初始 checkpoint 的特征版本或 hidden-size 与训练参数不一致")
        if residual_from_candidate:
            assert isinstance(initial_agent.network, CandidatePolicyValueNetwork)
            network = ResidualPublicSequencePolicyValueNetwork(
                feature_dim,
                args.hidden_size,
                attention_heads=args.attention_heads,
            ).to(device)
            network.initialize_from_candidate(initial_agent.network)
        else:
            network = initial_agent.network
    elif args.architecture == ARCHITECTURE_CANDIDATE_MLP:
        network = CandidatePolicyValueNetwork(feature_dim, args.hidden_size).to(device)
    elif args.architecture == ARCHITECTURE_PUBLIC_SEQUENCE_TRANSFORMER:
        network = PublicSequencePolicyValueNetwork(
            feature_dim,
            args.hidden_size,
            attention_heads=args.attention_heads,
        ).to(device)
    else:
        network = ResidualPublicSequencePolicyValueNetwork(
            feature_dim,
            args.hidden_size,
            attention_heads=args.attention_heads,
        ).to(device)
    if (
        has_action_value_targets
        and args.action_value_regression_weight > 0
        and not isinstance(network, CandidatePolicyValueNetwork)
    ):
        raise ValueError(
            "直接动作 Q 回归当前仅支持 candidate_mlp；"
            "请设 --action-value-regression-weight 0，或使用 candidate_mlp"
        )
    if isinstance(network, ResidualPublicSequencePolicyValueNetwork):
        base_parameters = list(network.candidate_encoder.parameters()) + list(
            network.base_policy_head.parameters()
        ) + list(network.base_value_head.parameters())
        base_parameter_ids = {id(parameter) for parameter in base_parameters}
        residual_parameters = [
            parameter
            for parameter in network.parameters()
            if id(parameter) not in base_parameter_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": base_parameters,
                    "lr": args.learning_rate * args.base_learning_rate_scale,
                },
                {"params": residual_parameters, "lr": args.learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    history = []
    best_epoch: int | None = None
    best_validation: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    indices = list(range(len(train)))
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(indices)
        network.train()
        policy_total = 0.0
        value_total = 0.0
        action_value_total = 0.0
        batches = 0
        for start in range(0, len(indices), args.batch_size):
            rows = [train[index] for index in indices[start : start + args.batch_size]]
            (
                candidates,
                mask,
                events,
                event_mask,
                chosen,
                action_values,
                action_value_mask,
                policy_weights,
                values,
                value_mask,
                action_value_stderrs,
            ) = tensors(
                rows, feature_dim=feature_dim, device=device, weights=weights
            )
            logits, predicted_values, predicted_action_values = forward_network(
                network, candidates, mask, events, event_mask
            )
            policy_loss_values = policy_preference_loss(
                logits,
                chosen=chosen,
                action_values=action_values,
                action_value_mask=action_value_mask,
                action_mask=mask,
                temperature=args.action_value_temperature,
            )
            policy_loss = (policy_loss_values * policy_weights).sum() / policy_weights.sum().clamp_min(1.0)
            value_loss = (
                F.smooth_l1_loss(predicted_values[value_mask], values[value_mask])
                if value_mask.any()
                else torch.zeros((), device=device)
            )
            if args.action_value_regression_weight > 0 and action_value_mask.any():
                if predicted_action_values is None:
                    raise RuntimeError("当前网络没有动作 Q 头")
                action_value_loss_values, _absolute_errors, _valid = (
                    action_value_regression_loss(
                        predicted_action_values,
                        action_values=action_values,
                        action_value_mask=action_value_mask,
                        action_mask=mask,
                        target_scale=args.action_value_target_scale,
                        action_value_stderrs=action_value_stderrs,
                        stderr_scale=args.action_value_stderr_scale,
                    )
                )
                action_value_weights = policy_weights * action_value_mask.to(
                    policy_weights.dtype
                )
                action_value_loss = (
                    action_value_loss_values * action_value_weights
                ).sum() / action_value_weights.sum().clamp_min(1.0)
            else:
                action_value_loss = torch.zeros((), device=device)
            loss = (
                policy_loss
                + args.value_weight * value_loss
                + args.action_value_regression_weight * action_value_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
            optimizer.step()
            policy_total += float(policy_loss.detach().cpu())
            value_total += float(value_loss.detach().cpu())
            action_value_total += float(action_value_loss.detach().cpu())
            batches += 1
        validation_metrics = evaluate(
            network,
            validation,
            feature_dim=feature_dim,
            device=device,
            batch_size=args.batch_size,
            class_weight_map=weights,
            action_value_temperature=args.action_value_temperature,
            action_value_target_scale=args.action_value_target_scale,
            action_value_stderr_scale=args.action_value_stderr_scale,
        )
        history.append(
            {
                "epoch": epoch,
                "train_policy_loss": policy_total / batches,
                "train_value_loss": value_total / batches,
                "train_action_value_huber_loss": action_value_total / batches,
                "validation": validation_metrics["overall"],
            }
        )
        selection_metrics = checkpoint_selection_metrics(
            validation_metrics,
            source=args.checkpoint_selection_source,
            minimum_decisions=args.minimum_selection_decisions,
            metric=args.checkpoint_selection_metric,
        )
        if better_validation_checkpoint(
            selection_metrics,
            best_validation,
            metric=args.checkpoint_selection_metric,
        ):
            best_epoch = epoch
            best_validation = selection_metrics
            best_state = copy.deepcopy(network.state_dict())
    if best_state is None or best_epoch is None or best_validation is None:
        raise RuntimeError("没有生成可用于选择 checkpoint 的验证指标")
    network.load_state_dict(best_state)
    validation_metrics = evaluate(
        network,
        validation,
        feature_dim=feature_dim,
        device=device,
        batch_size=args.batch_size,
        class_weight_map=weights,
        action_value_temperature=args.action_value_temperature,
        action_value_target_scale=args.action_value_target_scale,
        action_value_stderr_scale=args.action_value_stderr_scale,
    )
    test_metrics = evaluate(
        network,
        test,
        feature_dim=feature_dim,
        device=device,
        batch_size=args.batch_size,
        class_weight_map=weights,
        action_value_temperature=args.action_value_temperature,
        action_value_target_scale=args.action_value_target_scale,
        action_value_stderr_scale=args.action_value_stderr_scale,
    )
    report = {
        "model": "candidate_policy_value",
        "feature_version": args.feature_version,
        "feature_dim": feature_dim,
        "hidden_size": args.hidden_size,
        "architecture": args.architecture,
        "attention_heads": args.attention_heads,
        "device": str(device),
        "seed": args.seed,
        "value_scale": args.value_scale,
        "value_weight": args.value_weight,
        "action_value_regression_weight": args.action_value_regression_weight,
        "action_value_target_scale": args.action_value_target_scale,
        "action_value_temperature": args.action_value_temperature,
        "action_value_stderr_scale": args.action_value_stderr_scale,
        "action_value_margin_scale": args.action_value_margin_scale,
        "base_learning_rate_scale": args.base_learning_rate_scale,
        "inputs": {
            "train": [str(args.train), *(str(path) for path in args.additional_train)],
            "validation": [
                str(args.validation),
                *(str(path) for path in args.additional_validation),
            ],
            "test": str(args.test),
            "additional_test": [str(path) for path in args.additional_test],
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        },
        "class_weights": weights,
        "source_weights": {
            "teacher_self_play": 1.0,
            "candidate_vs_teacher_dagger": 1.0,
            "random_legal_teacher_labeled": args.exploration_weight,
            "physical_response_pass_search": args.synthetic_weight,
            "engine_validated_tour_curriculum": args.synthetic_weight,
            "counterfactual_action_value_rollout": args.action_value_weight,
        },
        "policy_targets": {
            "teacher_and_curriculum": "hard_teacher_action",
            "counterfactual_action_value_rollout": "softmax_terminal_score_q_pi",
        },
        "checkpoint_selection": {
            "split": "validation",
            "source": args.checkpoint_selection_source,
            "minimum_decisions": args.minimum_selection_decisions,
            "metric": (
                "lowest_action_value_huber_loss_then_highest_action_value_rank_accuracy"
                if args.checkpoint_selection_metric == "action_value_huber_loss"
                else "lowest_policy_loss_then_highest_accuracy"
            ),
            "selected_epoch": best_epoch,
            "selected_validation": best_validation,
            "final_epoch": args.epochs,
        },
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = TorchPolicyValueAgent(
        feature_version=args.feature_version,
        hidden_size=args.hidden_size,
        architecture=args.architecture,
        attention_heads=args.attention_heads,
        device=str(device),
        network=network,
    )
    agent.save(output_dir / "policy-value.pt", metadata=report)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Leave-one-rotation wall controls for randomized low-margin decisions."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .causal_residual import (
    CAUSAL_FEATURE_VERSION,
    CAUSAL_HIDDEN_SIZE,
    CAUSAL_PAIR_FEATURE_DIM,
    CAUSAL_SCORE_SCALE,
    LowMarginCausalNetwork,
    causal_pair_features,
)
from .low_margin_intervention import LowMarginCausalRecord


WALL_CONTROL_MODEL_VERSION = "xiamen-low-margin-wall-control-residual-v2"
TRAIN_CONTROL_COEFFICIENT = -1.0
CONTROL_COEFFICIENT_ABS_LIMIT = 4.0


def leave_one_rotation_baselines(
    records: Sequence[LowMarginCausalRecord],
) -> list[float]:
    """Return the mean outcome of the other three rotations in each wall.

    The four records of a physical wall may be stored in opaque UUID order.
    Only group membership is used; row or candidate-seat order is irrelevant.
    """

    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if not record.group_id:
            raise ValueError("wall-control 记录缺少 opaque group")
        if not math.isfinite(record.terminal_candidate_score):
            raise ValueError("wall-control 终局分数必须有限")
        groups[record.group_id].append(index)
    if not groups or any(len(indices) != 4 for indices in groups.values()):
        raise ValueError("wall-control 要求每个物理墙严格四条轮换记录")

    baselines = [0.0] * len(records)
    for indices in groups.values():
        total = sum(records[index].terminal_candidate_score for index in indices)
        for index in indices:
            baselines[index] = (
                total - records[index].terminal_candidate_score
            ) / 3.0
    return baselines


def wall_control_training_tensors(
    records: Sequence[LowMarginCausalRecord],
    *,
    control_coefficient: float = TRAIN_CONTROL_COEFFICIENT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build actor-visible features and a wall-controlled factual target."""

    if not math.isfinite(control_coefficient):
        raise ValueError("wall-control 训练系数必须有限")
    baselines = leave_one_rotation_baselines(records)
    eligible = [
        (record, baselines[index])
        for index, record in enumerate(records)
        if record.executed_arm != "none"
    ]
    if not eligible:
        raise ValueError("wall-control 训练记录没有随机干预")
    features = torch.tensor(
        [causal_pair_features(record) for record, _baseline in eligible],
        dtype=torch.float32,
    )
    treatment = torch.tensor(
        [
            0.5 if record.executed_arm == "alternative" else -0.5
            for record, _baseline in eligible
        ],
        dtype=torch.float32,
    )
    outcome = torch.tensor(
        [
            (
                record.terminal_candidate_score
                - control_coefficient * baseline
            )
            / CAUSAL_SCORE_SCALE
            for record, baseline in eligible
        ],
        dtype=torch.float32,
    )
    return features, treatment, outcome


def policy_wall_components(
    records: Sequence[LowMarginCausalRecord],
    effects: Sequence[float | None],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Build raw HT values Z and zero-mean wall controls Q for one gate."""

    if len(records) != len(effects) or not math.isfinite(threshold):
        raise ValueError("wall-control gate 输入长度或阈值无效")
    baselines = leave_one_rotation_baselines(records)
    group_z: dict[str, list[float]] = defaultdict(list)
    group_q: dict[str, list[float]] = defaultdict(list)
    override_groups: set[str] = set()
    logged = defaultdict(int)
    overrides = 0
    eligible = 0
    for record, effect, baseline in zip(records, effects, baselines):
        if record.executed_arm == "none":
            if effect is not None:
                raise ValueError("no_intervention 行不能出现 effect")
            override = False
        else:
            if effect is None or not math.isfinite(effect):
                raise ValueError("eligible wall-control 行缺少有限 effect")
            eligible += 1
            override = effect > threshold
        if override:
            overrides += 1
            override_groups.add(record.group_id)
            logged[record.executed_arm] += 1
            weight = (
                1.0 / record.propensity
                if record.executed_arm == "alternative"
                else -1.0 / (1.0 - record.propensity)
            )
            z_value = weight * record.terminal_candidate_score
            q_value = weight * baseline
        else:
            z_value = 0.0
            q_value = 0.0
        group_z[record.group_id].append(z_value)
        group_q[record.group_id].append(q_value)

    group_ids = sorted(group_z)
    if set(group_ids) != set(group_q) or any(
        len(group_z[group_id]) != 4 or len(group_q[group_id]) != 4
        for group_id in group_ids
    ):
        raise ValueError("wall-control 出现非四座墙组")
    return {
        "group_ids": group_ids,
        "raw_wall_values": [
            sum(group_z[group_id]) / 4.0 for group_id in group_ids
        ],
        "control_wall_values": [
            sum(group_q[group_id]) / 4.0 for group_id in group_ids
        ],
        "eligible_records": eligible,
        "overrides": overrides,
        "override_groups": len(override_groups),
        "logged_alternative_assignments": logged["alternative"],
        "logged_teacher_assignments": logged["teacher"],
    }


def estimate_control_coefficient(
    raw_wall_values: Sequence[float],
    control_wall_values: Sequence[float],
    *,
    absolute_limit: float = CONTROL_COEFFICIENT_ABS_LIMIT,
) -> float:
    """Fit and clip c in Z-cQ using train wall groups only."""

    if (
        len(raw_wall_values) != len(control_wall_values)
        or len(raw_wall_values) < 2
        or not math.isfinite(absolute_limit)
        or absolute_limit <= 0.0
        or any(not math.isfinite(value) for value in raw_wall_values)
        or any(not math.isfinite(value) for value in control_wall_values)
    ):
        raise ValueError("wall-control 系数估计输入无效")
    raw_mean = sum(raw_wall_values) / len(raw_wall_values)
    control_mean = sum(control_wall_values) / len(control_wall_values)
    covariance = sum(
        (raw - raw_mean) * (control - control_mean)
        for raw, control in zip(raw_wall_values, control_wall_values)
    ) / (len(raw_wall_values) - 1)
    variance = sum(
        (control - control_mean) ** 2 for control in control_wall_values
    ) / (len(control_wall_values) - 1)
    if variance <= 0.0:
        return 0.0
    coefficient = covariance / variance
    return max(-absolute_limit, min(absolute_limit, coefficient))


def adjusted_wall_values(
    raw_wall_values: Sequence[float],
    control_wall_values: Sequence[float],
    *,
    control_coefficient: float,
) -> list[float]:
    if (
        len(raw_wall_values) != len(control_wall_values)
        or not math.isfinite(control_coefficient)
    ):
        raise ValueError("wall-control 调整输入无效")
    return [
        raw - control_coefficient * control
        for raw, control in zip(raw_wall_values, control_wall_values)
    ]


def save_wall_control_checkpoint(
    network: LowMarginCausalNetwork,
    path: str | Path,
    *,
    metadata: dict[str, Any],
) -> None:
    destination = Path(path)
    if destination.exists():
        raise ValueError("wall-control checkpoint 已存在，拒绝覆盖")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": WALL_CONTROL_MODEL_VERSION,
            "feature_version": CAUSAL_FEATURE_VERSION,
            "input_dim": network.input_dim,
            "hidden_size": network.hidden_size,
            "score_scale": CAUSAL_SCORE_SCALE,
            "train_control_coefficient": TRAIN_CONTROL_COEFFICIENT,
            "state_dict": network.state_dict(),
            "metadata": metadata,
        },
        destination,
    )


def load_wall_control_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[LowMarginCausalNetwork, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("version") != WALL_CONTROL_MODEL_VERSION
        or payload.get("feature_version") != CAUSAL_FEATURE_VERSION
        or payload.get("input_dim") != CAUSAL_PAIR_FEATURE_DIM
        or payload.get("hidden_size") != CAUSAL_HIDDEN_SIZE
        or payload.get("score_scale") != CAUSAL_SCORE_SCALE
        or payload.get("train_control_coefficient")
        != TRAIN_CONTROL_COEFFICIENT
        or not isinstance(payload.get("state_dict"), dict)
        or not isinstance(payload.get("metadata"), dict)
    ):
        raise ValueError("不兼容的 wall-control checkpoint")
    network = LowMarginCausalNetwork()
    network.load_state_dict(payload["state_dict"], strict=True)
    network.to(device)
    network.eval()
    return network, dict(payload["metadata"])


def minimum_ensemble_effects(
    member_effects: Iterable[Sequence[float | None]],
) -> list[float | None]:
    members = [list(values) for values in member_effects]
    if not members or len({len(values) for values in members}) != 1:
        raise ValueError("wall-control ensemble 输入无效")
    result: list[float | None] = []
    for values in zip(*members):
        present = [value for value in values if value is not None]
        if not present:
            result.append(None)
        elif len(present) != len(values):
            raise ValueError("wall-control ensemble eligible 认定不一致")
        elif any(not math.isfinite(value) for value in present):
            raise ValueError("wall-control ensemble effect 非有限")
        else:
            result.append(min(present))
    return result

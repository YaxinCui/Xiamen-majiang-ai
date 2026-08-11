"""Small CPU causal residual for low-margin Teacher top-two decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError as error:  # pragma: no cover - local ML dependency
    raise RuntimeError("因果 residual 需要项目 PyTorch 虚拟环境") from error

from .low_margin_intervention import LowMarginCausalRecord
from .training import NEURAL_FEATURE_DIMS, _dense_action_features


LOW_MARGIN_CAUSAL_MODEL_VERSION = "xiamen-low-margin-causal-residual-v1"
CAUSAL_FEATURE_VERSION = 4
CAUSAL_ACTION_FEATURE_DIM = NEURAL_FEATURE_DIMS[CAUSAL_FEATURE_VERSION]
CAUSAL_PAIR_FEATURE_DIM = CAUSAL_ACTION_FEATURE_DIM * 3
CAUSAL_HIDDEN_SIZE = 64
CAUSAL_SCORE_SCALE = 40.0


def causal_pair_features(record: LowMarginCausalRecord) -> list[float]:
    """Encode Teacher, alternative and their difference without treatment."""

    if (
        record.executed_arm == "none"
        or record.state is None
        or record.teacher_action is None
        or record.alternative_action is None
    ):
        raise ValueError("no_intervention 记录没有可训练动作对")
    teacher = _dense_action_features(
        record.state,
        record.teacher_action,
        feature_version=CAUSAL_FEATURE_VERSION,
    )
    alternative = _dense_action_features(
        record.state,
        record.alternative_action,
        feature_version=CAUSAL_FEATURE_VERSION,
    )
    difference = [right - left for left, right in zip(teacher, alternative)]
    features = [*teacher, *alternative, *difference]
    if len(features) != CAUSAL_PAIR_FEATURE_DIM:
        raise RuntimeError("低分差因果 pair 特征维度错误")
    return features


class LowMarginCausalNetwork(nn.Module):
    """Shared public encoder with nuisance-outcome and treatment-effect heads."""

    def __init__(
        self,
        *,
        input_dim: int = CAUSAL_PAIR_FEATURE_DIM,
        hidden_size: int = CAUSAL_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_size <= 0:
            raise ValueError("因果 residual 网络维度必须为正数")
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.nuisance_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.GELU(), nn.Linear(32, 1)
        )
        self.effect_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.encoder(features)
        nuisance = self.nuisance_head(encoded).squeeze(-1)
        effect = self.effect_head(encoded).squeeze(-1)
        return nuisance, effect


def causal_training_tensors(
    records: Sequence[LowMarginCausalRecord],
) -> tuple[Tensor, Tensor, Tensor]:
    eligible = [record for record in records if record.executed_arm != "none"]
    if not eligible:
        raise ValueError("因果 residual 训练记录没有随机干预")
    features = torch.tensor(
        [causal_pair_features(record) for record in eligible], dtype=torch.float32
    )
    centered_treatment = torch.tensor(
        [0.5 if record.executed_arm == "alternative" else -0.5 for record in eligible],
        dtype=torch.float32,
    )
    outcomes = torch.tensor(
        [record.terminal_candidate_score / CAUSAL_SCORE_SCALE for record in eligible],
        dtype=torch.float32,
    )
    return features, centered_treatment, outcomes


def predict_causal_effects(
    network: LowMarginCausalNetwork,
    records: Sequence[LowMarginCausalRecord],
    *,
    device: str = "cpu",
    batch_size: int = 1024,
) -> list[float | None]:
    if batch_size <= 0:
        raise ValueError("因果 residual 推理 batch_size 必须为正数")
    resolved = torch.device(device)
    network.to(resolved)
    network.eval()
    effects: list[float | None] = [None] * len(records)
    eligible_indices = [
        index for index, record in enumerate(records)
        if record.executed_arm != "none"
    ]
    with torch.no_grad():
        for start in range(0, len(eligible_indices), batch_size):
            indices = eligible_indices[start : start + batch_size]
            features = torch.tensor(
                [causal_pair_features(records[index]) for index in indices],
                dtype=torch.float32,
                device=resolved,
            )
            _nuisance, effect = network(features)
            for index, value in zip(indices, effect.cpu().tolist()):
                effects[index] = float(value) * CAUSAL_SCORE_SCALE
    return effects


def save_causal_checkpoint(
    network: LowMarginCausalNetwork,
    path: str | Path,
    *,
    metadata: dict[str, Any],
) -> None:
    destination = Path(path)
    if destination.exists():
        raise ValueError("因果 residual checkpoint 已存在，拒绝覆盖")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": LOW_MARGIN_CAUSAL_MODEL_VERSION,
            "feature_version": CAUSAL_FEATURE_VERSION,
            "input_dim": network.input_dim,
            "hidden_size": network.hidden_size,
            "score_scale": CAUSAL_SCORE_SCALE,
            "state_dict": network.state_dict(),
            "metadata": metadata,
        },
        destination,
    )


def load_causal_checkpoint(
    path: str | Path, *, device: str = "cpu"
) -> tuple[LowMarginCausalNetwork, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("version") != LOW_MARGIN_CAUSAL_MODEL_VERSION
        or payload.get("feature_version") != CAUSAL_FEATURE_VERSION
        or payload.get("input_dim") != CAUSAL_PAIR_FEATURE_DIM
        or payload.get("hidden_size") != CAUSAL_HIDDEN_SIZE
        or payload.get("score_scale") != CAUSAL_SCORE_SCALE
        or not isinstance(payload.get("state_dict"), dict)
        or not isinstance(payload.get("metadata"), dict)
    ):
        raise ValueError("不兼容的低分差因果 residual checkpoint")
    network = LowMarginCausalNetwork()
    network.load_state_dict(payload["state_dict"], strict=True)
    network.to(device)
    network.eval()
    return network, dict(payload["metadata"])


def ensemble_causal_effects(
    networks: Iterable[LowMarginCausalNetwork],
    records: Sequence[LowMarginCausalRecord],
    *,
    device: str = "cpu",
) -> list[float | None]:
    members = list(networks)
    if not members:
        raise ValueError("因果 residual ensemble 不能为空")
    predictions = [
        predict_causal_effects(member, records, device=device)
        for member in members
    ]
    effects: list[float | None] = []
    for values in zip(*predictions):
        present = [value for value in values if value is not None]
        if not present:
            effects.append(None)
        elif len(present) != len(values):
            raise RuntimeError("因果 ensemble 对 eligible 记录认定不一致")
        else:
            effects.append(sum(present) / len(present))
    return effects

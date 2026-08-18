"""Teacher-relative advantage model with a zeroed Teacher baseline.

This is deliberately separate from historical terminal-score/Q heads.  Its
output is ``r(action) - r(Teacher action)`` at one observed information set,
so the Teacher action is identically zero.  A checkpoint is diagnostic until
it passes newly collected, unshrunk off-policy gates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

try:
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError as error:  # pragma: no cover - environment dependent.
    raise RuntimeError("relative advantage model 需要 PyTorch") from error

from .training import NEURAL_FEATURE_DIMS, TeacherDecision, _dense_action_features


RELATIVE_ADVANTAGE_VERSION = "xiamen-relative-advantage-model-v1"


class RelativeAdvantageNetwork(nn.Module):
    """Score legal actions, then center every row on its Teacher action."""

    def __init__(self, feature_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        candidates: Tensor,
        action_mask: Tensor,
        teacher_indices: Tensor,
    ) -> Tensor:
        if candidates.ndim != 3 or action_mask.shape != candidates.shape[:2]:
            raise ValueError("候选张量和合法动作掩码形状不匹配")
        if teacher_indices.shape != (candidates.shape[0],):
            raise ValueError("Teacher 索引批次形状不匹配")
        counts = action_mask.sum(dim=1)
        if torch.any(counts <= 0) or torch.any(teacher_indices < 0) or torch.any(
            teacher_indices >= counts
        ):
            raise ValueError("Teacher 索引不属于合法动作集合")
        raw = self.scorer(candidates).squeeze(-1)
        rows = torch.arange(candidates.shape[0], device=candidates.device)
        centered = raw - raw[rows, teacher_indices].unsqueeze(1)
        return centered.masked_fill(~action_mask, 0.0)


class RelativeAdvantageAgent:
    """Inference wrapper that exposes only Teacher-relative score estimates."""

    def __init__(
        self,
        *,
        feature_version: int = 3,
        hidden_size: int = 128,
        device: str | None = None,
        network: RelativeAdvantageNetwork | None = None,
    ) -> None:
        if feature_version not in NEURAL_FEATURE_DIMS:
            raise ValueError("不支持的 relative advantage 特征版本")
        if hidden_size <= 0:
            raise ValueError("hidden_size 必须为正数")
        self.feature_version = feature_version
        self.feature_dim = NEURAL_FEATURE_DIMS[feature_version]
        self.hidden_size = hidden_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.network = network or RelativeAdvantageNetwork(self.feature_dim, hidden_size)
        self.network.to(self.device)
        self.network.eval()

    def relative_advantages(self, decision: TeacherDecision) -> list[float]:
        vectors = [
            _dense_action_features(
                decision.state, action, feature_version=self.feature_version
            )
            for action in decision.legal_actions
        ]
        if not vectors:
            raise ValueError("没有合法动作")
        if not 0 <= decision.chosen_index < len(vectors):
            raise ValueError("Teacher 动作索引不属于合法动作")
        candidates = torch.tensor([vectors], dtype=torch.float32, device=self.device)
        mask = torch.ones((1, len(vectors)), dtype=torch.bool, device=self.device)
        teacher = torch.tensor([decision.chosen_index], dtype=torch.long, device=self.device)
        with torch.no_grad():
            values = self.network(candidates, mask, teacher)
        return [float(item) for item in values[0].detach().cpu()]

    def save(self, path: str | Path, *, metadata: dict[str, Any]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": RELATIVE_ADVANTAGE_VERSION,
                "model": "teacher_relative_advantage_centered_mlp",
                "feature_version": self.feature_version,
                "feature_dim": self.feature_dim,
                "hidden_size": self.hidden_size,
                "state_dict": self.network.state_dict(),
                "metadata": metadata,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> "RelativeAdvantageAgent":
        payload = torch.load(Path(path), map_location=device or "cpu", weights_only=True)
        if payload.get("version") != RELATIVE_ADVANTAGE_VERSION:
            raise ValueError("不支持的 relative advantage checkpoint 版本")
        feature_version = int(payload["feature_version"])
        expected_dim = NEURAL_FEATURE_DIMS.get(feature_version)
        if expected_dim is None or int(payload["feature_dim"]) != expected_dim:
            raise ValueError("relative advantage checkpoint 特征维度不匹配")
        agent = cls(
            feature_version=feature_version,
            hidden_size=int(payload["hidden_size"]),
            device=device,
        )
        agent.network.load_state_dict(payload["state_dict"], strict=True)
        agent.network.eval()
        return agent

"""GPU-capable policy-value model over engine-legal Mahjong actions.

This module is deliberately optional: the browser game and rule engine remain
dependency-free.  Import it only from the project virtual environment that
contains PyTorch when training or evaluating a ``.pt`` checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

try:
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError as error:  # pragma: no cover - depends on local ML env
    raise RuntimeError(
        "policy-value 模型需要 PyTorch；请使用项目 .venv/bin/python 运行"
    ) from error

from .agents import GameAction
from .game import XiamenMahjongGame
from .training import (
    NEURAL_FEATURE_DIMS,
    PUBLIC_ACTION_SEQUENCE_DIM,
    PUBLIC_ACTION_SEQUENCE_LENGTH,
    TeacherDecision,
    _decision,
    _dense_action_features,
    _turn_actions,
    public_action_sequence_features,
)


TORCH_POLICY_VALUE_VERSION = "xiamen-candidate-policy-value-v3"
_SUPPORTED_TORCH_POLICY_VALUE_VERSIONS = {
    "xiamen-candidate-policy-value-v1",
    "xiamen-candidate-policy-value-v2",
    TORCH_POLICY_VALUE_VERSION,
}
ARCHITECTURE_CANDIDATE_MLP = "candidate_mlp"
ARCHITECTURE_PUBLIC_SEQUENCE_TRANSFORMER = "public_sequence_transformer"
ARCHITECTURE_PUBLIC_SEQUENCE_RESIDUAL = "public_sequence_residual"
ACTION_SELECTION_POLICY = "policy"
ACTION_SELECTION_ACTION_VALUE = "action_value"
ACTION_SELECTION_RESPONSE_ACTION_VALUE = "response_action_value"
_SUPPORTED_ACTION_SELECTIONS = {
    ACTION_SELECTION_POLICY,
    ACTION_SELECTION_ACTION_VALUE,
    ACTION_SELECTION_RESPONSE_ACTION_VALUE,
}


class CandidatePolicyValueNetwork(nn.Module):
    """Encode every legal candidate, then pool them for a state value."""

    def __init__(self, feature_dim: int, hidden_size: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.candidate_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.policy_head = nn.Linear(hidden_size, 1)
        # This head estimates the expected terminal net score for *each* legal
        # action, in the same normalized units used by the trainer.  Starting
        # from zero is intentional: loading a v1/v2 checkpoint then retains
        # its established policy exactly until explicit Q supervision arrives.
        self.action_value_head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.action_value_head.weight)
        nn.init.zeros_(self.action_value_head.bias)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward_with_action_values(
        self, candidates: Tensor, action_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return policy logits, state value and per-action Q estimates.

        The Q estimates are deliberately not used by :meth:`forward`.  Older
        callers therefore retain their policy-only interface, while a trained
        experimental agent can opt into Q action selection explicitly.
        """

        encoded = self.candidate_encoder(candidates)
        logits = self.policy_head(encoded).squeeze(-1)
        logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        action_values = self.action_value_head(encoded).squeeze(-1)
        action_values = action_values.masked_fill(~action_mask, 0.0)
        denominator = action_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (encoded * action_mask.unsqueeze(-1)).sum(dim=1) / denominator
        value = self.value_head(pooled).squeeze(-1)
        return logits, value, action_values

    def forward(self, candidates: Tensor, action_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return masked policy logits and a value for each decision batch."""

        logits, value, _action_values = self.forward_with_action_values(
            candidates, action_mask
        )
        return logits, value


class PublicSequencePolicyValueNetwork(nn.Module):
    """Score legal actions using a Transformer over public event order.

    Candidate features still contain the current public aggregates.  The
    Transformer adds the information those aggregates lose: who acted, in what
    order, and which exposed public tiles were involved.  Its inputs never
    include a wall, another hand, or an ``an_kan`` face value.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int = 128,
        *,
        event_feature_dim: int = PUBLIC_ACTION_SEQUENCE_DIM,
        event_length: int = PUBLIC_ACTION_SEQUENCE_LENGTH,
        attention_heads: int = 4,
    ):
        super().__init__()
        if hidden_size % attention_heads:
            raise ValueError("hidden_size 必须能被 attention_heads 整除")
        if event_feature_dim <= 0 or event_length <= 0:
            raise ValueError("公开事件维度和长度必须为正数")
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.event_feature_dim = event_feature_dim
        self.event_length = event_length
        self.attention_heads = attention_heads
        self.candidate_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.event_encoder = nn.Sequential(
            nn.Linear(event_feature_dim, hidden_size),
            nn.GELU(),
        )
        self.position_embedding = nn.Parameter(
            torch.empty(event_length, hidden_size)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2, enable_nested_tensor=False
        )
        self.history_norm = nn.LayerNorm(hidden_size)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        candidates: Tensor,
        action_mask: Tensor,
        public_events: Tensor,
        event_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return legal-action logits and value from candidate/event batches."""

        if public_events.shape[-2:] != (self.event_length, self.event_feature_dim):
            raise ValueError("公开事件序列维度与网络配置不匹配")
        if event_mask.shape != public_events.shape[:2]:
            raise ValueError("公开事件掩码维度不匹配")
        candidate_encoded = self.candidate_encoder(candidates)
        event_encoded = self.event_encoder(public_events)
        event_encoded = event_encoded + self.position_embedding.unsqueeze(0)
        # A state with no prior public action receives one zero event token so
        # that Transformer attention never sees an all-padding sequence.
        safe_event_mask = event_mask.clone()
        empty_rows = ~safe_event_mask.any(dim=1)
        safe_event_mask[empty_rows, 0] = True
        history = self.history_encoder(
            event_encoded, src_key_padding_mask=~safe_event_mask
        )
        denominator = safe_event_mask.sum(dim=1, keepdim=True).clamp_min(1)
        history_context = self.history_norm(
            (history * safe_event_mask.unsqueeze(-1)).sum(dim=1) / denominator
        )
        action_count = action_mask.sum(dim=1, keepdim=True).clamp_min(1)
        candidate_context = (
            (candidate_encoded * action_mask.unsqueeze(-1)).sum(dim=1) / action_count
        )
        expanded_history = history_context.unsqueeze(1).expand(
            -1, candidate_encoded.shape[1], -1
        )
        logits = self.policy_head(
            torch.cat((candidate_encoded, expanded_history), dim=-1)
        ).squeeze(-1)
        logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        value = self.value_head(
            torch.cat((candidate_context, history_context), dim=-1)
        ).squeeze(-1)
        return logits, value


class ResidualPublicSequencePolicyValueNetwork(nn.Module):
    """Add ordered public-event corrections to a pretrained candidate policy.

    ``residual_scale`` starts at zero.  Once initialized from a candidate MLP,
    the first inference is therefore exactly the established policy; training
    can only introduce sequence effects when validation evidence supports it.
    This is safer than replacing a well-calibrated legal-action ranker with a
    randomly initialized Transformer on a still-small corpus.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_size: int = 128,
        *,
        event_feature_dim: int = PUBLIC_ACTION_SEQUENCE_DIM,
        event_length: int = PUBLIC_ACTION_SEQUENCE_LENGTH,
        attention_heads: int = 4,
    ):
        super().__init__()
        if hidden_size % attention_heads:
            raise ValueError("hidden_size 必须能被 attention_heads 整除")
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.event_feature_dim = event_feature_dim
        self.event_length = event_length
        self.attention_heads = attention_heads
        self.candidate_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.base_policy_head = nn.Linear(hidden_size, 1)
        self.base_value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.event_encoder = nn.Sequential(
            nn.Linear(event_feature_dim, hidden_size),
            nn.GELU(),
        )
        self.position_embedding = nn.Parameter(torch.empty(event_length, hidden_size))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2, enable_nested_tensor=False
        )
        self.history_norm = nn.LayerNorm(hidden_size)
        self.candidate_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.history_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.history_bias = nn.Linear(hidden_size, 1)
        self.value_delta = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def initialize_from_candidate(self, source: CandidatePolicyValueNetwork) -> None:
        """Copy the proven candidate-only path before residual fine-tuning."""

        if source.feature_dim != self.feature_dim or source.hidden_size != self.hidden_size:
            raise ValueError("候选 MLP 的特征维度或 hidden-size 不匹配")
        self.candidate_encoder.load_state_dict(source.candidate_encoder.state_dict())
        self.base_policy_head.load_state_dict(source.policy_head.state_dict())
        self.base_value_head.load_state_dict(source.value_head.state_dict())

    def forward(
        self,
        candidates: Tensor,
        action_mask: Tensor,
        public_events: Tensor,
        event_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if public_events.shape[-2:] != (self.event_length, self.event_feature_dim):
            raise ValueError("公开事件序列维度与网络配置不匹配")
        if event_mask.shape != public_events.shape[:2]:
            raise ValueError("公开事件掩码维度不匹配")
        candidate_encoded = self.candidate_encoder(candidates)
        event_encoded = self.event_encoder(public_events) + self.position_embedding.unsqueeze(0)
        safe_event_mask = event_mask.clone()
        empty_rows = ~safe_event_mask.any(dim=1)
        safe_event_mask[empty_rows, 0] = True
        history = self.history_encoder(
            event_encoded, src_key_padding_mask=~safe_event_mask
        )
        history_context = self.history_norm(
            (history * safe_event_mask.unsqueeze(-1)).sum(dim=1)
            / safe_event_mask.sum(dim=1, keepdim=True).clamp_min(1)
        )
        base_logits = self.base_policy_head(candidate_encoded).squeeze(-1)
        interaction = (
            self.candidate_query(candidate_encoded)
            * self.history_key(history_context).unsqueeze(1)
        ).sum(dim=-1) / (self.hidden_size**0.5)
        delta_logits = interaction + self.history_bias(history_context).squeeze(-1).unsqueeze(1)
        logits = base_logits + self.residual_scale * delta_logits
        logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        action_count = action_mask.sum(dim=1, keepdim=True).clamp_min(1)
        candidate_context = (
            (candidate_encoded * action_mask.unsqueeze(-1)).sum(dim=1) / action_count
        )
        base_value = self.base_value_head(candidate_context).squeeze(-1)
        value_delta = self.value_delta(
            torch.cat((candidate_context, history_context), dim=-1)
        ).squeeze(-1)
        value = base_value + self.residual_scale * value_delta
        return logits, value


class TorchPolicyValueAgent:
    """Inference wrapper that preserves the existing Teacher agent interface."""

    def __init__(
        self,
        *,
        feature_version: int = 3,
        hidden_size: int = 128,
        architecture: str = ARCHITECTURE_CANDIDATE_MLP,
        attention_heads: int = 4,
        action_selection: str = ACTION_SELECTION_POLICY,
        device: str | None = None,
        network: (
            CandidatePolicyValueNetwork
            | PublicSequencePolicyValueNetwork
            | ResidualPublicSequencePolicyValueNetwork
            | None
        ) = None,
    ):
        if feature_version not in NEURAL_FEATURE_DIMS:
            raise ValueError("不支持的 policy-value 特征版本")
        self.feature_version = feature_version
        self.feature_dim = NEURAL_FEATURE_DIMS[feature_version]
        self.hidden_size = hidden_size
        if architecture not in {
            ARCHITECTURE_CANDIDATE_MLP,
            ARCHITECTURE_PUBLIC_SEQUENCE_TRANSFORMER,
            ARCHITECTURE_PUBLIC_SEQUENCE_RESIDUAL,
        }:
            raise ValueError("不支持的 policy-value 网络结构")
        self.architecture = architecture
        self.attention_heads = attention_heads
        if action_selection not in _SUPPORTED_ACTION_SELECTIONS:
            raise ValueError("不支持的动作选择模式")
        self.action_selection = action_selection
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if network is not None:
            self.network = network
            if isinstance(network, PublicSequencePolicyValueNetwork):
                self.architecture = ARCHITECTURE_PUBLIC_SEQUENCE_TRANSFORMER
                self.attention_heads = network.attention_heads
            elif isinstance(network, ResidualPublicSequencePolicyValueNetwork):
                self.architecture = ARCHITECTURE_PUBLIC_SEQUENCE_RESIDUAL
                self.attention_heads = network.attention_heads
            else:
                self.architecture = ARCHITECTURE_CANDIDATE_MLP
        elif architecture == ARCHITECTURE_CANDIDATE_MLP:
            self.network = CandidatePolicyValueNetwork(self.feature_dim, hidden_size)
        elif architecture == ARCHITECTURE_PUBLIC_SEQUENCE_TRANSFORMER:
            self.network = PublicSequencePolicyValueNetwork(
                self.feature_dim,
                hidden_size,
                attention_heads=attention_heads,
            )
        else:
            self.network = ResidualPublicSequencePolicyValueNetwork(
                self.feature_dim,
                hidden_size,
                attention_heads=attention_heads,
            )
        self.network.to(self.device)
        self.network.eval()

    def _public_events(self, decision: TeacherDecision) -> tuple[Tensor, Tensor]:
        sequence = public_action_sequence_features(decision.state)
        events = torch.zeros(
            (1, PUBLIC_ACTION_SEQUENCE_LENGTH, PUBLIC_ACTION_SEQUENCE_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        event_mask = torch.zeros(
            (1, PUBLIC_ACTION_SEQUENCE_LENGTH), dtype=torch.bool, device=self.device
        )
        if sequence:
            events[0, : len(sequence)] = torch.tensor(
                sequence, dtype=torch.float32, device=self.device
            )
            event_mask[0, : len(sequence)] = True
        return events, event_mask

    def _public_events_batch(
        self, decisions: Sequence[TeacherDecision]
    ) -> tuple[Tensor, Tensor]:
        events = torch.zeros(
            (len(decisions), PUBLIC_ACTION_SEQUENCE_LENGTH, PUBLIC_ACTION_SEQUENCE_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        event_mask = torch.zeros(
            (len(decisions), PUBLIC_ACTION_SEQUENCE_LENGTH),
            dtype=torch.bool,
            device=self.device,
        )
        for row, decision in enumerate(decisions):
            sequence = public_action_sequence_features(decision.state)
            if sequence:
                events[row, : len(sequence)] = torch.tensor(
                    sequence, dtype=torch.float32, device=self.device
                )
                event_mask[row, : len(sequence)] = True
        return events, event_mask

    def _forward(
        self, candidates: Tensor, action_mask: Tensor, decision: TeacherDecision
    ) -> tuple[Tensor, Tensor]:
        if self.architecture in {
            ARCHITECTURE_PUBLIC_SEQUENCE_TRANSFORMER,
            ARCHITECTURE_PUBLIC_SEQUENCE_RESIDUAL,
        }:
            events, event_mask = self._public_events(decision)
            assert isinstance(
                self.network,
                (PublicSequencePolicyValueNetwork, ResidualPublicSequencePolicyValueNetwork),
            )
            return self.network(candidates, action_mask, events, event_mask)
        return self.network(candidates, action_mask)

    def _forward_with_action_values(
        self, candidates: Tensor, action_mask: Tensor, decision: TeacherDecision
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Return the optional Q head without changing legacy architectures."""

        if isinstance(self.network, CandidatePolicyValueNetwork):
            return self.network.forward_with_action_values(candidates, action_mask)
        logits, value = self._forward(candidates, action_mask, decision)
        return logits, value, None

    def _scores_for_selection(
        self, decision: TeacherDecision, *, action_selection: str
    ) -> list[float]:
        if action_selection == ACTION_SELECTION_POLICY:
            scores, _value = self.policy_value(decision)
            return scores
        action_values = self.action_value_scores(decision)
        if action_values is None:
            raise ValueError("当前网络结构没有动作价值头，不能使用动作价值选牌")
        return action_values

    def scores(self, decision: TeacherDecision) -> list[float]:
        """Return the general selector score without inferring game phase.

        ``response_action_value`` is intentionally policy here: callers that
        know they are resolving a response must use :meth:`choose_response`.
        This prevents a response-only Q head from silently ranking discard
        actions in generic batched/analysis code.
        """

        selection = (
            ACTION_SELECTION_POLICY
            if self.action_selection == ACTION_SELECTION_RESPONSE_ACTION_VALUE
            else self.action_selection
        )
        return self._scores_for_selection(decision, action_selection=selection)

    def policy_value(self, decision: TeacherDecision) -> tuple[list[float], float]:
        """Run one legal candidate set once and return policy logits plus value."""

        vectors = [
            _dense_action_features(
                decision.state, action, feature_version=self.feature_version
            )
            for action in decision.legal_actions
        ]
        if not vectors:
            raise ValueError("策略没有可用的合法动作")
        candidates = torch.tensor([vectors], dtype=torch.float32, device=self.device)
        mask = torch.ones((1, len(vectors)), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits, value = self._forward(candidates, mask, decision)
        return (
            [float(item) for item in logits[0].detach().cpu()],
            float(value[0].detach().cpu()),
        )

    def action_value_scores(self, decision: TeacherDecision) -> list[float] | None:
        """Return direct per-action Q estimates, if this architecture has them.

        Values use the normalized terminal-score unit configured by the
        trainer.  They are exposed separately rather than silently mixed with
        policy logits, so paired evaluation can test each decision rule.
        """

        vectors = [
            _dense_action_features(
                decision.state, action, feature_version=self.feature_version
            )
            for action in decision.legal_actions
        ]
        if not vectors:
            raise ValueError("策略没有可用的合法动作")
        candidates = torch.tensor([vectors], dtype=torch.float32, device=self.device)
        mask = torch.ones((1, len(vectors)), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            _logits, _value, action_values = self._forward_with_action_values(
                candidates, mask, decision
            )
        if action_values is None:
            return None
        return [float(item) for item in action_values[0].detach().cpu()]

    def policy_values_batch(
        self, decisions: Sequence[TeacherDecision]
    ) -> list[tuple[list[float], float]]:
        """Evaluate heterogeneous legal candidate sets in one network pass.

        Game transitions remain sequential and rule-validated; batching only
        combines independent model inferences from concurrently active hands.
        """

        if not decisions:
            return []
        counts = [len(decision.legal_actions) for decision in decisions]
        if any(count <= 0 for count in counts):
            raise ValueError("策略没有可用的合法动作")
        max_actions = max(counts)
        candidates = torch.zeros(
            (len(decisions), max_actions, self.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        action_mask = torch.zeros(
            (len(decisions), max_actions), dtype=torch.bool, device=self.device
        )
        for row, decision in enumerate(decisions):
            vectors = [
                _dense_action_features(
                    decision.state, action, feature_version=self.feature_version
                )
                for action in decision.legal_actions
            ]
            candidates[row, : counts[row]] = torch.tensor(
                vectors, dtype=torch.float32, device=self.device
            )
            action_mask[row, : counts[row]] = True
        with torch.no_grad():
            if self.architecture in {
                ARCHITECTURE_PUBLIC_SEQUENCE_TRANSFORMER,
                ARCHITECTURE_PUBLIC_SEQUENCE_RESIDUAL,
            }:
                events, event_mask = self._public_events_batch(decisions)
                assert isinstance(
                    self.network,
                    (
                        PublicSequencePolicyValueNetwork,
                        ResidualPublicSequencePolicyValueNetwork,
                    ),
                )
                logits, values = self.network(
                    candidates, action_mask, events, event_mask
                )
            else:
                logits, values = self.network(candidates, action_mask)
        logits_cpu = logits.detach().cpu()
        values_cpu = values.detach().cpu()
        return [
            (
                [float(value) for value in logits_cpu[row, :count]],
                float(values_cpu[row]),
            )
            for row, count in enumerate(counts)
        ]

    def action_value_scores_batch(
        self, decisions: Sequence[TeacherDecision]
    ) -> list[list[float]] | None:
        """Return direct Q scores for heterogeneous legal sets in one pass.

        This is the Q-selection counterpart to :meth:`policy_values_batch`.
        It deliberately remains unavailable for the historical sequence
        architectures, which do not have an action-value head.
        """

        if not decisions:
            return []
        if not isinstance(self.network, CandidatePolicyValueNetwork):
            return None
        counts = [len(decision.legal_actions) for decision in decisions]
        if any(count <= 0 for count in counts):
            raise ValueError("策略没有可用的合法动作")
        max_actions = max(counts)
        candidates = torch.zeros(
            (len(decisions), max_actions, self.feature_dim),
            dtype=torch.float32,
            device=self.device,
        )
        action_mask = torch.zeros(
            (len(decisions), max_actions),
            dtype=torch.bool,
            device=self.device,
        )
        for row, decision in enumerate(decisions):
            vectors = [
                _dense_action_features(
                    decision.state, action, feature_version=self.feature_version
                )
                for action in decision.legal_actions
            ]
            candidates[row, : counts[row]] = torch.tensor(
                vectors, dtype=torch.float32, device=self.device
            )
            action_mask[row, : counts[row]] = True
        with torch.no_grad():
            _logits, _values, action_values = (
                self.network.forward_with_action_values(candidates, action_mask)
            )
        action_values_cpu = action_values.detach().cpu()
        return [
            [float(value) for value in action_values_cpu[row, :count]]
            for row, count in enumerate(counts)
        ]

    def scores_batch(self, decisions: Sequence[TeacherDecision]) -> list[list[float]]:
        """Batch the active deployment action-selection rule.

        Collectors use this optional interface to batch independent games
        without accidentally changing an agent configured for Q selection
        back into a policy-logit agent.
        """

        if self.action_selection in {
            ACTION_SELECTION_POLICY,
            ACTION_SELECTION_RESPONSE_ACTION_VALUE,
        }:
            return [scores for scores, _value in self.policy_values_batch(decisions)]
        action_values = self.action_value_scores_batch(decisions)
        if action_values is None:
            raise ValueError("当前网络结构没有动作价值头，不能使用动作价值选牌")
        return action_values

    def value(self, decision: TeacherDecision) -> float:
        _scores, value = self.policy_value(decision)
        return value

    def predict_index(self, decision: TeacherDecision) -> int:
        scores = self.scores(decision)
        return max(range(len(scores)), key=lambda index: (scores[index], -index))

    def predict_action(self, decision: TeacherDecision) -> GameAction:
        return decision.legal_actions[self.predict_index(decision)]

    def choose_turn_action(self, game: XiamenMahjongGame, player_id: int) -> GameAction:
        legal = tuple(_turn_actions(game, player_id))
        decision = _decision(game, game.seed or 0, player_id, legal, legal[0])
        # Q labels currently cover only public response states. Never apply a
        # response-only checkpoint to turn/discard choices without turn Q data.
        if self.action_selection == ACTION_SELECTION_RESPONSE_ACTION_VALUE:
            scores, _value = self.policy_value(decision)
            return decision.legal_actions[
                max(range(len(scores)), key=lambda index: (scores[index], -index))
            ]
        return self.predict_action(decision)

    def choose_response(
        self, game: XiamenMahjongGame, player_id: int, options: Sequence[GameAction]
    ) -> GameAction:
        legal = tuple(options)
        decision = _decision(game, game.seed or 0, player_id, legal, legal[0])
        if self.action_selection == ACTION_SELECTION_RESPONSE_ACTION_VALUE:
            scores = self._scores_for_selection(
                decision, action_selection=ACTION_SELECTION_ACTION_VALUE
            )
            return decision.legal_actions[
                max(range(len(scores)), key=lambda index: (scores[index], -index))
            ]
        return self.predict_action(decision)

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": TORCH_POLICY_VALUE_VERSION,
                "model": "candidate_policy_value",
                "architecture": self.architecture,
                "feature_version": self.feature_version,
                "feature_dim": self.feature_dim,
                "hidden_size": self.hidden_size,
                "attention_heads": self.attention_heads,
                "action_selection": self.action_selection,
                "state_dict": self.network.state_dict(),
                "metadata": metadata or {},
            },
            destination,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
        action_selection: str | None = None,
    ) -> "TorchPolicyValueAgent":
        """Load a checkpoint, optionally overriding its experimental selector."""

        payload = torch.load(Path(path), map_location=device or "cpu", weights_only=True)
        if payload.get("version") not in _SUPPORTED_TORCH_POLICY_VALUE_VERSIONS:
            raise ValueError("不支持的 policy-value 检查点版本")
        feature_version = int(payload["feature_version"])
        expected_dim = NEURAL_FEATURE_DIMS.get(feature_version)
        if expected_dim is None or int(payload["feature_dim"]) != expected_dim:
            raise ValueError("policy-value 特征版本或维度不匹配")
        architecture = str(payload.get("architecture", ARCHITECTURE_CANDIDATE_MLP))
        agent = cls(
            feature_version=feature_version,
            hidden_size=int(payload["hidden_size"]),
            architecture=architecture,
            attention_heads=int(payload.get("attention_heads", 4)),
            action_selection=action_selection
            or str(payload.get("action_selection", ACTION_SELECTION_POLICY)),
            device=device,
        )
        state_dict = payload["state_dict"]
        if (
            architecture == ARCHITECTURE_CANDIDATE_MLP
            and payload.get("version")
            in {"xiamen-candidate-policy-value-v1", "xiamen-candidate-policy-value-v2"}
        ):
            missing, unexpected = agent.network.load_state_dict(state_dict, strict=False)
            allowed_missing = {
                "action_value_head.weight",
                "action_value_head.bias",
            }
            if set(missing) != allowed_missing or unexpected:
                raise ValueError("旧 policy-value checkpoint 的参数不完整或不匹配")
        else:
            agent.network.load_state_dict(state_dict)
        agent.network.to(agent.device)
        agent.network.eval()
        return agent

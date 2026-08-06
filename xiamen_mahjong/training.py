"""Rule-Teacher imitation-learning baseline for Xiamen Mahjong.

The browser game remains the authority for legality.  This module only learns
how to rank the actions that the engine already says are legal, so a policy
checkpoint cannot invent an illegal discard, claim, or kong.
"""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from .agents import GameAction, HeuristicTeacherAgent
from .belief import smoothed_deterministic_likelihood, smoothed_policy_likelihood
from .game import XiamenMahjongGame
from .hand import hand_quality, wait_tiles
from .rules import XiamenRules
from .tiles import BASE_TILE_COUNT, WHITE_DRAGON, base_wall, is_base_tile


DATASET_VERSION = "xiamen-rule-teacher-v1"
TRAJECTORY_DATASET_VERSION = "xiamen-training-trajectory-v3"
_SUPPORTED_TRAJECTORY_DATASET_VERSIONS = {
    "xiamen-training-trajectory-v2",
    TRAJECTORY_DATASET_VERSION,
}
ACTION_KINDS = (
    "discard",
    "hu",
    "an_kan",
    "add_kan",
    "advance_tour",
    "pass",
    "pong",
    "ming_kan",
    "chi",
)
ACTION_KIND_INDEX = {kind: index for index, kind in enumerate(ACTION_KINDS)}
NO_TILE = BASE_TILE_COUNT
TILE_SLOTS = BASE_TILE_COUNT + 1

# A target-tile row sees the actor's hand, public river, and exposed melds.
# This is deliberately a small linear policy: it trains quickly in pure Python
# and is a reproducible baseline before adding a neural network dependency.
_BIAS = 0
_KIND = 1
_TARGET = _KIND + len(ACTION_KINDS)
_KIND_TARGET = _TARGET + TILE_SLOTS
_TARGET_HAND = _KIND_TARGET + len(ACTION_KINDS) * TILE_SLOTS
_TARGET_RIVER = _TARGET_HAND + TILE_SLOTS * BASE_TILE_COUNT
_TARGET_MELD = _TARGET_RIVER + TILE_SLOTS * BASE_TILE_COUNT
_CONSUMED = _TARGET_MELD + TILE_SLOTS * BASE_TILE_COUNT
_KIND_TOUR = _CONSUMED + BASE_TILE_COUNT
_KIND_WALL = _KIND_TOUR + len(ACTION_KINDS) * 4
_KIND_PHASE = _KIND_WALL + len(ACTION_KINDS) * 8
FEATURE_DIM = _KIND_PHASE + len(ACTION_KINDS) * 2
NEURAL_FEATURE_DIMS = {1: 76, 2: 80, 3: 145}
DEFAULT_NEURAL_FEATURE_VERSION = 2
NEURAL_FEATURE_DIM = NEURAL_FEATURE_DIMS[DEFAULT_NEURAL_FEATURE_VERSION]
PUBLIC_ACTION_KINDS = (
    "draw",
    "discard",
    "chi",
    "pong",
    "ming_kan",
    "an_kan",
    "add_kan",
    "advance_tour",
    "hu",
    "result",
)
PUBLIC_ACTION_SEQUENCE_LENGTH = 24
# kind (including unknown), actor relative seat (including no actor), primary
# public tile (including no tile), public consumed/open tiles, and normalized
# relative position in the history window.
PUBLIC_ACTION_SEQUENCE_DIM = (
    len(PUBLIC_ACTION_KINDS) + 1 + 5 + TILE_SLOTS + BASE_TILE_COUNT + 1
)


@dataclass(frozen=True)
class TeacherDecision:
    """One perspective-correct rule-Teacher decision and its legal choices."""

    profile: str
    # ``None`` is used when a decision came from a safe trajectory export.
    # It is replay metadata, never an inference feature.
    seed: int | None
    seat: int
    state: dict[str, Any]
    legal_actions: tuple[GameAction, ...]
    chosen_index: int
    # Optional counterfactual rollout returns, one for every legal action.
    # They are intentionally *targets*, never model features.  A record may
    # have been evaluated with the simulator's hidden state, but the exported
    # observation remains strictly actor-visible (see ``_perspective_state``).
    # The trainer converts this vector into a soft action-preference target.
    action_values: tuple[float, ...] | None = None
    # Standard errors of those values across independent frozen-opponent
    # replicates.  ``None`` means the collector only ran one replicate, not
    # that the action value is known without uncertainty.
    action_value_stderrs: tuple[float, ...] | None = None

    @property
    def chosen_action(self) -> GameAction:
        return self.legal_actions[self.chosen_index]

    def payload(self) -> dict[str, Any]:
        payload = {
            "version": DATASET_VERSION,
            "profile": self.profile,
            "seat": self.seat,
            "state": self.state,
            "legal_actions": [_action_payload(action) for action in self.legal_actions],
            "chosen_index": self.chosen_index,
        }
        if self.action_values is not None:
            payload["action_values"] = list(self.action_values)
        if self.action_value_stderrs is not None:
            payload["action_value_stderrs"] = list(self.action_value_stderrs)
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TeacherDecision":
        if payload.get("version") != DATASET_VERSION:
            raise ValueError("不支持的 Teacher 轨迹版本")
        actions = tuple(_action_from_payload(item) for item in payload["legal_actions"])
        chosen_index = int(payload["chosen_index"])
        if not 0 <= chosen_index < len(actions):
            raise ValueError("Teacher 轨迹的目标动作索引无效")
        raw_action_values = payload.get("action_values")
        action_values = None
        if raw_action_values is not None:
            if (
                not isinstance(raw_action_values, list)
                or len(raw_action_values) != len(actions)
                or not all(
                    isinstance(value, (int, float)) and math.isfinite(float(value))
                    for value in raw_action_values
                )
            ):
                raise ValueError("动作价值目标必须与合法动作逐项对应")
            action_values = tuple(float(value) for value in raw_action_values)
        raw_action_value_stderrs = payload.get("action_value_stderrs")
        action_value_stderrs = None
        if raw_action_value_stderrs is not None:
            if (
                action_values is None
                or not isinstance(raw_action_value_stderrs, list)
                or len(raw_action_value_stderrs) != len(actions)
                or not all(
                    isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    and float(value) >= 0
                    for value in raw_action_value_stderrs
                )
            ):
                raise ValueError("动作价值标准误必须与动作价值逐项对应且非负")
            action_value_stderrs = tuple(
                float(value) for value in raw_action_value_stderrs
            )
        return cls(
            profile=str(payload["profile"]),
            seed=int(payload["seed"]) if payload.get("seed") is not None else None,
            seat=int(payload["seat"]),
            state=dict(payload["state"]),
            legal_actions=actions,
            chosen_index=chosen_index,
            action_values=action_values,
            action_value_stderrs=action_value_stderrs,
        )


@dataclass(frozen=True)
class DatasetSummary:
    hands: int
    decisions: int
    wins: int
    draws: int
    action_counts: dict[str, int]

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActionValueDatasetSummary:
    """Coverage report for public-state counterfactual action-value data."""

    hands: int
    decisions: int
    branch_rollouts: int
    belief_resampled_worlds: int
    belief_resample_skipped: int
    response_decisions: int
    repeated_decisions: int
    action_counts: dict[str, int]
    mean_action_value_span: float
    mean_action_value_stderr: float | None
    rollout_batch_size: int
    batched_inference_calls: int
    batched_inference_decisions: int
    max_batched_inference_decisions: int

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _CounterfactualBatchStats:
    """Non-private scheduling diagnostics for a collector invocation."""

    inference_calls: int = 0
    inference_decisions: int = 0
    max_inference_decisions: int = 0


@dataclass(frozen=True)
class PolicyStep:
    """One sampled, engine-legal action in a policy-gradient episode."""

    decision: TeacherDecision
    action_index: int


@dataclass(frozen=True)
class PolicyEpisode:
    """One candidate-versus-Teacher hand used for on-policy optimization."""

    seed: int
    candidate_seat: int
    reward: int
    winner: int | None
    win_type: str | None
    steps: tuple[PolicyStep, ...]


@dataclass(frozen=True)
class TrainingTrajectory:
    """One complete training hand with a safe/public export boundary.

    A decision snapshot contains the acting player's hand plus public state;
    the terminal outcome is stored separately for policy-value targets.  The
    physical wall and opponents' concealed hands are intentionally absent.
    ``seed`` is retained only while collecting/replaying locally.  The default
    JSONL export omits it (and all source seed metadata), because a seed could
    reconstruct the hidden wall even though the policy never consumes it.
    """

    profile: str
    rules_version: str
    rules: dict[str, Any]
    seed: int | None
    hand_number: int
    agent_profiles: tuple[str, ...]
    source_metadata: dict[str, Any]
    decisions: tuple[TeacherDecision, ...]
    outcome: dict[str, Any]
    public_actions: tuple[dict[str, Any], ...]
    trajectory_id: str = field(default_factory=lambda: uuid4().hex)
    # An opaque ID may tie seat rotations of one physical wall together for a
    # split without putting the replay seed in the training corpus.
    split_group_id: str | None = None

    def payload(self, *, include_replay_metadata: bool = False) -> dict[str, Any]:
        """Return a trainable trajectory without hidden-state replay data.

        Set ``include_replay_metadata`` only when writing a locally protected
        replay archive.  Normal training does not require any seed.
        """

        decisions = []
        for decision in self.decisions:
            decision_payload = decision.payload()
            if not include_replay_metadata:
                decision_payload.pop("seed", None)
            decisions.append(decision_payload)
        payload = {
            "version": TRAJECTORY_DATASET_VERSION,
            "trajectory_id": self.trajectory_id,
            "split_group_id": self.split_group_id or self.trajectory_id,
            "profile": self.profile,
            "rules_version": self.rules_version,
            "rules": self.rules,
            "hand_number": self.hand_number,
            "agent_profiles": list(self.agent_profiles),
            "source_metadata": (
                self.source_metadata
                if include_replay_metadata
                else _public_source_metadata(self.source_metadata)
            ),
            "decisions": decisions,
            "outcome": self.outcome,
            "public_actions": list(self.public_actions),
        }
        if include_replay_metadata and self.seed is not None:
            payload["seed"] = self.seed
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TrainingTrajectory":
        version = payload.get("version")
        if version not in _SUPPORTED_TRAJECTORY_DATASET_VERSIONS:
            raise ValueError("不支持的训练轨迹版本")
        decisions = tuple(
            TeacherDecision.from_payload(dict(item)) for item in payload["decisions"]
        )
        profile = str(payload["profile"])
        seed = int(payload["seed"]) if payload.get("seed") is not None else None
        if any(
            decision.profile != profile
            or (seed is not None and decision.seed is not None and decision.seed != seed)
            for decision in decisions
        ):
            raise ValueError("训练轨迹的决策与牌局元数据不一致")
        outcome = dict(payload["outcome"])
        scores = outcome.get("scores")
        if not isinstance(scores, list) or len(scores) != 4:
            raise ValueError("训练轨迹缺少四家终局得分")
        return cls(
            profile=profile,
            rules_version=str(payload["rules_version"]),
            rules=dict(payload["rules"]),
            seed=seed,
            hand_number=int(payload["hand_number"]),
            agent_profiles=tuple(str(value) for value in payload["agent_profiles"]),
            source_metadata=dict(payload.get("source_metadata", {})),
            decisions=decisions,
            outcome=outcome,
            public_actions=tuple(dict(item) for item in payload["public_actions"]),
            trajectory_id=str(
                payload.get("trajectory_id")
                or _legacy_trajectory_id(
                    profile, str(payload["rules_version"]), seed, int(payload["hand_number"])
                )
            ),
            split_group_id=(
                str(payload["split_group_id"])
                if payload.get("split_group_id") is not None
                else None
            ),
        )


def _public_source_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Remove replay-only random-state fields from a distributable corpus."""

    def redact(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): redact(item)
                for key, item in value.items()
                if "seed" not in str(key).lower()
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, tuple):
            return [redact(item) for item in value]
        return value

    return redact(metadata)


def _legacy_trajectory_id(
    profile: str, rules_version: str, seed: int | None, hand_number: int
) -> str:
    """Stable identity for old v2 records that have no opaque trajectory ID."""

    digest = hashlib.blake2b(
        f"legacy|{profile}|{rules_version}|{seed}|{hand_number}".encode("utf-8"),
        digest_size=16,
    ).hexdigest()
    return f"legacy-{digest}"


def _trajectory_from_game(
    game: XiamenMahjongGame,
    decisions: Sequence[TeacherDecision],
    *,
    agent_profiles: Sequence[str],
    source_metadata: Mapping[str, Any] | None = None,
) -> TrainingTrajectory:
    if game.phase != "over":
        raise RuntimeError("只能在牌局结算后导出训练轨迹")
    if len(agent_profiles) != game.rules.player_count:
        raise ValueError("agent_profiles 必须覆盖每个座位")
    return TrainingTrajectory(
        profile=game.rules.profile,
        rules_version=game.rules.version,
        rules=asdict(game.rules),
        seed=int(game.seed or 0),
        hand_number=game.hand_number,
        agent_profiles=tuple(agent_profiles),
        source_metadata=dict(source_metadata or {}),
        decisions=tuple(decisions),
        outcome={
            "winner": game.winner,
            "win_type": game.win_type,
            "win_pattern": game.win_pattern,
            "scores": [player.score for player in game.players],
            "score_breakdown": game.score_breakdown,
            "turn_count": game.turn_count,
        },
        public_actions=tuple(dict(action) for action in game.public_actions),
    )


def collect_teacher_decisions(
    *,
    hands: int,
    profile: str = "classic",
    seed: int = 20260804,
) -> tuple[list[TeacherDecision], DatasetSummary]:
    """Return flattened decisions from the versioned Teacher trajectories."""

    trajectories, summary = collect_teacher_trajectories(
        hands=hands, profile=profile, seed=seed
    )
    return [
        decision for trajectory in trajectories for decision in trajectory.decisions
    ], summary


def collect_teacher_trajectories(
    *,
    hands: int,
    profile: str = "classic",
    seed: int = 20260804,
) -> tuple[list[TrainingTrajectory], DatasetSummary]:
    """Play deterministic Teacher hands and export complete safe trajectories.

    Every decision is observed from one player's private hand plus public
    facts.  The terminal score is retained only as a learning target after
    the hand ends; wall order and opponents' concealed hands are never stored.
    """

    if hands <= 0:
        raise ValueError("hands 必须为正数")
    rules = XiamenRules.from_profile(profile)
    trajectories: list[TrainingTrajectory] = []
    action_counts: Counter[str] = Counter()
    wins = 0
    draws = 0
    for hand_index in range(hands):
        hand_seed = seed + hand_index
        game = XiamenMahjongGame(seed=hand_seed, rules=rules, auto_advance=False)
        hand_decisions: list[TeacherDecision] = []
        steps = 0
        while game.phase != "over":
            steps += 1
            if steps > 600:
                raise RuntimeError("Teacher 自博弈超过安全步数")
            if game.phase == "discard":
                player_id = game.current_player
                legal = tuple(_turn_actions(game, player_id))
                chosen = game.teacher.choose_turn_action(game, player_id)
                decision = _decision(game, hand_seed, player_id, legal, chosen)
                hand_decisions.append(decision)
                action_counts[chosen.kind] += 1
                game._apply_turn_action(player_id, chosen)
                continue
            if game.phase == "response":
                for player_id, options in sorted(game.response_options.items()):
                    legal = tuple(options)
                    chosen = game.teacher.choose_response(game, player_id, list(legal))
                    decision = _decision(game, hand_seed, player_id, legal, chosen)
                    hand_decisions.append(decision)
                    action_counts[chosen.kind] += 1
                    game.response_choices[player_id] = chosen
                game._resolve_responses()
                continue
            raise RuntimeError(f"未知训练阶段：{game.phase}")
        trajectories.append(
            _trajectory_from_game(
                game,
                hand_decisions,
                agent_profiles=("heuristic_teacher",) * rules.player_count,
                source_metadata={
                    "collector": "teacher_self_play",
                    "behavior_seed": hand_seed,
                },
            )
        )
        if game.win_type == "draw":
            draws += 1
        else:
            wins += 1
    return trajectories, DatasetSummary(
        hands=hands,
        decisions=sum(len(trajectory.decisions) for trajectory in trajectories),
        wins=wins,
        draws=draws,
        action_counts=dict(sorted(action_counts.items())),
    )


def collect_exploration_trajectories(
    *,
    hands: int,
    profile: str = "classic",
    seed: int = 20262804,
    behavior_seed: int | None = None,
) -> tuple[list[TrainingTrajectory], DatasetSummary]:
    """Label legally reachable random-policy states with the frozen Teacher.

    This is an offline exploration source, not a stronger opponent.  Random
    legal actions diversify claims, declined calls and damaged hands that do
    not occur in deterministic Teacher self-play; labels remain the same
    actor-visible Teacher decision used by behavioral cloning.
    """

    if hands <= 0:
        raise ValueError("hands 必须为正数")
    rules = XiamenRules.from_profile(profile)
    behavior_base = behavior_seed if behavior_seed is not None else seed + 50_000_000
    trajectories: list[TrainingTrajectory] = []
    action_counts: Counter[str] = Counter()
    wins = 0
    draws = 0
    for hand_index in range(hands):
        hand_seed = seed + hand_index
        hand_behavior_seed = behavior_base + hand_index
        rng = random.Random(hand_behavior_seed)
        game = XiamenMahjongGame(seed=hand_seed, rules=rules, auto_advance=False)
        hand_decisions: list[TeacherDecision] = []
        steps = 0
        while game.phase != "over":
            steps += 1
            if steps > 600:
                raise RuntimeError("探索自博弈超过安全步数")
            if game.phase == "discard":
                player_id = game.current_player
                legal = tuple(_turn_actions(game, player_id))
                teacher_action = game.teacher.choose_turn_action(game, player_id)
                hand_decisions.append(
                    _decision(game, hand_seed, player_id, legal, teacher_action)
                )
                action_counts[teacher_action.kind] += 1
                game._apply_turn_action(player_id, legal[rng.randrange(len(legal))])
                continue
            if game.phase == "response":
                for player_id, options in sorted(game.response_options.items()):
                    legal = tuple(options)
                    teacher_action = game.teacher.choose_response(game, player_id, list(legal))
                    hand_decisions.append(
                        _decision(game, hand_seed, player_id, legal, teacher_action)
                    )
                    action_counts[teacher_action.kind] += 1
                    game.response_choices[player_id] = legal[rng.randrange(len(legal))]
                game._resolve_responses()
                continue
            raise RuntimeError(f"未知探索训练阶段：{game.phase}")
        trajectories.append(
            _trajectory_from_game(
                game,
                hand_decisions,
                agent_profiles=("random_legal",) * rules.player_count,
                source_metadata={
                    "collector": "random_legal_teacher_labeled",
                    "behavior_seed": hand_behavior_seed,
                },
            )
        )
        if game.win_type == "draw":
            draws += 1
        else:
            wins += 1
    return trajectories, DatasetSummary(
        hands=hands,
        decisions=sum(len(trajectory.decisions) for trajectory in trajectories),
        wins=wins,
        draws=draws,
        action_counts=dict(sorted(action_counts.items())),
    )


def collect_response_pass_curriculum(
    *,
    examples: int,
    profile: str = "classic",
    seed: int = 20263804,
) -> list[TrainingTrajectory]:
    """Find physical response states where the Teacher elects to pass.

    Deterministic Teacher games almost always claim useful melds, leaving the
    response ``pass`` class absent from the corpus.  This generator searches
    actual shuffled hands, moves one physically held tile to another seat's
    public river, and retains only states in which the unmodified Teacher
    chooses ``pass`` among real legal claim options.  No label is handwritten.
    """

    if examples <= 0:
        raise ValueError("examples 必须为正数")
    rules = XiamenRules.from_profile(profile)
    trajectories: list[TrainingTrajectory] = []
    attempts = 0
    max_attempts = examples * 2_000
    while len(trajectories) < examples and attempts < max_attempts:
        hand_seed = seed + attempts
        attempts += 1
        game = XiamenMahjongGame(seed=hand_seed, rules=rules, auto_advance=False)
        accepted = False
        for player_id in range(rules.player_count):
            if accepted:
                break
            player = game.players[player_id]
            for discarder in range(rules.player_count):
                if discarder == player_id or accepted:
                    continue
                source = game.players[discarder]
                for tile in sorted(set(player.hand)):
                    if (
                        accepted
                        or tile == game.gold_tile
                        or player.hand.count(tile) < 2
                        or tile not in source.hand
                    ):
                        continue
                    source.hand.remove(tile)
                    source.hand.sort()
                    source.discards.append(tile)
                    previous = (
                        game.phase,
                        game.current_player,
                        game.last_discard,
                        game.discarder,
                        game.latest_discard,
                        game.latest_discard_seat,
                    )
                    game.phase = "response"
                    game.current_player = discarder
                    game.last_discard = tile
                    game.discarder = discarder
                    game.latest_discard = tile
                    game.latest_discard_seat = discarder
                    game._record_public_action("discard", seat=discarder, tile=tile)
                    legal = tuple(game._response_actions(player_id))
                    chosen = (
                        game.teacher.choose_response(game, player_id, list(legal))
                        if legal
                        else None
                    )
                    if chosen is not None and chosen.kind == "pass" and len(legal) > 1:
                        decision = _decision(game, hand_seed, player_id, legal, chosen)
                        trajectories.append(
                            TrainingTrajectory(
                                profile=game.rules.profile,
                                rules_version=game.rules.version,
                                rules=asdict(game.rules),
                                seed=hand_seed,
                                hand_number=game.hand_number,
                                agent_profiles=("response_pass_curriculum",)
                                * rules.player_count,
                                source_metadata={
                                    "collector": "physical_response_pass_search",
                                    "attempt": attempts,
                                    "actor_seat": player_id,
                                    "discarder_seat": discarder,
                                },
                                decisions=(decision,),
                                outcome={
                                    "winner": None,
                                    "win_type": "synthetic_response_pass",
                                    "win_pattern": None,
                                    "scores": [0] * rules.player_count,
                                    "score_breakdown": None,
                                    "turn_count": game.turn_count,
                                    "synthetic": True,
                                },
                                public_actions=tuple(
                                    dict(action) for action in game.public_actions
                                ),
                            )
                        )
                        accepted = True
                    game.public_actions.pop()
                    (
                        game.phase,
                        game.current_player,
                        game.last_discard,
                        game.discarder,
                        game.latest_discard,
                        game.latest_discard_seat,
                    ) = previous
                    source.discards.pop()
                    source.hand.append(tile)
                    source.hand.sort()
    if len(trajectories) != examples:
        raise RuntimeError("未能生成足够的 Teacher 选择 pass 的响应课程")
    return trajectories


def collect_dagger_decisions(
    policy: Any,
    *,
    hands: int,
    profile: str = "classic",
    seed: int = 20261004,
) -> tuple[list[TeacherDecision], DatasetSummary]:
    """Label the states visited by ``policy`` with the frozen rule Teacher.

    Ordinary Teacher trajectories do not include mistakes that a learned policy
    makes after drifting from the Teacher distribution.  DAgger closes that
    gap: actions used to advance the game come from ``policy``, while the
    stored action label always comes from the engine-compatible Teacher.  The
    snapshot remains perspective-correct and contains no wall order or opponent
    concealed hands.
    """

    if hands <= 0:
        raise ValueError("hands 必须为正数")
    rules = XiamenRules.from_profile(profile)
    decisions: list[TeacherDecision] = []
    action_counts: Counter[str] = Counter()
    wins = 0
    draws = 0
    for hand_index in range(hands):
        hand_seed = seed + hand_index
        game = XiamenMahjongGame(
            seed=hand_seed,
            rules=rules,
            auto_advance=False,
            agents={seat: policy for seat in range(rules.player_count)},
            human_seat=-1,
        )
        steps = 0
        while game.phase != "over":
            steps += 1
            if steps > 600:
                raise RuntimeError("DAgger 自博弈超过安全步数")
            if game.phase == "discard":
                player_id = game.current_player
                legal = tuple(_turn_actions(game, player_id))
                teacher_action = game.teacher.choose_turn_action(game, player_id)
                decisions.append(_decision(game, hand_seed, player_id, legal, teacher_action))
                action_counts[teacher_action.kind] += 1
                chosen = policy.choose_turn_action(game, player_id)
                if chosen not in legal:
                    raise RuntimeError("DAgger 策略选择了规则引擎未提供的摸牌后动作")
                game._apply_turn_action(player_id, chosen)
                continue
            if game.phase == "response":
                for player_id, options in sorted(game.response_options.items()):
                    legal = tuple(options)
                    teacher_action = game.teacher.choose_response(game, player_id, list(legal))
                    decisions.append(
                        _decision(game, hand_seed, player_id, legal, teacher_action)
                    )
                    action_counts[teacher_action.kind] += 1
                    chosen = policy.choose_response(game, player_id, legal)
                    if chosen not in legal:
                        raise RuntimeError("DAgger 策略选择了规则引擎未提供的响应动作")
                    game.response_choices[player_id] = chosen
                game._resolve_responses()
                continue
            raise RuntimeError(f"未知 DAgger 阶段：{game.phase}")
        if game.win_type == "draw":
            draws += 1
        else:
            wins += 1
    return decisions, DatasetSummary(
        hands=hands,
        decisions=len(decisions),
        wins=wins,
        draws=draws,
        action_counts=dict(sorted(action_counts.items())),
    )


def collect_candidate_teacher_dagger_trajectories(
    policy: Any,
    *,
    seed_count: int,
    profile: str = "classic",
    seed: int = 20264804,
) -> tuple[list[TrainingTrajectory], DatasetSummary]:
    """Collect one candidate seat versus three frozen Teachers per wall.

    For every physical shuffle, the candidate is rotated through all four
    seats.  Only states actually controlled by the candidate are labelled by
    the frozen Teacher and retained.  This is the deployment distribution used
    by paired evaluation, unlike all-four-seat policy self-play.

    The settled candidate score is a valid on-policy value target for these
    steps: the candidate controls its own subsequent actions and all opponents
    remain the same frozen Teacher policies.  ``split_group_id`` keeps all four
    rotations of a wall in one train/validation/test partition without
    exporting the underlying seed.
    """

    if seed_count <= 0:
        raise ValueError("seed_count 必须为正数")
    rules = XiamenRules.from_profile(profile)
    baseline = HeuristicTeacherAgent()
    trajectories: list[TrainingTrajectory] = []
    action_counts: Counter[str] = Counter()
    wins = 0
    draws = 0
    for seed_offset in range(seed_count):
        hand_seed = seed + seed_offset
        split_group_id = uuid4().hex
        for candidate_seat in range(rules.player_count):
            game = XiamenMahjongGame(
                seed=hand_seed,
                rules=rules,
                auto_advance=False,
                human_seat=-1,
            )
            decisions: list[TeacherDecision] = []
            steps = 0
            while game.phase != "over":
                steps += 1
                if steps > 600:
                    raise RuntimeError("候选对 Teacher 的 DAgger 对局超过安全步数")
                if game.phase == "discard":
                    player_id = game.current_player
                    legal = tuple(_turn_actions(game, player_id))
                    if player_id == candidate_seat:
                        teacher_action = baseline.choose_turn_action(game, player_id)
                        decisions.append(
                            _decision(game, hand_seed, player_id, legal, teacher_action)
                        )
                        action_counts[teacher_action.kind] += 1
                        action = policy.choose_turn_action(game, player_id)
                    else:
                        action = baseline.choose_turn_action(game, player_id)
                    if action not in legal:
                        raise RuntimeError("候选策略选择了规则引擎未提供的摸牌后动作")
                    game._apply_turn_action(player_id, action)
                    continue
                if game.phase == "response":
                    for player_id, options in sorted(game.response_options.items()):
                        legal = tuple(options)
                        if player_id == candidate_seat:
                            teacher_action = baseline.choose_response(game, player_id, list(legal))
                            decisions.append(
                                _decision(
                                    game, hand_seed, player_id, legal, teacher_action
                                )
                            )
                            action_counts[teacher_action.kind] += 1
                            action = policy.choose_response(game, player_id, legal)
                        else:
                            action = baseline.choose_response(game, player_id, list(legal))
                        if action not in legal:
                            raise RuntimeError("候选策略选择了规则引擎未提供的响应动作")
                        game.response_choices[player_id] = action
                    game._resolve_responses()
                    continue
                raise RuntimeError(f"未知候选 DAgger 阶段：{game.phase}")
            agent_profiles = ["heuristic_teacher"] * rules.player_count
            agent_profiles[candidate_seat] = "candidate_policy"
            trajectories.append(
                _trajectory_from_game(
                    game,
                    decisions,
                    agent_profiles=agent_profiles,
                    source_metadata={
                        "collector": "candidate_vs_teacher_dagger",
                        "candidate_seat": candidate_seat,
                        "wall_rotation": candidate_seat,
                    },
                )
            )
            # Retain the opaque group after construction; it is deliberately
            # distinct from both the data record ID and the replay seed.
            trajectories[-1] = replace(
                trajectories[-1], split_group_id=split_group_id
            )
            if game.win_type == "draw":
                draws += 1
            else:
                wins += 1
    return trajectories, DatasetSummary(
        hands=len(trajectories),
        decisions=sum(len(trajectory.decisions) for trajectory in trajectories),
        wins=wins,
        draws=draws,
        action_counts=dict(sorted(action_counts.items())),
    )


def _weighted_choice(
    options: Sequence[tuple[int, int]], rng: random.Random
) -> int:
    """Choose an item from non-negative integer combinatorial weights."""

    total = sum(weight for _item, weight in options)
    if total <= 0:
        raise ValueError("加权采样需要至少一个正权重候选")
    threshold = rng.randrange(total)
    cumulative = 0
    for item, weight in options:
        cumulative += weight
        if threshold < cumulative:
            return item
    return options[-1][0]  # pragma: no cover - integer arithmetic fallback


def _resample_private_world_for_actor(
    game: XiamenMahjongGame,
    *,
    actor_seat: int,
    rng: random.Random,
) -> XiamenMahjongGame | None:
    """Sample a hidden world consistent with a decision's actor-visible facts.

    Counterfactual rollouts on the original cloned deal leak a *target* that is
    conditioned on one particular wall and set of enemy hands.  The actor does
    not know those facts.  This routine keeps every actor-visible fact fixed
    (own hand, rivers, public melds, flower counts, indicator, turn state and
    public history), then redistributes all remaining base tiles and flowers
    across opponents and the wall.  Opponents' concealed-kong faces are also
    redrawn, because the actor only observes the existence of such a kong.

    It is a prior belief sampler rather than a full action-conditioned
    posterior: public opponent actions are preserved but not likelihood-
    weighted by each sampled hidden hand.  States with another player's public
    opening-wait flag are skipped, since faithfully conditioning that special
    historical predicate would require a dedicated posterior sampler.
    """

    if not 0 <= actor_seat < game.rules.player_count:
        raise ValueError("actor_seat 超出玩家范围")
    if any(
        seat != actor_seat for seat in game.opening_wait_seats
    ):
        return None

    sampled = copy.deepcopy(game)
    pool = Counter(base_wall())

    def consume(tiles: Iterable[int]) -> bool:
        values = list(tiles)
        counts = Counter(values)
        if any(pool[tile] < count for tile, count in counts.items()):
            return False
        pool.subtract(counts)
        return True

    actor = sampled.players[actor_seat]
    if not consume(actor.hand) or not consume(actor.flowers):
        return None
    if sampled.gold_indicator is not None and not consume([sampled.gold_indicator]):
        return None

    concealed_kongs: list[dict[str, Any]] = []
    opponent_flower_slots: list[tuple[int, int]] = []
    for player in sampled.players:
        if not consume(player.discards):
            return None
        for meld in player.melds:
            if meld["kind"] == "an_kan" and player.seat != actor_seat:
                if len(meld["tiles"]) != 4:
                    return None
                concealed_kongs.append(meld)
            elif not consume(meld["tiles"]):
                return None
        if player.seat != actor_seat:
            opponent_flower_slots.append((player.seat, len(player.flowers)))

    # Draw a face for each opponent's concealed kong.  The weight is the
    # number of physical four-of-a-kind choices remaining for that face.
    for meld in concealed_kongs:
        candidates = [
            (tile, math.comb(count, 4))
            for tile, count in pool.items()
            if is_base_tile(tile) and tile != sampled.gold_tile and count >= 4
        ]
        if not candidates:
            return None
        tile = _weighted_choice(candidates, rng)
        pool[tile] -= 4
        meld["tiles"] = [tile] * 4
        meld["value"] = sampled._tile_value(tile)

    base_tiles = [
        tile for tile, count in pool.items() if is_base_tile(tile) for _ in range(count)
    ]
    flower_tiles = [
        tile
        for tile, count in pool.items()
        if not is_base_tile(tile)
        for _ in range(count)
    ]
    rng.shuffle(base_tiles)
    rng.shuffle(flower_tiles)
    base_offset = 0
    for player in sampled.players:
        if player.seat == actor_seat:
            continue
        hand_count = len(player.hand)
        if base_offset + hand_count > len(base_tiles):
            return None
        player.hand = sorted(base_tiles[base_offset : base_offset + hand_count])
        base_offset += hand_count
    flower_offset = 0
    for seat, flower_count in opponent_flower_slots:
        if flower_offset + flower_count > len(flower_tiles):
            return None
        sampled.players[seat].flowers = sorted(
            flower_tiles[flower_offset : flower_offset + flower_count]
        )
        flower_offset += flower_count
    remaining_wall = [*base_tiles[base_offset:], *flower_tiles[flower_offset:]]
    if len(remaining_wall) != len(sampled.wall):
        return None
    rng.shuffle(remaining_wall)
    sampled.wall = remaining_wall
    # Only the actor knows its drawn tile; future opponent turns will replace
    # their entries with a newly sampled draw before that information matters.
    for seat in range(sampled.rules.player_count):
        if seat != actor_seat:
            sampled.last_drawn_tiles[seat] = None
    sampled.random = random.Random(rng.randrange(2**63))

    if sampled.phase == "response":
        sampled.response_choices = {}
        sampled.response_options = {}
        for seat in range(sampled.rules.player_count):
            options = sampled._response_actions(seat)
            if options:
                sampled.response_options[seat] = options
    return sampled


def _sample_rollout_opponents(
    *,
    player_count: int,
    candidate_seat: int,
    teacher: HeuristicTeacherAgent,
    opponents: Sequence[tuple[str, Any]],
    teacher_opponent_probability: float,
    rng: random.Random,
) -> dict[int, tuple[str, Any]]:
    """Choose a frozen opponent for each non-candidate seat.

    The returned map is reused for every legal action in a rollout replicate.
    This common-random-numbers treatment makes an action-value comparison much
    less noisy than evaluating every action against a different opponent draw.
    """

    selected: dict[int, tuple[str, Any]] = {}
    for seat in range(player_count):
        if seat == candidate_seat:
            continue
        if not opponents or rng.random() < teacher_opponent_probability:
            selected[seat] = ("heuristic_teacher", teacher)
        else:
            selected[seat] = opponents[rng.randrange(len(opponents))]
    return selected


def _check_rollout_action(action: GameAction, legal: Sequence[GameAction]) -> None:
    if action not in legal:
        raise RuntimeError("反事实 rollout 策略选择了规则引擎未提供的动作")


@dataclass
class _CounterfactualRolloutJob:
    """One private forced-action branch that may share inference with peers.

    The game instance, including its RNG, is intentionally owned by exactly
    one job.  Batching therefore changes only neural inference scheduling;
    it never merges state transitions or random draws between worlds.
    """

    game: XiamenMahjongGame
    candidate_seat: int
    candidate_policy: Any
    opponents: Mapping[int, tuple[str, Any]]
    forced_action: GameAction
    pending_forced_action: bool = True
    safety: int = 0


@dataclass(frozen=True)
class _ActorPrivateDrawObservation:
    """One actor-known draw positioned relative to public event history.

    This is private collector memory, not a dataset record.  ``flowers`` are
    the zero or more flower replacements the actor saw before receiving the
    final playable tile.  A replacement draw after a kong has no public
    ``draw`` event, so ``public_draw_event_index`` is intentionally optional.
    """

    after_public_action_count: int
    public_draw_event_index: int | None
    turn_count: int
    tile: int
    flowers: tuple[int, ...]


@dataclass(frozen=True)
class _ActorPrivateActionObservation:
    """One actor action, including facts intentionally absent from public log."""

    phase: str
    action: GameAction
    before_public_action_count: int
    after_public_action_count: int


@dataclass(frozen=True)
class _ActorPrivateReplayTrace:
    """Runtime-only facts needed to replay a candidate's information set.

    The trace stores only the candidate's own initial hand/flowers, draws,
    and actions.  The action trace is necessary because a concealed kong face
    is intentionally absent from the public log but known to its owner.  This
    class deliberately has no payload method and must never be copied into a
    ``TrainingTrajectory`` or ``TeacherDecision``.
    """

    actor_seat: int
    dealer: int
    gold_indicator: int | None
    gold_tile: int | None
    initial_hand: tuple[int, ...]
    initial_flowers: tuple[int, ...]
    draws: tuple[_ActorPrivateDrawObservation, ...]
    actions: tuple[_ActorPrivateActionObservation, ...]
    public_action_count: int


@dataclass(frozen=True)
class _CounterfactualDecisionSnapshot:
    """A private simulation snapshot plus runtime-only replay prerequisites.

    ``initial_game`` is the table immediately after setup (and its opening
    dealer draw).  It is retained solely to validate/reconstruct a particle's
    public-history replay.  Like ``actor_trace``, it is deliberately absent
    from every exportable dataset type.
    """

    game: XiamenMahjongGame
    legal_actions: tuple[GameAction, ...]
    actor_trace: _ActorPrivateReplayTrace
    initial_game: XiamenMahjongGame


@dataclass(frozen=True)
class _HistoryReplayResult:
    """Private runtime result of replaying one public-history particle.

    The result reports only a scalar likelihood and a rejection label.  It
    intentionally does not expose a particle game, wall, or any concealed
    opponent tile to collector metadata or training JSONL.
    """

    accepted: bool
    likelihood: float
    log_likelihood: float
    public_action_count: int
    rejection_reason: str | None = None


@dataclass(frozen=True)
class _HistoryReplayAudit:
    """Non-sensitive health report for a proposed history-conditioned belief.

    Only aggregate acceptance and likelihood diagnostics are retained.  In
    particular, proposed worlds, their hidden hands, wall orders and RNG
    states stay local to the audit loop.
    """

    proposed_particles: int
    accepted_particles: int
    mean_accepted_log_likelihood: float | None
    rejection_counts: dict[str, int]

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_particles / self.proposed_particles

    def payload(self) -> dict[str, Any]:
        return {
            "proposed_particles": self.proposed_particles,
            "accepted_particles": self.accepted_particles,
            "acceptance_rate": self.acceptance_rate,
            "mean_accepted_log_likelihood": self.mean_accepted_log_likelihood,
            "rejection_counts": dict(self.rejection_counts),
        }


def _public_events_match(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Compare one public event without treating dict insertion order as state."""

    fields = ("index", "turn", "kind", "seat", "tile", "tiles", "result")
    return all(actual.get(field) == expected.get(field) for field in fields)


def _event_compatible_actions(
    legal: Sequence[GameAction], event: Mapping[str, Any]
) -> tuple[GameAction, ...]:
    """Return detailed legal actions compatible with one public event.

    A concealed kong is deliberately a coarse public event: the owner knows
    its face, while every other seat observes only that an ``an_kan``
    occurred.  The returned set consequently marginalizes its legal faces.
    All other currently supported action events retain their public tile(s).
    """

    kind = event.get("kind")
    if not isinstance(kind, str):
        return ()
    matches = [action for action in legal if action.kind == kind]
    if kind == "an_kan":
        return tuple(matches)
    if "tile" in event:
        matches = [action for action in matches if action.tile == event["tile"]]
    if "tiles" in event:
        expected_tiles = tuple(event["tiles"])
        matches = [action for action in matches if action.tiles == expected_tiles]
    return tuple(matches)


def _frozen_behavior_action_likelihood(
    agent: Any,
    game: XiamenMahjongGame,
    *,
    player_id: int,
    legal: Sequence[GameAction],
    compatible: Sequence[GameAction],
    is_response: bool,
    temperature: float,
    uniform_mixture: float,
) -> tuple[GameAction, float] | None:
    """Select a compatible detailed action and score its public observation.

    When an agent exposes legal-action ``scores``, their smoothed softmax is
    used as the behavior likelihood; otherwise its deterministic chosen action
    is a conservative fallback. ``uniform_mixture`` gives every legal detailed
    action positive mass, so an imperfect behavior model does not destroy the
    whole particle set. For an opponent concealed kong the likelihood is the
    sum over all matching faces because that face is not a public observation.
    """

    if not compatible:
        return None
    matching_indices = [
        index for index, action in enumerate(legal) if action in compatible
    ]
    scores = getattr(agent, "scores", None)
    if callable(scores):
        try:
            decision = _policy_decision(
                game, game.seed or 0, player_id, legal
            )
            score_values = tuple(float(value) for value in scores(decision))
        except (TypeError, ValueError, RuntimeError):
            score_values = ()
        if len(score_values) == len(legal) and all(
            math.isfinite(value) for value in score_values
        ):
            likelihood = sum(
                smoothed_policy_likelihood(
                    score_values,
                    observed_index=index,
                    temperature=temperature,
                    uniform_mixture=uniform_mixture,
                )
                for index in matching_indices
            )
            action = legal[
                max(matching_indices, key=lambda index: (score_values[index], -index))
            ]
            return action, likelihood
    selected = (
        agent.choose_response(game, player_id, list(legal))
        if is_response
        else agent.choose_turn_action(game, player_id)
    )
    if selected not in legal:
        return None
    selected_index = legal.index(selected)
    likelihood = sum(
        smoothed_deterministic_likelihood(
            selected_index=selected_index,
            observed_index=index,
            action_count=len(legal),
            uniform_mixture=uniform_mixture,
        )
        for index in matching_indices
    )
    # Where the public event hides a detailed action (currently only an_kan),
    # use the frozen policy's matching face when available.  If it chose a
    # different public event, any compatible face is a valid uniform-noise
    # proposal; the likelihood above still accounts for that mismatch.
    action = selected if selected in compatible else compatible[0]
    return action, likelihood


_ACTOR_DRAW_RESERVATION = -1


def _sample_replay_setup_for_actor(
    snapshot: _CounterfactualDecisionSnapshot,
    *,
    rng: random.Random,
) -> XiamenMahjongGame | None:
    """Propose a setup-world conditional on the actor's private draw trace.

    The proposal samples unknown opponent initial hands/flowers and the
    unknown portion of the wall without using their source-world identities.
    Every later actor-known draw is removed from that unknown pool and
    represented by an in-memory reservation.  During replay the reservation
    supplies the recorded private tile (and any flower replacements); all
    other draws remain a uniform permutation of the remaining physical tiles.

    This helper deliberately supports the ``core`` profile only.  It is not a
    public serializer and the returned game must stay inside particle/replay
    code.  A caller must still replay and legality-check the entire public
    prefix before treating this as an accepted belief particle.
    """

    trace = snapshot.actor_trace
    source = snapshot.initial_game
    if source.rules.profile != "core":
        return None
    if any(seat != trace.actor_seat for seat in source.opening_wait_seats):
        return None
    sampled = copy.deepcopy(source)
    pool = Counter(base_wall())

    def consume(tiles: Iterable[int]) -> bool:
        counts = Counter(tiles)
        if any(pool[tile] < count for tile, count in counts.items()):
            return False
        pool.subtract(counts)
        return True

    actor = sampled.players[trace.actor_seat]
    if not consume(actor.hand) or not consume(actor.flowers):
        return None
    if sampled.gold_indicator is not None and not consume([sampled.gold_indicator]):
        return None

    setup_public_count = len(sampled.public_actions)
    later_actor_draws = [
        draw
        for draw in trace.draws
        if draw.after_public_action_count > setup_public_count
    ]
    reserved_count = 0
    for draw in later_actor_draws:
        if not consume([*draw.flowers, draw.tile]):
            return None
        reserved_count += len(draw.flowers) + 1

    # At setup the actor's concealed hand is known, while each opposing hand
    # and flower *count* is public.  Allocate their physical identities from
    # the remaining pool without retaining any identity from the source game.
    base_tiles = [
        tile
        for tile, count in pool.items()
        if is_base_tile(tile)
        for _ in range(count)
    ]
    flower_tiles = [
        tile
        for tile, count in pool.items()
        if not is_base_tile(tile)
        for _ in range(count)
    ]
    rng.shuffle(base_tiles)
    rng.shuffle(flower_tiles)
    base_offset = 0
    flower_offset = 0
    for player in sampled.players:
        if player.seat == trace.actor_seat:
            continue
        hand_count = len(player.hand)
        flower_count = len(player.flowers)
        if (
            base_offset + hand_count > len(base_tiles)
            or flower_offset + flower_count > len(flower_tiles)
        ):
            return None
        player.hand = sorted(base_tiles[base_offset : base_offset + hand_count])
        player.flowers = sorted(flower_tiles[flower_offset : flower_offset + flower_count])
        base_offset += hand_count
        flower_offset += flower_count
    unknown_wall = [
        *base_tiles[base_offset:],
        *flower_tiles[flower_offset:],
    ]
    if len(unknown_wall) + reserved_count != len(sampled.wall):
        return None
    sampled.wall = [*unknown_wall, *([_ACTOR_DRAW_RESERVATION] * reserved_count)]
    rng.shuffle(sampled.wall)
    # A non-actor's opening drawn tile is not privately visible to the actor;
    # core rules do not use it after setup.  Keeping the source value would be
    # a needless hidden-state dependency.
    for seat in range(sampled.rules.player_count):
        if seat != trace.actor_seat:
            sampled.last_drawn_tiles[seat] = None
    sampled.random = random.Random(rng.randrange(2**63))
    return sampled


def _replay_snapshot_public_history(
    snapshot: _CounterfactualDecisionSnapshot,
    *,
    opponents: Mapping[int, tuple[str, Any]],
    behavior_temperature: float = 1.0,
    uniform_mixture: float = 0.02,
    initial_game: XiamenMahjongGame | None = None,
    condition_actor_draws: bool = False,
) -> _HistoryReplayResult:
    """Strictly replay one snapshot's public prefix in its original world.

    This is the safety oracle for the forthcoming particle proposal: it starts
    from the private setup state, forces only the candidate's recorded private
    actions, and makes frozen opponents reproduce the observed public actions.
    Every generated public event must match the source prefix exactly.  The
    routine is runtime-only; it never returns the replay game or its hidden
    state.

    It deliberately supports the core event vocabulary only.  A new profile
    feature must gain a precise replay transition before the collector is
    allowed to condition particles on it.
    """

    if not math.isfinite(behavior_temperature) or behavior_temperature <= 0.0:
        raise ValueError("behavior_temperature 必须为正且有限")
    if not 0.0 <= uniform_mixture < 1.0:
        raise ValueError("uniform_mixture 必须在 0（含）到 1（不含）之间")
    trace = snapshot.actor_trace
    game = copy.deepcopy(initial_game or snapshot.initial_game)
    target_events = tuple(dict(event) for event in snapshot.game.public_actions)
    cursor = len(game.public_actions)
    if cursor > len(target_events) or any(
        not _public_events_match(game.public_actions[index], target_events[index])
        for index in range(cursor)
    ):
        return _HistoryReplayResult(False, 0.0, float("-inf"), cursor, "setup_prefix")
    if game.rules.profile != "core":
        return _HistoryReplayResult(False, 0.0, float("-inf"), cursor, "profile")

    # The opening dealer draw already exists in ``initial_game``.  Later
    # candidate draws must agree with the private trace even when no public
    # event reveals their face (replacement draws after kongs).
    remaining_draws = iter(
        draw
        for draw in trace.draws
        if draw.after_public_action_count > len(game.public_actions)
    )
    next_draw: _ActorPrivateDrawObservation | None = None
    known_actor_flowers = len(game.players[trace.actor_seat].flowers)
    original_draw = game._draw_for_player

    def draw_with_trace(player):
        nonlocal next_draw, known_actor_flowers
        before_public_count = len(game.public_actions)
        if player.seat != trace.actor_seat:
            if not condition_actor_draws:
                return original_draw(player)
            # Draw the next unknown physical tile; reservations represent
            # later actor-known cards and must never be handed to an opponent.
            while game.wall:
                try:
                    index = next(
                        index
                        for index, candidate in enumerate(game.wall)
                        if candidate != _ACTOR_DRAW_RESERVATION
                    )
                except StopIteration:
                    return None
                tile = game.wall.pop(index)
                if tile >= BASE_TILE_COUNT:
                    player.flowers.append(tile)
                    game._event("补花", f"{game._seat_name(player.seat)}补到花牌")
                    continue
                player.hand.append(tile)
                player.hand.sort()
                return tile
            return None
        if next_draw is None:
            next_draw = next(remaining_draws, None)
        if next_draw is None:
            raise RuntimeError("actor_draw_trace_exhausted")
        if condition_actor_draws:
            for flower in next_draw.flowers:
                try:
                    game.wall.pop(game.wall.index(_ACTOR_DRAW_RESERVATION))
                except ValueError as error:
                    raise RuntimeError("actor_draw_reservation_exhausted") from error
                player.flowers.append(flower)
                game._event("补花", f"{game._seat_name(player.seat)}补到花牌")
            try:
                game.wall.pop(game.wall.index(_ACTOR_DRAW_RESERVATION))
            except ValueError as error:
                raise RuntimeError("actor_draw_reservation_exhausted") from error
            tile = next_draw.tile
            player.hand.append(tile)
            player.hand.sort()
        else:
            tile = original_draw(player)
        observed_flowers = tuple(player.flowers[known_actor_flowers:])
        expected_public_count = before_public_count + (
            1 if next_draw.public_draw_event_index is not None else 0
        )
        if (
            tile != next_draw.tile
            or observed_flowers != next_draw.flowers
            or next_draw.after_public_action_count != expected_public_count
            or (
                next_draw.public_draw_event_index is not None
                and next_draw.public_draw_event_index != before_public_count
            )
        ):
            raise RuntimeError("actor_private_draw_mismatch")
        known_actor_flowers = len(player.flowers)
        next_draw = None
        return tile

    # Bound onto this private clone only.  The source snapshot and any
    # exported TrainingTrajectory retain no callback or hidden game object.
    game._draw_for_player = draw_with_trace  # type: ignore[method-assign]

    def actor_action(phase: str) -> GameAction | None:
        before = len(game.public_actions)
        matches = [
            item.action
            for item in trace.actions
            if item.phase == phase and item.before_public_action_count == before
        ]
        return matches[0] if len(matches) == 1 else None

    def consume_generated_events() -> str | None:
        nonlocal cursor
        while cursor < len(game.public_actions):
            if cursor >= len(target_events) or not _public_events_match(
                game.public_actions[cursor], target_events[cursor]
            ):
                return "public_event_mismatch"
            cursor += 1
        return None

    log_likelihood = 0.0
    try:
        while cursor < len(target_events):
            if game.phase == "over":
                return _HistoryReplayResult(
                    False, 0.0, float("-inf"), cursor, "ended_before_prefix"
                )
            if game.phase == "discard":
                player_id = game.current_player
                legal = tuple(_turn_actions(game, player_id))
                expected = target_events[cursor]
                compatible = _event_compatible_actions(legal, expected)
                if player_id == trace.actor_seat:
                    action = actor_action("discard")
                    if action not in compatible:
                        return _HistoryReplayResult(
                            False, 0.0, float("-inf"), cursor, "actor_turn_action"
                        )
                else:
                    opponent = opponents.get(player_id)
                    if opponent is None:
                        return _HistoryReplayResult(
                            False, 0.0, float("-inf"), cursor, "opponent_missing"
                        )
                    proposal = _frozen_behavior_action_likelihood(
                        opponent[1],
                        game,
                        player_id=player_id,
                        legal=legal,
                        compatible=compatible,
                        is_response=False,
                        temperature=behavior_temperature,
                        uniform_mixture=uniform_mixture,
                    )
                    if proposal is None:
                        return _HistoryReplayResult(
                            False, 0.0, float("-inf"), cursor, "opponent_turn_action"
                        )
                    action, likelihood = proposal
                    log_likelihood += math.log(likelihood)
                _check_rollout_action(action, legal)
                game._apply_turn_action(player_id, action)
            elif game.phase == "response":
                expected = target_events[cursor]
                expected_kind = expected.get("kind")
                response_actions: dict[int, GameAction] = {}
                if expected_kind in {"pong", "chi", "ming_kan"}:
                    claimant = expected.get("seat")
                    if not isinstance(claimant, int):
                        return _HistoryReplayResult(
                            False, 0.0, float("-inf"), cursor, "claimant_missing"
                        )
                elif expected_kind == "result" and expected.get("result") == "discard":
                    claimant = expected.get("seat")
                    if not isinstance(claimant, int):
                        return _HistoryReplayResult(
                            False, 0.0, float("-inf"), cursor, "winner_missing"
                        )
                    expected_kind = "hu"
                elif expected_kind in {"draw", "result"}:
                    # A draw result can be created by all players passing when
                    # the next normal draw reaches the dead wall.
                    claimant = None
                    expected_kind = "pass"
                else:
                    return _HistoryReplayResult(
                        False, 0.0, float("-inf"), cursor, "response_event"
                    )
                for player_id, options in sorted(game.response_options.items()):
                    legal = tuple(options)
                    if player_id == trace.actor_seat:
                        action = actor_action("response")
                        if action is None:
                            return _HistoryReplayResult(
                                False, 0.0, float("-inf"), cursor, "actor_response"
                            )
                    else:
                        if player_id == claimant:
                            event = dict(expected)
                            event["kind"] = expected_kind
                            compatible = _event_compatible_actions(legal, event)
                        else:
                            compatible = tuple(
                                action for action in legal if action.kind == "pass"
                            )
                        opponent = opponents.get(player_id)
                        if opponent is None:
                            return _HistoryReplayResult(
                                False, 0.0, float("-inf"), cursor, "opponent_missing"
                            )
                        proposal = _frozen_behavior_action_likelihood(
                            opponent[1],
                            game,
                            player_id=player_id,
                            legal=legal,
                            compatible=compatible,
                            is_response=True,
                            temperature=behavior_temperature,
                            uniform_mixture=uniform_mixture,
                        )
                        if proposal is None:
                            return _HistoryReplayResult(
                                False,
                                0.0,
                                float("-inf"),
                                cursor,
                                "opponent_response",
                            )
                        action, likelihood = proposal
                        log_likelihood += math.log(likelihood)
                    if action not in legal:
                        return _HistoryReplayResult(
                            False, 0.0, float("-inf"), cursor, "response_illegal"
                        )
                    response_actions[player_id] = action
                game.response_choices = response_actions
                game._resolve_responses()
            else:
                return _HistoryReplayResult(
                    False, 0.0, float("-inf"), cursor, "unknown_phase"
                )
            mismatch = consume_generated_events()
            if mismatch:
                return _HistoryReplayResult(
                    False, 0.0, float("-inf"), cursor, mismatch
                )
    except (RuntimeError, ValueError):
        return _HistoryReplayResult(
            False, 0.0, float("-inf"), cursor, "private_trace_mismatch"
        )
    likelihood = math.exp(log_likelihood) if log_likelihood > -745.0 else 0.0
    return _HistoryReplayResult(True, likelihood, log_likelihood, cursor)


def _audit_resampled_history_prefix(
    snapshot: _CounterfactualDecisionSnapshot,
    *,
    opponents: Mapping[int, tuple[str, Any]],
    particle_count: int,
    rng: random.Random,
    behavior_temperature: float = 1.0,
    uniform_mixture: float = 0.02,
) -> _HistoryReplayAudit:
    """Measure a proposal before it can influence any training rollout.

    This currently audits the intentionally conservative, rejection-sampling
    setup proposal.  It must demonstrate a healthy acceptance rate and ESS in
    a separately calibrated successor before it is wired into the collector.
    The function itself makes no dataset/model mutation and returns no hidden
    particles.
    """

    if particle_count <= 0:
        raise ValueError("particle_count 必须为正数")
    accepted_log_likelihoods: list[float] = []
    rejection_counts: Counter[str] = Counter()
    for _ in range(particle_count):
        initial_game = _sample_replay_setup_for_actor(snapshot, rng=rng)
        if initial_game is None:
            rejection_counts["setup_proposal"] += 1
            continue
        result = _replay_snapshot_public_history(
            snapshot,
            opponents=opponents,
            behavior_temperature=behavior_temperature,
            uniform_mixture=uniform_mixture,
            initial_game=initial_game,
            condition_actor_draws=True,
        )
        if result.accepted:
            accepted_log_likelihoods.append(result.log_likelihood)
        else:
            rejection_counts[result.rejection_reason or "unknown"] += 1
    return _HistoryReplayAudit(
        proposed_particles=particle_count,
        accepted_particles=len(accepted_log_likelihoods),
        mean_accepted_log_likelihood=(
            sum(accepted_log_likelihoods) / len(accepted_log_likelihoods)
            if accepted_log_likelihoods
            else None
        ),
        rejection_counts=dict(sorted(rejection_counts.items())),
    )


@dataclass(frozen=True)
class _CounterfactualActionRequest:
    """A legal policy choice waiting for a possibly batched inference call."""

    job: _CounterfactualRolloutJob
    player_id: int
    legal: tuple[GameAction, ...]
    agent: Any
    is_response: bool


def _select_counterfactual_actions(
    requests: Sequence[_CounterfactualActionRequest],
    *,
    batch_stats: _CounterfactualBatchStats | None = None,
) -> list[tuple[_CounterfactualActionRequest, GameAction]]:
    """Choose legal actions, batching agents that explicitly support it.

    ``scores_batch`` is an opt-in protocol implemented by
    :class:`TorchPolicyValueAgent`.  Other frozen agents keep their existing
    single-game methods.  This keeps the rule Teacher dependency-free and
    avoids assuming a particular ML framework in this module.
    """

    selected: list[tuple[_CounterfactualActionRequest, GameAction]] = []
    by_agent: dict[int, tuple[Any, list[_CounterfactualActionRequest]]] = {}
    for request in requests:
        by_agent.setdefault(id(request.agent), (request.agent, []))[1].append(request)
    for agent, agent_requests in by_agent.values():
        scores_batch = getattr(agent, "scores_batch", None)
        if callable(scores_batch):
            if batch_stats is not None:
                batch_stats.inference_calls += 1
                batch_stats.inference_decisions += len(agent_requests)
                batch_stats.max_inference_decisions = max(
                    batch_stats.max_inference_decisions, len(agent_requests)
                )
            decisions = [
                _policy_decision(
                    request.job.game,
                    request.job.game.seed or 0,
                    request.player_id,
                    request.legal,
                )
                for request in agent_requests
            ]
            scores_rows = scores_batch(decisions)
            if len(scores_rows) != len(agent_requests):
                raise RuntimeError("批量策略返回的动作集合数量不匹配")
            for request, scores in zip(agent_requests, scores_rows):
                if len(scores) != len(request.legal):
                    raise RuntimeError("批量策略返回的动作分数数量不匹配")
                action = request.legal[
                    max(range(len(scores)), key=lambda index: (scores[index], -index))
                ]
                selected.append((request, action))
            continue
        for request in agent_requests:
            if request.is_response:
                action = agent.choose_response(
                    request.job.game, request.player_id, list(request.legal)
                )
            else:
                action = agent.choose_turn_action(request.job.game, request.player_id)
            selected.append((request, action))
    return selected


def _continue_counterfactual_rollout_batch(
    jobs: Sequence[_CounterfactualRolloutJob],
    *,
    batch_stats: _CounterfactualBatchStats | None = None,
) -> list[int]:
    """Settle independent forced branches while sharing optional inference.

    All jobs must already be clones of compatible decision snapshots.  This
    function intentionally does not export the games, hidden hands, walls, or
    their RNG states; callers receive only the candidate terminal score.
    """

    active = list(jobs)
    while active:
        action_requests: list[_CounterfactualActionRequest] = []
        response_jobs: list[_CounterfactualRolloutJob] = []
        for job in active:
            game = job.game
            if game.phase == "over":
                continue
            job.safety += 1
            if job.safety > 600:
                raise RuntimeError("反事实 rollout 超过安全步数")
            if game.phase == "discard":
                player_id = game.current_player
                legal = tuple(_turn_actions(game, player_id))
                if player_id == job.candidate_seat and job.pending_forced_action:
                    action = job.forced_action
                    job.pending_forced_action = False
                    _check_rollout_action(action, legal)
                    game._apply_turn_action(player_id, action)
                else:
                    agent = (
                        job.candidate_policy
                        if player_id == job.candidate_seat
                        else job.opponents[player_id][1]
                    )
                    action_requests.append(
                        _CounterfactualActionRequest(
                            job, player_id, legal, agent, False
                        )
                    )
                continue
            if game.phase == "response":
                response_jobs.append(job)
                for player_id, options in sorted(game.response_options.items()):
                    legal = tuple(options)
                    if player_id == job.candidate_seat and job.pending_forced_action:
                        action = job.forced_action
                        job.pending_forced_action = False
                        _check_rollout_action(action, legal)
                        game.response_choices[player_id] = action
                    else:
                        agent = (
                            job.candidate_policy
                            if player_id == job.candidate_seat
                            else job.opponents[player_id][1]
                        )
                        action_requests.append(
                            _CounterfactualActionRequest(
                                job, player_id, legal, agent, True
                            )
                        )
                continue
            raise RuntimeError(f"未知反事实 rollout 阶段：{game.phase}")
        for request, action in _select_counterfactual_actions(
            action_requests, batch_stats=batch_stats
        ):
            _check_rollout_action(action, request.legal)
            if request.is_response:
                request.job.game.response_choices[request.player_id] = action
            else:
                request.job.game._apply_turn_action(request.player_id, action)
        for job in response_jobs:
            job.game._resolve_responses()
        active = [job for job in active if job.game.phase != "over"]
    if any(job.pending_forced_action for job in jobs):
        raise RuntimeError("反事实动作没有在快照状态中得到执行")
    return [int(job.game.players[job.candidate_seat].score) for job in jobs]


def _continue_counterfactual_rollouts_batched(
    jobs: Sequence[_CounterfactualRolloutJob],
    *,
    rollout_batch_size: int,
    batch_stats: _CounterfactualBatchStats | None = None,
) -> list[int]:
    """Run a bounded job queue to avoid retaining all private worlds at once."""

    if rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size 必须为正数")
    scores: list[int] = []
    for start in range(0, len(jobs), rollout_batch_size):
        scores.extend(
            _continue_counterfactual_rollout_batch(
                jobs[start : start + rollout_batch_size], batch_stats=batch_stats
            )
        )
    return scores


def _continue_counterfactual_rollout(
    game: XiamenMahjongGame,
    *,
    candidate_seat: int,
    candidate_policy: Any,
    opponents: Mapping[int, tuple[str, Any]],
    forced_action: GameAction,
) -> int:
    """Apply one candidate action, then settle a legal frozen-policy game.

    ``game`` is a private in-memory clone used only to calculate the target.
    The caller exports neither its wall nor other players' hands.  Subsequent
    candidate decisions use the current policy, so the result estimates
    :math:`Q^pi(s, a)`, not a hidden-state heuristic supplied to the model.
    """

    pending_forced_action = True
    safety = 0
    while game.phase != "over":
        safety += 1
        if safety > 600:
            raise RuntimeError("反事实 rollout 超过安全步数")
        if game.phase == "discard":
            player_id = game.current_player
            legal = tuple(_turn_actions(game, player_id))
            if player_id == candidate_seat:
                if pending_forced_action:
                    action = forced_action
                    pending_forced_action = False
                else:
                    action = candidate_policy.choose_turn_action(game, player_id)
            else:
                action = opponents[player_id][1].choose_turn_action(game, player_id)
            _check_rollout_action(action, legal)
            game._apply_turn_action(player_id, action)
            continue
        if game.phase == "response":
            for player_id, options in sorted(game.response_options.items()):
                legal = tuple(options)
                if player_id == candidate_seat:
                    if pending_forced_action:
                        action = forced_action
                        pending_forced_action = False
                    else:
                        action = candidate_policy.choose_response(
                            game, player_id, list(legal)
                        )
                else:
                    action = opponents[player_id][1].choose_response(
                        game, player_id, list(legal)
                    )
                _check_rollout_action(action, legal)
                game.response_choices[player_id] = action
            game._resolve_responses()
            continue
        raise RuntimeError(f"未知反事实 rollout 阶段：{game.phase}")
    if pending_forced_action:
        raise RuntimeError("反事实动作没有在快照状态中得到执行")
    return int(game.players[candidate_seat].score)


def _run_candidate_base_hand(
    game: XiamenMahjongGame,
    *,
    candidate_seat: int,
    candidate_policy: Any,
    opponents: Mapping[int, tuple[str, Any]],
) -> list[_CounterfactualDecisionSnapshot]:
    """Play one deployment-like hand and retain private replay snapshots.

    The returned actor trace is retained only while the collector runs.  It is
    the minimum information needed for a future sequential belief replayer to
    respect the actor's own initial hand, flower replacements, and later
    draws; the safe exported trajectory remains actor/public observation only.
    """

    snapshots: list[_CounterfactualDecisionSnapshot] = []
    # The same immutable-in-practice setup clone is shared by the hand's
    # snapshots.  Replay code always deep-copies it before applying an event;
    # it is never surfaced through a trajectory or collector summary.
    initial_game = copy.deepcopy(game)
    initial_actor = game.players[candidate_seat]
    initial_hand = list(initial_actor.hand)
    initial_draw = (
        game.last_drawn_tiles[candidate_seat]
        if game.phase == "discard" and game.current_player == candidate_seat
        else None
    )
    if initial_draw is not None and initial_draw in initial_hand:
        initial_hand.remove(initial_draw)
    initial_flowers = tuple(initial_actor.flowers)
    private_draws: list[_ActorPrivateDrawObservation] = []
    private_actions: list[_ActorPrivateActionObservation] = []
    known_flower_count = len(initial_flowers)
    last_draw_marker: tuple[int, int, int, int, int] | None = None

    def record_actor_draw_if_needed() -> None:
        nonlocal known_flower_count, last_draw_marker
        if game.phase != "discard" or game.current_player != candidate_seat:
            return
        player = game.players[candidate_seat]
        tile = game.last_drawn_tiles[candidate_seat]
        if tile is None or tile not in player.hand:
            return
        marker = (
            game.turn_count,
            len(game.public_actions),
            tile,
            len(player.hand),
            len(player.flowers),
        )
        if marker == last_draw_marker:
            return
        public_draw_event_index = None
        if game.public_actions:
            latest = game.public_actions[-1]
            if latest.get("kind") == "draw" and latest.get("seat") == candidate_seat:
                public_draw_event_index = int(latest["index"])
        flowers = tuple(player.flowers[known_flower_count:])
        private_draws.append(
            _ActorPrivateDrawObservation(
                after_public_action_count=len(game.public_actions),
                public_draw_event_index=public_draw_event_index,
                turn_count=game.turn_count,
                tile=tile,
                flowers=flowers,
            )
        )
        known_flower_count = len(player.flowers)
        last_draw_marker = marker

    def actor_trace() -> _ActorPrivateReplayTrace:
        return _ActorPrivateReplayTrace(
            actor_seat=candidate_seat,
            dealer=game.dealer,
            gold_indicator=game.gold_indicator,
            gold_tile=game.gold_tile,
            initial_hand=tuple(sorted(initial_hand)),
            initial_flowers=initial_flowers,
            draws=tuple(private_draws),
            actions=tuple(private_actions),
            public_action_count=len(game.public_actions),
        )

    def record_actor_action(
        phase: str, action: GameAction, *, before_public_action_count: int
    ) -> None:
        private_actions.append(
            _ActorPrivateActionObservation(
                phase=phase,
                action=action,
                before_public_action_count=before_public_action_count,
                after_public_action_count=len(game.public_actions),
            )
        )

    safety = 0
    while game.phase != "over":
        safety += 1
        if safety > 600:
            raise RuntimeError("动作价值采样对局超过安全步数")
        record_actor_draw_if_needed()
        if game.phase == "discard":
            player_id = game.current_player
            legal = tuple(_turn_actions(game, player_id))
            if player_id == candidate_seat:
                if len(legal) > 1:
                    snapshots.append(
                        _CounterfactualDecisionSnapshot(
                            game=copy.deepcopy(game),
                            legal_actions=legal,
                            actor_trace=actor_trace(),
                            initial_game=initial_game,
                        )
                    )
                action = candidate_policy.choose_turn_action(game, player_id)
                before_public_action_count = len(game.public_actions)
            else:
                action = opponents[player_id][1].choose_turn_action(game, player_id)
            _check_rollout_action(action, legal)
            game._apply_turn_action(player_id, action)
            if player_id == candidate_seat:
                record_actor_action(
                    "discard",
                    action,
                    before_public_action_count=before_public_action_count,
                )
            continue
        if game.phase == "response":
            actor_response: GameAction | None = None
            response_public_action_count = len(game.public_actions)
            for player_id, options in sorted(game.response_options.items()):
                legal = tuple(options)
                if player_id == candidate_seat:
                    if len(legal) > 1:
                        snapshots.append(
                        _CounterfactualDecisionSnapshot(
                            game=copy.deepcopy(game),
                            legal_actions=legal,
                            actor_trace=actor_trace(),
                            initial_game=initial_game,
                        )
                    )
                    action = candidate_policy.choose_response(game, player_id, list(legal))
                    actor_response = action
                else:
                    action = opponents[player_id][1].choose_response(
                        game, player_id, list(legal)
                    )
                _check_rollout_action(action, legal)
                game.response_choices[player_id] = action
            game._resolve_responses()
            if actor_response is not None:
                record_actor_action(
                    "response",
                    actor_response,
                    before_public_action_count=response_public_action_count,
                )
            continue
        raise RuntimeError(f"未知动作价值采样阶段：{game.phase}")
    return snapshots


def collect_counterfactual_action_value_trajectories(
    policy: Any,
    *,
    seed_count: int,
    profile: str = "classic",
    seed: int = 20266804,
    samples_per_hand: int = 1,
    rollouts_per_action: int = 1,
    response_sample_probability: float = 0.35,
    opponents: Sequence[tuple[str, Any]] = (),
    teacher_opponent_probability: float = 1.0,
    belief_resample: bool = False,
    rollout_batch_size: int = 1,
) -> tuple[list[TrainingTrajectory], ActionValueDatasetSummary]:
    """Build safe public observations with rollout-ranked legal actions.

    A base hand is played by one candidate seat against frozen opponents and
    the candidate rotates through all four seats for each physical wall.  A
    small number of *multi-action* candidate snapshots is then selected.  For
    every selected snapshot, each legal action is forced once (or more) on an
    in-memory clone and played to settlement by the frozen continuation
    policy.  The export stores only actor-visible state plus the resulting
    candidate score for each legal action.

    ``rollout_batch_size`` only groups independent private branches for agents
    that implement the optional ``scores_batch`` protocol.  Each branch keeps
    its own rule state and RNG, so changing this setting never relaxes legal
    action validation or exposes hidden state.

    The target is ``Q^pi(s, a)`` under the current candidate and frozen
    opponent mixture.  With ``belief_resample=True``, each rollout first
    redraws the unknown wall, opponent hands, flowers and concealed-kong faces
    from the actor-visible information set.  It is not an oracle: the model
    never receives the private simulation state.  Recollecting after policy
    updates is a conservative form of generalized policy iteration.
    """

    if seed_count <= 0:
        raise ValueError("seed_count 必须为正数")
    if samples_per_hand <= 0 or rollouts_per_action <= 0:
        raise ValueError("每局样本数和每动作 rollout 数必须为正数")
    if rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size 必须为正数")
    if not 0.0 <= response_sample_probability <= 1.0:
        raise ValueError("response_sample_probability 必须在 0 和 1 之间")
    if not 0.0 <= teacher_opponent_probability <= 1.0:
        raise ValueError("teacher_opponent_probability 必须在 0 和 1 之间")
    if teacher_opponent_probability < 1.0 and not opponents:
        raise ValueError("混入非 Teacher 对手时必须提供 opponents")

    rules = XiamenRules.from_profile(profile)
    teacher = HeuristicTeacherAgent()
    rng = random.Random(seed)
    batch_stats = _CounterfactualBatchStats()
    trajectories: list[TrainingTrajectory] = []
    action_counts: Counter[str] = Counter()
    branch_rollouts = 0
    belief_resampled_worlds = 0
    belief_resample_skipped = 0
    response_decisions = 0
    repeated_decisions = 0
    action_value_spans: list[float] = []
    action_value_stderrs: list[float] = []
    for seed_offset in range(seed_count):
        hand_seed = seed + seed_offset
        split_group_id = uuid4().hex
        for candidate_seat in range(rules.player_count):
            base_opponents = _sample_rollout_opponents(
                player_count=rules.player_count,
                candidate_seat=candidate_seat,
                teacher=teacher,
                opponents=opponents,
                teacher_opponent_probability=teacher_opponent_probability,
                rng=rng,
            )
            game = XiamenMahjongGame(
                seed=hand_seed,
                rules=rules,
                auto_advance=False,
                human_seat=-1,
            )
            snapshots = _run_candidate_base_hand(
                game,
                candidate_seat=candidate_seat,
                candidate_policy=policy,
                opponents=base_opponents,
            )
            response_snapshots = [
                snapshot for snapshot in snapshots if snapshot.game.phase == "response"
            ]
            selected_pool = (
                response_snapshots
                if response_snapshots and rng.random() < response_sample_probability
                else snapshots
            )
            if not selected_pool:
                continue
            selected_count = min(samples_per_hand, len(selected_pool))
            for selected_snapshot in rng.sample(selected_pool, selected_count):
                snapshot = selected_snapshot.game
                legal = selected_snapshot.legal_actions
                action_return_samples = [[] for _ in legal]
                usable_snapshot = True
                pending_batched_branches: list[
                    tuple[int, _CounterfactualRolloutJob]
                ] = []
                for _ in range(rollouts_per_action):
                    rollout_snapshot = snapshot
                    if belief_resample:
                        rollout_snapshot = _resample_private_world_for_actor(
                            snapshot, actor_seat=candidate_seat, rng=rng
                        )
                        if rollout_snapshot is None:
                            usable_snapshot = False
                            belief_resample_skipped += 1
                            break
                        if rollout_snapshot.phase == "discard":
                            resampled_legal = tuple(
                                _turn_actions(rollout_snapshot, candidate_seat)
                            )
                        else:
                            resampled_legal = tuple(
                                rollout_snapshot.response_options.get(
                                    candidate_seat, []
                                )
                            )
                        if resampled_legal != legal:
                            usable_snapshot = False
                            belief_resample_skipped += 1
                            break
                        belief_resampled_worlds += 1
                    rollout_opponents = _sample_rollout_opponents(
                        player_count=rules.player_count,
                        candidate_seat=candidate_seat,
                        teacher=teacher,
                        opponents=opponents,
                        teacher_opponent_probability=teacher_opponent_probability,
                        rng=rng,
                    )
                    if rollout_batch_size == 1:
                        for action_index, action in enumerate(legal):
                            branch = copy.deepcopy(rollout_snapshot)
                            action_return_samples[action_index].append(
                                _continue_counterfactual_rollout(
                                    branch,
                                    candidate_seat=candidate_seat,
                                    candidate_policy=policy,
                                    opponents=rollout_opponents,
                                    forced_action=action,
                                )
                            )
                            branch_rollouts += 1
                    else:
                        pending_batched_branches.extend(
                            (
                                action_index,
                                _CounterfactualRolloutJob(
                                    game=copy.deepcopy(rollout_snapshot),
                                    candidate_seat=candidate_seat,
                                    candidate_policy=policy,
                                    opponents=rollout_opponents,
                                    forced_action=action,
                                ),
                            )
                            for action_index, action in enumerate(legal)
                        )
                if not usable_snapshot:
                    continue
                if pending_batched_branches:
                    branch_scores = _continue_counterfactual_rollouts_batched(
                        [job for _action_index, job in pending_batched_branches],
                        rollout_batch_size=rollout_batch_size,
                        batch_stats=batch_stats,
                    )
                    for (action_index, _job), score in zip(
                        pending_batched_branches, branch_scores
                    ):
                        action_return_samples[action_index].append(score)
                    branch_rollouts += len(pending_batched_branches)
                action_values = tuple(
                    sum(samples) / len(samples) for samples in action_return_samples
                )
                action_value_errors = None
                if rollouts_per_action > 1:
                    action_value_errors = tuple(
                        math.sqrt(
                            sum((value - action_values[index]) ** 2 for value in samples)
                            / (len(samples) * (len(samples) - 1))
                        )
                        for index, samples in enumerate(action_return_samples)
                    )
                    repeated_decisions += 1
                    action_value_stderrs.extend(action_value_errors)
                chosen_index = max(
                    range(len(legal)), key=lambda index: (action_values[index], -index)
                )
                decision = TeacherDecision(
                    profile=rules.profile,
                    seed=hand_seed,
                    seat=candidate_seat,
                    state=_perspective_state(snapshot, candidate_seat),
                    legal_actions=legal,
                    chosen_index=chosen_index,
                    action_values=action_values,
                    action_value_stderrs=action_value_errors,
                )
                action_counts[decision.chosen_action.kind] += 1
                if snapshot.phase == "response":
                    response_decisions += 1
                action_value_spans.append(max(action_values) - min(action_values))
                trajectories.append(
                    TrainingTrajectory(
                        profile=rules.profile,
                        rules_version=rules.version,
                        rules=asdict(rules),
                        seed=hand_seed,
                        hand_number=snapshot.hand_number,
                        agent_profiles=tuple(
                            "counterfactual_candidate"
                            if seat == candidate_seat
                            else base_opponents[seat][0]
                            for seat in range(rules.player_count)
                        ),
                        source_metadata={
                            "collector": "counterfactual_action_value_rollout",
                            "candidate_seat": candidate_seat,
                            "wall_rotation": candidate_seat,
                            "samples_per_hand": samples_per_hand,
                            "rollouts_per_action": rollouts_per_action,
                            "rollout_batch_size": rollout_batch_size,
                            "return_semantics": "candidate_terminal_net_score_q_pi",
                            "continuation": "frozen_candidate_and_opponents",
                            "belief_resample": belief_resample,
                        },
                        decisions=(decision,),
                        outcome={
                            "winner": None,
                            "win_type": "synthetic_counterfactual_action_value",
                            "win_pattern": None,
                            "scores": [0] * rules.player_count,
                            "score_breakdown": None,
                            "turn_count": snapshot.turn_count,
                            "synthetic": True,
                        },
                        public_actions=tuple(
                            dict(action) for action in snapshot.public_actions
                        ),
                        split_group_id=split_group_id,
                    )
                )
    return trajectories, ActionValueDatasetSummary(
        hands=len(trajectories),
        decisions=len(trajectories),
        branch_rollouts=branch_rollouts,
        belief_resampled_worlds=belief_resampled_worlds,
        belief_resample_skipped=belief_resample_skipped,
        response_decisions=response_decisions,
        repeated_decisions=repeated_decisions,
        action_counts=dict(sorted(action_counts.items())),
        mean_action_value_span=(
            sum(action_value_spans) / len(action_value_spans)
            if action_value_spans
            else 0.0
        ),
        mean_action_value_stderr=(
            sum(action_value_stderrs) / len(action_value_stderrs)
            if action_value_stderrs
            else None
        ),
        rollout_batch_size=rollout_batch_size,
        batched_inference_calls=batch_stats.inference_calls,
        batched_inference_decisions=batch_stats.inference_decisions,
        max_batched_inference_decisions=batch_stats.max_inference_decisions,
    )


def collect_policy_episodes(
    policy: "NeuralRulePolicyModel",
    *,
    episodes: int,
    profile: str = "classic",
    seed: int = 20262004,
) -> list[PolicyEpisode]:
    """Sample legal candidate actions against three frozen Teacher seats.

    Candidate seats rotate for every initial shuffle.  Rewards are the
    candidate's settled net hand score, not any hidden-state heuristic.  This
    produces the first on-policy data source for the project; it deliberately
    has no access to the wall order or opponent concealed hands.
    """

    if episodes <= 0:
        raise ValueError("episodes 必须为正数")
    rules = XiamenRules.from_profile(profile)
    rng = random.Random(seed)
    baseline = HeuristicTeacherAgent()
    collected: list[PolicyEpisode] = []
    for episode_index in range(episodes):
        hand_seed = seed + episode_index // rules.player_count
        candidate_seat = episode_index % rules.player_count
        game = XiamenMahjongGame(
            seed=hand_seed,
            rules=rules,
            auto_advance=False,
            human_seat=-1,
        )
        steps: list[PolicyStep] = []
        safety = 0
        while game.phase != "over":
            safety += 1
            if safety > 600:
                raise RuntimeError("策略梯度对局超过安全步数")
            if game.phase == "discard":
                player_id = game.current_player
                legal = tuple(_turn_actions(game, player_id))
                if player_id == candidate_seat:
                    decision = _policy_decision(game, hand_seed, player_id, legal)
                    action_index = policy.sample_index(decision, rng)
                    steps.append(PolicyStep(decision, action_index))
                    action = legal[action_index]
                else:
                    action = baseline.choose_turn_action(game, player_id)
                game._apply_turn_action(player_id, action)
                continue
            if game.phase == "response":
                for player_id, options in sorted(game.response_options.items()):
                    legal = tuple(options)
                    if player_id == candidate_seat:
                        decision = _policy_decision(game, hand_seed, player_id, legal)
                        action_index = policy.sample_index(decision, rng)
                        steps.append(PolicyStep(decision, action_index))
                        game.response_choices[player_id] = legal[action_index]
                    else:
                        game.response_choices[player_id] = baseline.choose_response(
                            game, player_id, list(legal)
                        )
                game._resolve_responses()
                continue
            raise RuntimeError(f"未知策略梯度阶段：{game.phase}")
        collected.append(
            PolicyEpisode(
                seed=hand_seed,
                candidate_seat=candidate_seat,
                reward=game.players[candidate_seat].score,
                winner=game.winner,
                win_type=game.win_type,
                steps=tuple(steps),
            )
        )
    return collected


def collect_tour_curriculum(
    *, examples: int = 136, seed: int = 20260904
) -> list[TeacherDecision]:
    """Create engine-validated rare-action samples for 游金 and 双游.

    Random self-play rarely enters a tour state, so merely generating more
    ordinary games starves ``advance_tour`` of supervision.  Every sample here
    is built through the classic engine's own ``_turn_actions`` and Teacher;
    the expected label is never handwritten.
    """

    if examples <= 0:
        raise ValueError("examples 必须为正数")
    rules = XiamenRules.classic()
    decisions: list[TeacherDecision] = []
    for index in range(examples):
        hand_seed = seed + index
        game = XiamenMahjongGame(seed=hand_seed, rules=rules, auto_advance=False)
        player_id = 0
        gold_tile = index % BASE_TILE_COUNT
        game.gold_tile = gold_tile
        game.gold_indicator = gold_tile
        game.phase = "discard"
        game.current_player = player_id
        game.last_discard = None
        game.discarder = None
        game.gold_discard_lock_seat = None
        game.tour_state = {
            "owner": player_id,
            "level": 1 if index % 2 == 0 else 2,
            "locked": False,
            "remaining": [],
        }
        game.players[player_id].hand = _tour_ready_hand(gold_tile)
        game.players[player_id].melds = []
        game.players[player_id].discards = []
        legal = tuple(_turn_actions(game, player_id))
        chosen = game.teacher.choose_turn_action(game, player_id)
        if chosen.kind != "advance_tour":
            raise RuntimeError("游金课程状态没有得到 Teacher 的升级动作")
        decisions.append(_decision(game, hand_seed, player_id, legal, chosen))
    return decisions


def collect_tour_trajectories(
    *, examples: int = 136, seed: int = 20260904
) -> list[TrainingTrajectory]:
    """Wrap engine-validated 游金 decisions in the common trajectory schema."""

    decisions = collect_tour_curriculum(examples=examples, seed=seed)
    rules = XiamenRules.classic()
    return [
        TrainingTrajectory(
            profile=rules.profile,
            rules_version=rules.version,
            rules=asdict(rules),
            seed=decision.seed,
            hand_number=1,
            agent_profiles=("tour_curriculum",) * rules.player_count,
            source_metadata={
                "collector": "engine_validated_tour_curriculum",
                "synthetic": True,
            },
            decisions=(decision,),
            outcome={
                "winner": None,
                "win_type": "synthetic_tour_curriculum",
                "win_pattern": None,
                "scores": [0] * rules.player_count,
                "score_breakdown": None,
                "turn_count": int(decision.state.get("turn_count", 0)),
                "synthetic": True,
            },
            public_actions=tuple(
                dict(action) for action in decision.state.get("recent_public_actions", [])
            ),
        )
        for decision in decisions
    ]


def write_jsonl(decisions: Iterable[TeacherDecision], path: str | Path) -> int:
    """Export a portable Teacher data set without serializing hidden state."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision.payload(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[TeacherDecision]:
    decisions: list[TeacherDecision] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                decisions.append(TeacherDecision.from_payload(payload))
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                raise ValueError(f"第 {line_number} 行 Teacher 轨迹无效：{error}") from error
    return decisions


def write_trajectory_jsonl(
    trajectories: Iterable[TrainingTrajectory],
    path: str | Path,
    *,
    include_replay_metadata: bool = False,
) -> int:
    """Write one complete training hand per JSONL line.

    The safe default excludes all deal/behaviour seeds.  Use a distinct local
    replay index when exact deal reproduction is needed.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(
                json.dumps(
                    trajectory.payload(
                        include_replay_metadata=include_replay_metadata
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")
            count += 1
    return count


def write_trajectory_replay_index(
    trajectories: Iterable[TrainingTrajectory], path: str | Path
) -> int:
    """Write the opt-in private mapping required for deterministic replay.

    This file contains deal/behaviour seeds and must never be supplied to a
    model trainer or committed alongside a distributable training corpus.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            if trajectory.seed is None:
                raise ValueError("安全导出的轨迹不含可建立 replay 索引的种子")
            handle.write(
                json.dumps(
                    {
                        "trajectory_id": trajectory.trajectory_id,
                        "split_group_id": trajectory.split_group_id,
                        "profile": trajectory.profile,
                        "rules_version": trajectory.rules_version,
                        "seed": trajectory.seed,
                        "hand_number": trajectory.hand_number,
                        "source_metadata": trajectory.source_metadata,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            handle.write("\n")
            count += 1
    return count


def read_trajectory_jsonl(path: str | Path) -> list[TrainingTrajectory]:
    """Read and validate a versioned trajectory JSONL data set."""

    trajectories: list[TrainingTrajectory] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                trajectories.append(TrainingTrajectory.from_payload(json.loads(line)))
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                raise ValueError(f"第 {line_number} 行训练轨迹无效：{error}") from error
    return trajectories


def split_trajectories_by_hand(
    trajectories: Iterable[TrainingTrajectory],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    split_salt: str = "xiamen-trajectory-v2",
) -> dict[str, list[TrainingTrajectory]]:
    """Deterministically split complete hands, never individual decisions."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction 必须在 0 和 1 之间")
    if not 0.0 < validation_fraction < 1.0 - train_fraction:
        raise ValueError("validation_fraction 必须为正且须保留测试集")
    train_cutoff = int(train_fraction * 10_000)
    validation_cutoff = int((train_fraction + validation_fraction) * 10_000)
    partitions: dict[str, list[TrainingTrajectory]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    seen_trajectory_ids: set[str] = set()
    for trajectory in trajectories:
        if trajectory.trajectory_id in seen_trajectory_ids:
            raise ValueError("训练轨迹包含重复 trajectory_id")
        seen_trajectory_ids.add(trajectory.trajectory_id)
        identity = trajectory.split_group_id or (
            str(trajectory.seed)
            if trajectory.seed is not None
            else trajectory.trajectory_id
        )
        digest = hashlib.blake2b(
            f"{split_salt}|{trajectory.profile}|{trajectory.rules_version}|"
            f"{identity}|{trajectory.hand_number}".encode("utf-8"),
            digest_size=8,
        ).digest()
        bucket = int.from_bytes(digest, "big") % 10_000
        partition = (
            "train"
            if bucket < train_cutoff
            else "validation"
            if bucket < validation_cutoff
            else "test"
        )
        partitions[partition].append(trajectory)
    return partitions


def trajectory_manifest(trajectories: Iterable[TrainingTrajectory]) -> dict[str, Any]:
    """Summarize coverage and outcomes for data-quality gates."""

    records = list(trajectories)
    action_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    rules_versions: Counter[str] = Counter()
    agent_profiles: Counter[str] = Counter()
    score_values: list[int] = []
    action_value_decisions = 0
    action_value_spans: list[float] = []
    action_value_stderrs: list[float] = []
    for trajectory in records:
        rules_versions[trajectory.rules_version] += 1
        agent_profiles.update(trajectory.agent_profiles)
        outcome_counts[str(trajectory.outcome.get("win_type"))] += 1
        score_values.extend(int(value) for value in trajectory.outcome["scores"])
        for decision in trajectory.decisions:
            phase_counts[str(decision.state["phase"])] += 1
            action_counts[decision.chosen_action.kind] += 1
            if decision.action_values is not None:
                action_value_decisions += 1
                action_value_spans.append(
                    max(decision.action_values) - min(decision.action_values)
                )
            if decision.action_value_stderrs is not None:
                action_value_stderrs.extend(decision.action_value_stderrs)
    return {
        "version": TRAJECTORY_DATASET_VERSION,
        "hands": len(records),
        "decisions": sum(len(trajectory.decisions) for trajectory in records),
        "rules_versions": dict(sorted(rules_versions.items())),
        "agent_profiles": dict(sorted(agent_profiles.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "score_mean": sum(score_values) / len(score_values) if score_values else 0.0,
        "score_min": min(score_values) if score_values else 0,
        "score_max": max(score_values) if score_values else 0,
        "action_value_decisions": action_value_decisions,
        "mean_action_value_span": (
            sum(action_value_spans) / len(action_value_spans)
            if action_value_spans
            else None
        ),
        "action_value_stderr_observations": len(action_value_stderrs),
        "mean_action_value_stderr": (
            sum(action_value_stderrs) / len(action_value_stderrs)
            if action_value_stderrs
            else None
        ),
    }


class RulePolicyModel:
    """A legal-action softmax ranker trained by rule-Teacher imitation."""

    def __init__(self, weights: Sequence[float] | None = None):
        if weights is None:
            self.weights = [0.0] * FEATURE_DIM
        else:
            if len(weights) != FEATURE_DIM:
                raise ValueError("策略权重维度与当前特征定义不匹配")
            self.weights = [float(weight) for weight in weights]

    def scores(self, decision: TeacherDecision) -> list[float]:
        return [self._score(_action_features(decision.state, action)) for action in decision.legal_actions]

    def predict_index(self, decision: TeacherDecision) -> int:
        scores = self.scores(decision)
        return max(range(len(scores)), key=lambda index: (scores[index], -index))

    def sample_index(self, decision: TeacherDecision, rng: random.Random) -> int:
        """Sample from the masked legal-action policy for exploration."""

        probabilities = _softmax(self.scores(decision))
        threshold = rng.random()
        total = 0.0
        for index, probability in enumerate(probabilities):
            total += probability
            if threshold < total:
                return index
        return len(probabilities) - 1

    def predict_action(self, decision: TeacherDecision) -> GameAction:
        return decision.legal_actions[self.predict_index(decision)]

    def choose_turn_action(self, game: XiamenMahjongGame, player_id: int) -> GameAction:
        legal = tuple(_turn_actions(game, player_id))
        return self.predict_action(_decision(game, game.seed or 0, player_id, legal, legal[0]))

    def choose_response(
        self, game: XiamenMahjongGame, player_id: int, options: Sequence[GameAction]
    ) -> GameAction:
        legal = tuple(options)
        return self.predict_action(_decision(game, game.seed or 0, player_id, legal, legal[0]))

    def fit(
        self,
        decisions: Sequence[TeacherDecision],
        *,
        epochs: int = 8,
        learning_rate: float = 0.035,
        l2: float = 0.00001,
        seed: int = 20260804,
    ) -> list[dict[str, float]]:
        if not decisions:
            raise ValueError("训练数据为空")
        if epochs <= 0 or learning_rate <= 0:
            raise ValueError("epochs 和 learning_rate 必须为正数")
        history: list[dict[str, float]] = []
        order = list(range(len(decisions)))
        rng = random.Random(seed)
        for epoch in range(1, epochs + 1):
            rng.shuffle(order)
            for index in order:
                decision = decisions[index]
                vectors = [_action_features(decision.state, action) for action in decision.legal_actions]
                probabilities = _softmax([self._score(vector) for vector in vectors])
                for action_index, vector in enumerate(vectors):
                    gradient = probabilities[action_index] - (
                        1.0 if action_index == decision.chosen_index else 0.0
                    )
                    if not gradient:
                        continue
                    for feature_index, value in vector:
                        self.weights[feature_index] -= learning_rate * (
                            gradient * value + l2 * self.weights[feature_index]
                        )
            metrics = self.evaluate(decisions)
            history.append({"epoch": float(epoch), **metrics})
        return history

    def evaluate(self, decisions: Sequence[TeacherDecision]) -> dict[str, float]:
        if not decisions:
            return {"accuracy": 0.0, "loss": 0.0, "decisions": 0.0}
        correct = 0
        total_loss = 0.0
        for decision in decisions:
            scores = self.scores(decision)
            probabilities = _softmax(scores)
            correct += self.predict_index(decision) == decision.chosen_index
            total_loss -= math.log(max(probabilities[decision.chosen_index], 1e-12))
        count = len(decisions)
        return {
            "accuracy": correct / count,
            "loss": total_loss / count,
            "decisions": float(count),
        }

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": DATASET_VERSION,
            "model": "legal_action_linear_softmax",
            "feature_dim": FEATURE_DIM,
            "weights": self.weights,
            "metadata": metadata or {},
        }
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RulePolicyModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != DATASET_VERSION:
            raise ValueError("不支持的策略检查点版本")
        if payload.get("model") != "legal_action_linear_softmax":
            raise ValueError("不支持的策略检查点类型")
        return cls(payload["weights"])

    def _score(self, vector: Sequence[tuple[int, float]]) -> float:
        return sum(self.weights[index] * value for index, value in vector)


class NeuralRulePolicyModel:
    """Tiny ReLU MLP that ranks only rules-engine legal actions.

    The model intentionally consumes a compact state/action representation
    rather than the large sparse linear feature map.  It is dependency-free
    so the baseline can train in the same Python-only environment as the
    playable rules engine.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 12,
        seed: int = 20260804,
        feature_version: int = DEFAULT_NEURAL_FEATURE_VERSION,
        input_weights: Sequence[Sequence[float]] | None = None,
        hidden_bias: Sequence[float] | None = None,
        output_weights: Sequence[float] | None = None,
        direct_weights: Sequence[float] | None = None,
        output_bias: float = 0.0,
    ):
        if hidden_size <= 0:
            raise ValueError("hidden_size 必须为正数")
        if feature_version not in NEURAL_FEATURE_DIMS:
            raise ValueError("不支持的 MLP 特征版本")
        self.hidden_size = hidden_size
        self.feature_version = feature_version
        self.feature_dim = NEURAL_FEATURE_DIMS[feature_version]
        if input_weights is None:
            rng = random.Random(seed)
            scale = 1.0 / math.sqrt(self.feature_dim)
            self.input_weights = [
                [rng.uniform(-scale, scale) for _ in range(self.feature_dim)]
                for _ in range(hidden_size)
            ]
            self.hidden_bias = [0.0] * hidden_size
            output_scale = 1.0 / math.sqrt(hidden_size)
            self.output_weights = [
                rng.uniform(-output_scale, output_scale) for _ in range(hidden_size)
            ]
            # A wide path lets compact, rule-derived numerical features (such
            # as candidate wait count) contribute directly, while the ReLU
            # path continues to learn interactions between tiles and actions.
            self.direct_weights = [0.0] * self.feature_dim
            self.output_bias = 0.0
            return
        if (
            len(input_weights) != hidden_size
            or any(len(row) != self.feature_dim for row in input_weights)
            or hidden_bias is None
            or len(hidden_bias) != hidden_size
            or output_weights is None
            or len(output_weights) != hidden_size
            or (direct_weights is not None and len(direct_weights) != self.feature_dim)
        ):
            raise ValueError("MLP 检查点维度与当前特征定义不匹配")
        self.input_weights = [[float(value) for value in row] for row in input_weights]
        self.hidden_bias = [float(value) for value in hidden_bias]
        self.output_weights = [float(value) for value in output_weights]
        self.direct_weights = (
            [0.0] * self.feature_dim
            if direct_weights is None
            else [float(value) for value in direct_weights]
        )
        self.output_bias = float(output_bias)

    def scores(self, decision: TeacherDecision) -> list[float]:
        return [self._forward(features)[2] for features in self._feature_vectors(decision)]

    def predict_index(self, decision: TeacherDecision) -> int:
        scores = self.scores(decision)
        return max(range(len(scores)), key=lambda index: (scores[index], -index))

    def sample_index(self, decision: TeacherDecision, rng: random.Random) -> int:
        """Sample from the masked legal-action policy for exploration."""

        probabilities = _softmax(self.scores(decision))
        threshold = rng.random()
        total = 0.0
        for index, probability in enumerate(probabilities):
            total += probability
            if threshold < total:
                return index
        return len(probabilities) - 1

    def predict_action(self, decision: TeacherDecision) -> GameAction:
        return decision.legal_actions[self.predict_index(decision)]

    def choose_turn_action(self, game: XiamenMahjongGame, player_id: int) -> GameAction:
        legal = tuple(_turn_actions(game, player_id))
        return self.predict_action(_decision(game, game.seed or 0, player_id, legal, legal[0]))

    def choose_response(
        self, game: XiamenMahjongGame, player_id: int, options: Sequence[GameAction]
    ) -> GameAction:
        legal = tuple(options)
        return self.predict_action(_decision(game, game.seed or 0, player_id, legal, legal[0]))

    def fit(
        self,
        decisions: Sequence[TeacherDecision],
        *,
        epochs: int = 6,
        learning_rate: float = 0.012,
        l2: float = 0.00001,
        seed: int = 20260804,
        class_weights: Mapping[str, float] | None = None,
        sample_weights: Sequence[float] | None = None,
    ) -> list[dict[str, float]]:
        if not decisions:
            raise ValueError("训练数据为空")
        if epochs <= 0 or learning_rate <= 0:
            raise ValueError("epochs 和 learning_rate 必须为正数")
        if sample_weights is not None and len(sample_weights) != len(decisions):
            raise ValueError("sample_weights 数量必须与训练决策一致")
        # Version 2 contains deterministic candidate look-ahead features. They
        # do not change while fitting, so materialize them once instead of
        # repeatedly solving the same ready-hand calculation every epoch.
        prepared = [self._feature_vectors(decision) for decision in decisions]
        order = list(range(len(decisions)))
        rng = random.Random(seed)
        history: list[dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            rng.shuffle(order)
            for decision_index in order:
                decision = decisions[decision_index]
                candidates = [self._forward(features) for features in prepared[decision_index]]
                probabilities = _softmax([candidate[2] for candidate in candidates])
                example_weight = (
                    float(class_weights.get(decision.chosen_action.kind, 1.0))
                    if class_weights
                    else 1.0
                )
                if sample_weights is not None:
                    example_weight *= float(sample_weights[decision_index])
                if example_weight <= 0:
                    raise ValueError("动作类别与样本权重必须为正数")
                for action_index, (features, hidden, _score) in enumerate(candidates):
                    gradient = example_weight * (
                        probabilities[action_index]
                        - (1.0 if action_index == decision.chosen_index else 0.0)
                    )
                    if not gradient:
                        continue
                    previous_output = list(self.output_weights)
                    for feature_index, value in enumerate(features):
                        self.direct_weights[feature_index] -= learning_rate * (
                            gradient * value + l2 * self.direct_weights[feature_index]
                        )
                    for hidden_index, hidden_value in enumerate(hidden):
                        self.output_weights[hidden_index] -= learning_rate * (
                            gradient * hidden_value + l2 * self.output_weights[hidden_index]
                        )
                    self.output_bias -= learning_rate * gradient
                    for hidden_index, hidden_value in enumerate(hidden):
                        if hidden_value <= 0.0:
                            continue
                        hidden_gradient = gradient * previous_output[hidden_index]
                        self.hidden_bias[hidden_index] -= learning_rate * hidden_gradient
                        row = self.input_weights[hidden_index]
                        for feature_index, value in enumerate(features):
                            row[feature_index] -= learning_rate * (
                                hidden_gradient * value + l2 * row[feature_index]
                            )
            history.append(
                {"epoch": float(epoch), **self._evaluate_prepared(decisions, prepared)}
            )
        return history

    def evaluate(self, decisions: Sequence[TeacherDecision]) -> dict[str, float]:
        return self._evaluate_prepared(
            decisions, [self._feature_vectors(decision) for decision in decisions]
        )

    def reinforce(
        self,
        episodes: Sequence[PolicyEpisode],
        *,
        learning_rate: float = 0.0005,
        reward_scale: float = 40.0,
        l2: float = 0.00001,
        advantages: Sequence[Sequence[float]] | None = None,
    ) -> dict[str, float]:
        """Apply one REINFORCE update from settled candidate hand scores.

        ``advantages`` can supply a state-value baseline for every sampled
        step.  When omitted, the method falls back to a batch-normalized final
        return.  In both cases the update remains on-policy: sampled actions,
        their legal masks, and final rewards originate from the same frozen
        policy batch.
        """

        if not episodes:
            raise ValueError("策略梯度 episode 不能为空")
        if learning_rate <= 0 or reward_scale <= 0:
            raise ValueError("learning_rate 和 reward_scale 必须为正数")
        normalized_rewards = [episode.reward / reward_scale for episode in episodes]
        mean_reward = sum(normalized_rewards) / len(normalized_rewards)
        reward_variance = sum(
            (reward - mean_reward) ** 2 for reward in normalized_rewards
        ) / max(1, len(normalized_rewards) - 1)
        if advantages is None:
            raw_advantages = [
                [reward - mean_reward for _step in episode.steps]
                for episode, reward in zip(episodes, normalized_rewards)
            ]
        else:
            if len(advantages) != len(episodes):
                raise ValueError("advantage 的 episode 数量不匹配")
            raw_advantages = []
            for episode, values in zip(episodes, advantages):
                if len(values) != len(episode.steps):
                    raise ValueError("advantage 的决策数量不匹配")
                raw_advantages.append([float(value) for value in values])
        flat_advantages = [value for values in raw_advantages for value in values]
        advantage_mean = (
            sum(flat_advantages) / len(flat_advantages) if flat_advantages else 0.0
        )
        advantage_variance = (
            sum((value - advantage_mean) ** 2 for value in flat_advantages)
            / max(1, len(flat_advantages) - 1)
            if flat_advantages
            else 0.0
        )
        advantage_std = math.sqrt(advantage_variance)
        decisions = 0
        for episode, values in zip(episodes, raw_advantages):
            for step, raw_advantage in zip(episode.steps, values):
                advantage = (raw_advantage - advantage_mean) / max(advantage_std, 1.0)
                vectors = self._feature_vectors(step.decision)
                candidates = [self._forward(features) for features in vectors]
                probabilities = _softmax([candidate[2] for candidate in candidates])
                for action_index, (features, hidden, _score) in enumerate(candidates):
                    gradient = advantage * (
                        probabilities[action_index]
                        - (1.0 if action_index == step.action_index else 0.0)
                    )
                    if not gradient:
                        continue
                    previous_output = list(self.output_weights)
                    for feature_index, value in enumerate(features):
                        self.direct_weights[feature_index] -= learning_rate * (
                            gradient * value + l2 * self.direct_weights[feature_index]
                        )
                    for hidden_index, hidden_value in enumerate(hidden):
                        self.output_weights[hidden_index] -= learning_rate * (
                            gradient * hidden_value + l2 * self.output_weights[hidden_index]
                        )
                    self.output_bias -= learning_rate * gradient
                    for hidden_index, hidden_value in enumerate(hidden):
                        if hidden_value <= 0.0:
                            continue
                        hidden_gradient = gradient * previous_output[hidden_index]
                        self.hidden_bias[hidden_index] -= learning_rate * hidden_gradient
                        row = self.input_weights[hidden_index]
                        for feature_index, value in enumerate(features):
                            row[feature_index] -= learning_rate * (
                                hidden_gradient * value + l2 * row[feature_index]
                            )
                decisions += 1
        wins = sum(episode.winner == episode.candidate_seat for episode in episodes)
        return {
            "episodes": float(len(episodes)),
            "decisions": float(decisions),
            "mean_reward": sum(episode.reward for episode in episodes) / len(episodes),
            "reward_stderr": math.sqrt(reward_variance / len(episodes)) * reward_scale,
            "win_rate": wins / len(episodes),
            "advantage_mean": advantage_mean,
            "advantage_std": advantage_std,
        }

    def _feature_vectors(self, decision: TeacherDecision) -> list[list[float]]:
        return [
            _dense_action_features(
                decision.state, action, feature_version=self.feature_version
            )
            for action in decision.legal_actions
        ]

    def _evaluate_prepared(
        self,
        decisions: Sequence[TeacherDecision],
        prepared: Sequence[Sequence[Sequence[float]]],
    ) -> dict[str, float]:
        if not decisions:
            return {"accuracy": 0.0, "loss": 0.0, "decisions": 0.0}
        correct = 0
        total_loss = 0.0
        for decision, vectors in zip(decisions, prepared):
            scores = [self._forward(features)[2] for features in vectors]
            probabilities = _softmax(scores)
            predicted_index = max(range(len(scores)), key=lambda index: (scores[index], -index))
            correct += predicted_index == decision.chosen_index
            total_loss -= math.log(max(probabilities[decision.chosen_index], 1e-12))
        count = len(decisions)
        return {
            "accuracy": correct / count,
            "loss": total_loss / count,
            "decisions": float(count),
        }

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": DATASET_VERSION,
            "model": "legal_action_relu_mlp",
            "feature_version": self.feature_version,
            "feature_dim": self.feature_dim,
            "hidden_size": self.hidden_size,
            "input_weights": self.input_weights,
            "hidden_bias": self.hidden_bias,
            "output_weights": self.output_weights,
            "direct_weights": self.direct_weights,
            "output_bias": self.output_bias,
            "metadata": metadata or {},
        }
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NeuralRulePolicyModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != DATASET_VERSION:
            raise ValueError("不支持的策略检查点版本")
        if payload.get("model") != "legal_action_relu_mlp":
            raise ValueError("不支持的策略检查点类型")
        feature_version = int(payload.get("feature_version", 1))
        expected_feature_dim = NEURAL_FEATURE_DIMS.get(feature_version)
        if expected_feature_dim is None or payload.get("feature_dim") != expected_feature_dim:
            raise ValueError("MLP 检查点特征版本或维度不匹配")
        return cls(
            hidden_size=int(payload["hidden_size"]),
            feature_version=feature_version,
            input_weights=payload["input_weights"],
            hidden_bias=payload["hidden_bias"],
            output_weights=payload["output_weights"],
            direct_weights=payload.get("direct_weights"),
            output_bias=float(payload["output_bias"]),
        )

    def _forward(self, features: Sequence[float]) -> tuple[list[float], list[float], float]:
        pre_activation = [
            bias + sum(weight * value for weight, value in zip(row, features))
            for row, bias in zip(self.input_weights, self.hidden_bias)
        ]
        hidden = [max(value, 0.0) for value in pre_activation]
        score = (
            self.output_bias
            + sum(weight * value for weight, value in zip(self.output_weights, hidden))
            + sum(weight * value for weight, value in zip(self.direct_weights, features))
        )
        return list(features), hidden, score


class StateValueBaseline:
    """Small action-independent critic for variance-reduced policy gradients.

    The critic sees the same actor-visible state as the policy.  It averages
    feature vectors across *all* engine-legal actions, so it cannot receive the
    sampled action as an input.  Its scalar target is the candidate's settled
    hand score divided by the configured reward scale.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        weights: Sequence[float] | None = None,
        bias: float = 0.0,
    ):
        if feature_dim <= 0:
            raise ValueError("价值基线特征维度必须为正数")
        if weights is not None and len(weights) != feature_dim:
            raise ValueError("价值基线检查点维度不匹配")
        self.feature_dim = feature_dim
        self.weights = [0.0] * feature_dim if weights is None else [float(value) for value in weights]
        self.bias = float(bias)

    def predict(self, policy: NeuralRulePolicyModel, decision: TeacherDecision) -> float:
        features = self._features(policy, decision)
        return self.bias + sum(weight * value for weight, value in zip(self.weights, features))

    def advantages(
        self,
        policy: NeuralRulePolicyModel,
        episodes: Sequence[PolicyEpisode],
        *,
        reward_scale: float,
    ) -> list[list[float]]:
        if reward_scale <= 0:
            raise ValueError("reward_scale 必须为正数")
        return [
            [
                episode.reward / reward_scale - self.predict(policy, step.decision)
                for step in episode.steps
            ]
            for episode in episodes
        ]

    def fit(
        self,
        policy: NeuralRulePolicyModel,
        episodes: Sequence[PolicyEpisode],
        *,
        reward_scale: float,
        epochs: int = 4,
        learning_rate: float = 0.02,
        l2: float = 0.00001,
    ) -> dict[str, float]:
        """Fit a linear Monte-Carlo value baseline to sampled policy states."""

        if reward_scale <= 0 or epochs <= 0 or learning_rate <= 0:
            raise ValueError("reward_scale、epochs 和 learning_rate 必须为正数")
        samples = [
            (self._features(policy, step.decision), episode.reward / reward_scale)
            for episode in episodes
            for step in episode.steps
        ]
        if not samples:
            return {"value_samples": 0.0, "value_mse_before": 0.0, "value_mse_after": 0.0}
        mse_before = self._mse(samples)
        for _epoch in range(epochs):
            gradient_weights = [0.0] * self.feature_dim
            gradient_bias = 0.0
            for features, target in samples:
                error = self.bias + sum(
                    weight * value for weight, value in zip(self.weights, features)
                ) - target
                gradient_bias += error
                for index, value in enumerate(features):
                    gradient_weights[index] += error * value
            scale = 1.0 / len(samples)
            self.bias -= learning_rate * gradient_bias * scale
            for index, gradient in enumerate(gradient_weights):
                self.weights[index] -= learning_rate * (
                    gradient * scale + l2 * self.weights[index]
                )
        return {
            "value_samples": float(len(samples)),
            "value_mse_before": mse_before,
            "value_mse_after": self._mse(samples),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "version": DATASET_VERSION,
                    "model": "state_value_linear",
                    "feature_dim": self.feature_dim,
                    "weights": self.weights,
                    "bias": self.bias,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "StateValueBaseline":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != DATASET_VERSION or payload.get("model") != "state_value_linear":
            raise ValueError("不支持的价值基线检查点")
        return cls(
            feature_dim=int(payload["feature_dim"]),
            weights=payload["weights"],
            bias=float(payload["bias"]),
        )

    def _features(
        self, policy: NeuralRulePolicyModel, decision: TeacherDecision
    ) -> list[float]:
        candidates = policy._feature_vectors(decision)
        if not candidates:
            raise RuntimeError("价值基线没有可用的合法候选动作")
        if len(candidates[0]) != self.feature_dim:
            raise ValueError("策略与价值基线特征维度不匹配")
        return [
            sum(features[index] for features in candidates) / len(candidates)
            for index in range(self.feature_dim)
        ]

    def _mse(self, samples: Sequence[tuple[Sequence[float], float]]) -> float:
        return sum(
            (
                self.bias
                + sum(weight * value for weight, value in zip(self.weights, features))
                - target
            )
            ** 2
            for features, target in samples
        ) / len(samples)


def _turn_actions(game: XiamenMahjongGame, player_id: int) -> list[GameAction]:
    """Convert the engine's turn payloads back into exact action identities."""

    actions: list[GameAction] = []
    for payload in game._turn_actions(player_id):
        actions.append(
            GameAction(
                str(payload["kind"]),
                payload.get("tile"),
                tuple(payload.get("tiles", [])),
            )
        )
    if not actions:
        raise RuntimeError("规则引擎没有提供可训练的合法动作")
    return actions


def _tour_ready_hand(gold_tile: int) -> list[int]:
    """Return five natural triplets plus two gold tiles (17 tiles total)."""

    triplet_tiles = [tile for tile in range(BASE_TILE_COUNT) if tile != gold_tile][:5]
    hand = [tile for tile in triplet_tiles for _ in range(3)]
    hand.extend([gold_tile, gold_tile])
    return sorted(hand)


def _decision(
    game: XiamenMahjongGame,
    seed: int,
    player_id: int,
    legal: Sequence[GameAction],
    chosen: GameAction,
) -> TeacherDecision:
    legal_actions = tuple(legal)
    try:
        chosen_index = legal_actions.index(chosen)
    except ValueError as error:
        raise RuntimeError("Teacher 选择了规则引擎未列出的动作") from error
    return TeacherDecision(
        profile=game.rules.profile,
        seed=seed,
        seat=player_id,
        state=_perspective_state(game, player_id),
        legal_actions=legal_actions,
        chosen_index=chosen_index,
    )


def _policy_decision(
    game: XiamenMahjongGame,
    seed: int,
    player_id: int,
    legal: Sequence[GameAction],
) -> TeacherDecision:
    """Build a policy-gradient observation without pretending it has a label."""

    legal_actions = tuple(legal)
    if not legal_actions:
        raise RuntimeError("规则引擎没有提供可采样的动作")
    return TeacherDecision(
        profile=game.rules.profile,
        seed=seed,
        seat=player_id,
        state=_perspective_state(game, player_id),
        legal_actions=legal_actions,
        # This field is unused by policy gradient; it keeps the reusable
        # decision container valid and intentionally points at no privileged
        # Teacher target.
        chosen_index=0,
    )


def _perspective_state(game: XiamenMahjongGame, player_id: int) -> dict[str, Any]:
    player = game.players[player_id]
    river_counts = Counter(
        tile
        for other in game.players
        for tile in other.discards
        if is_base_tile(tile)
    )
    meld_counts = Counter(
        int(meld.get("value", meld["tiles"][0]))
        for meld in player.melds
        if is_base_tile(int(meld.get("value", meld["tiles"][0])))
    )
    tour_level = 0
    if game.tour_state and game.tour_state["owner"] == player_id:
        tour_level = int(game.tour_state["level"])
    player_count = game.rules.player_count

    def relative_seat(seat: int | None) -> int | None:
        return None if seat is None else (seat - player_id) % player_count

    public_players = []
    for offset in range(player_count):
        seat = (player_id + offset) % player_count
        observed = game.players[seat]
        melds = []
        for meld in observed.melds:
            concealed = meld["kind"] == "an_kan" and seat != player_id
            melds.append(
                {
                    "kind": meld["kind"],
                    "tiles": [] if concealed else list(meld["tiles"]),
                    "tile_count": len(meld["tiles"]),
                    "concealed": concealed,
                }
            )
        public_players.append(
            {
                "relative_seat": offset,
                "discards": list(observed.discards),
                "melds": melds,
                "flowers": len(observed.flowers),
                "hand_count": len(observed.hand),
                "score_delta": observed.score - player.score,
                "opening_wait": seat in game.opening_wait_seats,
                "gold_locked": game.gold_discard_lock_seat == seat,
            }
        )
    recent_public_actions = []
    for action in game.public_actions[-24:]:
        public_action = dict(action)
        if "seat" in public_action:
            public_action["relative_seat"] = relative_seat(int(public_action.pop("seat")))
        recent_public_actions.append(public_action)
    public_tour = None
    if game.tour_state:
        public_tour = {
            "owner_relative": relative_seat(int(game.tour_state["owner"])),
            "level": int(game.tour_state["level"]),
            "locked": bool(game.tour_state["locked"]),
            "remaining_relative": [
                relative_seat(int(seat)) for seat in game.tour_state["remaining"]
            ],
        }
    return {
        "rules_profile": game.rules.profile,
        "rules_version": game.rules.version,
        "phase": game.phase,
        "hand": list(player.hand),
        "drawn_tile": game.last_drawn_tiles[player_id]
        if game.last_drawn_tiles[player_id] in player.hand
        else None,
        "river_counts": [river_counts[tile] for tile in range(BASE_TILE_COUNT)],
        "meld_counts": [meld_counts[tile] for tile in range(BASE_TILE_COUNT)],
        "gold_tile": game.gold_tile,
        "gold_indicator": game.gold_indicator,
        "last_discard": game.last_discard,
        "discarder_relative": relative_seat(game.discarder),
        "latest_discard_seat_relative": relative_seat(game.latest_discard_seat),
        "wall_remaining": len(game.wall),
        "turn_count": game.turn_count,
        "hand_number": game.hand_number,
        "flowers": len(player.flowers),
        "tour_level": tour_level,
        "tour": public_tour,
        "gold_locked": game.gold_discard_lock_seat == player_id,
        "is_dealer": game.dealer == player_id,
        "dealer_relative": relative_seat(game.dealer),
        "current_player_relative": relative_seat(game.current_player),
        "dealer_streak": game.dealer_streak,
        "opening_wait_relative_seats": [
            relative_seat(seat) for seat in sorted(game.opening_wait_seats)
        ],
        "public_players": public_players,
        "recent_public_actions": recent_public_actions,
    }


def _action_features(state: dict[str, Any], action: GameAction) -> list[tuple[int, float]]:
    kind_index = ACTION_KIND_INDEX.get(action.kind)
    if kind_index is None:
        raise ValueError(f"不支持的训练动作：{action.kind}")
    target = action.tile if action.tile is not None and is_base_tile(action.tile) else NO_TILE
    hand_counts = Counter(tile for tile in state["hand"] if is_base_tile(tile))
    river_counts = state["river_counts"]
    meld_counts = state["meld_counts"]
    values: list[tuple[int, float]] = [
        (_BIAS, 1.0),
        (_KIND + kind_index, 1.0),
        (_TARGET + target, 1.0),
        (_KIND_TARGET + kind_index * TILE_SLOTS + target, 1.0),
    ]
    for tile, count in hand_counts.items():
        values.append((_TARGET_HAND + target * BASE_TILE_COUNT + tile, float(count)))
    for tile, count in enumerate(river_counts):
        if count:
            values.append((_TARGET_RIVER + target * BASE_TILE_COUNT + tile, float(count)))
    for tile, count in enumerate(meld_counts):
        if count:
            values.append((_TARGET_MELD + target * BASE_TILE_COUNT + tile, float(count)))
    for tile in set(action.tiles):
        if is_base_tile(tile):
            values.append((_CONSUMED + tile, float(action.tiles.count(tile))))
    tour_level = min(max(int(state.get("tour_level", 0)), 0), 3)
    values.append((_KIND_TOUR + kind_index * 4 + tour_level, 1.0))
    wall_bucket = min(max(int(state["wall_remaining"]) // 18, 0), 7)
    values.append((_KIND_WALL + kind_index * 8 + wall_bucket, 1.0))
    phase_index = 0 if state["phase"] == "discard" else 1
    values.append((_KIND_PHASE + kind_index * 2 + phase_index, 1.0))
    return values


def _dense_action_features(
    state: dict[str, Any],
    action: GameAction,
    *,
    feature_version: int = DEFAULT_NEURAL_FEATURE_VERSION,
) -> list[float]:
    """Compact normalized features for the MLP candidate scorer."""

    kind_index = ACTION_KIND_INDEX.get(action.kind)
    if kind_index is None:
        raise ValueError(f"不支持的训练动作：{action.kind}")
    target = _candidate_target(state, action)
    hand_counts = Counter(tile for tile in state["hand"] if is_base_tile(tile))
    features: list[float] = [hand_counts[tile] / 4.0 for tile in range(BASE_TILE_COUNT)]

    features.extend(1.0 if index == kind_index else 0.0 for index in range(len(ACTION_KINDS)))
    features.extend(1.0 if state["phase"] == phase else 0.0 for phase in ("discard", "response"))

    target_category = 4 if target == NO_TILE else (target // 9 if target < 27 else 3)
    features.extend(1.0 if index == target_category else 0.0 for index in range(5))
    target_rank = target % 9 if target < 27 else None
    features.extend(1.0 if rank == target_rank else 0.0 for rank in range(9))

    local_tiles = [target]
    if target < 27:
        suit_base = target // 9 * 9
        local_tiles.extend(
            candidate
            for candidate in (target - 2, target - 1, target + 1, target + 2)
            if suit_base <= candidate < suit_base + 9
        )
    while len(local_tiles) < 5:
        local_tiles.append(NO_TILE)
    features.extend(
        hand_counts[tile] / 4.0 if is_base_tile(tile) else 0.0 for tile in local_tiles[:5]
    )
    target_index = target if is_base_tile(target) else 0
    features.extend(
        [
            state["river_counts"][target_index] / 4.0 if is_base_tile(target) else 0.0,
            state["meld_counts"][target_index] / 4.0 if is_base_tile(target) else 0.0,
            len(action.tiles) / 3.0,
        ]
    )
    tour_level = min(max(int(state.get("tour_level", 0)), 0), 3)
    features.extend(1.0 if level == tour_level else 0.0 for level in range(4))
    features.extend(
        [
            min(int(state["wall_remaining"]), 144) / 144.0,
            min(int(state.get("flowers", 0)), 8) / 8.0,
            1.0 if state.get("gold_locked") else 0.0,
            1.0 if state.get("is_dealer") else 0.0,
            1.0 if target == state.get("gold_tile") else 0.0,
        ]
    )
    if feature_version == 2:
        features.extend(_lookahead_features(state, action))
    elif feature_version == 3:
        features.extend(_lookahead_features(state, action))
        features.extend(_public_context_features(state, target))
    expected_dimension = NEURAL_FEATURE_DIMS.get(feature_version)
    if expected_dimension is None:
        raise ValueError("不支持的 MLP 特征版本")
    if len(features) != expected_dimension:
        raise RuntimeError(f"MLP 特征维度错误：{len(features)}")
    return features


def _public_context_features(state: dict[str, Any], target: int) -> list[float]:
    """Encode all actor-visible public information added in trajectory v2.

    Seats are normalized relative to the acting player, preserving rotation
    invariance while retaining which opponent exposed or discarded each tile.
    Concealed-kong identities are redacted by ``_perspective_state``.
    """

    players = list(state.get("public_players", []))
    by_relative_seat = {
        int(player.get("relative_seat", -1)): dict(player) for player in players
    }
    features: list[float] = []
    for relative_seat in range(4):
        player = by_relative_seat.get(relative_seat, {})
        discards = [
            int(tile) for tile in player.get("discards", []) if is_base_tile(int(tile))
        ]
        melds = [dict(meld) for meld in player.get("melds", [])]
        open_target_count = sum(
            sum(1 for tile in meld.get("tiles", []) if tile == target)
            for meld in melds
        )
        features.extend(
            [
                min(len(discards), 30) / 30.0,
                min(len(melds), 5) / 5.0,
                min(int(player.get("flowers", 0)), 8) / 8.0,
                min(int(player.get("hand_count", 0)), 17) / 17.0,
                (max(-256, min(256, int(player.get("score_delta", 0)))) + 256) / 512.0,
                1.0 if player.get("opening_wait") else 0.0,
                1.0 if player.get("gold_locked") else 0.0,
                discards.count(target) / 4.0 if is_base_tile(target) else 0.0,
                min(open_target_count, 4) / 4.0 if is_base_tile(target) else 0.0,
            ]
        )

    def one_hot(value: Any, size: int, *, include_none: bool = False) -> list[float]:
        index = size if include_none and value is None else value
        return [1.0 if index == candidate else 0.0 for candidate in range(size + int(include_none))]

    features.extend(one_hot(state.get("dealer_relative"), 4))
    features.extend(one_hot(state.get("current_player_relative"), 4))
    features.extend(one_hot(state.get("discarder_relative"), 4, include_none=True))
    last_action = list(state.get("recent_public_actions", []))
    last = dict(last_action[-1]) if last_action else {}
    action_kind = last.get("kind")
    features.extend(
        [1.0 if action_kind == kind else 0.0 for kind in PUBLIC_ACTION_KINDS]
        + [1.0 if action_kind not in PUBLIC_ACTION_KINDS else 0.0]
    )
    features.extend(one_hot(last.get("relative_seat"), 4, include_none=True))
    return features


def public_action_sequence_features(
    state: Mapping[str, Any], *, length: int = PUBLIC_ACTION_SEQUENCE_LENGTH
) -> tuple[tuple[float, ...], ...]:
    """Encode an ordered, actor-visible public action history for sequence models.

    The game exporter records ``an_kan`` without its face value, so the
    resulting token cannot leak another player's concealed kong.  The output
    contains only real events (no padding); callers build the attention mask.
    """

    if length <= 0:
        raise ValueError("公开动作序列长度必须为正数")
    actions = list(state.get("recent_public_actions", []))[-length:]
    encoded: list[tuple[float, ...]] = []
    kind_index = {kind: index for index, kind in enumerate(PUBLIC_ACTION_KINDS)}
    for position, raw_action in enumerate(actions):
        action = dict(raw_action)
        values: list[float] = []
        kind = str(action.get("kind", ""))
        values.extend(
            1.0 if kind_index.get(kind, len(PUBLIC_ACTION_KINDS)) == index else 0.0
            for index in range(len(PUBLIC_ACTION_KINDS) + 1)
        )
        relative_seat = action.get("relative_seat")
        seat_index = (
            int(relative_seat)
            if isinstance(relative_seat, int) and 0 <= relative_seat < 4
            else 4
        )
        values.extend(1.0 if seat_index == index else 0.0 for index in range(5))
        # A concealed kong is public as an action but not by face value.  The
        # normal exporter already removes these fields; this guard also makes
        # old/malformed records safe for training.
        concealed_kong = kind == "an_kan"
        primary_tile = None if concealed_kong else action.get("tile")
        tile_index = (
            int(primary_tile)
            if isinstance(primary_tile, int) and is_base_tile(primary_tile)
            else NO_TILE
        )
        values.extend(1.0 if tile_index == index else 0.0 for index in range(TILE_SLOTS))
        public_tiles = Counter(
            int(tile)
            for tile in (() if concealed_kong else action.get("tiles", []))
            if isinstance(tile, int) and is_base_tile(tile)
        )
        values.extend(public_tiles[tile] / 4.0 for tile in range(BASE_TILE_COUNT))
        values.append((position + 1) / max(len(actions), 1))
        if len(values) != PUBLIC_ACTION_SEQUENCE_DIM:
            raise RuntimeError("公开动作序列特征维度错误")
        encoded.append(tuple(values))
    return tuple(encoded)


def _lookahead_features(state: dict[str, Any], action: GameAction) -> list[float]:
    """Return actor-visible, deterministic candidate features for MLP v2."""

    hand = list(state["hand"])
    gold_tile = state.get("gold_tile")
    meld_count = sum(int(value) for value in state["meld_counts"])
    if action.kind == "discard" and action.tile in hand:
        hand.remove(action.tile)
    elif action.kind in {"chi", "pong", "ming_kan"}:
        for tile in action.tiles:
            if tile in hand:
                hand.remove(tile)
        meld_count += 1
    elif action.kind == "an_kan" and action.tile is not None:
        for _ in range(4):
            if action.tile in hand:
                hand.remove(action.tile)
        meld_count += 1
    elif action.kind == "add_kan" and action.tile in hand:
        hand.remove(action.tile)
    elif action.kind == "advance_tour" and gold_tile in hand:
        hand.remove(gold_tile)

    rules = XiamenRules.from_profile(str(state.get("rules_profile", "classic")))
    proxy_tile = (
        WHITE_DRAGON
        if rules.white_dragon_is_gold_proxy and gold_tile is not None and gold_tile != WHITE_DRAGON
        else None
    )
    wildcard_tiles = {gold_tile} if rules.gold_is_wildcard and gold_tile is not None else set()
    quality = hand_quality(
        hand,
        gold_tile,
        meld_count=meld_count,
        melds_required=rules.melds_required,
        wildcard_tiles=wildcard_tiles,
        proxy_tile=proxy_tile,
        proxy_as=gold_tile,
    )
    waits = []
    if action.kind == "discard":
        waits = wait_tiles(
            hand,
            gold_tile,
            meld_count=meld_count,
            melds_required=rules.melds_required,
            allow_seven_pairs=rules.allow_seven_pairs,
            wildcard_tiles=wildcard_tiles,
            proxy_tile=proxy_tile,
            proxy_as=gold_tile,
        )
    gold_count = hand.count(gold_tile) if isinstance(gold_tile, int) else 0
    gold_discard = action.kind == "discard" and action.tile == gold_tile
    return [
        min(quality, 200.0) / 200.0,
        len(waits) / BASE_TILE_COUNT,
        min(gold_count, 4) / 4.0,
        1.0 if gold_discard else 0.0,
    ]


def _candidate_target(state: dict[str, Any], action: GameAction) -> int:
    if action.tile is not None and is_base_tile(action.tile):
        return action.tile
    last_discard = state.get("last_discard")
    if action.kind == "hu" and isinstance(last_discard, int) and is_base_tile(last_discard):
        return last_discard
    return NO_TILE


def _softmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        raise ValueError("动作集合不能为空")
    maximum = max(scores)
    exponentials = [math.exp(min(score - maximum, 0.0)) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _action_payload(action: GameAction) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": action.kind, "tiles": list(action.tiles)}
    if action.tile is not None:
        payload["tile"] = action.tile
    return payload


def _action_from_payload(payload: dict[str, Any]) -> GameAction:
    tile = payload.get("tile")
    if tile is not None and not isinstance(tile, int):
        raise ValueError("动作牌值无效")
    tiles = payload.get("tiles", [])
    if not isinstance(tiles, list) or not all(isinstance(item, int) for item in tiles):
        raise ValueError("动作组合无效")
    return GameAction(str(payload["kind"]), tile, tuple(tiles))

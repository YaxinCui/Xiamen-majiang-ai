"""Rule-Teacher imitation-learning baseline for Xiamen Mahjong.

The browser game remains the authority for legality.  This module only learns
how to rank the actions that the engine already says are legal, so a policy
checkpoint cannot invent an illegal discard, claim, or kong.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

from .agents import GameAction
from .game import XiamenMahjongGame
from .rules import XiamenRules
from .tiles import BASE_TILE_COUNT, is_base_tile


DATASET_VERSION = "xiamen-rule-teacher-v1"
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
NEURAL_FEATURE_DIM = 76


@dataclass(frozen=True)
class TeacherDecision:
    """One perspective-correct rule-Teacher decision and its legal choices."""

    profile: str
    seed: int
    seat: int
    state: dict[str, Any]
    legal_actions: tuple[GameAction, ...]
    chosen_index: int

    @property
    def chosen_action(self) -> GameAction:
        return self.legal_actions[self.chosen_index]

    def payload(self) -> dict[str, Any]:
        return {
            "version": DATASET_VERSION,
            "profile": self.profile,
            "seed": self.seed,
            "seat": self.seat,
            "state": self.state,
            "legal_actions": [_action_payload(action) for action in self.legal_actions],
            "chosen_index": self.chosen_index,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TeacherDecision":
        if payload.get("version") != DATASET_VERSION:
            raise ValueError("不支持的 Teacher 轨迹版本")
        actions = tuple(_action_from_payload(item) for item in payload["legal_actions"])
        chosen_index = int(payload["chosen_index"])
        if not 0 <= chosen_index < len(actions):
            raise ValueError("Teacher 轨迹的目标动作索引无效")
        return cls(
            profile=str(payload["profile"]),
            seed=int(payload["seed"]),
            seat=int(payload["seat"]),
            state=dict(payload["state"]),
            legal_actions=actions,
            chosen_index=chosen_index,
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


def collect_teacher_decisions(
    *,
    hands: int,
    profile: str = "classic",
    seed: int = 20260804,
) -> tuple[list[TeacherDecision], DatasetSummary]:
    """Play deterministic all-Teacher hands and return every decision.

    Each player is observed from only that player's private hand plus public
    information.  No wall order or opponents' concealed tiles enter a sample.
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
        game = XiamenMahjongGame(seed=hand_seed, rules=rules, auto_advance=False)
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
                decisions.append(decision)
                action_counts[chosen.kind] += 1
                game._apply_turn_action(player_id, chosen)
                continue
            if game.phase == "response":
                for player_id, options in sorted(game.response_options.items()):
                    legal = tuple(options)
                    chosen = game.teacher.choose_response(game, player_id, list(legal))
                    decision = _decision(game, hand_seed, player_id, legal, chosen)
                    decisions.append(decision)
                    action_counts[chosen.kind] += 1
                    game.response_choices[player_id] = chosen
                game._resolve_responses()
                continue
            raise RuntimeError(f"未知训练阶段：{game.phase}")
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
        input_weights: Sequence[Sequence[float]] | None = None,
        hidden_bias: Sequence[float] | None = None,
        output_weights: Sequence[float] | None = None,
        output_bias: float = 0.0,
    ):
        if hidden_size <= 0:
            raise ValueError("hidden_size 必须为正数")
        self.hidden_size = hidden_size
        if input_weights is None:
            rng = random.Random(seed)
            scale = 1.0 / math.sqrt(NEURAL_FEATURE_DIM)
            self.input_weights = [
                [rng.uniform(-scale, scale) for _ in range(NEURAL_FEATURE_DIM)]
                for _ in range(hidden_size)
            ]
            self.hidden_bias = [0.0] * hidden_size
            output_scale = 1.0 / math.sqrt(hidden_size)
            self.output_weights = [
                rng.uniform(-output_scale, output_scale) for _ in range(hidden_size)
            ]
            self.output_bias = 0.0
            return
        if (
            len(input_weights) != hidden_size
            or any(len(row) != NEURAL_FEATURE_DIM for row in input_weights)
            or hidden_bias is None
            or len(hidden_bias) != hidden_size
            or output_weights is None
            or len(output_weights) != hidden_size
        ):
            raise ValueError("MLP 检查点维度与当前特征定义不匹配")
        self.input_weights = [[float(value) for value in row] for row in input_weights]
        self.hidden_bias = [float(value) for value in hidden_bias]
        self.output_weights = [float(value) for value in output_weights]
        self.output_bias = float(output_bias)

    def scores(self, decision: TeacherDecision) -> list[float]:
        return [self._forward(_dense_action_features(decision.state, action))[2] for action in decision.legal_actions]

    def predict_index(self, decision: TeacherDecision) -> int:
        scores = self.scores(decision)
        return max(range(len(scores)), key=lambda index: (scores[index], -index))

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
    ) -> list[dict[str, float]]:
        if not decisions:
            raise ValueError("训练数据为空")
        if epochs <= 0 or learning_rate <= 0:
            raise ValueError("epochs 和 learning_rate 必须为正数")
        order = list(range(len(decisions)))
        rng = random.Random(seed)
        history: list[dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            rng.shuffle(order)
            for decision_index in order:
                decision = decisions[decision_index]
                candidates = [
                    self._forward(_dense_action_features(decision.state, action))
                    for action in decision.legal_actions
                ]
                probabilities = _softmax([candidate[2] for candidate in candidates])
                for action_index, (features, hidden, _score) in enumerate(candidates):
                    gradient = probabilities[action_index] - (
                        1.0 if action_index == decision.chosen_index else 0.0
                    )
                    if not gradient:
                        continue
                    previous_output = list(self.output_weights)
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
            history.append({"epoch": float(epoch), **self.evaluate(decisions)})
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
            "model": "legal_action_relu_mlp",
            "feature_dim": NEURAL_FEATURE_DIM,
            "hidden_size": self.hidden_size,
            "input_weights": self.input_weights,
            "hidden_bias": self.hidden_bias,
            "output_weights": self.output_weights,
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
        return cls(
            hidden_size=int(payload["hidden_size"]),
            input_weights=payload["input_weights"],
            hidden_bias=payload["hidden_bias"],
            output_weights=payload["output_weights"],
            output_bias=float(payload["output_bias"]),
        )

    def _forward(self, features: Sequence[float]) -> tuple[list[float], list[float], float]:
        pre_activation = [
            bias + sum(weight * value for weight, value in zip(row, features))
            for row, bias in zip(self.input_weights, self.hidden_bias)
        ]
        hidden = [max(value, 0.0) for value in pre_activation]
        score = self.output_bias + sum(
            weight * value for weight, value in zip(self.output_weights, hidden)
        )
        return list(features), hidden, score


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
    return {
        "phase": game.phase,
        "hand": list(player.hand),
        "river_counts": [river_counts[tile] for tile in range(BASE_TILE_COUNT)],
        "meld_counts": [meld_counts[tile] for tile in range(BASE_TILE_COUNT)],
        "gold_tile": game.gold_tile,
        "last_discard": game.last_discard,
        "wall_remaining": len(game.wall),
        "flowers": len(player.flowers),
        "tour_level": tour_level,
        "gold_locked": game.gold_discard_lock_seat == player_id,
        "is_dealer": game.dealer == player_id,
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


def _dense_action_features(state: dict[str, Any], action: GameAction) -> list[float]:
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
    if len(features) != NEURAL_FEATURE_DIM:
        raise RuntimeError(f"MLP 特征维度错误：{len(features)}")
    return features


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

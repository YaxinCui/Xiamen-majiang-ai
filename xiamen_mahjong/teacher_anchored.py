"""Rule-anchored residual policy wrapper for conservative self-play research.

The wrapped neural policy supplies only a logit correction.  A deterministic
``HeuristicTeacherAgent`` supplies a fixed prior for the legal action it would
take from the same actor-visible game state.  This keeps an untrained residual
actor behaviorally identical to the rule Teacher when its policy head is zero,
without feeding private wall or opponent-hand information to the neural model.
"""

from __future__ import annotations

from typing import Any, Sequence

from .agents import GameAction, HeuristicTeacherAgent
from .game import XiamenMahjongGame
from .training import _decision, _turn_actions


def teacher_prior_logits(
    teacher_action: GameAction,
    legal_actions: Sequence[GameAction],
    *,
    margin: float,
) -> tuple[float, ...]:
    """Give the rule action a fixed advantage without changing legality."""

    if margin <= 0:
        raise ValueError("Teacher prior margin 必须为正数")
    try:
        teacher_index = tuple(legal_actions).index(teacher_action)
    except ValueError as error:
        raise RuntimeError("规则 Teacher 选择了不合法动作") from error
    return tuple(0.0 if index == teacher_index else -margin for index in range(len(legal_actions)))


class TeacherAnchoredPolicyAgent:
    """Combine a deployable residual policy with a visible rule prior.

    It intentionally implements the same action methods as regular agents but
    has no training logic.  PPO collectors construct exactly the same prior
    logits before sampling and updating, preventing train/inference mismatch.
    """

    def __init__(self, residual_policy: Any, *, margin: float):
        if margin <= 0:
            raise ValueError("Teacher prior margin 必须为正数")
        self.residual_policy = residual_policy
        self.margin = float(margin)
        self.teacher = HeuristicTeacherAgent()

    def _combined_index(
        self,
        game: XiamenMahjongGame,
        player_id: int,
        legal_actions: Sequence[GameAction],
        *,
        response: bool,
    ) -> int:
        legal = tuple(legal_actions)
        if not legal:
            raise ValueError("Teacher-anchored policy 没有合法动作")
        teacher_action = (
            self.teacher.choose_response(game, player_id, list(legal))
            if response
            else self.teacher.choose_turn_action(game, player_id)
        )
        prior = teacher_prior_logits(teacher_action, legal, margin=self.margin)
        decision = _decision(game, game.seed or 0, player_id, legal, legal[0])
        residual_logits, _value = self.residual_policy.policy_value(decision)
        if len(residual_logits) != len(legal):
            raise RuntimeError("residual policy 的合法动作 logits 长度不匹配")
        return max(
            range(len(legal)),
            key=lambda index: (residual_logits[index] + prior[index], -index),
        )

    def choose_turn_action(self, game: XiamenMahjongGame, player_id: int) -> GameAction:
        legal = tuple(_turn_actions(game, player_id))
        return legal[self._combined_index(game, player_id, legal, response=False)]

    def choose_response(
        self,
        game: XiamenMahjongGame,
        player_id: int,
        options: Sequence[GameAction],
    ) -> GameAction:
        legal = tuple(options)
        return legal[self._combined_index(game, player_id, legal, response=True)]


class ConfidenceGatedTeacherAgent:
    """Let a compact policy override Teacher only with a strict logit gap.

    This is a conservative intermediate policy, not an additive residual
    trained end to end.  The frozen Teacher owns every decision by default.
    The compact policy may replace it only when the Teacher action kind is in
    ``allowed_teacher_kinds`` and its best alternative exceeds the Teacher
    logit by more than ``minimum_policy_advantage``.

    The default scope is deliberately just ordinary discard.  Tour, gold-lock,
    response, win and kong decisions therefore remain frozen even if a neural
    checkpoint is confidently wrong on one of those rare rule branches.  A
    response experiment may explicitly set ``ordinary_response_only``; this
    additionally requires a classic, non-special legal set containing only
    pass/chi/pong before the policy is even evaluated.
    """

    def __init__(
        self,
        policy: Any,
        *,
        minimum_policy_advantage: float,
        allowed_teacher_kinds: Sequence[str] = ("discard",),
        allowed_alternative_kinds: Sequence[str] = ("discard",),
        ordinary_response_only: bool = False,
    ) -> None:
        if minimum_policy_advantage < 0:
            raise ValueError("minimum_policy_advantage 不能为负数")
        allowed = frozenset(str(kind) for kind in allowed_teacher_kinds)
        if not allowed:
            raise ValueError("allowed_teacher_kinds 不能为空")
        alternatives = frozenset(str(kind) for kind in allowed_alternative_kinds)
        if not alternatives:
            raise ValueError("allowed_alternative_kinds 不能为空")
        self.policy = policy
        self.minimum_policy_advantage = float(minimum_policy_advantage)
        self.allowed_teacher_kinds = allowed
        self.allowed_alternative_kinds = alternatives
        self.ordinary_response_only = bool(ordinary_response_only)
        self.teacher = HeuristicTeacherAgent()
        self.eligible_decisions = 0
        self.override_count = 0

    def _policy_decision(
        self,
        game: XiamenMahjongGame,
        player_id: int,
        legal: tuple[GameAction, ...],
    ) -> Any:
        builder = getattr(self.policy, "_decision_for_game", None)
        if callable(builder):
            return builder(game, player_id, legal)
        return _decision(game, game.seed or 0, player_id, legal, legal[0])

    def _gated_action(
        self,
        game: XiamenMahjongGame,
        player_id: int,
        legal_actions: Sequence[GameAction],
        *,
        response: bool,
    ) -> GameAction:
        legal = tuple(legal_actions)
        if not legal:
            raise ValueError("置信门控策略没有合法动作")
        teacher_action = (
            self.teacher.choose_response(game, player_id, list(legal))
            if response
            else self.teacher.choose_turn_action(game, player_id)
        )
        if teacher_action not in legal:
            raise RuntimeError("规则 Teacher 选择了不合法动作")
        if teacher_action.kind not in self.allowed_teacher_kinds:
            return teacher_action
        if self.ordinary_response_only and response:
            kinds = tuple(action.kind for action in legal)
            if (
                game.rules.profile != "classic"
                or game.phase != "response"
                or game.tour_state is not None
                or game.gold_discard_lock_seat == player_id
                or player_id in game.opening_wait_seats
                or not set(kinds).issubset({"pass", "chi", "pong"})
                or kinds.count("pass") != 1
                or not any(kind in {"chi", "pong"} for kind in kinds)
            ):
                return teacher_action
        if (
            teacher_action.kind == "discard"
            and (
                game.tour_state is not None
                or game.gold_discard_lock_seat == player_id
            )
        ):
            return teacher_action
        if len(legal) < 2:
            return teacher_action

        self.eligible_decisions += 1
        decision = self._policy_decision(game, player_id, legal)
        logits, _value = self.policy.policy_value(decision)
        if len(logits) != len(legal):
            raise RuntimeError("置信门控策略的合法动作 logits 长度不匹配")
        teacher_index = legal.index(teacher_action)
        alternative_indices = tuple(
            index
            for index, action in enumerate(legal)
            if index != teacher_index
            and action.kind in self.allowed_alternative_kinds
        )
        if not alternative_indices:
            return teacher_action
        alternative_index = max(
            alternative_indices,
            key=lambda index: (float(logits[index]), -index),
        )
        advantage = float(logits[alternative_index]) - float(logits[teacher_index])
        if advantage <= self.minimum_policy_advantage:
            return teacher_action
        self.override_count += 1
        return legal[alternative_index]

    def choose_turn_action(
        self, game: XiamenMahjongGame, player_id: int
    ) -> GameAction:
        legal = tuple(_turn_actions(game, player_id))
        return self._gated_action(game, player_id, legal, response=False)

    def choose_response(
        self,
        game: XiamenMahjongGame,
        player_id: int,
        options: Sequence[GameAction],
    ) -> GameAction:
        return self._gated_action(game, player_id, options, response=True)

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

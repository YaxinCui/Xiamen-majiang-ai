"""Small information-set rollout policy for controlled Mahjong experiments.

The agent never evaluates the hidden state attached to the live game object.
Before every decision it redraws opponent hands and the wall from the acting
seat's visible information, then forces each engine-legal action in the same
sampled world and lets the rule Teacher finish the hand.  This is deliberately
a low-budget one-step policy-improvement baseline, not an oracle or a trained
model, and is useful only after independent paired evaluation.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Sequence

from .agents import GameAction, HeuristicTeacherAgent
from .game import XiamenMahjongGame


class InformationSetRolloutAgent:
    """Rank legal actions by hidden-state-safe Teacher continuation rollouts.

    ``belief_worlds`` is intentionally small because one policy decision
    evaluates every legal action in every world. By default only response
    actions use rollouts; ``rollout_turn_actions=True`` is an explicit slower
    research mode. A sampled world is accepted only when it presents precisely
    the same legal action set as the player observes in the live game. Any
    unsupported/sparse state falls back to the deterministic Teacher rather
    than retaining the actual hidden deal.
    """

    def __init__(
        self,
        *,
        belief_worlds: int = 1,
        seed: int = 20260807,
        rollout_turn_actions: bool = False,
        selection_mode: str = "mean",
        paired_lcb_z: float = 1.0,
    ) -> None:
        if belief_worlds <= 0:
            raise ValueError("belief_worlds 必须为正数")
        if selection_mode not in {"mean", "paired_lcb"}:
            raise ValueError("selection_mode 必须为 mean 或 paired_lcb")
        if paired_lcb_z < 0.0:
            raise ValueError("paired_lcb_z 不能为负数")
        self.belief_worlds = int(belief_worlds)
        self.rollout_turn_actions = bool(rollout_turn_actions)
        self.selection_mode = selection_mode
        self.paired_lcb_z = float(paired_lcb_z)
        self._rng = random.Random(seed)
        self._teacher = HeuristicTeacherAgent()
        self.last_diagnostics: dict[str, int] = {
            "requested_worlds": self.belief_worlds,
            "usable_worlds": 0,
            "skipped_worlds": 0,
        }

    def choose_turn_action(self, game: XiamenMahjongGame, player_id: int) -> GameAction:
        from .training import _turn_actions

        legal = tuple(_turn_actions(game, player_id))
        fallback = self._teacher.choose_turn_action(game, player_id)
        if not self.rollout_turn_actions:
            self.last_diagnostics = {
                "requested_worlds": self.belief_worlds,
                "usable_worlds": 0,
                "skipped_worlds": 0,
            }
            return fallback
        return self._choose(game, player_id, legal, fallback)

    def choose_response(
        self,
        game: XiamenMahjongGame,
        player_id: int,
        options: Sequence[GameAction],
    ) -> GameAction:
        legal = tuple(options)
        fallback = self._teacher.choose_response(game, player_id, list(legal))
        return self._choose(game, player_id, legal, fallback)

    def _choose(
        self,
        game: XiamenMahjongGame,
        player_id: int,
        legal: tuple[GameAction, ...],
        fallback: GameAction,
    ) -> GameAction:
        if not legal or fallback not in legal:
            raise RuntimeError("rollout 策略未得到规则引擎提供的回退动作")
        if game.rules.profile not in {"core", "classic"}:
            return fallback

        # Import locally to keep the default browser/Teacher path free from
        # this experiment's training helpers and optional rollout machinery.
        from .training import (
            _continue_counterfactual_rollout,
            _resample_private_world_for_actor,
            _turn_actions,
        )

        returns: list[list[int]] = [[] for _ in legal]
        usable_worlds = 0
        skipped_worlds = 0
        opponents = {
            seat: ("heuristic_teacher", self._teacher)
            for seat in range(game.rules.player_count)
            if seat != player_id
        }
        for _ in range(self.belief_worlds):
            world = _resample_private_world_for_actor(
                game,
                actor_seat=player_id,
                rng=self._rng,
            )
            if world is None:
                skipped_worlds += 1
                continue
            world_legal = (
                tuple(_turn_actions(world, player_id))
                if world.phase == "discard"
                else tuple(world.response_options.get(player_id, []))
                if world.phase == "response"
                else ()
            )
            if world_legal != legal:
                skipped_worlds += 1
                continue
            usable_worlds += 1
            for action_index, action in enumerate(legal):
                score = _continue_counterfactual_rollout(
                    copy.deepcopy(world),
                    candidate_seat=player_id,
                    # The forced action is the only planned decision. Future
                    # decisions use the frozen Teacher to avoid recursive
                    # search and to keep every compared branch symmetric.
                    candidate_policy=self._teacher,
                    opponents=opponents,
                    forced_action=action,
                )
                returns[action_index].append(score)
        self.last_diagnostics = {
            "requested_worlds": self.belief_worlds,
            "usable_worlds": usable_worlds,
            "skipped_worlds": skipped_worlds,
        }
        if not usable_worlds:
            return fallback
        if self.selection_mode == "paired_lcb":
            return self._choose_paired_lcb(legal, fallback, returns)
        return legal[
            max(
                range(len(legal)),
                key=lambda index: (sum(returns[index]) / len(returns[index]), -index),
            )
        ]

    def _choose_paired_lcb(
        self,
        legal: tuple[GameAction, ...],
        fallback: GameAction,
        returns: Sequence[Sequence[int]],
    ) -> GameAction:
        """Override Teacher only on a paired lower-confidence improvement.

        Every action is rolled out in the same determinized worlds, so a
        candidate-vs-Teacher score difference removes much of the world
        variance.  Requiring a positive lower confidence bound specifically
        controls the maximization bias of selecting the largest noisy rollout
        average.  A single world cannot estimate this uncertainty and thus
        deliberately falls back to Teacher.
        """

        fallback_index = legal.index(fallback)
        reference = returns[fallback_index]
        if len(reference) < 2:
            return fallback
        best_index = fallback_index
        best_lower_bound = 0.0
        for index, action_returns in enumerate(returns):
            if index == fallback_index:
                continue
            deltas = [
                float(value - baseline)
                for value, baseline in zip(action_returns, reference)
            ]
            if len(deltas) != len(reference):
                raise RuntimeError("rollout 分支没有共享的 world 数")
            mean_delta = sum(deltas) / len(deltas)
            variance = sum((value - mean_delta) ** 2 for value in deltas) / (
                len(deltas) - 1
            )
            lower_bound = mean_delta - self.paired_lcb_z * math.sqrt(
                variance / len(deltas)
            )
            if lower_bound > best_lower_bound:
                best_index = index
                best_lower_bound = lower_bound
        return legal[best_index]

import copy
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from xiamen_mahjong.agents import GameAction, HeuristicTeacherAgent
from xiamen_mahjong.game import XiamenMahjongGame
from xiamen_mahjong.rules import XiamenRules


@unittest.skipIf(torch is None, "PyTorch 仅在项目 .venv 中安装")
class TeacherAnchoredPolicyTests(unittest.TestCase):
    def test_zero_residual_exactly_matches_teacher_and_ignores_hidden_world(self):
        from scripts.init_fresh_selfplay_policy import create_fresh_policy
        from xiamen_mahjong.agents import HeuristicTeacherAgent
        from xiamen_mahjong.game import XiamenMahjongGame
        from xiamen_mahjong.rules import XiamenRules
        from xiamen_mahjong.teacher_anchored import TeacherAnchoredPolicyAgent

        game = XiamenMahjongGame(
            seed=202611702,
            rules=XiamenRules.classic(),
            auto_advance=False,
            human_seat=-1,
        )
        policy = create_fresh_policy(
            seed=202611703,
            feature_version=3,
            hidden_size=16,
            device="cpu",
            zero_policy_head=True,
        )
        anchored = TeacherAnchoredPolicyAgent(policy, margin=5.0)
        teacher = HeuristicTeacherAgent()
        player_id = game.current_player
        altered = copy.deepcopy(game)
        opponent = next(player for player in altered.players if player.seat != player_id)
        wall_index = next(
            index for index, tile in enumerate(altered.wall) if tile != opponent.hand[0]
        )
        opponent.hand[0], altered.wall[wall_index] = altered.wall[wall_index], opponent.hand[0]
        opponent.hand.sort()

        self.assertEqual(
            anchored.choose_turn_action(game, player_id),
            teacher.choose_turn_action(game, player_id),
        )
        self.assertEqual(
            anchored.choose_turn_action(game, player_id),
            anchored.choose_turn_action(altered, player_id),
        )

    def test_prior_requires_a_positive_margin(self):
        from xiamen_mahjong.agents import GameAction
        from xiamen_mahjong.teacher_anchored import teacher_prior_logits

        with self.assertRaisesRegex(ValueError, "正数"):
            teacher_prior_logits(GameAction("pass"), (GameAction("pass"),), margin=0)

    def test_confidence_gate_requires_strict_advantage(self):
        from xiamen_mahjong.teacher_anchored import ConfidenceGatedTeacherAgent

        class ControlledPolicy:
            def __init__(self, alternative_gap):
                self.alternative_gap = alternative_gap

            def policy_value(self, decision):
                teacher_index = decision.legal_actions.index(
                    HeuristicTeacherAgent().choose_turn_action(game, player_id)
                )
                logits = [0.0] * len(decision.legal_actions)
                alternative = next(
                    index
                    for index in range(len(logits))
                    if index != teacher_index
                )
                logits[alternative] = self.alternative_gap
                return logits, 0.0

        game = XiamenMahjongGame(
            seed=202618000,
            rules=XiamenRules.classic(),
            auto_advance=False,
            human_seat=-1,
        )
        player_id = game.current_player
        teacher = HeuristicTeacherAgent().choose_turn_action(game, player_id)
        tied = ConfidenceGatedTeacherAgent(
            ControlledPolicy(2.0), minimum_policy_advantage=2.0
        )
        self.assertEqual(tied.choose_turn_action(game, player_id), teacher)
        self.assertEqual(tied.eligible_decisions, 1)
        self.assertEqual(tied.override_count, 0)
        overriding = ConfidenceGatedTeacherAgent(
            ControlledPolicy(2.001), minimum_policy_advantage=2.0
        )
        self.assertNotEqual(overriding.choose_turn_action(game, player_id), teacher)
        self.assertEqual(overriding.eligible_decisions, 1)
        self.assertEqual(overriding.override_count, 1)

    def test_confidence_gate_keeps_responses_on_teacher_by_default(self):
        from xiamen_mahjong.teacher_anchored import ConfidenceGatedTeacherAgent

        class FailingPolicy:
            def policy_value(self, _decision):
                raise AssertionError("默认 discard-only gate 不应调用响应模型")

        game = XiamenMahjongGame(seed=202618001, rules=XiamenRules.classic())
        game.phase = "response"
        options = [GameAction("pass"), GameAction("pong", 4, (4, 4))]
        game.players[1].hand = [
            0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
        ]
        agent = ConfidenceGatedTeacherAgent(
            FailingPolicy(), minimum_policy_advantage=0.0
        )
        self.assertEqual(
            agent.choose_response(game, 1, options),
            HeuristicTeacherAgent().choose_response(game, 1, options),
        )

    def test_response_gate_overrides_only_ordinary_pass_chi_pong(self):
        from xiamen_mahjong.teacher_anchored import ConfidenceGatedTeacherAgent

        game = XiamenMahjongGame(seed=202633100, rules=XiamenRules.classic())
        game.phase = "response"
        player_id = 1
        game.players[player_id].hand = [
            0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
        ]
        options = [GameAction("pass"), GameAction("pong", 4, (4, 4))]
        teacher = HeuristicTeacherAgent().choose_response(game, player_id, options)

        class ControlledPolicy:
            def policy_value(self, decision):
                teacher_index = decision.legal_actions.index(teacher)
                logits = [3.0] * len(decision.legal_actions)
                logits[teacher_index] = 0.0
                return logits, 0.0

        agent = ConfidenceGatedTeacherAgent(
            ControlledPolicy(),
            minimum_policy_advantage=2.0,
            allowed_teacher_kinds=("pass", "chi", "pong"),
            allowed_alternative_kinds=("pass", "chi", "pong"),
            ordinary_response_only=True,
        )
        self.assertNotEqual(
            agent.choose_response(game, player_id, options), teacher
        )
        self.assertEqual(agent.override_count, 1)

        game.opening_wait_seats.add(player_id)
        before = agent.eligible_decisions
        self.assertEqual(
            agent.choose_response(game, player_id, options), teacher
        )
        self.assertEqual(agent.eligible_decisions, before)

        game.opening_wait_seats.clear()
        hu_options = [*options, GameAction("hu", 4)]
        hu_teacher = HeuristicTeacherAgent().choose_response(
            game, player_id, hu_options
        )
        self.assertEqual(
            agent.choose_response(game, player_id, hu_options), hu_teacher
        )

    def test_confidence_gate_rejects_invalid_configuration(self):
        from xiamen_mahjong.teacher_anchored import ConfidenceGatedTeacherAgent

        with self.assertRaises(ValueError):
            ConfidenceGatedTeacherAgent(object(), minimum_policy_advantage=-0.1)
        with self.assertRaises(ValueError):
            ConfidenceGatedTeacherAgent(
                object(),
                minimum_policy_advantage=0.0,
                allowed_teacher_kinds=(),
            )
        with self.assertRaises(ValueError):
            ConfidenceGatedTeacherAgent(
                object(),
                minimum_policy_advantage=0.0,
                allowed_alternative_kinds=(),
            )

    def test_confidence_gate_never_uses_non_discard_alternative(self):
        from xiamen_mahjong.teacher_anchored import ConfidenceGatedTeacherAgent

        game = XiamenMahjongGame(
            seed=202618002,
            rules=XiamenRules.classic(),
            auto_advance=False,
            human_seat=-1,
        )
        player_id = game.current_player
        teacher_action = HeuristicTeacherAgent().choose_turn_action(game, player_id)
        legal = (
            teacher_action,
            GameAction("an_kan", 99),
            next(
                action
                for action in game.teacher.explain_discard(game, player_id)
                if action["tile"] != teacher_action.tile
            ),
        )
        legal = (
            legal[0],
            legal[1],
            GameAction("discard", int(legal[2]["tile"])),
        )

        class ControlledPolicy:
            def policy_value(self, _decision):
                return [0.0, 100.0, 3.0], 0.0

        agent = ConfidenceGatedTeacherAgent(
            ControlledPolicy(), minimum_policy_advantage=2.0
        )
        self.assertEqual(
            agent._gated_action(
                game,
                player_id,
                legal,
                response=False,
            ),
            legal[2],
        )

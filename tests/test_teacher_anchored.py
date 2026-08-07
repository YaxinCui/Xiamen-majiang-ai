import copy
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


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

import copy
import unittest

from xiamen_mahjong.game import XiamenMahjongGame
from xiamen_mahjong.rollout_agent import InformationSetRolloutAgent
from xiamen_mahjong.rules import XiamenRules
from xiamen_mahjong.training import _turn_actions


class InformationSetRolloutAgentTests(unittest.TestCase):
    def _initial_game(self) -> XiamenMahjongGame:
        return XiamenMahjongGame(
            seed=20260877,
            rules=XiamenRules.from_profile("core"),
            auto_advance=False,
            human_seat=-1,
        )

    def test_selects_an_engine_legal_action_without_mutating_live_hidden_state(self):
        game = self._initial_game()
        player_id = game.current_player
        legal = tuple(_turn_actions(game, player_id))
        before_hand = tuple(game.players[player_id].hand)
        before_wall = tuple(game.wall)
        before_opponent_hands = tuple(
            tuple(player.hand) for player in game.players if player.seat != player_id
        )

        agent = InformationSetRolloutAgent(
            belief_worlds=1, seed=5, rollout_turn_actions=True
        )
        action = agent.choose_turn_action(game, player_id)

        self.assertIn(action, legal)
        self.assertEqual(agent.last_diagnostics["usable_worlds"], 1)
        self.assertEqual(tuple(game.players[player_id].hand), before_hand)
        self.assertEqual(tuple(game.wall), before_wall)
        self.assertEqual(
            tuple(
                tuple(player.hand) for player in game.players if player.seat != player_id
            ),
            before_opponent_hands,
        )

    def test_action_is_invariant_to_replacing_unseen_live_cards(self):
        game = self._initial_game()
        player_id = game.current_player
        altered = copy.deepcopy(game)
        opponent = next(player for player in altered.players if player.seat != player_id)
        wall_index = next(
            index for index, tile in enumerate(altered.wall) if tile != opponent.hand[0]
        )
        opponent.hand[0], altered.wall[wall_index] = (
            altered.wall[wall_index],
            opponent.hand[0],
        )
        opponent.hand.sort()

        original_action = InformationSetRolloutAgent(
            belief_worlds=1, seed=7, rollout_turn_actions=True
        ).choose_turn_action(game, player_id)
        altered_action = InformationSetRolloutAgent(
            belief_worlds=1, seed=7, rollout_turn_actions=True
        ).choose_turn_action(altered, player_id)

        self.assertEqual(original_action, altered_action)

    def test_default_mode_keeps_turn_decisions_on_the_teacher_path(self):
        game = self._initial_game()
        player_id = game.current_player
        rollout = InformationSetRolloutAgent(belief_worlds=1, seed=11)
        action = rollout.choose_turn_action(game, player_id)
        self.assertEqual(action, game.teacher.choose_turn_action(game, player_id))
        self.assertEqual(rollout.last_diagnostics["usable_worlds"], 0)

    def test_rejects_non_positive_belief_world_count(self):
        with self.assertRaisesRegex(ValueError, "belief_worlds"):
            InformationSetRolloutAgent(belief_worlds=0)


if __name__ == "__main__":
    unittest.main()

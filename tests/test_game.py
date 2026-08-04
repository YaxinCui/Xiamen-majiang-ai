import unittest

from xiamen_mahjong.game import XiamenMahjongGame
from xiamen_mahjong.agents import GameAction
from xiamen_mahjong.tiles import BASE_TILE_COUNT


class GameTests(unittest.TestCase):
    def test_public_state_never_exposes_ai_hands(self):
        game = XiamenMahjongGame(seed=20260803)
        state = game.public_state()
        self.assertEqual(len(state["players"][0]["hand"]), state["players"][0]["hand_count"])
        self.assertTrue(all(player["hand"] is None for player in state["players"][1:]))
        self.assertLess(state["gold_tile"]["id"], BASE_TILE_COUNT)

    def test_human_legal_action_advances_or_requests_a_response(self):
        game = XiamenMahjongGame(seed=7)
        state = game.public_state()
        self.assertTrue(state["actions"])
        action = next(
            (candidate for candidate in state["actions"] if candidate["kind"] == "discard"),
            state["actions"][0],
        )
        game.apply_human_action(action)
        self.assertIn(game.phase, {"discard", "response", "over"})
        self.assertNotEqual(game.phase, "setup")

    def test_game_can_run_until_end_with_first_legal_human_actions(self):
        game = XiamenMahjongGame(seed=41)
        steps = 0
        while game.phase != "over" and steps < 300:
            actions = game.human_actions()
            self.assertTrue(actions)
            preferred = next((action for action in actions if action["kind"] == "discard"), actions[0])
            game.apply_human_action(preferred)
            steps += 1
        self.assertEqual(game.phase, "over")
        self.assertLess(steps, 300)

    def test_gold_discard_skips_all_claims(self):
        game = XiamenMahjongGame(seed=5)
        game.phase = "discard"
        game.current_player = 0
        game.gold_tile = 33
        game.players[0].hand = [33, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        game._apply_turn_action(0, GameAction("discard", 33))
        self.assertEqual(game.phase, "discard")
        self.assertEqual(game.current_player, 1)
        self.assertIsNone(game.last_discard)

    def test_discard_win_applies_simplified_payment(self):
        game = XiamenMahjongGame(seed=11)
        game.gold_tile = 33
        game.players[0].score = 0
        game.players[1].score = 0
        game.players[1].flowers = []
        game.players[1].hand = [0, 1, 2, 3, 4, 5, 9, 10, 11, 18, 18, 18, 27]
        game.last_discard = 27
        game.discarder = 0
        game._finish_win(1, "discard")
        self.assertEqual(game.players[0].score, -24)
        self.assertEqual(game.players[1].score, 24)
        self.assertEqual(game.win_pattern, "标准和")


if __name__ == "__main__":
    unittest.main()

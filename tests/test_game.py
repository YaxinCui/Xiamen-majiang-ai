import unittest

from xiamen_mahjong.game import XiamenMahjongGame
from xiamen_mahjong.agents import AvailabilityTeacherAgent, GameAction
from xiamen_mahjong.rules import XiamenRules
from xiamen_mahjong.tiles import BASE_TILE_COUNT, WHITE_DRAGON


class GameTests(unittest.TestCase):
    def test_public_state_never_exposes_ai_hands(self):
        game = XiamenMahjongGame(seed=20260803)
        state = game.public_state()
        self.assertEqual(len(state["players"][0]["hand"]), state["players"][0]["hand_count"])
        self.assertTrue(all(player["hand"] is None for player in state["players"][1:]))
        self.assertLess(state["gold_tile"]["id"], BASE_TILE_COUNT)

    def test_public_draw_records_replacement_flower_faces(self):
        game = XiamenMahjongGame(seed=20260807, auto_advance=False)
        player = game.players[1]
        # A normal draw may reveal one or more flower replacements before the
        # concealed playable tile.  The table sees the former, never the latter.
        game.wall = [34, 35, 0]

        game._start_turn(1)

        event = game.public_actions[-1]
        self.assertEqual(event["kind"], "draw")
        self.assertEqual(event["seat"], 1)
        self.assertEqual(event["tiles"], [34, 35])
        self.assertNotIn("tile", event)
        self.assertEqual(game.last_drawn_flowers[1], (34, 35))
        self.assertEqual(player.flowers[-2:], [34, 35])
        self.assertEqual(game.last_drawn_tiles[1], 0)

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

    def test_classic_game_can_run_to_a_settlement(self):
        game = XiamenMahjongGame(seed=20260804, rules=XiamenRules.classic())
        steps = 0
        while game.phase != "over" and steps < 400:
            actions = game.human_actions()
            self.assertTrue(actions)
            preferred = next(
                (action for action in actions if action["kind"] in {"hu", "advance_tour"}),
                next((action for action in actions if action["kind"] == "discard"), actions[0]),
            )
            game.apply_human_action(preferred)
            steps += 1
        self.assertEqual(game.phase, "over")
        self.assertLess(steps, 400)

    def test_gold_discard_skips_all_claims(self):
        game = XiamenMahjongGame(seed=5)
        game.phase = "discard"
        game.current_player = 0
        game.gold_tile = 33
        for player in game.players:
            player.score = 0
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

    def test_classic_profile_stops_at_the_sixteen_tile_dead_wall(self):
        game = XiamenMahjongGame(seed=17, rules=XiamenRules.classic())
        self.assertEqual(game.gold_indicator, game.gold_tile)
        game.wall = list(range(16))
        game._start_turn(0)
        self.assertEqual(game.phase, "over")
        self.assertEqual(game.win_type, "draw")

    def test_classic_profile_blocks_discard_win_with_gold_in_hand(self):
        game = XiamenMahjongGame(seed=19, rules=XiamenRules.classic())
        game.phase = "response"
        game.gold_tile = 33
        game.last_discard = 27
        game.discarder = 0
        game.players[1].hand = [0, 1, 2, 3, 4, 33, 9, 10, 11, 18, 18, 18, 27]
        actions = game._response_actions(1)
        self.assertNotIn("hu", [action.kind for action in actions])

    def test_classic_profile_offers_white_dragon_substitute_for_pong(self):
        game = XiamenMahjongGame(seed=23, rules=XiamenRules.classic())
        game.gold_tile = 8
        game.players[1].hand = [WHITE_DRAGON, WHITE_DRAGON]
        self.assertIn(
            [WHITE_DRAGON, WHITE_DRAGON],
            game._claim_consumptions(game.players[1].hand, 8, 2),
        )

    def test_classic_deals_sixteen_tiles_before_the_dealer_draws(self):
        game = XiamenMahjongGame(seed=31, rules=XiamenRules.classic())
        self.assertEqual(sum(len(player.hand) for player in game.players), 65)
        self.assertEqual(len(game.players[game.current_player].hand), 17)
        self.assertTrue(
            all(
                len(player.hand) == (17 if player.seat == game.current_player else 16)
                for player in game.players
            )
        )

    def test_classic_discard_win_is_paid_by_all_three_players(self):
        game = XiamenMahjongGame(seed=37, rules=XiamenRules.classic())
        game.gold_tile = 33
        game.dealer = 0
        game.dealer_streak = 0
        for player in game.players:
            player.score = 0
        game.players[1].flowers = []
        game.players[1].melds = []
        game.players[1].hand = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 31]
        game.last_discard = 31
        game.discarder = 0
        game._finish_win(1, "discard")
        self.assertEqual(game.players[0].score, -8)
        self.assertEqual(game.players[2].score, -8)
        self.assertEqual(game.players[3].score, -8)
        self.assertEqual(game.players[1].score, 24)

    def test_travelling_gold_declaration_and_upgrade(self):
        game = XiamenMahjongGame(seed=41, rules=XiamenRules.classic())
        game.gold_tile = 33
        five_melds = [0, 1, 2, 3, 4, 5, 9, 10, 11, 18, 19, 20, 27, 27, 27]
        game.phase = "discard"
        game.current_player = 0
        game.players[0].hand = [*five_melds, 31, 33]
        for player in game.players[1:]:
            player.hand = []
        game._apply_turn_action(0, GameAction("discard", 31))
        self.assertEqual(game.tour_state["owner"], 0)
        self.assertEqual(game.tour_state["level"], 1)

        game.phase = "discard"
        game.current_player = 0
        game.players[0].hand = [*five_melds, 33, 33]
        self.assertTrue(game._can_advance_tour(0))
        kinds = {action["kind"] for action in game._turn_actions(0)}
        self.assertEqual(kinds, {"hu", "advance_tour"})

    def test_three_gold_instant_win_uses_opening_multiplier(self):
        game = XiamenMahjongGame(seed=43, rules=XiamenRules.classic())
        game.phase = "discard"
        game.gold_tile = 33
        game.players[0].hand = [33, 33, 33, *range(13)]
        game.first_turn_pending.add(0)
        game._start_turn(0)
        self.assertEqual(game.win_type, "three_gold_open")
        self.assertEqual(game.win_pattern, "三金倒（开局）")
        self.assertEqual(game.score_breakdown["multiplier"], 4)

    def test_dealer_base_doubles_with_each_continuation(self):
        game = XiamenMahjongGame(
            seed=47,
            rules=XiamenRules.classic(),
            dealer=0,
            dealer_streak=2,
        )
        for player in game.players:
            player.score = 0
        game.gold_tile = 33
        game.players[0].flowers = []
        game.players[0].melds = []
        game.players[0].hand = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 31, 31
        ]
        game._finish_win(0, "self_draw")
        self.assertEqual(game.score_breakdown["base"], 64)
        self.assertEqual(game.score_breakdown["multiplier"], 2)

    def test_opening_gold_capture_scores_four_times_and_one_gold_water(self):
        game = XiamenMahjongGame(seed=59, rules=XiamenRules.classic())
        game.dealer = 0
        game.gold_indicator = 31
        game.gold_tile = 31
        for player in game.players:
            player.hand = []
            player.flowers = []
            player.melds = []
            player.score = 0
        game.players[1].hand = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 32]
        self.assertEqual(game._opening_gold_winner(), 1)
        game._finish_win(1, "opening_gold")
        self.assertEqual(game.win_pattern, "抢金")
        self.assertEqual(game.score_breakdown["multiplier"], 4)
        self.assertIn("真金", [item["label"] for item in game.score_breakdown["items"]])

    def test_complete_flower_family_totals_eight_water(self):
        game = XiamenMahjongGame(seed=61, rules=XiamenRules.classic())
        game.dealer = 0
        game.gold_tile = 33
        for player in game.players:
            player.score = 0
        game.players[1].flowers = [34, 35, 36, 37]
        game.players[1].melds = []
        game.players[1].hand = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 31, 31
        ]
        game._finish_win(1, "self_draw")
        self.assertEqual(game.score_breakdown["water"], 8)

    def test_white_dragon_substitute_pong_cannot_be_promoted_to_added_kong(self):
        game = XiamenMahjongGame(seed=27, rules=XiamenRules.classic())
        game.phase = "discard"
        game.current_player = 0
        game.gold_tile = 8
        game.players[0].hand = [0]
        game.players[0].melds = [{"kind": "pong", "tiles": [0, 0, WHITE_DRAGON]}]
        self.assertNotIn("add_kan", [action["kind"] for action in game.human_actions()])

    def test_classic_profile_enforces_honor_follow(self):
        game = XiamenMahjongGame(seed=29, rules=XiamenRules.classic())
        game.phase = "discard"
        game.current_player = 0
        game.gold_tile = 8
        game.players[1].discards = [27]
        game.players[0].hand = [27, 0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]
        actions = game.human_actions()
        discards = [action["tile"] for action in actions if action["kind"] == "discard"]
        self.assertEqual(discards, [27])

    def test_teacher_respects_forced_honor_follow_before_kong(self):
        game = XiamenMahjongGame(seed=67, rules=XiamenRules.classic())
        game.phase = "discard"
        game.current_player = 0
        game.gold_tile = 25
        game.players[1].discards = [27]
        game.players[0].hand = [
            0, 3, 4, 9, 9, 9, 9, 10, 10, 11, 14, 16, 18, 22, 27, 30, 30
        ]
        action = game.teacher.choose_turn_action(game, 0)
        self.assertEqual(action, GameAction("discard", 27))

    def test_availability_teacher_counts_only_public_remaining_waits(self):
        game = XiamenMahjongGame(seed=73, rules=XiamenRules.classic())
        game.phase = "discard"
        game.current_player = 0
        game.gold_tile = 33
        game.gold_indicator = 32
        game.players[0].hand = [
            0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 13
        ]
        game.players[1].discards = [0, 0, 0]
        ranked = AvailabilityTeacherAgent().explain_discard(game, 0)
        self.assertTrue(ranked)
        self.assertTrue(all("wait_availability" in item for item in ranked))
        self.assertTrue(all(int(item["wait_availability"]) >= 0 for item in ranked))

    def test_availability_teacher_rejects_negative_wait_value(self):
        with self.assertRaises(ValueError):
            AvailabilityTeacherAgent(wait_copy_value=-0.1)

    def test_public_state_marks_the_current_drawn_tile(self):
        game = XiamenMahjongGame(seed=71, rules=XiamenRules.classic(), dealer=0)
        state = game.public_state()
        current = state["players"][state["current_player"]]
        self.assertIsNotNone(current["drawn_tile"])
        self.assertIn(current["drawn_tile"], current["hand"])

    def test_latest_discard_stays_in_river_while_responses_are_offered(self):
        game = XiamenMahjongGame(seed=73, rules=XiamenRules.classic())
        game.phase = "discard"
        game.current_player = 0
        discarded = next(tile for tile in game.players[0].hand if tile != game.gold_tile)
        game._apply_turn_action(0, GameAction("discard", discarded))
        self.assertEqual(game.latest_discard, discarded)
        self.assertEqual(game.latest_discard_seat, 0)
        self.assertEqual(game.players[0].discards[-1], discarded)

    def test_claimed_discard_moves_from_river_into_meld(self):
        game = XiamenMahjongGame(seed=79, rules=XiamenRules.classic())
        game.gold_tile = 8
        game.last_discard = 5
        game.discarder = 0
        game.latest_discard = 5
        game.latest_discard_seat = 0
        game.players[0].discards = [5]
        game.players[1].hand = [5, 5]
        game._apply_claim(1, GameAction("pong", 5, (5, 5)))
        self.assertEqual(game.players[0].discards, [])
        self.assertIsNone(game.latest_discard)
        self.assertIsNone(game.latest_discard_seat)
        self.assertEqual(game.players[1].melds[-1]["tiles"], [5, 5, 5])


if __name__ == "__main__":
    unittest.main()

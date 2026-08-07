import copy
import unittest

from xiamen_mahjong.game import XiamenMahjongGame
from xiamen_mahjong.agents import (
    AvailabilityTeacherAgent,
    ExactOneDrawTenpaiTieBreakTeacherAgent,
    GameAction,
    HeuristicTeacherAgent,
    MeldContinuationTeacherAgent,
    OnePlyLookaheadTeacherAgent,
    RiskAwareTeacherAgent,
)
from xiamen_mahjong.rules import XiamenRules
from xiamen_mahjong.tiles import BASE_TILE_COUNT, WHITE_DRAGON, gold_indicator_index


class GameTests(unittest.TestCase):
    def test_gold_indicator_scan_wraps_and_skips_flowers(self):
        # Dice 1+1 starts at index 3 and scans 3, 2, 1, 0, then wraps.
        self.assertEqual(gold_indicator_index([34, 0, 35, 36, 1], (1, 1)), 1)
        self.assertEqual(gold_indicator_index([34, 35, 36, 37, 1], (1, 1)), 4)
        self.assertIsNone(gold_indicator_index([34, 35, 36], (1, 1)))

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

    def test_slow_mode_advances_exactly_until_a_manual_seat_is_needed(self):
        game = XiamenMahjongGame(
            seed=202608071,
            rules=XiamenRules.classic(),
            auto_advance=False,
            human_seat=0,
        )
        steps = 0
        while game.phase != "over" and game.manual_action_seat() is None:
            self.assertTrue(game.awaiting_ai_action())
            self.assertTrue(game.advance_one_ai())
            steps += 1
            self.assertLess(steps, 24)
        self.assertGreater(steps, 0)
        if game.phase != "over":
            self.assertEqual(game.manual_action_seat(), 0)
            self.assertTrue(game.human_actions())

    def test_four_manual_seats_show_only_the_current_actor_hand(self):
        game = XiamenMahjongGame(
            seed=202608072,
            rules=XiamenRules.classic(),
            human_seats=range(4),
        )
        self.assertNotEqual(game.phase, "over")
        actor = game.manual_action_seat()
        self.assertIsNotNone(actor)
        state = game.public_state()
        self.assertEqual(state["viewer_seat"], actor)
        self.assertEqual(state["action_seat"], actor)
        self.assertEqual(
            [player["seat"] for player in state["players"] if player["hand"] is not None],
            [actor],
        )
        action = next(
            (item for item in game.human_actions() if item["kind"] == "discard"),
            game.human_actions()[0],
        )
        game.apply_human_action(action)
        if game.phase != "over":
            next_actor = game.manual_action_seat()
            self.assertIsNotNone(next_actor)
            next_state = game.public_state()
            self.assertEqual(next_state["viewer_seat"], next_actor)
            self.assertEqual(
                [
                    player["seat"]
                    for player in next_state["players"]
                    if player["hand"] is not None
                ],
                [next_actor],
            )

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

    def test_meld_continuation_teacher_can_decline_a_low_value_call(self):
        game = XiamenMahjongGame(seed=79, rules=XiamenRules.classic())
        game.gold_tile = 33
        game.phase = "response"
        game.discarder = 0
        game.last_discard = 4
        game.players[1].hand = [
            0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
        ]
        options = [GameAction("pass"), GameAction("pong", 4, (4, 4))]
        agent = MeldContinuationTeacherAgent(minimum_claim_gain=10_000)
        self.assertEqual(agent.choose_response(game, 1, options), GameAction("pass"))
        explanation = agent.explain_response(game, 1, options)
        self.assertEqual(explanation[0]["kind"], "pong")
        self.assertIn("post_call_discard", explanation[0])

    def test_meld_continuation_teacher_rejects_negative_gain(self):
        with self.assertRaises(ValueError):
            MeldContinuationTeacherAgent(minimum_claim_gain=-0.1)

    def test_meld_continuation_response_ignores_hidden_wall_and_other_hands(self):
        game = XiamenMahjongGame(seed=81, rules=XiamenRules.classic())
        game.gold_tile = 33
        game.phase = "response"
        game.discarder = 0
        game.last_discard = 4
        game.players[1].hand = [
            0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
        ]
        options = [GameAction("pass"), GameAction("pong", 4, (4, 4))]
        altered = copy.deepcopy(game)
        altered.players[2].hand[0], altered.wall[0] = (
            altered.wall[0],
            altered.players[2].hand[0],
        )
        altered.players[2].hand.sort()
        agent = MeldContinuationTeacherAgent(minimum_claim_gain=0)
        self.assertEqual(
            agent.choose_response(game, 1, options),
            agent.choose_response(altered, 1, options),
        )

    def test_risk_aware_teacher_uses_only_public_information(self):
        game = XiamenMahjongGame(
            seed=202611001,
            rules=XiamenRules.classic(),
            auto_advance=False,
            human_seat=-1,
        )
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

        agent = RiskAwareTeacherAgent(risk_weight=4.0)
        original = agent.choose_turn_action(game, player_id)
        changed_hidden_state = agent.choose_turn_action(altered, player_id)
        ranked = agent.explain_discard(game, player_id)

        self.assertEqual(original, changed_hidden_state)
        self.assertTrue(ranked)
        self.assertTrue(all("public_danger" in item for item in ranked))
        self.assertTrue(all(float(item["risk_penalty"]) >= 0 for item in ranked))

    def test_risk_aware_teacher_rejects_negative_weight(self):
        with self.assertRaises(ValueError):
            RiskAwareTeacherAgent(risk_weight=-0.1)

    def test_one_ply_teacher_uses_only_public_information(self):
        game = XiamenMahjongGame(
            seed=202608391,
            rules=XiamenRules.classic(),
            auto_advance=False,
            human_seat=-1,
        )
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

        agent = OnePlyLookaheadTeacherAgent()
        original = agent.choose_turn_action(game, player_id)
        changed_hidden_state = agent.choose_turn_action(altered, player_id)
        ranked = agent.explain_discard(game, player_id)

        self.assertEqual(original, changed_hidden_state)
        self.assertTrue(ranked)
        self.assertTrue(all("next_draw_score" in item for item in ranked))
        self.assertTrue(
            all("immediate_shape_score" in item for item in ranked)
        )
        if original.kind == "discard":
            self.assertIn(original.tile, game.players[player_id].hand)

    def test_exact_one_draw_tiebreak_uses_only_public_information(self):
        game = XiamenMahjongGame(
            seed=202614200,
            rules=XiamenRules.classic(),
            auto_advance=False,
            human_seat=-1,
        )
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

        agent = ExactOneDrawTenpaiTieBreakTeacherAgent()
        self.assertEqual(
            agent.choose_turn_action(game, player_id),
            agent.choose_turn_action(altered, player_id),
        )
        # An unreachable threshold must exactly preserve the frozen Teacher.
        strict = ExactOneDrawTenpaiTieBreakTeacherAgent(
            minimum_live_advantage=1_000_000
        )
        self.assertEqual(
            strict.choose_turn_action(game, player_id),
            HeuristicTeacherAgent().choose_turn_action(game, player_id),
        )

    def test_exact_one_draw_tiebreak_reorders_only_after_a_strict_gate(self):
        game = XiamenMahjongGame(
            seed=202614200,
            rules=XiamenRules.classic(),
            auto_advance=False,
            human_seat=-1,
        )
        player_id = game.current_player
        frozen = HeuristicTeacherAgent().explain_discard(game, player_id)
        self.assertFalse(frozen[0]["waits"])
        self.assertEqual(float(frozen[0]["score"]), float(frozen[1]["score"]))
        frozen_tile = int(frozen[0]["tile"])
        selected_tile = int(frozen[1]["tile"])

        agent = ExactOneDrawTenpaiTieBreakTeacherAgent()

        def controlled_potential(_game, _player_id, hand_after_discard, _visible):
            return 2 if selected_tile not in hand_after_discard else 0

        agent._live_route_potential = controlled_potential  # type: ignore[method-assign]
        ranked = agent.explain_discard(game, player_id)
        self.assertEqual(int(ranked[0]["tile"]), selected_tile)
        self.assertNotEqual(int(ranked[0]["tile"]), frozen_tile)
        self.assertTrue(ranked[0]["selected_by_exact_one_draw_tiebreak"])
        self.assertEqual(ranked[0]["teacher_score"], frozen[1]["score"])

    def test_exact_one_draw_tiebreak_rejects_invalid_gates(self):
        with self.assertRaises(ValueError):
            ExactOneDrawTenpaiTieBreakTeacherAgent(score_margin=-0.1)
        with self.assertRaises(ValueError):
            ExactOneDrawTenpaiTieBreakTeacherAgent(minimum_live_advantage=0)

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

import unittest

from xiamen_mahjong.hand import (
    hand_quality,
    hand_quality_components,
    is_winning_hand,
    one_draw_tenpai_profile,
    one_draw_tenpai_routes,
    public_draw_improvement_profile,
    standard_hand_shanten,
    wait_tiles,
    winning_pattern,
)
from xiamen_mahjong.tiles import WHITE_DRAGON


class HandTests(unittest.TestCase):
    def test_standard_hand_is_winning(self):
        hand = [0, 1, 2, 3, 4, 5, 9, 10, 11, 18, 18, 18, 27, 27]
        self.assertTrue(is_winning_hand(hand, gold_tile=33))
        self.assertEqual(winning_pattern(hand, gold_tile=33), "标准和")

    def test_gold_completes_a_missing_sequence_tile(self):
        hand = [0, 1, 33, 3, 4, 5, 9, 10, 11, 18, 18, 18, 27, 27]
        self.assertTrue(is_winning_hand(hand, gold_tile=33))

    def test_seven_pairs_is_winning(self):
        hand = [0, 0, 1, 1, 9, 9, 10, 10, 18, 18, 19, 19, 27, 27]
        self.assertTrue(is_winning_hand(hand, gold_tile=33))
        self.assertEqual(winning_pattern(hand, gold_tile=33), "七对")

    def test_wait_tiles_detects_completion(self):
        hand = [0, 1, 2, 3, 4, 5, 9, 10, 11, 18, 18, 18, 27]
        self.assertIn(27, wait_tiles(hand, gold_tile=33))

    def test_one_draw_tenpai_profile_finds_a_real_progressing_draw(self):
        # This 16-tile classic hand is one structural step behind the
        # five-meld-plus-pair shape.  Drawing 3万 creates a real tenpai route;
        # the helper may find a stronger wait than a hand-written route, but
        # it must do so without looking at a wall or another player's hand.
        hand = [
            0, 1, 8, 3, 4, 5, 9, 10, 11, 18, 19, 20, 27, 27, 27, 31
        ]
        profile = one_draw_tenpai_profile(
            hand,
            gold_tile=33,
            melds_required=5,
            allow_seven_pairs=False,
        )
        self.assertIn(2, profile)
        self.assertGreaterEqual(len(profile[2]), 1)

    def test_one_draw_routes_honor_a_follow_discard_constraint(self):
        hand = [
            0, 1, 8, 3, 4, 5, 9, 10, 11, 18, 19, 20, 27, 27, 27, 31
        ]
        unconstrained = one_draw_tenpai_routes(
            hand,
            gold_tile=33,
            melds_required=5,
            allow_seven_pairs=False,
        )
        constrained = one_draw_tenpai_routes(
            hand,
            gold_tile=33,
            melds_required=5,
            allow_seven_pairs=False,
            legal_discards=lambda after_draw: [31],
        )
        self.assertIn(2, unconstrained)
        self.assertIn(2, constrained)
        self.assertTrue(all(discarded == 31 for discarded, _waits in constrained[2]))

    def test_white_dragon_can_be_an_explicit_gold_proxy(self):
        hand = [0, 1, WHITE_DRAGON, 3, 4, 5, 9, 10, 11, 18, 18, 18, 27, 27]
        self.assertTrue(
            is_winning_hand(
                hand,
                gold_tile=8,
                wildcard_tiles={8},
                proxy_tile=WHITE_DRAGON,
                proxy_as=2,
            )
        )

    def test_classic_seventeen_tile_hand_needs_five_melds_and_a_pair(self):
        hand = [0, 1, 2, 3, 4, 5, 9, 10, 11, 18, 19, 20, 27, 27, 27, 31, 31]
        self.assertTrue(
            is_winning_hand(
                hand,
                gold_tile=33,
                melds_required=5,
                allow_seven_pairs=False,
            )
        )
        self.assertFalse(is_winning_hand(hand[:-3], gold_tile=33, melds_required=5))

    def test_white_proxy_is_fixed_face_value_not_a_wildcard(self):
        hand = [0, 1, WHITE_DRAGON, 3, 4, 5, 9, 10, 11, 18, 19, 20, 27, 27]
        self.assertFalse(
            is_winning_hand(
                hand,
                gold_tile=8,
                wildcard_tiles={8},
                proxy_tile=WHITE_DRAGON,
                proxy_as=8,
            )
        )

    def test_classic_regular_hand_shanten_distinguishes_three_distances(self):
        complete = [
            0, 1, 2, 3, 4, 5, 9, 10, 11, 18, 19, 20, 27, 27, 27, 31, 31
        ]
        tenpai = complete[:-1]
        one_away = [
            0, 1, 8, 3, 4, 5, 9, 10, 11, 18, 19, 20, 27, 27, 27, 31
        ]
        self.assertEqual(
            standard_hand_shanten(complete, 33, melds_required=5), -1
        )
        self.assertEqual(
            standard_hand_shanten(tenpai, 33, melds_required=5), 0
        )
        self.assertEqual(
            standard_hand_shanten(one_away, 33, melds_required=5), 1
        )

    def test_gold_reduces_regular_hand_shanten_as_a_real_wildcard(self):
        hand = [
            0, 1, 33, 3, 4, 5, 9, 10, 11, 18, 19, 20, 27, 27, 27, 31, 31
        ]
        self.assertEqual(
            standard_hand_shanten(hand, 33, melds_required=5), -1
        )
        self.assertGreaterEqual(
            standard_hand_shanten(hand, None, melds_required=5), 0
        )

    def test_white_proxy_has_a_fixed_face_in_regular_hand_shanten(self):
        hand = [
            0, 1, WHITE_DRAGON, 3, 4, 5, 9, 10, 11,
            18, 19, 20, 27, 27, 27, 31, 31,
        ]
        self.assertEqual(
            standard_hand_shanten(
                hand,
                8,
                melds_required=5,
                wildcard_tiles={8},
                proxy_tile=WHITE_DRAGON,
                proxy_as=2,
            ),
            -1,
        )
        self.assertGreaterEqual(
            standard_hand_shanten(
                hand,
                8,
                melds_required=5,
                wildcard_tiles={8},
                proxy_tile=WHITE_DRAGON,
                proxy_as=8,
            ),
            0,
        )

    def test_public_draw_profile_respects_live_counts_and_discard_constraint(self):
        hand = [
            0, 1, 8, 3, 4, 5, 9, 10, 11, 18, 19, 20, 27, 27, 27, 31
        ]
        only_three_wan_live = [0] * 34
        only_three_wan_live[2] = 3
        profile = public_draw_improvement_profile(
            hand,
            only_three_wan_live,
            33,
            melds_required=5,
            legal_discards=lambda _after_draw: [31],
        )
        self.assertEqual(profile.shanten, 1)
        self.assertEqual(profile.improving_faces, (2,))
        self.assertEqual(profile.improving_live_copies, 3)

    def test_hand_quality_components_exactly_reconstruct_frozen_score(self):
        hand = [0, 0, 0, 1, 2, 3, 9, 10, 18, 20, 27, 27, 33]
        parts = hand_quality_components(hand, 33, meld_count=1)
        rebuilt = (
            parts.fixed_melds * 28
            + parts.gold_tiles * 12
            + parts.triplet_groups * 14
            + parts.pair_remainders * 5
            + parts.adjacent_overlap * 2.5
            + parts.gap_overlap * 1.25
            + parts.sequence_overlap * 5
        )
        self.assertEqual(rebuilt, hand_quality(hand, 33, meld_count=1))


if __name__ == "__main__":
    unittest.main()

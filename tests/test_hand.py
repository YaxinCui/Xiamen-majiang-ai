import unittest

from xiamen_mahjong.hand import is_winning_hand, wait_tiles, winning_pattern
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


if __name__ == "__main__":
    unittest.main()

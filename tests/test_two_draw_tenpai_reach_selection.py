import unittest

from scripts.select_two_draw_tenpai_reach_teacher import (
    MINIMUM_PROBABILITY_ADVANTAGE,
    SCORE_MARGIN,
    SELECTION_HANDS,
    SELECTION_SEED,
    STRATIFIED_SCENARIOS,
    fixed_candidate,
)


class TwoDrawTenpaiReachSelectionTests(unittest.TestCase):
    def test_fixed_candidate_matches_the_prespecified_gate(self):
        candidate = fixed_candidate()
        self.assertEqual(candidate.score_margin, SCORE_MARGIN)
        self.assertEqual(
            candidate.minimum_probability_advantage,
            MINIMUM_PROBABILITY_ADVANTAGE,
        )
        self.assertEqual(candidate.STRATIFIED_SCENARIOS, STRATIFIED_SCENARIOS)

    def test_selection_is_fixed_to_one_hundred_new_walls(self):
        self.assertEqual(SELECTION_HANDS, 100)
        self.assertEqual(SELECTION_SEED, 202630000)


if __name__ == "__main__":
    unittest.main()

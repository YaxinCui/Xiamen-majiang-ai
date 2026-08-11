import unittest

from scripts.select_public_tenpai_value_teacher import (
    MINIMUM_WEIGHTED_VALUE_ADVANTAGE,
    SCORE_MARGIN,
    SELECTION_HANDS,
    SELECTION_SEED,
    TERMINAL_HANDS,
    TERMINAL_SEED,
    fixed_candidate,
)


class PublicTenpaiValueSelectionTests(unittest.TestCase):
    def test_fixed_candidate_matches_the_prespecified_gate(self):
        candidate = fixed_candidate()
        self.assertEqual(candidate.score_margin, SCORE_MARGIN)
        self.assertEqual(
            candidate.minimum_weighted_value_advantage,
            MINIMUM_WEIGHTED_VALUE_ADVANTAGE,
        )

    def test_selection_and_terminal_walls_are_disjoint(self):
        selection = set(range(SELECTION_SEED, SELECTION_SEED + SELECTION_HANDS))
        terminal = set(range(TERMINAL_SEED, TERMINAL_SEED + TERMINAL_HANDS))
        self.assertTrue(selection.isdisjoint(terminal))


if __name__ == "__main__":
    unittest.main()

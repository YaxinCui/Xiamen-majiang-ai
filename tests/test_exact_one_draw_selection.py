import unittest

from scripts.select_exact_one_draw_tenpai_teacher import (
    MINIMUM_LIVE_ADVANTAGE,
    SCORE_MARGIN,
    fixed_candidate,
)


class ExactOneDrawSelectionTests(unittest.TestCase):
    def test_fixed_candidate_matches_the_prespecified_gate(self):
        candidate = fixed_candidate()
        self.assertEqual(candidate.score_margin, SCORE_MARGIN)
        self.assertEqual(candidate.minimum_live_advantage, MINIMUM_LIVE_ADVANTAGE)


if __name__ == "__main__":
    unittest.main()

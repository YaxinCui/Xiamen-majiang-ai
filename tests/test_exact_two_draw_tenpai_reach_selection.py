import unittest

from scripts.select_exact_two_draw_tenpai_reach_teacher_v2 import (
    MINIMUM_PROBABILITY_ADVANTAGE,
    SCORE_MARGIN,
    SELECTION_HANDS,
    SELECTION_SEED,
    STRATIFIED_PROPOSAL_SCENARIOS,
    fixed_candidate,
)


class ExactTwoDrawSelectionTests(unittest.TestCase):
    def test_candidate_is_the_frozen_two_stage_gate(self):
        candidate = fixed_candidate()
        self.assertEqual(candidate.score_margin, SCORE_MARGIN)
        self.assertEqual(
            candidate.minimum_probability_advantage,
            MINIMUM_PROBABILITY_ADVANTAGE,
        )
        self.assertEqual(
            candidate._stratified_proposal.STRATIFIED_SCENARIOS,
            STRATIFIED_PROPOSAL_SCENARIOS,
        )
        self.assertEqual(
            candidate.SOLVER_VERSION, "public-two-draw-exact-dp-v2"
        )

    def test_selection_uses_one_hundred_fresh_walls(self):
        self.assertEqual(SELECTION_HANDS, 100)
        self.assertEqual(SELECTION_SEED, 202636100)


if __name__ == "__main__":
    unittest.main()

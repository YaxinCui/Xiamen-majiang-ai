import unittest

from scripts.select_deficiency_meld_teacher import (
    SELECTION_HANDS,
    SELECTION_SEED,
    fixed_candidate,
)
from xiamen_mahjong.agents import DeficiencyMeldTeacherAgent


class DeficiencyMeldSelectionTests(unittest.TestCase):
    def test_fixed_candidate_is_the_exact_response_policy(self):
        self.assertIsInstance(fixed_candidate(), DeficiencyMeldTeacherAgent)

    def test_selection_is_fixed_to_one_hundred_fresh_walls(self):
        self.assertEqual(SELECTION_HANDS, 100)
        self.assertEqual(SELECTION_SEED, 202632000)


if __name__ == "__main__":
    unittest.main()

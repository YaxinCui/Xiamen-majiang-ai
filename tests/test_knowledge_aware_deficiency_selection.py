import unittest

from scripts.select_knowledge_aware_deficiency_teacher import (
    SCORE_MARGIN,
    SELECTION_HANDS,
    SELECTION_SEED,
    fixed_candidate,
)


class KnowledgeAwareDeficiencySelectionTests(unittest.TestCase):
    def test_fixed_candidate_matches_the_prespecified_gate(self):
        self.assertEqual(fixed_candidate().score_margin, SCORE_MARGIN)

    def test_selection_is_the_fixed_one_hundred_wall_screen(self):
        self.assertEqual(SELECTION_HANDS, 100)
        self.assertEqual(SELECTION_SEED, 202625000)


if __name__ == "__main__":
    unittest.main()

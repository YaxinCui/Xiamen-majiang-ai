import unittest

from scripts.select_two_draw_targeted_intervention_v1 import (
    CANDIDATE_ACTION_PROBABILITY,
    FIRST_SEED,
    MINIMUM_INTERVENTIONS,
    PHYSICAL_WALLS,
)


class TwoDrawTargetedInterventionSelectionTests(unittest.TestCase):
    def test_selection_budget_and_binary_support_are_frozen(self):
        self.assertEqual(PHYSICAL_WALLS, 100)
        self.assertEqual(FIRST_SEED, 202635100)
        self.assertEqual(CANDIDATE_ACTION_PROBABILITY, 0.5)
        self.assertEqual(MINIMUM_INTERVENTIONS, 100)


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.select_targeted_kan_intervention_v1 import (
    BEHAVIOR_SEED,
    FALLBACK_PROBABILITY,
    FIRST_SEED,
    MINIMUM_INTERVENTIONS,
    MINIMUM_KIND_INTERVENTIONS,
    PHYSICAL_WALLS,
)


class TargetedKanSelectionTests(unittest.TestCase):
    def test_budget_and_support_are_frozen(self):
        self.assertEqual(PHYSICAL_WALLS, 100)
        self.assertEqual(FIRST_SEED, 202637000)
        self.assertEqual(BEHAVIOR_SEED, 202637200)
        self.assertEqual(FALLBACK_PROBABILITY, 0.5)
        self.assertEqual(MINIMUM_INTERVENTIONS, 100)
        self.assertEqual(MINIMUM_KIND_INTERVENTIONS, 30)


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.select_risk_aware_teacher import choose_candidate


class RiskAwareSelectionTests(unittest.TestCase):
    def test_choose_uses_lower_bound_then_smaller_risk_weight(self):
        def audit(weight, lower, passes=True):
            return {
                "risk_weight": weight,
                "paired_against_heuristic_teacher": {
                    "paired_seed_score_delta_95pct_low": lower
                },
                "passes": passes,
            }

        chosen = choose_candidate((audit(4.0, 1.0), audit(2.0, 1.0)))
        self.assertEqual(chosen["risk_weight"], 2.0)
        self.assertIsNone(choose_candidate((audit(1.0, -0.1, False),)))


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.select_meld_continuation_teacher import choose_candidate


class MeldContinuationSelectionTests(unittest.TestCase):
    def test_choose_uses_lower_bound_then_smaller_minimum_gain(self):
        def audit(gain, lower, passes=True):
            return {
                "minimum_claim_gain": gain,
                "paired_against_heuristic_teacher": {
                    "paired_seed_score_delta_95pct_low": lower
                },
                "passes": passes,
            }

        chosen = choose_candidate((audit(32.0, 1.0), audit(16.0, 1.0)))
        self.assertEqual(chosen["minimum_claim_gain"], 16.0)
        self.assertIsNone(choose_candidate((audit(0.0, -0.1, False),)))


if __name__ == "__main__":
    unittest.main()

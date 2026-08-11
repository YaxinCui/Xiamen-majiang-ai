import unittest

from scripts.audit_confidence_override_belief_advantage import paired_delta_summary


class ConfidenceOverrideBeliefAdvantageTests(unittest.TestCase):
    def test_paired_delta_summary_uses_world_differences(self):
        summary = paired_delta_summary([2.0, 4.0, 6.0], confidence_z=1.0)
        self.assertEqual(summary["worlds"], 3)
        self.assertEqual(summary["mean"], 4.0)
        self.assertAlmostEqual(summary["stderr"], 2.0 / (3.0**0.5))
        self.assertAlmostEqual(summary["low"], 4.0 - 2.0 / (3.0**0.5))

    def test_empty_summary_is_explicit(self):
        self.assertEqual(
            paired_delta_summary([]),
            {
                "worlds": 0,
                "mean": None,
                "stderr": None,
                "low": None,
                "high": None,
            },
        )

    def test_negative_confidence_is_rejected(self):
        with self.assertRaises(ValueError):
            paired_delta_summary([1.0], confidence_z=-0.1)


if __name__ == "__main__":
    unittest.main()

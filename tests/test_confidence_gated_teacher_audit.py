import unittest

from scripts.audit_confidence_gated_teacher import choose_strict_margin


class ConfidenceGatedTeacherAuditTests(unittest.TestCase):
    def test_margin_hits_the_largest_coverage_not_above_target(self):
        margin = choose_strict_margin(
            [5.0, 4.0, 3.0, 2.0],
            eligible_decisions=100,
            target_override_rate=0.02,
        )
        self.assertEqual(margin, 3.5)
        self.assertEqual(sum(gap > margin for gap in [5.0, 4.0, 3.0, 2.0]), 2)

    def test_tied_boundary_fails_closed(self):
        margin = choose_strict_margin(
            [5.0, 4.0, 4.0, 3.0],
            eligible_decisions=100,
            target_override_rate=0.02,
        )
        self.assertEqual(margin, 4.0)
        self.assertEqual(sum(gap > margin for gap in [5.0, 4.0, 4.0, 3.0]), 1)

    def test_sparse_disagreements_keep_all_positive_gaps(self):
        margin = choose_strict_margin(
            [1.5], eligible_decisions=100, target_override_rate=0.02
        )
        self.assertEqual(margin, 0.0)

    def test_invalid_or_empty_inputs_fail_closed(self):
        self.assertIsNone(
            choose_strict_margin(
                [], eligible_decisions=100, target_override_rate=0.02
            )
        )
        with self.assertRaises(ValueError):
            choose_strict_margin(
                [1.0], eligible_decisions=100, target_override_rate=0.0
            )


if __name__ == "__main__":
    unittest.main()

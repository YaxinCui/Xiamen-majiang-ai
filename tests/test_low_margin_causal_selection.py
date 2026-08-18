import unittest

from scripts.select_low_margin_top2_causal_residual_v1 import (
    BONFERRONI_Z,
    policy_difference_contribution,
)


class LowMarginCausalSelectionTests(unittest.TestCase):
    def test_binary_randomization_contribution_is_unbiased(self):
        teacher_outcome = -3.0
        alternative_outcome = 7.0
        expectation = 0.5 * policy_difference_contribution(
            terminal_score=teacher_outcome,
            logged_arm="teacher",
            override=True,
        ) + 0.5 * policy_difference_contribution(
            terminal_score=alternative_outcome,
            logged_arm="alternative",
            override=True,
        )
        self.assertEqual(expectation, alternative_outcome - teacher_outcome)

    def test_teacher_fallback_and_no_intervention_are_exact_zero(self):
        for arm in ("teacher", "alternative", "none"):
            self.assertEqual(
                policy_difference_contribution(
                    terminal_score=99.0,
                    logged_arm=arm,
                    override=False,
                    propensity=1.0 if arm == "none" else 0.5,
                ),
                0.0,
            )

    def test_bonferroni_critical_value_is_frozen(self):
        self.assertAlmostEqual(BONFERRONI_Z, 2.4977054744123737)


if __name__ == "__main__":
    unittest.main()

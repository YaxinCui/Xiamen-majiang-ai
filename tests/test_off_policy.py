import unittest

from xiamen_mahjong.off_policy import (
    LoggedIntervention,
    doubly_robust_delta,
    effective_sample_size,
    intervention_estimates,
    ips_delta,
)
from scripts.audit_teacher_response_intervention_ope import (
    teacher_epsilon_propensities,
)


class OffPolicyTests(unittest.TestCase):
    def test_ips_and_dr_recover_a_two_action_grouped_delta(self):
        # Under a uniform behavior policy, one logged target reward of 10 and
        # one logged baseline reward of 0 estimate a +10 target improvement.
        rows = (
            LoggedIntervention(
                group_id="wall-a",
                logged_index=1,
                propensities=(0.5, 0.5),
                reward=10.0,
                baseline_index=0,
                target_index=1,
                direct_values=(0.0, 10.0),
            ),
            LoggedIntervention(
                group_id="wall-a",
                logged_index=0,
                propensities=(0.5, 0.5),
                reward=0.0,
                baseline_index=0,
                target_index=1,
                direct_values=(0.0, 10.0),
            ),
            LoggedIntervention(
                group_id="wall-b",
                logged_index=1,
                propensities=(0.5, 0.5),
                reward=12.0,
                baseline_index=0,
                target_index=1,
                direct_values=(2.0, 12.0),
            ),
            LoggedIntervention(
                group_id="wall-b",
                logged_index=0,
                propensities=(0.5, 0.5),
                reward=2.0,
                baseline_index=0,
                target_index=1,
                direct_values=(2.0, 12.0),
            ),
        )

        self.assertEqual(ips_delta(rows[0]), 20.0)
        self.assertEqual(doubly_robust_delta(rows[0]), 10.0)
        report = intervention_estimates(rows)
        self.assertEqual(report["ips"]["groups"], 2)
        self.assertAlmostEqual(report["ips"]["mean"], 10.0)
        self.assertAlmostEqual(report["doubly_robust"]["mean"], 10.0)
        self.assertEqual(report["support"]["target_matched_logged_action_count"], 2)
        self.assertEqual(report["support"]["baseline_matched_logged_action_count"], 2)

    def test_same_target_and_baseline_has_exact_zero_delta(self):
        row = LoggedIntervention(
            group_id="wall-a",
            logged_index=0,
            propensities=(0.75, 0.25),
            reward=-18.0,
            baseline_index=0,
            target_index=0,
            direct_values=(-12.0, 4.0),
        )
        self.assertEqual(ips_delta(row), 0.0)
        self.assertEqual(doubly_robust_delta(row), 0.0)

    def test_rejects_invalid_support_and_reports_zero_weight_ess(self):
        with self.assertRaisesRegex(ValueError, "归一化"):
            LoggedIntervention(
                group_id="wall-a",
                logged_index=0,
                propensities=(0.2, 0.2),
                reward=0.0,
                baseline_index=0,
                target_index=1,
            )
        self.assertEqual(effective_sample_size((0.0, 0.0)), 0.0)

    def test_teacher_mixture_keeps_exact_logged_action_support(self):
        self.assertEqual(
            teacher_epsilon_propensities(
                action_count=3, teacher_index=1, epsilon=0.4
            ),
            (0.4 / 3.0, 0.4 / 3.0 + 0.6, 0.4 / 3.0),
        )
        with self.assertRaisesRegex(ValueError, "随机化"):
            teacher_epsilon_propensities(action_count=2, teacher_index=0, epsilon=0.0)


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.audit_exact_tie_randomized_ope import (
    deterministic_policy_difference_contribution,
    teacher_epsilon_propensities,
)


class ExactTieRandomizedOpeTests(unittest.TestCase):
    def test_propensities_and_ht_difference_are_unbiased_in_enumeration(self):
        probabilities = teacher_epsilon_propensities(
            action_count=4, teacher_index=0, epsilon=0.8
        )
        self.assertAlmostEqual(sum(probabilities), 1.0)
        for actual, expected in zip(probabilities, (0.4, 0.2, 0.2, 0.2)):
            self.assertAlmostEqual(actual, expected)
        rewards = (3.0, 9.0, -4.0, 1.0)
        expectation = sum(
            probability
            * deterministic_policy_difference_contribution(
                reward=rewards[index],
                logged_index=index,
                baseline_index=0,
                target_index=1,
                propensities=probabilities,
            )
            for index, probability in enumerate(probabilities)
        )
        self.assertAlmostEqual(expectation, rewards[1] - rewards[0])

    def test_no_override_has_exact_zero_contribution(self):
        probabilities = teacher_epsilon_propensities(
            action_count=3, teacher_index=1, epsilon=0.6
        )
        for logged in range(3):
            self.assertEqual(
                deterministic_policy_difference_contribution(
                    reward=99.0,
                    logged_index=logged,
                    baseline_index=1,
                    target_index=1,
                    propensities=probabilities,
                ),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()

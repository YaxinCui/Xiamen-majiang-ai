import unittest

from scripts.collect_low_margin_top2_wall_control_validation_v2 import (
    VALIDATION_BEHAVIOR_SEED,
    VALIDATION_CHUNKS,
    VALIDATION_FIRST_SEED,
)
from scripts.select_low_margin_top2_wall_control_v2 import (
    evaluate_wall_control_gate,
)
from scripts.train_low_margin_top2_wall_control_v2 import (
    _controlled_outcome,
    _inner_fit_mask,
)
from tests.test_wall_control import wall_records


class WallControlV2SelectionTests(unittest.TestCase):
    def test_new_validation_seed_domains_are_disjoint_from_v1(self):
        self.assertEqual(VALIDATION_FIRST_SEED, 202658000)
        self.assertEqual(VALIDATION_CHUNKS, 20)
        self.assertEqual(VALIDATION_BEHAVIOR_SEED, 202673000)
        self.assertGreater(VALIDATION_FIRST_SEED, 202657999)
        self.assertLess(
            VALIDATION_FIRST_SEED + VALIDATION_CHUNKS * 100 - 1,
            202660000,
        )

    def test_inner_split_never_separates_one_wall_group(self):
        groups = ["a" * 32] * 4 + [f"{index:032x}" for index in range(1, 100)]
        mask = _inner_fit_mask(groups).tolist()
        self.assertEqual(len(set(mask[:4])), 1)
        self.assertTrue(any(mask))
        self.assertTrue(any(not value for value in mask))

    def test_controlled_outcome_uses_only_targets_from_same_complete_group(self):
        records = wall_records([10.0, 20.0, 30.0, 40.0])
        outcome, group_ids = _controlled_outcome(records)
        self.assertEqual(group_ids, ["a" * 32] * 4)
        self.assertAlmostEqual(outcome.tolist()[0], 1.0)
        self.assertAlmostEqual(outcome.tolist()[3], 1.5)

    def test_gate_reports_adjusted_and_unadjusted_estimates(self):
        records = wall_records([10.0, 20.0, 30.0, 40.0])
        report = evaluate_wall_control_gate(
            records,
            [2.0, 2.0, 2.0, 2.0],
            target_coverage=1.0,
            threshold=1.0,
            control_coefficient=-1.0,
        )
        self.assertEqual(report["overrides"], 4)
        self.assertIn("insufficient_overrides", report["gate_reasons"])
        self.assertIn("unadjusted_ht_nominal_95pct_diagnostic", report)
        self.assertIn("wall_control_policy_minus_teacher_bonferroni", report)

    def test_gate_rejects_unfrozen_extreme_coefficient(self):
        with self.assertRaises(ValueError):
            evaluate_wall_control_gate(
                wall_records([1.0, 2.0, 3.0, 4.0]),
                [1.0] * 4,
                target_coverage=1.0,
                threshold=0.0,
                control_coefficient=4.01,
            )


if __name__ == "__main__":
    unittest.main()

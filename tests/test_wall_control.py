import itertools
import unittest

from tests.test_causal_residual import causal_record
from xiamen_mahjong.low_margin_intervention import LowMarginCausalRecord
from xiamen_mahjong.wall_control import (
    adjusted_wall_values,
    estimate_control_coefficient,
    leave_one_rotation_baselines,
    minimum_ensemble_effects,
    policy_wall_components,
    wall_control_training_tensors,
)


def wall_records(scores, arms=None):
    base = causal_record()
    arms = arms or ["alternative", "teacher", "alternative", "teacher"]
    return [
        LowMarginCausalRecord(
            **{
                **base.__dict__,
                "record_id": f"{index + 1:032x}",
                "group_id": "a" * 32,
                "candidate_seat": index,
                "executed_arm": arms[index],
                "terminal_candidate_score": float(scores[index]),
            }
        )
        for index in range(4)
    ]


class WallControlTests(unittest.TestCase):
    def test_leave_one_rotation_baseline_uses_group_not_row_order(self):
        records = wall_records([10.0, 20.0, 30.0, 40.0])
        baselines = leave_one_rotation_baselines(records)
        self.assertEqual(baselines, [30.0, 80.0 / 3.0, 70.0 / 3.0, 20.0])
        shuffled = [records[index] for index in (2, 0, 3, 1)]
        by_record = dict(
            zip(
                [record.record_id for record in shuffled],
                leave_one_rotation_baselines(shuffled),
            )
        )
        self.assertEqual(by_record[records[0].record_id], 30.0)

    def test_training_target_adds_other_rotation_mean_but_not_to_features(self):
        records = wall_records([10.0, 20.0, 30.0, 40.0])
        features, treatment, outcome = wall_control_training_tensors(records)
        self.assertEqual(features.shape[0], 4)
        self.assertEqual(treatment.tolist(), [0.5, -0.5, 0.5, -0.5])
        expected = [(10.0 + 30.0) / 40.0, (20.0 + 80.0 / 3.0) / 40.0]
        self.assertAlmostEqual(outcome.tolist()[0], expected[0])
        self.assertAlmostEqual(outcome.tolist()[1], expected[1])

    def test_control_adjustment_is_exactly_unbiased_over_random_assignment(self):
        potential_teacher = [-5.0, 10.0, 20.0, -2.0]
        potential_alternative = [3.0, 14.0, 18.0, 6.0]
        raw_values = []
        adjusted_values = []
        for assignment in itertools.product(("teacher", "alternative"), repeat=4):
            observed = [
                potential_alternative[index]
                if arm == "alternative"
                else potential_teacher[index]
                for index, arm in enumerate(assignment)
            ]
            records = wall_records(observed, list(assignment))
            components = policy_wall_components(
                records, [1.0, 1.0, 1.0, 1.0], threshold=0.0
            )
            raw_values.extend(components["raw_wall_values"])
            adjusted_values.extend(
                adjusted_wall_values(
                    components["raw_wall_values"],
                    components["control_wall_values"],
                    control_coefficient=-2.5,
                )
            )
        expected = sum(
            alternative - teacher
            for teacher, alternative in zip(
                potential_teacher, potential_alternative
            )
        ) / 4.0
        self.assertAlmostEqual(sum(raw_values) / len(raw_values), expected)
        self.assertAlmostEqual(sum(adjusted_values) / len(adjusted_values), expected)

    def test_coefficient_and_minimum_ensemble_are_deterministic(self):
        coefficient = estimate_control_coefficient(
            [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]
        )
        self.assertAlmostEqual(coefficient, 0.5)
        self.assertEqual(
            minimum_ensemble_effects(
                [[1.0, None, 4.0], [2.0, None, 3.0], [0.5, None, 5.0]]
            ),
            [0.5, None, 3.0],
        )


if __name__ == "__main__":
    unittest.main()

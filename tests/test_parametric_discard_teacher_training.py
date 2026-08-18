import math
import unittest

from scripts.train_parametric_discard_teacher_v1 import (
    GENERATIONS,
    POPULATION,
    SEARCH_FIELDS,
    TRAIN_SEGMENTS,
    VALIDATION_HANDS,
    weights_from_logs,
)
from xiamen_mahjong.agents import DiscardShapeWeights


class ParametricDiscardTeacherTrainingTests(unittest.TestCase):
    def test_zero_log_vector_is_exactly_frozen_weights(self):
        self.assertEqual(
            weights_from_logs((0.0,) * len(SEARCH_FIELDS)),
            DiscardShapeWeights(),
        )

    def test_search_budget_and_splits_are_frozen(self):
        self.assertEqual(GENERATIONS * POPULATION, 48)
        self.assertEqual(TRAIN_SEGMENTS, ((202626500, 20), (202626600, 20)))
        self.assertEqual(VALIDATION_HANDS, 100)

    def test_log_parameterization_stays_positive(self):
        weights = weights_from_logs(tuple(-1.5 for _ in SEARCH_FIELDS))
        self.assertTrue(
            all(
                math.isfinite(getattr(weights, field))
                and getattr(weights, field) > 0
                for field in SEARCH_FIELDS
            )
        )


if __name__ == "__main__":
    unittest.main()

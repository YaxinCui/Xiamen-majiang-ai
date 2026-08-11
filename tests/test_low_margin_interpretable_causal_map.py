import unittest

from scripts.analyze_low_margin_interpretable_causal_map_v1 import (
    CONTROL_COEFFICIENT,
    _evaluate_category,
    interpretable_category_flags,
)
from tests.test_causal_residual import causal_record
from tests.test_wall_control import wall_records
from xiamen_mahjong.agents import GameAction
from xiamen_mahjong.low_margin_intervention import LowMarginCausalRecord


class LowMarginInterpretableCausalMapTests(unittest.TestCase):
    def test_flags_use_only_actor_visible_action_and_state_fields(self):
        base = causal_record()
        state = {
            **base.state,
            "hand": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17, 19, 21, 23],
            "drawn_tile": 9,
            "wall_remaining": 60,
            "is_dealer": True,
        }
        record = LowMarginCausalRecord(
            **{
                **base.__dict__,
                "state": state,
                "teacher_action": GameAction("discard", 1),
                "alternative_action": GameAction("discard", 9),
                "teacher_margin": 0.0,
            }
        )
        flags = interpretable_category_flags(record)
        self.assertTrue(flags["stage_early"])
        self.assertTrue(flags["margin_exact_tie"])
        self.assertTrue(flags["actor_dealer"])
        self.assertTrue(flags["transition_suit_to_suit"])
        self.assertTrue(flags["alternative_lower_multiplicity"])
        self.assertTrue(flags["alternative_is_drawn_teacher_is_not"])
        self.assertTrue(flags["exact_tie__alternative_lower_multiplicity"])

    def test_no_intervention_has_no_category(self):
        base = causal_record()
        none = LowMarginCausalRecord(
            **{
                **base.__dict__,
                "state": None,
                "teacher_action": None,
                "alternative_action": None,
                "teacher_margin": None,
                "executed_arm": "none",
                "propensity": 1.0,
            }
        )
        self.assertEqual(interpretable_category_flags(none), {})

    def test_category_evaluation_uses_fixed_control_and_group_support(self):
        records = wall_records([10.0, 20.0, 30.0, 40.0])
        report = _evaluate_category(records, "stage_early")
        self.assertEqual(CONTROL_COEFFICIENT, -2.5)
        self.assertEqual(report["wall_groups"], 1)
        self.assertEqual(report["overrides"], 4)
        self.assertFalse(report["support_ready"])
        self.assertIn("insufficient_overrides", report["support_reasons"])


if __name__ == "__main__":
    unittest.main()

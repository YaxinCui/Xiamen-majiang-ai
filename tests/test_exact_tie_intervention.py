import unittest

from xiamen_mahjong.exact_tie_intervention import (
    TargetedExactTieEnsembleInterventionBehavior,
    exact_tie_ensemble_action_from_state,
)
from xiamen_mahjong.exact_tie_rollout import collect_exact_tie_rollout_records


class _FixedPolicy:
    def __init__(self, index):
        self.index = index

    def predict_index(self, decision):
        return min(self.index, len(decision.legal_actions) - 1)


class ExactTieInterventionTests(unittest.TestCase):
    def test_unanimous_state_target_and_disagreement_fallback(self):
        records, _report = collect_exact_tie_rollout_records(
            seed_count=1,
            seed=202638714,
            samples_per_rotation=1,
            selection_seed=202638715,
        )
        record = records[0]
        target = exact_tie_ensemble_action_from_state(
            record.state,
            record.actions,
            record.teacher_index,
            [_FixedPolicy(1) for _ in range(5)],
        )
        self.assertEqual(target, record.actions[1])
        fallback = exact_tie_ensemble_action_from_state(
            record.state,
            record.actions,
            record.teacher_index,
            [_FixedPolicy(index % 2) for index in range(5)],
        )
        self.assertEqual(fallback, record.actions[record.teacher_index])

    def test_behavior_requires_five_models(self):
        with self.assertRaisesRegex(ValueError, "五个"):
            TargetedExactTieEnsembleInterventionBehavior(
                [_FixedPolicy(0)], seed=1
            )


if __name__ == "__main__":
    unittest.main()

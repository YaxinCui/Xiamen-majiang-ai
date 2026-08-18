import unittest

from scripts.train_exact_tie_rollout_ranker_v1 import (
    _pair_tensors,
    _train_member,
    evaluate_unanimous_future_average,
    evaluate_unanimous_ensemble,
)
from xiamen_mahjong.exact_tie_rollout import (
    collect_exact_tie_rollout_records,
    exact_tie_pairwise_examples,
)
from xiamen_mahjong.training import _dense_action_features


class ExactTieRolloutRankerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, _report = collect_exact_tie_rollout_records(
            seed_count=5,
            seed=202638710,
            samples_per_rotation=1,
            selection_seed=202638711,
        )
        cls.examples = exact_tie_pairwise_examples(cls.records)

    def test_pair_tensors_only_encode_two_actions(self):
        features, targets = _pair_tensors(self.examples)
        self.assertEqual(features.shape[0], len(self.examples))
        self.assertEqual(features.shape[1:], (2, 145))
        self.assertEqual(targets.shape, (len(self.examples),))

    def test_structured_v4_features_are_actor_visible_and_fixed_width(self):
        record = self.records[0]
        vectors = [
            _dense_action_features(record.state, action, feature_version=4)
            for action in record.actions
        ]
        self.assertTrue(all(len(vector) == 204 for vector in vectors))
        self.assertTrue(all(all(value == value for value in vector) for vector in vectors))

    def test_tiny_member_and_unanimous_evaluation(self):
        midpoint = max(1, len(self.examples) // 2)
        train = self.examples[:midpoint]
        validation = self.examples[midpoint:] or self.examples[:1]
        # Temporarily shorten the global protocol only for a functional unit
        # test; the CLI still fixes forty epochs.
        import scripts.train_exact_tie_rollout_ranker_v1 as module

        original = module.EPOCHS
        module.EPOCHS = 1
        try:
            policy, report = _train_member(
                train, validation, seed=12345, device="cpu"
            )
        finally:
            module.EPOCHS = original
        self.assertEqual(report["selected_epoch"], 1)
        audit = evaluate_unanimous_ensemble([policy], self.records)
        self.assertEqual(audit["records"], len(self.records))
        self.assertGreater(audit["wall_groups"], 0)

    def test_future_average_evaluator_collapses_repeated_rows(self):
        repeated, _report = collect_exact_tie_rollout_records(
            seed_count=1,
            seed=202638712,
            samples_per_rotation=1,
            selection_seed=202638713,
            future_wall_permutations=2,
        )
        midpoint = max(1, len(self.examples) // 2)
        import scripts.train_exact_tie_rollout_ranker_v1 as module

        policy, _member = _train_member(
            self.examples[:midpoint],
            self.examples[midpoint:] or self.examples[:1],
            seed=54321,
            device="cpu",
            epochs=1,
        )
        audit = evaluate_unanimous_future_average([policy], repeated)
        self.assertEqual(audit["records"], len(repeated))
        self.assertEqual(audit["minimum_future_permutations"], 2)
        self.assertEqual(audit["maximum_future_permutations"], 2)
        self.assertEqual(audit["distinct_public_decisions"] * 2, len(repeated))


if __name__ == "__main__":
    unittest.main()

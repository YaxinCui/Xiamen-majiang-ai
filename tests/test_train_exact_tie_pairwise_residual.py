import unittest

from scripts.train_exact_tie_pairwise_residual_v1 import (
    FEATURE_VERSION,
    HIDDEN_SIZE,
    build_pairwise_tensors,
    train_pairwise_model,
)
from tests.test_pairwise_review import pairwise_queue_item
from xiamen_mahjong.pairwise_review import (
    make_pairwise_label,
    pairwise_training_comparisons,
)
from xiamen_mahjong.training import NEURAL_FEATURE_DIMS


def _comparison(*, choose_candidate: bool):
    item = pairwise_queue_item()
    selected = (
        item["candidate_index"]
        if choose_candidate
        else item["reference_teacher_index"]
    )
    position = item["pair_action_indices"].index(selected)
    label = make_pairwise_label(item, chosen_position=position, confidence="confirmed")
    return pairwise_training_comparisons([label])[0]


class TrainExactTiePairwiseResidualTests(unittest.TestCase):
    def test_tensor_builder_contains_only_two_displayed_actions(self):
        features, target = build_pairwise_tensors([_comparison(choose_candidate=True)])
        self.assertEqual(features.shape[1], 2)
        self.assertEqual(features.shape[2], NEURAL_FEATURE_DIMS[FEATURE_VERSION])
        self.assertIn(int(target[0]), {0, 1})

    def test_tiny_training_uses_small_candidate_mlp_and_validation_selection(self):
        train = [_comparison(choose_candidate=True) for _ in range(4)]
        validation = [_comparison(choose_candidate=True) for _ in range(2)]
        policy, report = train_pairwise_model(
            train,
            validation,
            device="cpu",
            seed=7,
            hidden_size=16,
            epochs=2,
            batch_size=2,
        )
        self.assertEqual(policy.architecture, "candidate_mlp")
        self.assertEqual(policy.feature_version, FEATURE_VERSION)
        self.assertLess(report["trainable_parameters"], 500_000)
        self.assertIn(report["selected_epoch"], {1, 2})
        self.assertEqual(
            report["protocol"]["target_semantics"],
            "pairwise_only_not_full_action_classification",
        )
        self.assertEqual(HIDDEN_SIZE, 64)


if __name__ == "__main__":
    unittest.main()

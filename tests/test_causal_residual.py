import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from xiamen_mahjong.causal_residual import (
    CAUSAL_PAIR_FEATURE_DIM,
    LowMarginCausalNetwork,
    causal_pair_features,
    causal_training_tensors,
    load_causal_checkpoint,
    predict_causal_effects,
    save_causal_checkpoint,
)
from xiamen_mahjong.low_margin_intervention import LowMarginCausalRecord
from tests.test_human_review import priority_review_item
from xiamen_mahjong.agents import GameAction, HeuristicTeacherAgent
from xiamen_mahjong.human_review import _ActorVisibleReviewGame


def causal_record(arm="alternative"):
    item = priority_review_item()
    game = _ActorVisibleReviewGame({"state": item["state"]})
    ranked = HeuristicTeacherAgent().explain_discard(game, 0)
    return LowMarginCausalRecord(
        record_id="1" * 32,
        group_id="2" * 32,
        candidate_seat=0,
        state=item["state"],
        teacher_action=GameAction("discard", int(ranked[0]["tile"])),
        alternative_action=GameAction("discard", int(ranked[1]["tile"])),
        teacher_margin=float(ranked[0]["score"]) - float(ranked[1]["score"]),
        executed_arm=arm,
        propensity=0.5,
        terminal_candidate_score=8.0,
    )


class CausalResidualTests(unittest.TestCase):
    def test_pair_features_do_not_depend_on_randomized_arm_or_outcome(self):
        left = causal_record("alternative")
        right = LowMarginCausalRecord(
            **{
                **left.__dict__,
                "executed_arm": "teacher",
                "terminal_candidate_score": -100.0,
            }
        )
        left_features = causal_pair_features(left)
        right_features = causal_pair_features(right)
        self.assertEqual(len(left_features), CAUSAL_PAIR_FEATURE_DIM)
        self.assertEqual(left_features, right_features)

    def test_training_target_uses_centered_randomized_treatment(self):
        alternative = causal_record("alternative")
        teacher = LowMarginCausalRecord(
            **{
                **alternative.__dict__,
                "record_id": "4" * 32,
                "executed_arm": "teacher",
                "terminal_candidate_score": -4.0,
            }
        )
        features, treatment, outcomes = causal_training_tensors(
            [alternative, teacher]
        )
        self.assertEqual(tuple(features.shape), (2, CAUSAL_PAIR_FEATURE_DIM))
        self.assertEqual(treatment.tolist(), [0.5, -0.5])
        self.assertAlmostEqual(outcomes.tolist()[0], 0.2)
        self.assertAlmostEqual(outcomes.tolist()[1], -0.1)

    def test_checkpoint_roundtrip_and_no_intervention_prediction(self):
        torch.manual_seed(7)
        network = LowMarginCausalNetwork()
        row = causal_record()
        none = LowMarginCausalRecord(
            record_id="3" * 32,
            group_id="2" * 32,
            candidate_seat=1,
            state=None,
            teacher_action=None,
            alternative_action=None,
            teacher_margin=None,
            executed_arm="none",
            propensity=1.0,
            terminal_candidate_score=0.0,
        )
        before = predict_causal_effects(network, [row, none])
        self.assertIsInstance(before[0], float)
        self.assertIsNone(before[1])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_causal_checkpoint(network, path, metadata={"seed": 7})
            loaded, metadata = load_causal_checkpoint(path)
        after = predict_causal_effects(loaded, [row, none])
        self.assertEqual(metadata, {"seed": 7})
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

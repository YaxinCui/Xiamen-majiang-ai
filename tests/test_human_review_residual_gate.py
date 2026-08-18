import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.select_human_review_residual_gate_v1 import run_gate
import scripts.select_human_review_residual_gate_v1 as review_gate_module
from scripts.select_human_teacher_residual_gate import (
    HumanGateScan,
    scan_review_inputs,
)
from tests.test_human_review import review_item
from xiamen_mahjong.human_review import make_review_label


class FakePolicy:
    architecture = "candidate_mlp"
    feature_version = 3
    hidden_size = 16

    def policy_values_batch(self, decisions):
        return [([0.0, 1.0], 0.0) for _decision in decisions]


class HumanReviewResidualGateTests(unittest.TestCase):
    def test_review_scan_scores_only_confirmed_human_alternative(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "validation.review.jsonl"
            label = make_review_label(
                review_item(), chosen_index=1, confidence="confirmed"
            )
            source.write_text(json.dumps(label) + "\n", encoding="utf-8")
            scan = scan_review_inputs([source], policy=FakePolicy(), batch_size=8)
        self.assertEqual(len(scan.observations), 1)
        observation = scan.observations[0]
        self.assertFalse(observation.teacher_matches_human)
        self.assertTrue(observation.alternative_matches_human)
        self.assertEqual(observation.policy_gap, 1.0)

    def test_validation_failure_keeps_review_test_physically_unread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "candidate.pt"
            checkpoint.write_bytes(b"fixed-review-candidate")
            validation = root / "validation.review.jsonl"
            validation.write_text("validation", encoding="utf-8")
            test = root / "must-stay-unread.review.jsonl"
            output = root / "gate.json"
            empty = HumanGateScan(
                observations=(),
                trajectory_groups=frozenset({"validation-group"}),
                structural_audit={"confirmed_labels": 40},
            )
            with (
                patch.object(review_gate_module, "_sha256", return_value="frozen"),
                patch.object(
                    review_gate_module.TorchPolicyValueAgent,
                    "load",
                    return_value=FakePolicy(),
                ),
                patch.object(
                    review_gate_module,
                    "scan_review_inputs",
                    return_value=empty,
                ) as scan,
            ):
                payload = run_gate(
                    checkpoint=checkpoint,
                    checkpoint_sha256="frozen",
                    validation_input=validation,
                    test_input=test,
                    output=output,
                    device="cpu",
                    batch_size=8,
                )
            self.assertEqual(scan.call_count, 1)
            self.assertEqual(scan.call_args.args[0], [validation])
            self.assertEqual(payload["test"], {"status": "unread"})
            self.assertIn("failed_review_test_unread", payload["status"])


if __name__ == "__main__":
    unittest.main()

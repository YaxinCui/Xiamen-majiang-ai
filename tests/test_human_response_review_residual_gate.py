import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import scripts.select_human_response_review_residual_gate_v1 as gate_module
from scripts.select_human_response_review_residual_gate_v1 import (
    run_gate,
    scan_response_review_inputs,
)
from scripts.select_human_teacher_residual_gate import HumanGateScan
from tests.test_response_review import response_item
from xiamen_mahjong.response_review import make_response_review_label


class FakeResponsePolicy:
    architecture = "candidate_mlp"
    feature_version = 3
    hidden_size = 16

    def policy_values_batch(self, decisions):
        return [([1.0, 0.0], 0.0) for _decision in decisions]


class HumanResponseReviewResidualGateTests(unittest.TestCase):
    def test_response_scan_scores_best_nonteacher_alternative(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "validation.response-review.jsonl"
            label = make_response_review_label(
                response_item(), chosen_index=0, confidence="confirmed"
            )
            source.write_text(json.dumps(label) + "\n", encoding="utf-8")
            scan = scan_response_review_inputs(
                [source], policy=FakeResponsePolicy(), batch_size=8
            )
        self.assertEqual(len(scan.observations), 1)
        observation = scan.observations[0]
        self.assertFalse(observation.teacher_matches_human)
        self.assertTrue(observation.alternative_matches_human)
        self.assertEqual(observation.policy_gap, 1.0)

    def test_validation_failure_keeps_response_test_physically_unread(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "candidate.pt"
            checkpoint.write_bytes(b"fixed-response-review-candidate")
            validation = root / "validation.response-review.jsonl"
            validation.write_text("validation", encoding="utf-8")
            test = root / "must-stay-unread.response-review.jsonl"
            output = root / "gate.json"
            empty = HumanGateScan(
                observations=(),
                trajectory_groups=frozenset({"validation-group"}),
                structural_audit={"confirmed_labels": 30},
            )
            with (
                patch.object(gate_module, "_sha256", return_value="frozen"),
                patch.object(
                    gate_module.TorchPolicyValueAgent,
                    "load",
                    return_value=FakeResponsePolicy(),
                ),
                patch.object(
                    gate_module,
                    "scan_response_review_inputs",
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
            self.assertIn("failed_test_unread", payload["status"])


if __name__ == "__main__":
    unittest.main()

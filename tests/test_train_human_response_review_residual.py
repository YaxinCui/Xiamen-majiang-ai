from pathlib import Path
import json
from types import SimpleNamespace
import tempfile
import unittest

from scripts.train_policy_value import iter_response_review_examples
from scripts.train_human_response_review_residual_v1 import (
    build_training_command,
)
from tests.test_response_review import response_item
from xiamen_mahjong.response_review import make_response_review_label


class TrainHumanResponseReviewResidualTests(unittest.TestCase):
    def test_fixed_command_is_fresh_cpu_small_mlp_and_cannot_read_test(self):
        validated = {
            "teacher_paths": {
                split: Path(f"teacher/{split}.trajectories.jsonl")
                for split in ("train", "validation", "test")
            },
            "review_paths": {
                split: Path(f"private/{split}.response-review.jsonl")
                for split in ("train", "validation", "test")
            },
        }
        command = build_training_command(
            validated, output_dir=Path("artifacts/candidate"), device="cpu"
        )
        rendered = " ".join(command)
        self.assertIn("--architecture candidate_mlp", rendered)
        self.assertIn("--hidden-size 128", rendered)
        self.assertIn("--device cpu", rendered)
        self.assertIn(
            "--human-response-review-train private/train.response-review.jsonl",
            rendered,
        )
        self.assertIn(
            "--human-response-review-validation "
            "private/validation.response-review.jsonl",
            rendered,
        )
        self.assertNotIn("private/test.response-review.jsonl", rendered)
        self.assertNotIn("--init-checkpoint", command)
        self.assertIn(
            "--checkpoint-selection-source "
            "local_human_response_review_opt_in",
            rendered,
        )

    def test_response_review_encodes_behavior_only_with_declared_weight(self):
        label = make_response_review_label(
            response_item(), chosen_index=0, confidence="confirmed"
        )
        args = SimpleNamespace(
            allow_local_human_response_review_data=True,
            history_window=24,
            feature_version=3,
            human_response_review_weight=4.0,
            human_teacher_disagreement_weight=2.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.response-review.jsonl"
            source.write_text(json.dumps(label) + "\n", encoding="utf-8")
            examples = list(iter_response_review_examples(source, args))
        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example.source, "local_human_response_review_opt_in")
        self.assertEqual(example.action_kind, "pass")
        self.assertEqual(example.sample_weight, 8.0)
        self.assertIsNone(example.value_target)
        self.assertIsNone(example.action_values)


if __name__ == "__main__":
    unittest.main()

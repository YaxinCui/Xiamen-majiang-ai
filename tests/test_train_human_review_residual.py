from pathlib import Path
import unittest

from scripts.train_human_review_residual_v1 import build_training_command


class TrainHumanReviewResidualTests(unittest.TestCase):
    def test_fixed_command_uses_cpu_small_mlp_and_never_passes_review_test(self):
        validated = {
            "teacher_paths": {
                split: Path(f"teacher/{split}.trajectories.jsonl")
                for split in ("train", "validation", "test")
            },
            "review_paths": {
                split: Path(f"private/{split}.review.jsonl")
                for split in ("train", "validation", "test")
            },
        }
        command = build_training_command(
            validated, output_dir=Path("artifacts/candidate"), device="cpu"
        )
        rendered = " ".join(command)
        self.assertIn("--architecture candidate_mlp", rendered)
        self.assertIn("--device cpu", rendered)
        self.assertIn("--human-review-train private/train.review.jsonl", rendered)
        self.assertIn(
            "--human-review-validation private/validation.review.jsonl", rendered
        )
        self.assertNotIn("private/test.review.jsonl", rendered)
        self.assertNotIn("--init-checkpoint", command)
        self.assertIn(
            "--checkpoint-selection-source local_human_review_opt_in", rendered
        )


if __name__ == "__main__":
    unittest.main()

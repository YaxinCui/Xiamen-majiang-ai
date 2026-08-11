from pathlib import Path
import unittest

from scripts.train_human_teacher_residual_v1 import (
    EPOCHS,
    FRESH_SEED,
    HUMAN_DISAGREEMENT_WEIGHT,
    build_training_command,
)


class TrainHumanTeacherResidualTests(unittest.TestCase):
    def test_fixed_command_reserves_human_test_and_uses_no_old_checkpoint(self):
        teacher = {
            split: Path(f"teacher/{split}.trajectories.jsonl")
            for split in ("train", "validation", "test")
        }
        human = {
            split: Path(f"private-human/{split}.trajectories.jsonl")
            for split in ("train", "validation", "test")
        }
        command = build_training_command(
            {
                "teacher_paths": teacher,
                "human_paths": human,
            },
            output_dir=Path("artifacts/fixed-human-v1"),
            device="cpu",
        )
        rendered = " ".join(command)
        self.assertIn("--reserve-local-human-test-for-gate", command)
        self.assertIn("--checkpoint-selection-source local_human_opt_in", rendered)
        self.assertIn(f"--seed {FRESH_SEED}", rendered)
        self.assertIn(f"--epochs {EPOCHS}", rendered)
        self.assertIn(
            f"--human-teacher-disagreement-weight {HUMAN_DISAGREEMENT_WEIGHT}",
            rendered,
        )
        self.assertIn("--human-discard-corrections-only", command)
        self.assertIn(str(human["train"]), command)
        self.assertIn(str(human["validation"]), command)
        self.assertNotIn(str(human["test"]), command)
        self.assertNotIn("--init-checkpoint", command)


if __name__ == "__main__":
    unittest.main()

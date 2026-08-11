import copy
import unittest
from pathlib import Path


class TeacherReferenceKlPpoV2SelectionTests(unittest.TestCase):
    def _report(self):
        from scripts.select_teacher_reference_kl_ppo_v2 import (
            BASE_CHECKPOINT_SHA256,
            TRAINING_SOURCE_TREE_SHA256,
            expected_training_config,
        )

        checkpoint = Path("artifacts/example/policy-value-ppo-iteration-4.pt")
        iterations = []
        for index, decisions in enumerate((100, 110, 120, 130), start=1):
            iterations.append(
                {
                    "iteration": index,
                    "rollout_seed": 202646000 + index * 100_000,
                    "update_seed": 202646000 + index,
                    "rollout": {
                        "episodes": 1024,
                        "decisions": decisions,
                        "wins": 256,
                        "draws": 0,
                        "action_counts": {"discard": decisions},
                        "opponent_profile_counts": {"heuristic_teacher": 3072},
                    },
                    "update": {
                        "final_reference_kl_mean": 0.004,
                        "final_reference_argmax_disagreement_rate": 0.01,
                    },
                }
            )
        return checkpoint, {
            "algorithm": "legal_action_ppo_terminal_score_v1",
            "profile": "classic",
            "checkpoint_source_sha256": BASE_CHECKPOINT_SHA256,
            "source_tree_sha256": TRAINING_SOURCE_TREE_SHA256,
            "training_config": expected_training_config(),
            "final_checkpoint": str(checkpoint),
            "iterations": iterations,
        }

    def test_exact_frozen_report_passes_mechanics_gate(self):
        from scripts.select_teacher_reference_kl_ppo_v2 import (
            validate_training_report,
        )

        checkpoint, report = self._report()
        gate = validate_training_report(report, checkpoint=checkpoint)
        self.assertTrue(gate["passes"])
        self.assertEqual(gate["total_candidate_decisions"], 460)

    def test_changed_config_or_incomplete_action_accounting_fails_closed(self):
        from scripts.select_teacher_reference_kl_ppo_v2 import (
            validate_training_report,
        )

        checkpoint, report = self._report()
        changed = copy.deepcopy(report)
        changed["training_config"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "超参数"):
            validate_training_report(changed, checkpoint=checkpoint)

        incomplete = copy.deepcopy(report)
        incomplete["iterations"][0]["rollout"]["action_counts"]["discard"] -= 1
        with self.assertRaisesRegex(ValueError, "动作计数"):
            validate_training_report(incomplete, checkpoint=checkpoint)

    def test_mechanics_bounds_reject_without_throwing(self):
        from scripts.select_teacher_reference_kl_ppo_v2 import (
            validate_training_report,
        )

        checkpoint, report = self._report()
        report["iterations"][-1]["update"][
            "final_reference_argmax_disagreement_rate"
        ] = 0.001
        gate = validate_training_report(report, checkpoint=checkpoint)
        self.assertFalse(gate["passes"])


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_teacher_response_causal_error_map_v1 import analyze_paths


def _decision(*, teacher: str, logged: str, wall: int, tile: int, probability: float):
    actions = [{"kind": "pass", "tiles": []}, {"kind": teacher, "tiles": []}]
    return {
        "seat": 0,
        "state": {
            "phase": "response",
            "wall_remaining": wall,
            "last_discard": tile,
        },
        "legal_actions": actions,
        "chosen_index": 1,
        "executed_index": 0 if logged == "pass" else 1,
        "executed_probability": probability,
    }


def _trajectory(group: str, decision: dict):
    return {
        "split_group_id": group,
        "source_metadata": {
            "behavior_policy": "single_intervention_epsilon_uniform",
            "base_policy": "heuristic_teacher",
            "intervention_phase": "response",
            "uniform_exploration_probability": 0.4,
        },
        "outcome": {"scores": [12, -4, -4, -4]},
        "decisions": [decision],
    }


class TeacherResponseCausalErrorMapTests(unittest.TestCase):
    def _write(self, directory: str, name: str, rows: list[dict]) -> Path:
        path = Path(directory) / name
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    def test_discovery_taxonomy_is_mutually_exclusive_and_train_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                "train.trajectories.jsonl",
                [
                    _trajectory(
                        "g1",
                        _decision(
                            teacher="chi", logged="pass", wall=60, tile=3, probability=0.2
                        ),
                    ),
                    _trajectory(
                        "g2",
                        _decision(
                            teacher="pong", logged="pong", wall=40, tile=31, probability=0.8
                        ),
                    ),
                ],
            )
            report = analyze_paths([path], mode="discovery", partition="train")
        self.assertEqual(
            report["taxonomy"]["categories"], ["chi_early", "pong_middle_honor"]
        )
        self.assertEqual(report["categories"]["chi_early"]["ips"]["observations"], 1)
        self.assertFalse(report["per_decision_state_exported"])

    def test_validation_changes_only_teacher_chi(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                "validation.trajectories.jsonl",
                [
                    _trajectory(
                        "g1",
                        _decision(
                            teacher="chi", logged="pass", wall=60, tile=3, probability=0.2
                        ),
                    ),
                    _trajectory(
                        "g2",
                        _decision(
                            teacher="pong", logged="pong", wall=40, tile=31, probability=0.8
                        ),
                    ),
                ],
            )
            report = analyze_paths(
                [path], mode="validate-chi-pass", partition="validation"
            )
        self.assertEqual(report["candidate"]["override_observations"], 1)
        self.assertEqual(report["candidate"]["target_action_kinds"], {"pass": 1, "pong": 1})
        self.assertFalse(report["gate"]["passes"])

    def test_partition_and_propensity_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                "validation.trajectories.jsonl",
                [
                    _trajectory(
                        "g1",
                        _decision(
                            teacher="chi", logged="pass", wall=60, tile=3, probability=0.5
                        ),
                    )
                ],
            )
            with self.assertRaisesRegex(ValueError, "discovery"):
                analyze_paths([path], mode="discovery", partition="validation")
            with self.assertRaisesRegex(ValueError, "epsilon"):
                analyze_paths([path], mode="validate-chi-pass", partition="validation")


if __name__ == "__main__":
    unittest.main()

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.select_human_response_review_residual_gate_v1 import RESPONSE_SCOPE
from scripts.select_human_response_review_teacher_screen_v1 import (
    SELECTION_HANDS,
    SELECTION_SEED,
    TERMINAL_HANDS,
    TERMINAL_SEED,
    load_gate_identity,
)


class HumanResponseReviewTeacherScreenTests(unittest.TestCase):
    def test_accepts_only_passed_frozen_response_gate_report(self):
        payload = {
            "status": (
                "response_review_test_gate_passed_ready_for_100_wall_teacher_screen"
            ),
            "checkpoint": {"sha256": "checkpoint-sha"},
            "protocol": {
                "source": "local_human_response_review_opt_in",
                "scope": RESPONSE_SCOPE,
            },
            "validation": {"winner": {"strict_margin": 1.25}},
            "test": {"passes": True, "summary": {"scope": RESPONSE_SCOPE}},
        }
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "gate.json"
            report.write_text(json.dumps(payload), encoding="utf-8")
            digest = hashlib.sha256(report.read_bytes()).hexdigest()
            identity = load_gate_identity(
                report,
                gate_report_sha256=digest,
                checkpoint_sha256="checkpoint-sha",
            )
            self.assertEqual(identity["minimum_policy_advantage"], 1.25)
            payload["test"]["passes"] = False
            report.write_text(json.dumps(payload), encoding="utf-8")
            digest = hashlib.sha256(report.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "未通过"):
                load_gate_identity(
                    report,
                    gate_report_sha256=digest,
                    checkpoint_sha256="checkpoint-sha",
                )

    def test_fresh_wall_stages_are_fixed_and_disjoint(self):
        self.assertEqual(SELECTION_HANDS, 100)
        self.assertEqual(TERMINAL_HANDS, 400)
        selection = set(range(SELECTION_SEED, SELECTION_SEED + SELECTION_HANDS))
        terminal = set(range(TERMINAL_SEED, TERMINAL_SEED + TERMINAL_HANDS))
        self.assertFalse(selection & terminal)


if __name__ == "__main__":
    unittest.main()

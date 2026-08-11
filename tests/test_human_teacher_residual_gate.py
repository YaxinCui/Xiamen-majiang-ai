from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import scripts.select_human_teacher_residual_gate as gate_module
from scripts.select_human_teacher_residual_gate import (
    HumanGateScan,
    HumanGateObservation,
    choose_strict_margin,
    gate_passes,
    select_validation_gate,
    summarize_gate,
    wilson_lower_bound,
)


def _observations(*, helpful: bool) -> list[HumanGateObservation]:
    rows: list[HumanGateObservation] = []
    for index in range(400):
        selected = index < 20
        rows.append(
            HumanGateObservation(
                group_id=f"hand-{index % 40}",
                policy_gap=float(100 - index) if selected else float(-index),
                teacher_matches_human=selected and not helpful,
                alternative_matches_human=selected and helpful,
            )
        )
    return rows


class HumanTeacherResidualGateTests(unittest.TestCase):
    def test_strict_margin_never_splits_a_tied_boundary(self):
        margin = choose_strict_margin(
            [9.0, 8.0, 8.0, 7.0],
            eligible_decisions=10,
            target_override_rate=0.2,
        )
        self.assertEqual(margin, 8.0)
        self.assertEqual(sum(gap > margin for gap in [9.0, 8.0, 8.0, 7.0]), 1)

    def test_wilson_lower_bound_requires_real_override_support(self):
        self.assertIsNone(wilson_lower_bound(0, 0))
        self.assertGreater(wilson_lower_bound(20, 20), 0.5)
        self.assertLess(wilson_lower_bound(10, 20), 0.5)

    def test_helpful_gate_passes_hand_grouped_and_precision_bounds(self):
        summary = summarize_gate(_observations(helpful=True), margin=80.5)
        self.assertEqual(summary["overrides"], 20)
        self.assertEqual(summary["human_corrections"], 20)
        self.assertEqual(summary["teacher_to_human_harms"], 0)
        self.assertGreater(summary["hand_grouped_accuracy_gain_95pct_low"], 0.0)
        self.assertTrue(
            gate_passes(
                summary,
                minimum_overrides=20,
                minimum_groups=20,
                maximum_override_rate=0.05,
            )
        )

    def test_validation_selects_only_predeclared_rate_that_passes(self):
        winner, candidates = select_validation_gate(
            _observations(helpful=True),
            target_override_rates=(0.02, 0.05),
            minimum_overrides=20,
            minimum_groups=20,
        )
        self.assertEqual(len(candidates), 2)
        self.assertFalse(candidates[0]["passes"])
        self.assertTrue(candidates[1]["passes"])
        self.assertIsNotNone(winner)
        assert winner is not None
        self.assertEqual(winner["target_override_rate"], 0.05)

    def test_gate_that_overrides_correct_teacher_is_rejected(self):
        winner, candidates = select_validation_gate(
            _observations(helpful=False),
            target_override_rates=(0.05,),
            minimum_overrides=20,
            minimum_groups=20,
        )
        self.assertIsNone(winner)
        self.assertFalse(candidates[0]["passes"])
        self.assertLess(
            candidates[0]["hand_grouped_accuracy_gain_95pct_low"], 0.0
        )

    def test_selection_configuration_fails_closed(self):
        with self.assertRaises(ValueError):
            select_validation_gate(
                _observations(helpful=True),
                target_override_rates=(0.05, 0.05),
                minimum_overrides=20,
                minimum_groups=20,
            )

    def test_validation_failure_keeps_test_split_physically_unread(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidate.pt"
            checkpoint.write_bytes(b"frozen-test-checkpoint")
            output = Path(directory) / "report.json"
            args = SimpleNamespace(
                checkpoint=checkpoint,
                checkpoint_sha256="frozen-sha",
                validation_input=[Path(directory) / "validation.jsonl"],
                test_input=[Path(directory) / "must-stay-unread.jsonl"],
                target_override_rate=None,
                minimum_validation_overrides=20,
                minimum_validation_groups=20,
                minimum_test_overrides=20,
                minimum_test_groups=20,
                maximum_test_override_rate=0.075,
                batch_size=32,
                device="cpu",
                output=output,
            )
            empty_scan = HumanGateScan(
                observations=(),
                trajectory_groups=frozenset({"validation-hand"}),
                structural_audit={"valid_hands": 1},
            )
            fake_policy = SimpleNamespace(
                architecture="candidate_mlp",
                feature_version=3,
                hidden_size=16,
            )
            with (
                patch.object(gate_module, "parse_args", return_value=args),
                patch.object(gate_module, "_sha256", return_value="frozen-sha"),
                patch.object(
                    gate_module.TorchPolicyValueAgent,
                    "load",
                    return_value=fake_policy,
                ),
                patch.object(
                    gate_module,
                    "scan_human_inputs",
                    return_value=empty_scan,
                ) as scan,
            ):
                with redirect_stdout(StringIO()):
                    gate_module.main()
            self.assertEqual(scan.call_count, 1)
            self.assertEqual(
                scan.call_args.args[0], args.validation_input
            )
            report = output.read_text(encoding="utf-8")
            self.assertIn('"status": "validation_gate_failed_test_unread"', report)
            self.assertIn('"test": {\n    "status": "unread"', report)


if __name__ == "__main__":
    unittest.main()

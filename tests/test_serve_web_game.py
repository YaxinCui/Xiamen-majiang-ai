from pathlib import Path
import unittest

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None

from scripts.serve_web_game import load_explicit_ai


class ServeWebGameConfigurationTests(unittest.TestCase):
    def test_default_remains_rule_teacher(self):
        agent, profile, identity = load_explicit_ai(
            None,
            device="cpu",
            teacher_gate_margin=None,
        )
        self.assertIsNone(agent)
        self.assertEqual(profile, "heuristic_teacher")
        self.assertEqual(identity, "heuristic_teacher")

    def test_gate_requires_an_explicit_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "必须与 --ai-checkpoint"):
            load_explicit_ai(
                None,
                device="cpu",
                teacher_gate_margin=2.0,
            )

    @unittest.skipIf(torch is None, "PyTorch 仅在项目 .venv 中安装")
    def test_explicit_human_correction_gate_has_frozen_identity_and_scope(self):
        from xiamen_mahjong.teacher_anchored import ConfidenceGatedTeacherAgent

        checkpoint = Path("artifacts/policy-value-classic-v1-run3/policy-value.pt")
        agent, profile, identity = load_explicit_ai(
            checkpoint,
            device="cpu",
            teacher_gate_margin=2.5,
        )
        self.assertIsInstance(agent, ConfidenceGatedTeacherAgent)
        self.assertEqual(profile, "explicit_human_correction_teacher_gate")
        self.assertIn("sha256:", identity)
        self.assertIn("wrapper=human_correction_discard_gate_v1", identity)
        self.assertIn("strict_margin=2.5", identity)
        self.assertEqual(agent.allowed_teacher_kinds, {"discard"})
        self.assertEqual(agent.allowed_alternative_kinds, {"discard"})

    @unittest.skipIf(torch is None, "PyTorch 仅在项目 .venv 中安装")
    def test_gate_rejects_negative_margin(self):
        checkpoint = Path("artifacts/policy-value-classic-v1-run3/policy-value.pt")
        with self.assertRaisesRegex(ValueError, "不能为负数"):
            load_explicit_ai(
                checkpoint,
                device="cpu",
                teacher_gate_margin=-0.1,
            )


if __name__ == "__main__":
    unittest.main()

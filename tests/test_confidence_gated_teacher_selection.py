import unittest

from scripts.select_confidence_gated_teacher import (
    CHECKPOINT,
    CHECKPOINT_SHA256,
    MINIMUM_POLICY_ADVANTAGE,
    SELECTION_HANDS,
    SELECTION_SEED,
    TERMINAL_HANDS,
    TERMINAL_SEED,
    _sha256,
    fixed_candidate,
)


class ConfidenceGatedTeacherSelectionTests(unittest.TestCase):
    def test_checkpoint_identity_and_fixed_gate(self):
        self.assertEqual(_sha256(CHECKPOINT), CHECKPOINT_SHA256)
        candidate = fixed_candidate(device="cpu")
        self.assertEqual(
            candidate.minimum_policy_advantage, MINIMUM_POLICY_ADVANTAGE
        )
        self.assertEqual(candidate.allowed_teacher_kinds, frozenset({"discard"}))

    def test_selection_and_terminal_walls_are_disjoint(self):
        selection = set(range(SELECTION_SEED, SELECTION_SEED + SELECTION_HANDS))
        terminal = set(range(TERMINAL_SEED, TERMINAL_SEED + TERMINAL_HANDS))
        self.assertTrue(selection.isdisjoint(terminal))


if __name__ == "__main__":
    unittest.main()

import unittest

from scripts.select_public_progress_tiebreak_teacher_v1 import (
    SELECTION_HANDS,
    SELECTION_SEED,
    AuditedPublicProgressTieBreakTeacher,
    fixed_candidate,
)


class PublicProgressTieBreakSelectionTests(unittest.TestCase):
    def test_fixed_candidate_is_the_audited_exact_tie_agent(self):
        self.assertIsInstance(
            fixed_candidate(), AuditedPublicProgressTieBreakTeacher
        )

    def test_selection_uses_one_hundred_fresh_walls(self):
        self.assertEqual(SELECTION_HANDS, 100)
        self.assertEqual(SELECTION_SEED, 202638300)

    def test_coverage_payload_is_zero_before_play(self):
        payload = fixed_candidate().coverage_payload()
        self.assertEqual(payload["discard_decisions"], 0)
        self.assertEqual(payload["exact_top_ties"], 0)
        self.assertEqual(payload["overrides"], 0)
        self.assertEqual(payload["override_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

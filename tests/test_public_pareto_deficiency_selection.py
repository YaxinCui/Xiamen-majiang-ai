import unittest

from scripts.select_public_pareto_deficiency_teacher_v2 import (
    SCORE_MARGIN,
    SELECTION_HANDS,
    SELECTION_SEED,
    AuditedPublicParetoDeficiencyTeacher,
    fixed_candidate,
)


class PublicParetoDeficiencySelectionTests(unittest.TestCase):
    def test_fixed_candidate_matches_prespecified_gate(self):
        candidate = fixed_candidate()
        self.assertIsInstance(candidate, AuditedPublicParetoDeficiencyTeacher)
        self.assertEqual(candidate.score_margin, SCORE_MARGIN)

    def test_selection_uses_one_hundred_fresh_walls(self):
        self.assertEqual(SELECTION_HANDS, 100)
        self.assertEqual(SELECTION_SEED, 202638000)

    def test_coverage_payload_is_safe_and_zero_before_play(self):
        self.assertEqual(
            fixed_candidate().coverage_payload(),
            {
                "discard_decisions": 0,
                "v1_proposals": 0,
                "pareto_rejections": 0,
                "overrides": 0,
                "override_rate": 0.0,
                "v1_confirmation_rate": 0.0,
                "shanten_reduction_total": 0,
                "ukeire_change_total": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()

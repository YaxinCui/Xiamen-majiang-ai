import unittest

from scripts.audit_teacher_one_draw_tenpai import OneDrawAudit


class TeacherOneDrawAuditTests(unittest.TestCase):
    def test_empty_audit_is_explicitly_diagnostic_only(self):
        payload = OneDrawAudit(score_margin=2.0).payload(
            hands=1, seed=1, profile="classic"
        )
        self.assertEqual(
            payload["status"], "diagnostic_only_not_authorized_for_action_selection"
        )
        self.assertIsNone(payload["better_near_tied_alternative_rate"])
        self.assertNotIn("wall", payload)
        self.assertNotIn("hand", payload)


if __name__ == "__main__":
    unittest.main()

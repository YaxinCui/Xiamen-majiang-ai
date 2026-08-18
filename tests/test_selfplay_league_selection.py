import unittest


class SelfPlayLeagueSelectionTests(unittest.TestCase):
    def test_positive_lower_bound_requires_strict_improvement(self):
        from scripts.select_selfplay_league_candidate import positive_lower_bound

        self.assertFalse(
            positive_lower_bound({"paired_seed_score_delta_95pct_low": 0.0})
        )
        self.assertFalse(
            positive_lower_bound({"paired_seed_score_delta_95pct_low": -0.0001})
        )
        self.assertTrue(
            positive_lower_bound({"paired_seed_score_delta_95pct_low": 0.0001})
        )

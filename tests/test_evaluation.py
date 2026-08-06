import unittest

from xiamen_mahjong.evaluation import (
    PolicyEvaluation,
    evaluate_against_teacher,
    paired_score_comparison,
)
from xiamen_mahjong.training import NeuralRulePolicyModel


class EvaluationTests(unittest.TestCase):
    def test_paired_score_comparison_uses_seed_groups(self) -> None:
        candidate = PolicyEvaluation(
            profile="classic",
            first_seed=10,
            seed_count=2,
            games=8,
            candidate_wins=0,
            draws=0,
            candidate_score_total=20,
            candidate_score_mean=2.5,
            candidate_score_stderr=0.0,
            candidate_win_rate=0.0,
            candidate_scores=(1, 3, 5, 7, 1, 3, 5, 7),
        )
        reference = PolicyEvaluation(
            profile="classic",
            first_seed=10,
            seed_count=2,
            games=8,
            candidate_wins=0,
            draws=0,
            candidate_score_total=12,
            candidate_score_mean=1.5,
            candidate_score_stderr=0.0,
            candidate_win_rate=0.0,
            candidate_scores=(0, 2, 4, 6, 0, 2, 4, 6),
        )
        comparison = paired_score_comparison(candidate, reference)
        self.assertEqual(comparison["paired_seed_score_delta_mean"], 1.0)
        self.assertEqual(comparison["paired_seed_score_delta_stderr"], 0.0)

    def test_seat_rotated_evaluation_finishes_and_is_deterministic(self):
        candidate = NeuralRulePolicyModel(hidden_size=4, seed=617)
        first = evaluate_against_teacher(candidate, hands=2, profile="classic", seed=617)
        second = evaluate_against_teacher(candidate, hands=2, profile="classic", seed=617)
        self.assertEqual(first, second)
        self.assertEqual(first.games, 8)
        self.assertEqual(len(first.candidate_scores), 8)
        self.assertLessEqual(first.candidate_wins + first.draws, first.games)


if __name__ == "__main__":
    unittest.main()

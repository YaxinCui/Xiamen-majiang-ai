from dataclasses import replace
import unittest

from xiamen_mahjong.agents import HeuristicTeacherAgent
from xiamen_mahjong.evaluation import (
    PolicyEvaluation,
    _roster_agents_for_candidate_seat,
    evaluate_against_opponent_roster,
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

    def test_roster_evaluation_is_deterministic_and_records_identity(self):
        first = evaluate_against_opponent_roster(
            NeuralRulePolicyModel(hidden_size=4, seed=618),
            tuple(HeuristicTeacherAgent() for _ in range(3)),
            opponent_labels=("teacher-a", "teacher-b", "teacher-c"),
            hands=2,
            profile="classic",
            seed=618,
        )
        second = evaluate_against_opponent_roster(
            NeuralRulePolicyModel(hidden_size=4, seed=618),
            tuple(HeuristicTeacherAgent() for _ in range(3)),
            opponent_labels=("teacher-a", "teacher-b", "teacher-c"),
            hands=2,
            profile="classic",
            seed=618,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.games, 8)
        self.assertEqual(first.opponent_labels, ("teacher-a", "teacher-b", "teacher-c"))

    def test_roster_evaluation_requires_three_opponents(self):
        with self.assertRaisesRegex(ValueError, "3 名对手"):
            evaluate_against_opponent_roster(
                NeuralRulePolicyModel(hidden_size=4, seed=619),
                (HeuristicTeacherAgent(),),
                hands=1,
            )

    def test_roster_assignment_covers_each_relative_seat(self):
        candidate = object()
        opponents = (object(), object(), object())
        relative_positions = [set(), set(), set()]
        for candidate_seat in range(4):
            agents = _roster_agents_for_candidate_seat(
                candidate,
                opponents,
                candidate_seat=candidate_seat,
                player_count=4,
            )
            self.assertIs(agents[candidate_seat], candidate)
            for index, opponent in enumerate(opponents):
                seat = next(seat for seat, agent in agents.items() if agent is opponent)
                relative_positions[index].add((seat - candidate_seat) % 4)
        self.assertEqual(relative_positions, [{1, 2, 3}, {1, 2, 3}, {1, 2, 3}])

    def test_paired_comparison_rejects_different_opponent_rosters(self):
        candidate = PolicyEvaluation(
            profile="classic",
            first_seed=10,
            seed_count=1,
            games=4,
            candidate_wins=0,
            draws=0,
            candidate_score_total=0,
            candidate_score_mean=0.0,
            candidate_score_stderr=0.0,
            candidate_win_rate=0.0,
            candidate_scores=(0, 0, 0, 0),
            opponent_labels=("a", "b", "c"),
        )
        with self.assertRaisesRegex(ValueError, "对手阵容"):
            paired_score_comparison(
                candidate, replace(candidate, opponent_labels=("a", "b", "other"))
            )


if __name__ == "__main__":
    unittest.main()

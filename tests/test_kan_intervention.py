import unittest

from tests.test_human_review import priority_review_item
from xiamen_mahjong.agents import GameAction
from xiamen_mahjong.game import XiamenMahjongGame
from xiamen_mahjong.kan_intervention import (
    KAN_INTERVENTION_BEHAVIOR_VERSION,
    TargetedKanInterventionBehavior,
    audit_targeted_kan_interventions,
)
from xiamen_mahjong.rules import XiamenRules
from xiamen_mahjong.training import TeacherDecision, TrainingTrajectory, _turn_actions


class _ForcedAnKanTeacher:
    def choose_turn_action(self, game, player_id):
        return next(
            action for action in _turn_actions(game, player_id)
            if action.kind == "an_kan"
        )

    def _best_discard(self, game, player_id):
        return next(
            int(action.tile) for action in _turn_actions(game, player_id)
            if action.kind == "discard"
        )

    def choose_response(self, _game, _player_id, options):
        return options[0]


class KanInterventionTests(unittest.TestCase):
    def test_behavior_randomizes_once_then_restores_teacher(self):
        game = XiamenMahjongGame(
            seed=202637099,
            rules=XiamenRules.classic(),
            auto_advance=False,
            human_seat=-1,
        )
        seat = game.current_player
        game.gold_tile = 33
        game.players[seat].hand = [0, 0, 0, 0, *range(1, 14)]
        teacher = _ForcedAnKanTeacher()
        behavior = TargetedKanInterventionBehavior(seed=1, teacher=teacher)

        action = behavior.choose_turn_action(game, seat)
        legal = tuple(_turn_actions(game, seat))
        self.assertIn(action.kind, {"discard", "an_kan"})
        self.assertEqual(
            behavior.action_probability(
                game, seat, legal, action, is_response=False
            ),
            0.5,
        )
        second = behavior.choose_turn_action(game, seat)
        self.assertEqual(second.kind, "an_kan")
        self.assertEqual(
            behavior.action_probability(
                game, seat, legal, second, is_response=False
            ),
            1.0,
        )

    def test_audit_reproduces_response_fallback_and_ht_delta(self):
        state = priority_review_item()["state"]
        state = {**state, "phase": "response"}
        actions = (GameAction("pass"), GameAction("ming_kan", 3, (3, 3, 3)))
        rows = []
        for seat in range(4):
            fallback = seat < 2
            decision = TeacherDecision(
                profile="classic",
                seed=None,
                seat=seat,
                state=state,
                legal_actions=actions,
                chosen_index=1,
                executed_index=0 if fallback else 1,
                executed_probability=0.5,
            )
            scores = [0, 0, 0, 0]
            scores[seat] = 10 if fallback else 0
            rows.append(
                TrainingTrajectory(
                    profile="classic",
                    rules_version="xiamen-classic-full-v2",
                    rules={},
                    seed=None,
                    hand_number=1,
                    agent_profiles=("candidate_policy",) + ("heuristic_teacher",) * 3,
                    source_metadata={
                        "collector": "candidate_vs_teacher_dagger",
                        "candidate_seat": seat,
                        "behavior_policy": KAN_INTERVENTION_BEHAVIOR_VERSION,
                        "base_policy": "heuristic_teacher",
                        "fallback_policy": "teacher_non_kan_fallback_v1",
                        "fallback_probability": 0.5,
                        "maximum_interventions_per_trajectory": 1,
                    },
                    decisions=(decision,),
                    outcome={"scores": scores},
                    public_actions=(),
                    trajectory_id=f"{seat + 20:032x}",
                    split_group_id="e" * 32,
                )
            )
        report = audit_targeted_kan_interventions(
            rows,
            minimum_interventions=4,
            minimum_wall_groups=2,
        )
        self.assertEqual(report["issues"], {})
        self.assertEqual(report["fallback_assignments"], 2)
        self.assertEqual(report["teacher_assignments"], 2)
        interval = report["overall_skip_kan_minus_teacher_95pct_interval_per_wall"]
        self.assertEqual(interval["mean"], 10.0)


if __name__ == "__main__":
    unittest.main()

import unittest

from tests.test_human_review import priority_review_item
from xiamen_mahjong.agents import GameAction
from xiamen_mahjong.game import XiamenMahjongGame
from xiamen_mahjong.rules import XiamenRules
from xiamen_mahjong.targeted_intervention import (
    TARGETED_TWO_DRAW_BEHAVIOR_VERSION,
    TargetedTwoDrawInterventionBehavior,
    audit_targeted_two_draw_interventions,
)
from xiamen_mahjong.training import (
    TeacherDecision,
    TrainingTrajectory,
    _turn_actions,
)


class FirstDiscardPolicy:
    def choose_turn_action(self, game, player_id):
        return next(
            action for action in _turn_actions(game, player_id)
            if action.kind == "discard"
        )

    def choose_response(self, _game, _player_id, options):
        return options[0]


class SecondDiscardPolicy(FirstDiscardPolicy):
    def choose_turn_action(self, game, player_id):
        discards = [
            action for action in _turn_actions(game, player_id)
            if action.kind == "discard"
        ]
        return discards[1]


class FixedExplainCandidate:
    def __init__(self, tile):
        self.tile = tile

    def explain_discard(self, _game, _player_id):
        return [{"tile": self.tile}]


class TargetedInterventionTests(unittest.TestCase):
    def test_behavior_randomizes_once_then_restores_teacher(self):
        game = XiamenMahjongGame(
            seed=202635099,
            rules=XiamenRules.classic(),
            auto_advance=False,
            human_seat=-1,
        )
        player_id = game.current_player
        teacher = FirstDiscardPolicy()
        candidate = SecondDiscardPolicy()
        behavior = TargetedTwoDrawInterventionBehavior(
            seed=7,
            teacher=teacher,
            candidate=candidate,
        )
        action = behavior.choose_turn_action(game, player_id)
        self.assertIn(
            action,
            {
                teacher.choose_turn_action(game, player_id),
                candidate.choose_turn_action(game, player_id),
            },
        )
        legal = tuple(_turn_actions(game, player_id))
        self.assertEqual(
            behavior.action_probability(
                game, player_id, legal, action, is_response=False
            ),
            0.5,
        )
        second = behavior.choose_turn_action(game, player_id)
        self.assertEqual(second, teacher.choose_turn_action(game, player_id))
        self.assertEqual(
            behavior.action_probability(
                game, player_id, legal, second, is_response=False
            ),
            1.0,
        )

    def test_audit_reproduces_binary_support_and_grouped_ht_delta(self):
        item = priority_review_item()
        actions = tuple(
            GameAction("discard", int(action["tile"]))
            for action in item["legal_actions"]
        )
        teacher_index = int(item["reference_teacher_index"])
        target_index = next(
            index for index in range(len(actions)) if index != teacher_index
        )
        target_tile = int(actions[target_index].tile)
        rows = []
        for seat in range(4):
            use_candidate = seat < 2
            score = 10 if use_candidate else 0
            decision = TeacherDecision(
                profile="classic",
                seed=None,
                seat=seat,
                state=item["state"],
                legal_actions=actions,
                chosen_index=teacher_index,
                executed_index=target_index if use_candidate else teacher_index,
                executed_probability=0.5,
            )
            scores = [0, 0, 0, 0]
            scores[seat] = score
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
                        "behavior_policy": TARGETED_TWO_DRAW_BEHAVIOR_VERSION,
                        "base_policy": "heuristic_teacher",
                        "candidate_policy": "two_draw_tenpai_reach_v1",
                        "candidate_action_probability": 0.5,
                        "maximum_interventions_per_trajectory": 1,
                    },
                    decisions=(decision,),
                    outcome={"scores": scores},
                    public_actions=(),
                    trajectory_id=f"{seat + 10:032x}",
                    split_group_id="f" * 32,
                )
            )
        report = audit_targeted_two_draw_interventions(
            rows,
            minimum_interventions=4,
            minimum_wall_groups=1 + 1,
            candidate=FixedExplainCandidate(target_tile),
        )
        # The public API intentionally requires >=2 wall groups. Duplicate a
        # group below would be needed for readiness; this row still verifies
        # exact action support and the Horvitz-Thompson arithmetic.
        self.assertEqual(report["issues"], {})
        self.assertEqual(report["candidate_assignments"], 2)
        self.assertEqual(report["teacher_assignments"], 2)
        estimate = report["horvitz_thompson_candidate_minus_teacher_per_wall"]
        self.assertEqual(estimate["mean"], 10.0)


if __name__ == "__main__":
    unittest.main()

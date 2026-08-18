import unittest

from scripts.audit_teacher_tenpai_value import TenpaiValueAudit, _discard_rows
from xiamen_mahjong.training import _decision, _turn_actions
from xiamen_mahjong.game import XiamenMahjongGame
from xiamen_mahjong.rules import XiamenRules


class TeacherTenpaiValueAuditTest(unittest.TestCase):
    def test_controlled_rows_identify_a_near_tied_higher_value_alternative(self):
        audit = TenpaiValueAudit(score_margin=2.0)
        audit.inspect_rows(
            [
                {
                    "tile": 3,
                    "teacher_score": 100.0,
                    "wait_faces": 2,
                    "live_copies": 2,
                    "weighted_value": 48.0,
                    "mean_score": 24.0,
                },
                {
                    "tile": 7,
                    "teacher_score": 99.0,
                    "wait_faces": 1,
                    "live_copies": 4,
                    "weighted_value": 120.0,
                    "mean_score": 30.0,
                },
            ],
            teacher_tile=3,
        )
        self.assertEqual(audit.teacher_tenpai_decisions, 1)
        self.assertEqual(audit.near_tied_tenpai_choices, 1)
        self.assertEqual(audit.better_value_alternatives, 1)
        self.assertEqual(audit.reason_counts["more_live_and_higher_score"], 1)

    def test_score_margin_excludes_a_distant_alternative(self):
        audit = TenpaiValueAudit(score_margin=2.0)
        audit.inspect_rows(
            [
                {
                    "tile": 3,
                    "teacher_score": 100.0,
                    "wait_faces": 1,
                    "live_copies": 1,
                    "weighted_value": 12.0,
                    "mean_score": 12.0,
                },
                {
                    "tile": 7,
                    "teacher_score": 90.0,
                    "wait_faces": 1,
                    "live_copies": 4,
                    "weighted_value": 120.0,
                    "mean_score": 30.0,
                },
            ],
            teacher_tile=3,
        )
        self.assertEqual(audit.teacher_tenpai_decisions, 1)
        self.assertEqual(audit.near_tied_tenpai_choices, 0)
        self.assertEqual(audit.better_value_alternatives, 0)

    def test_actual_teacher_decision_can_be_scored_from_actor_visible_state(self):
        game = XiamenMahjongGame(
            seed=202608800,
            rules=XiamenRules.classic(),
            auto_advance=False,
            human_seat=-1,
        )
        player_id = game.current_player
        legal = tuple(_turn_actions(game, player_id))
        chosen = game.teacher.choose_turn_action(game, player_id)
        decision = _decision(game, 202608800, player_id, legal, chosen)
        rows = _discard_rows(decision)
        if chosen.kind == "discard":
            self.assertTrue(rows)
            self.assertEqual(int(rows[0]["tile"]), chosen.tile)

    def test_payload_is_aggregate_only(self):
        payload = TenpaiValueAudit(score_margin=2.0).payload(
            profile="classic", maximum_teacher_trajectories=10
        )

        def collect_keys(value):
            if isinstance(value, dict):
                result = set(value)
                for child in value.values():
                    result.update(collect_keys(child))
                return result
            if isinstance(value, (list, tuple)):
                result = set()
                for child in value:
                    result.update(collect_keys(child))
                return result
            return set()

        exported_keys = collect_keys(payload)
        self.assertTrue(
            {
                "hand",
                "opponent_hand",
                "wall_order",
                "public_history",
                "seed",
                "candidate_rows",
                "decision_traces",
            }.isdisjoint(exported_keys)
        )
        self.assertEqual(
            payload["status"], "diagnostic_only_not_authorized_for_action_selection"
        )


if __name__ == "__main__":
    unittest.main()

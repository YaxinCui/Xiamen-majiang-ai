import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.test_human_review import priority_review_item
from xiamen_mahjong.agents import GameAction
from xiamen_mahjong.game import XiamenMahjongGame
from xiamen_mahjong.low_margin_intervention import (
    LOW_MARGIN_TOP2_BEHAVIOR_VERSION,
    LOW_MARGIN_TOP2_POLICY_VERSION,
    LowMarginTop2InterventionBehavior,
    audit_low_margin_causal_records,
    audit_low_margin_top2_interventions,
    extract_low_margin_causal_records,
    read_low_margin_causal_jsonl,
    write_low_margin_causal_jsonl,
)
from xiamen_mahjong.rules import XiamenRules
from xiamen_mahjong.training import TeacherDecision, TrainingTrajectory, _turn_actions


class FixedTop2Teacher:
    def choose_turn_action(self, game, player_id):
        return next(
            action for action in _turn_actions(game, player_id)
            if action.kind == "discard"
        )

    def choose_response(self, _game, _player_id, options):
        return options[0]

    def explain_discard(self, game, player_id):
        discards = sorted(
            {
                int(action.tile)
                for action in _turn_actions(game, player_id)
                if action.kind == "discard"
            }
        )
        return [
            {"tile": discards[0], "score": 10.0},
            {"tile": discards[1], "score": 9.0},
        ]


class LowMarginInterventionTests(unittest.TestCase):
    def test_behavior_randomizes_once_then_restores_teacher(self):
        game = XiamenMahjongGame(
            seed=202649099,
            rules=XiamenRules.classic(),
            auto_advance=False,
            human_seat=-1,
        )
        teacher = FixedTop2Teacher()
        behavior = LowMarginTop2InterventionBehavior(seed=7, teacher=teacher)
        player_id = game.current_player
        legal = tuple(_turn_actions(game, player_id))
        first = behavior.choose_turn_action(game, player_id)
        self.assertIn(first, {legal[0], legal[1]})
        self.assertEqual(
            behavior.action_probability(
                game, player_id, legal, first, is_response=False
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

    def test_audit_rejects_non_zero_sum_and_accepts_exact_support(self):
        item = priority_review_item()
        actions = tuple(
            GameAction("discard", int(action["tile"]))
            for action in item["legal_actions"]
        )
        teacher_index = int(item["reference_teacher_index"])
        # Build a legal exported state whose actual frozen top-two indices are
        # reproduced by the production audit rather than a test stub.
        from xiamen_mahjong.human_review import _ActorVisibleReviewGame
        from xiamen_mahjong.agents import HeuristicTeacherAgent

        game = _ActorVisibleReviewGame({"state": item["state"]})
        ranked = HeuristicTeacherAgent().explain_discard(game, 0)
        top_tiles = [int(row["tile"]) for row in ranked[:2]]
        top_indices = [
            next(index for index, action in enumerate(actions) if action.tile == tile)
            for tile in top_tiles
        ]
        teacher_index = top_indices[0]
        rows = []
        for group_offset in range(2):
            for seat in range(4):
                use_alternative = (seat + group_offset) % 2 == 0
                score = 10 if use_alternative else -2
                decision = TeacherDecision(
                    profile="classic",
                    seed=None,
                    seat=seat,
                    state=item["state"],
                    legal_actions=actions,
                    chosen_index=teacher_index,
                    executed_index=top_indices[1] if use_alternative else teacher_index,
                    executed_probability=0.5,
                )
                scores = [0, 0, 0, 0]
                scores[seat] = score
                scores[(seat + 1) % 4] = -score
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
                            "behavior_policy": LOW_MARGIN_TOP2_BEHAVIOR_VERSION,
                            "base_policy": "heuristic_teacher",
                            "candidate_policy": LOW_MARGIN_TOP2_POLICY_VERSION,
                            "alternative_action_probability": 0.5,
                            "maximum_teacher_score_margin": 2.0,
                            "maximum_interventions_per_trajectory": 1,
                        },
                        decisions=(decision,),
                        outcome={"scores": scores},
                        public_actions=(),
                        trajectory_id=f"{group_offset * 4 + seat + 1:032x}",
                        split_group_id=f"{group_offset + 1:032x}",
                    )
                )
        report = audit_low_margin_top2_interventions(
            rows,
            maximum_margin=2.0,
            minimum_interventions=8,
            minimum_wall_groups=2,
        )
        self.assertEqual(report["issues"], {})
        self.assertTrue(report["structurally_ready"])
        self.assertEqual(report["alternative_assignments"], 4)
        self.assertEqual(report["teacher_assignments"], 4)

        bad = list(rows)
        bad[0] = TrainingTrajectory(
            **{
                **bad[0].__dict__,
                "outcome": {"scores": [1, 0, 0, 0]},
            }
        )
        rejected = audit_low_margin_top2_interventions(
            bad,
            maximum_margin=2.0,
            minimum_interventions=8,
            minimum_wall_groups=2,
        )
        self.assertIn("invalid_or_non_zero_sum_terminal_scores", rejected["issues"])
        self.assertFalse(rejected["structurally_ready"])

    def test_slim_record_roundtrip_keeps_only_actor_visible_contract(self):
        item = priority_review_item()
        actions = tuple(
            GameAction("discard", int(action["tile"]))
            for action in item["legal_actions"]
        )
        from xiamen_mahjong.human_review import _ActorVisibleReviewGame
        from xiamen_mahjong.agents import HeuristicTeacherAgent

        game = _ActorVisibleReviewGame({"state": item["state"]})
        ranked = HeuristicTeacherAgent().explain_discard(game, 0)
        indices = [
            next(
                index for index, action in enumerate(actions)
                if action.tile == int(row["tile"])
            )
            for row in ranked[:2]
        ]
        rows = []
        for seat in range(4):
            decision = TeacherDecision(
                profile="classic",
                seed=None,
                seat=seat,
                state=item["state"],
                legal_actions=actions,
                chosen_index=indices[0],
                executed_index=indices[1] if seat % 2 == 0 else indices[0],
                executed_probability=0.5,
            )
            scores = [0, 0, 0, 0]
            scores[seat] = 8 if seat % 2 == 0 else -4
            scores[(seat + 1) % 4] = -scores[seat]
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
                        "behavior_policy": LOW_MARGIN_TOP2_BEHAVIOR_VERSION,
                        "candidate_policy": LOW_MARGIN_TOP2_POLICY_VERSION,
                        "alternative_action_probability": 0.5,
                        "maximum_interventions_per_trajectory": 1,
                    },
                    decisions=(decision,),
                    outcome={"scores": scores},
                    public_actions=(),
                    trajectory_id=f"{seat + 101:032x}",
                    split_group_id="a" * 32,
                )
            )
        slim = extract_low_margin_causal_records(rows)
        self.assertEqual(len(slim), 4)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            write_low_margin_causal_jsonl(slim, path)
            loaded = read_low_margin_causal_jsonl(path)
        self.assertEqual(loaded, slim)
        audit = audit_low_margin_causal_records(loaded, expected_wall_groups=1)
        self.assertTrue(audit["ready"])
        payload = slim[0].payload()
        serialized = str(payload).lower()
        self.assertNotIn("seed", serialized)
        self.assertNotIn("walltiles", serialized)
        self.assertNotIn("opponent_hand", serialized)

        mismatched = dict(payload)
        mismatched["teacher_margin"] = (
            2.0 if float(payload["teacher_margin"]) == 0.0 else 0.0
        )
        from xiamen_mahjong.low_margin_intervention import LowMarginCausalRecord

        with self.assertRaises(ValueError):
            LowMarginCausalRecord.from_payload(mismatched)
        schema_only = LowMarginCausalRecord.from_payload(
            mismatched, validate_actor_visible_pair=False
        )
        self.assertEqual(
            schema_only.teacher_margin, mismatched["teacher_margin"]
        )


if __name__ == "__main__":
    unittest.main()

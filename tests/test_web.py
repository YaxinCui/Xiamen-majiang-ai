import json
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from urllib.request import Request, urlopen

from xiamen_mahjong.agents import HeuristicTeacherAgent
from xiamen_mahjong.human_data import (
    audit_local_human_evaluation,
    audit_local_human_teacher_corrections,
    audit_local_human_trajectories,
    is_eligible_human_teacher_discard_correction,
    require_local_human_training_approval,
    split_local_human_trajectories,
)
from xiamen_mahjong.web import GameStore, make_handler
from http.server import ThreadingHTTPServer


class WebTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(GameStore()))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_game_endpoint_exposes_public_state(self):
        with urlopen(f"{self.base_url}/api/game") as response:
            state = json.load(response)
        self.assertIn("actions", state)
        self.assertIsNone(state["players"][1]["hand"])
        self.assertEqual(state["rules"]["profile"], "classic")
        self.assertIn("classic", [profile["id"] for profile in state["rule_profiles"]])
        self.assertEqual(state["local_human_recording"]["scope"], "disabled")
        self.assertTrue(state["debug_ai_hands_allowed"])

    def test_debug_state_explicitly_reveals_ai_hands(self):
        with urlopen(f"{self.base_url}/api/game?debug=1") as response:
            state = json.load(response)
        self.assertTrue(all(player["hand"] for player in state["players"][1:]))

    def test_recording_forces_hidden_opponents_but_manual_rule_debug_stays_available(self):
        with tempfile.TemporaryDirectory() as directory:
            store = GameStore(
                human_log=Path(directory) / "human.jsonl",
                human_recording_purpose="training",
            )
            recorded_state = store._public_state(reveal_ai_hands=True)
            self.assertFalse(recorded_state["debug_ai_hands_allowed"])
            self.assertTrue(
                all(
                    player["hand"] is None
                    for player in recorded_state["players"][1:]
                )
            )
            store.new_game(
                seed=202608073,
                rules_profile="classic",
                reset_match=True,
                table_mode="four_player_manual",
            )
            manual_state = store._public_state(reveal_ai_hands=True)
            self.assertTrue(manual_state["debug_ai_hands_allowed"])
            self.assertTrue(all(player["hand"] for player in manual_state["players"]))

    def test_static_page_is_served(self):
        with urlopen(f"{self.base_url}/") as response:
            page = response.read().decode("utf-8")
        self.assertIn("厦门麻将", page)
        self.assertIn("显示 AI 手牌（调试）", page)
        with urlopen(f"{self.base_url}/app.js") as response:
            script = response.read().decode("utf-8")
        self.assertIn("let debugAiHands = false;", script)

    def test_rules_guide_page_is_served(self):
        with urlopen(f"{self.base_url}/guide.html") as response:
            page = response.read().decode("utf-8")
        self.assertIn("厦门麻将完整教学", page)
        self.assertIn("怎样才算胡牌？", page)
        self.assertIn("游金、三金倒、天听与天胡", page)
        self.assertIn("17 张牌", page)
        self.assertIn("听什么牌，什么时候能胡", page)
        self.assertIn("两面听", page)
        self.assertIn("底分、加水与付款", page)
        with urlopen(f"{self.base_url}/guide.js") as response:
            script = response.read().decode("utf-8")
        self.assertIn("updateGuideNavigation", script)

    def test_action_endpoint_accepts_a_legal_human_action(self):
        with urlopen(f"{self.base_url}/api/game") as response:
            state = json.load(response)
        action = next(
            (candidate for candidate in state["actions"] if candidate["kind"] == "discard"),
            state["actions"][0],
        )
        request = Request(
            f"{self.base_url}/api/game/action",
            data=json.dumps(action).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            next_state = json.load(response)
        self.assertIn(next_state["phase"], {"discard", "response", "over"})

    def test_new_classic_hand_carries_scores_and_continues_the_dealer(self):
        store = GameStore()
        previous = store.game
        previous.phase = "over"
        previous.winner = previous.dealer
        previous.win_type = "self_draw"
        previous.players[0].score = 88
        state = store.new_game(seed=53, rules_profile="classic")
        self.assertEqual(state["dealer"], previous.dealer)
        self.assertEqual(state["dealer_streak"], previous.dealer_streak + 1)
        self.assertEqual(state["hand_number"], previous.hand_number + 1)
        self.assertEqual(state["players"][0]["score"], 88)

    def test_explicit_ai_profile_is_reported_without_changing_default(self):
        default_state = GameStore().state()
        checkpoint_state = GameStore(
            ai_agent=HeuristicTeacherAgent(),
            ai_profile="explicit_test_checkpoint",
            ai_identity="sha256:test",
        ).state()
        self.assertEqual(default_state["ai_profile"], "heuristic_teacher")
        self.assertEqual(checkpoint_state["ai_profile"], "explicit_test_checkpoint")

    def test_slow_ai_mode_advances_one_server_authoritative_step(self):
        store = GameStore()
        state = None
        for seed in range(40):
            candidate = store.new_game(
                seed=seed,
                rules_profile="classic",
                reset_match=True,
                ai_delay_seconds=5,
            )
            if candidate["awaiting_ai_action"]:
                state = candidate
                break
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state["ai_delay_seconds"], 5)
        self.assertEqual(state["table_mode"], "solo_vs_ai")
        next_state = store.advance_ai()
        self.assertGreaterEqual(next_state["turn_count"], state["turn_count"])
        self.assertNotEqual(next_state["phase"], "setup")

    def test_four_player_manual_mode_rotates_the_private_view(self):
        store = GameStore()
        state = store.new_game(
            seed=202608072,
            rules_profile="classic",
            reset_match=True,
            table_mode="four_player_manual",
            ai_delay_seconds=10,
        )
        self.assertEqual(state["table_mode"], "four_player_manual")
        self.assertEqual(state["manual_seats"], [0, 1, 2, 3])
        self.assertEqual(state["ai_profile"], "four_player_manual")
        self.assertFalse(state["local_human_recording"]["enabled"])
        self.assertTrue(state["actions"])
        actor = state["action_seat"]
        self.assertEqual(state["viewer_seat"], actor)
        self.assertEqual(
            [player["seat"] for player in state["players"] if player["hand"] is not None],
            [actor],
        )

    def test_opt_in_human_recording_writes_only_safe_completed_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "human.jsonl"
            store = GameStore(
                human_log=output,
                human_recording_purpose="training",
                ai_agent=HeuristicTeacherAgent(),
                ai_profile="explicit_test_checkpoint",
                ai_identity="sha256:test",
            )
            store.new_game(seed=1, rules_profile="classic", reset_match=True)
            debug_state = store._public_state(reveal_ai_hands=True)
            self.assertFalse(debug_state["debug_ai_hands_allowed"])
            self.assertTrue(
                all(player["hand"] is None for player in debug_state["players"][1:])
            )
            reference = HeuristicTeacherAgent().choose_turn_action(
                store.game, store.game.human_seat
            )
            action = next(
                candidate
                for candidate in store.game.human_actions()
                if (
                    candidate["kind"],
                    candidate.get("tile"),
                    tuple(candidate.get("tiles", [])),
                )
                != (reference.kind, reference.tile, reference.tiles)
            )
            captured = store._capture_human_decision(action)
            self.assertTrue(
                is_eligible_human_teacher_discard_correction(captured)
            )
            self.assertFalse(
                is_eligible_human_teacher_discard_correction(
                    replace(
                        captured,
                        state={**captured.state, "phase": "response"},
                    )
                )
            )
            self.assertNotEqual(
                captured.reference_teacher_index, captured.chosen_index
            )
            store._human_decisions.append(captured)
            store._human_score_start = (10, -4, -3, -3)
            for player, score in zip(store.game.players, (26, -12, -7, -7)):
                player.score = score
            # The recorder only flushes completed hands. This synthetic finish
            # avoids relying on a particular random game length while keeping
            # the ordinary game state and safe export boundary intact.
            store.game.phase = "over"
            store.game.win_type = "draw"
            store._write_completed_human_hand()
            store._write_completed_human_hand()
            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertNotIn("seed", record)
            self.assertEqual(record["outcome"]["scores"], [16, -8, -4, -4])
            self.assertEqual(record["outcome"]["score_semantics"], "single_hand_delta")
            self.assertEqual(
                record["source_metadata"]["collector"], "local_human_opt_in"
            )
            self.assertEqual(
                record["source_metadata"]["training_default"],
                "excluded_until_separate_quality_review",
            )
            self.assertEqual(record["source_metadata"]["recording_purpose"], "training")
            self.assertRegex(
                record["source_metadata"]["recording_session_id"], r"^[0-9a-f]{32}$"
            )
            self.assertEqual(
                record["source_metadata"]["opponent_policy"],
                "sha256:test",
            )
            self.assertEqual(
                record["source_metadata"]["opponent_hand_reveal"],
                "server_forced_disabled",
            )
            self.assertEqual(
                record["source_metadata"]["reference_policy"],
                "heuristic_teacher_v1",
            )
            self.assertEqual(
                record["source_metadata"]["reference_label"],
                "same_state_frozen_teacher_action_index",
            )
            self.assertEqual(
                record["agent_profiles"],
                [
                    "local_human_opt_in",
                    "explicit_test_checkpoint",
                    "explicit_test_checkpoint",
                    "explicit_test_checkpoint",
                ],
            )
            decision = record["decisions"][0]
            self.assertEqual(decision["chosen_index"], decision["executed_index"])
            self.assertNotEqual(
                decision["reference_teacher_index"], decision["chosen_index"]
            )
            self.assertNotIn("wall", decision["state"])
            self.assertNotIn("opponent_hands", decision["state"])
            self.assertTrue(store._public_state()["local_human_recording"]["enabled"])
            recording_state = store._public_state()["local_human_recording"]
            self.assertEqual(recording_state["session_completed_hands"], 1)
            self.assertEqual(
                recording_state["session_eligible_discard_decisions"], 1
            )
            self.assertEqual(recording_state["session_discard_disagreements"], 1)
            audit = audit_local_human_trajectories([output], minimum_hands=1)
            self.assertTrue(audit["ready_for_manual_review"])
            self.assertEqual(audit["valid_hands"], 1)
            self.assertEqual(audit["opponent_policies"], {"sha256:test": 1})
            self.assertEqual(audit["recording_purposes"], {"training": 1})
            reference_summary = audit["reference_teacher_summary"]
            self.assertEqual(reference_summary["decisions_with_reference"], 1)
            self.assertEqual(reference_summary["disagreements"], 1)
            self.assertEqual(
                reference_summary[
                    "eligible_ordinary_discard_reference_decisions"
                ],
                1,
            )
            self.assertEqual(
                reference_summary[
                    "eligible_ordinary_discard_disagreements"
                ],
                1,
            )
            self.assertEqual(
                reference_summary["reference_policies"],
                {"heuristic_teacher_v1": 1},
            )
            summary = audit["human_match_summary"]
            self.assertEqual(summary["scope"], "structurally_valid_completed_hands_only")
            self.assertEqual(summary["hands"], 1)
            self.assertEqual(summary["human_score_delta_mean"], 16.0)
            self.assertIsNone(summary["human_score_delta_stderr"])
            self.assertEqual(summary["human_win_rate"], 0.0)
            self.assertEqual(summary["draw_rate"], 1.0)
            with self.assertRaisesRegex(ValueError, "默认禁止训练"):
                require_local_human_training_approval(
                    [output], manually_approved=False, minimum_hands=1
                )
            # The trainer itself must fail closed too; a caller cannot bypass
            # the recorder's metadata boundary merely by naming this file as
            # an additional training input.
            from scripts.train_policy_value import load_examples

            with self.assertRaisesRegex(ValueError, "默认禁止训练"):
                load_examples(
                    output, SimpleNamespace(allow_local_human_data=False)
                )
            approved = require_local_human_training_approval(
                [output], manually_approved=True, minimum_hands=1
            )
            self.assertTrue(approved["manual_training_approval"])
            self.assertEqual(approved["audit"]["valid_hands"], 1)
            tampered = Path(directory) / "human-with-q.jsonl"
            tampered_record = json.loads(json.dumps(record))
            tampered_record["decisions"][0]["action_values"] = [
                0.0 for _ in tampered_record["decisions"][0]["legal_actions"]
            ]
            tampered.write_text(json.dumps(tampered_record) + "\n", encoding="utf-8")
            tampered_audit = audit_local_human_trajectories(
                [tampered], minimum_hands=1
            )
            self.assertFalse(tampered_audit["ready_for_manual_review"])
            self.assertEqual(
                tampered_audit["issues"],
                {"human_record_must_not_have_action_value_targets": 1},
            )
            duplicate_audit = audit_local_human_trajectories(
                [output, output], minimum_hands=1
            )
            self.assertFalse(duplicate_audit["ready_for_manual_review"])
            self.assertEqual(duplicate_audit["duplicate_hands"], 1)
            duplicate_new_session = Path(directory) / "human-duplicate-new-session.jsonl"
            duplicate_new_session_record = json.loads(json.dumps(record))
            duplicate_new_session_record["trajectory_id"] = "duplicate-new-session"
            duplicate_new_session_record["split_group_id"] = "duplicate-new-session"
            duplicate_new_session_record["source_metadata"]["recording_session_id"] = "f" * 32
            duplicate_new_session.write_text(
                json.dumps(duplicate_new_session_record) + "\n", encoding="utf-8"
            )
            duplicate_cross_session_audit = audit_local_human_trajectories(
                [output, duplicate_new_session], minimum_hands=1
            )
            self.assertFalse(duplicate_cross_session_audit["ready_for_manual_review"])
            self.assertEqual(duplicate_cross_session_audit["duplicate_hands"], 1)

            # Split only complete, audited human hands.  The source fixture
            # has one decision per hand; cloning it with distinct opaque IDs
            # and hand numbers creates a deterministic 100-hand test corpus
            # without adding seeds, other-player hands or value targets.
            split_input = Path(directory) / "split-input.jsonl"
            split_rows = []
            for index in range(100):
                split_record = json.loads(json.dumps(record))
                split_record["trajectory_id"] = f"human-split-{index}"
                split_record["split_group_id"] = f"human-group-{index}"
                split_record["hand_number"] = index + 1
                split_rows.append(json.dumps(split_record, ensure_ascii=False))
            split_input.write_text("\n".join(split_rows) + "\n", encoding="utf-8")
            partitions, split_audit = split_local_human_trajectories(
                [split_input], minimum_hands=100
            )
            self.assertEqual(sum(len(records) for records in partitions.values()), 100)
            self.assertTrue(all(partitions[split] for split in partitions))
            self.assertEqual(split_audit["split_group_overlap"], 0)
            self.assertEqual(
                split_audit["split_version"], "xiamen-local-human-hand-split-v1"
            )
            split_group_sets = {
                split: {trajectory.split_group_id for trajectory in records}
                for split, records in partitions.items()
            }
            self.assertFalse(split_group_sets["train"] & split_group_sets["validation"])
            self.assertFalse(split_group_sets["train"] & split_group_sets["test"])
            self.assertFalse(split_group_sets["validation"] & split_group_sets["test"])

            correction_audit = audit_local_human_teacher_corrections(
                [split_input],
                minimum_hands=100,
                minimum_reference_decisions=100,
                minimum_disagreements=50,
            )
            self.assertTrue(
                correction_audit["ready_for_teacher_residual_experiment"]
            )
            self.assertEqual(
                correction_audit["correction_summary"]["teacher_disagreements"],
                100,
            )

            from scripts.train_policy_value import load_examples

            human_examples = load_examples(
                split_input,
                SimpleNamespace(
                    allow_local_human_data=True,
                    feature_version=3,
                    history_window=24,
                    full_public_history=False,
                    value_scale=80.0,
                    human_weight=1.0,
                    human_teacher_disagreement_weight=3.0,
                    human_discard_corrections_only=True,
                    exploration_weight=0.35,
                    synthetic_weight=1.0,
                    action_value_weight=1.0,
                    action_value_margin_scale=0.0,
                    action_value_stderr_scale=0.0,
                ),
            )
            self.assertEqual(len(human_examples), 100)
            self.assertTrue(
                all(example.sample_weight == 3.0 for example in human_examples)
            )
            noneligible_input = Path(directory) / "noneligible-human.jsonl"
            noneligible_record = json.loads(json.dumps(record))
            noneligible_record["trajectory_id"] = "noneligible-response"
            noneligible_record["split_group_id"] = "noneligible-response"
            noneligible_record["decisions"][0]["state"]["phase"] = "response"
            noneligible_input.write_text(
                json.dumps(noneligible_record, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            filtered = load_examples(
                noneligible_input,
                SimpleNamespace(
                    allow_local_human_data=True,
                    feature_version=3,
                    history_window=24,
                    full_public_history=False,
                    value_scale=80.0,
                    human_weight=1.0,
                    human_teacher_disagreement_weight=3.0,
                    human_discard_corrections_only=True,
                    exploration_weight=0.35,
                    synthetic_weight=1.0,
                    action_value_weight=1.0,
                    action_value_margin_scale=0.0,
                    action_value_stderr_scale=0.0,
                ),
            )
            self.assertEqual(filtered, [])

            # A frozen-AI human benchmark must be recorded evaluation-only;
            # this purpose is rejected by both training approval and the
            # hand splitter, but may be summarized by the strength auditor.
            evaluation_input = Path(directory) / "evaluation-input.jsonl"
            evaluation_rows = []
            for index in range(100):
                evaluation_record = json.loads(json.dumps(record))
                evaluation_record["trajectory_id"] = f"human-evaluation-{index}"
                evaluation_record["split_group_id"] = f"evaluation-group-{index}"
                evaluation_record["hand_number"] = index + 1
                evaluation_record["source_metadata"]["recording_purpose"] = "evaluation"
                evaluation_record["source_metadata"]["recording_session_id"] = (
                    f"{index // 10 + 1:032x}"
                )
                evaluation_record["outcome"]["scores"] = [-16, 6, 5, 5]
                evaluation_rows.append(json.dumps(evaluation_record, ensure_ascii=False))
            evaluation_input.write_text(
                "\n".join(evaluation_rows) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "不得用于训练"):
                require_local_human_training_approval(
                    [evaluation_input], manually_approved=True, minimum_hands=100
                )
            with self.assertRaisesRegex(ValueError, "不得切分为训练数据"):
                split_local_human_trajectories([evaluation_input], minimum_hands=100)
            evaluation = audit_local_human_evaluation(
                [evaluation_input], minimum_hands=100, minimum_sessions=10
            )
            self.assertTrue(evaluation["ready_for_manual_human_strength_review"])
            self.assertEqual(evaluation["comparison"]["recording_sessions"], 10)
            self.assertEqual(
                evaluation["comparison"]["ai_side_score_delta_mean"], 16.0
            )
            self.assertEqual(
                evaluation["comparison"]["ai_side_score_delta_95pct_low"], 16.0
            )
            self.assertTrue(evaluation["comparison"]["positive_ai_side_lcb"])
            self.assertEqual(evaluation["comparison"]["ai_roster_size"], 3)
            self.assertAlmostEqual(
                evaluation["comparison"]["ai_per_seat_score_delta_mean"],
                16.0 / 3.0,
            )
            self.assertAlmostEqual(
                evaluation["comparison"]["ai_per_seat_score_delta_95pct_low"],
                16.0 / 3.0,
            )
            self.assertTrue(
                evaluation["comparison"]["positive_ai_per_seat_lcb"]
            )
            insufficient_sessions = audit_local_human_evaluation(
                [evaluation_input], minimum_hands=100, minimum_sessions=11
            )
            self.assertFalse(insufficient_sessions["ready_for_manual_human_strength_review"])
            self.assertIn(
                "insufficient_evaluation_sessions",
                insufficient_sessions["gate_reasons"],
            )


if __name__ == "__main__":
    unittest.main()

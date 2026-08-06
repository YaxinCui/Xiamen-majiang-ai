import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen

from xiamen_mahjong.agents import HeuristicTeacherAgent
from xiamen_mahjong.human_data import audit_local_human_trajectories
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

    def test_debug_state_explicitly_reveals_ai_hands(self):
        with urlopen(f"{self.base_url}/api/game?debug=1") as response:
            state = json.load(response)
        self.assertTrue(all(player["hand"] for player in state["players"][1:]))

    def test_static_page_is_served(self):
        with urlopen(f"{self.base_url}/") as response:
            page = response.read().decode("utf-8")
        self.assertIn("厦门麻将", page)
        self.assertIn("隐藏 AI 手牌（调试）", page)
        with urlopen(f"{self.base_url}/app.js") as response:
            script = response.read().decode("utf-8")
        self.assertIn("let debugAiHands = true;", script)

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

    def test_opt_in_human_recording_writes_only_safe_completed_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "human.jsonl"
            store = GameStore(human_log=output)
            store.new_game(seed=1, rules_profile="classic", reset_match=True)
            action = next(
                candidate
                for candidate in store.game.human_actions()
                if candidate["kind"] == "discard"
            )
            store._human_decisions.append(store._capture_human_decision(action))
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
            self.assertEqual(
                record["source_metadata"]["opponent_policy"],
                "heuristic_teacher",
            )
            decision = record["decisions"][0]
            self.assertEqual(decision["chosen_index"], decision["executed_index"])
            self.assertNotIn("wall", decision["state"])
            self.assertNotIn("opponent_hands", decision["state"])
            self.assertTrue(store._public_state()["local_human_recording"]["enabled"])
            audit = audit_local_human_trajectories([output], minimum_hands=1)
            self.assertTrue(audit["ready_for_manual_review"])
            self.assertEqual(audit["valid_hands"], 1)
            self.assertEqual(audit["opponent_policies"], {"heuristic_teacher": 1})
            duplicate_audit = audit_local_human_trajectories(
                [output, output], minimum_hands=1
            )
            self.assertFalse(duplicate_audit["ready_for_manual_review"])
            self.assertEqual(duplicate_audit["duplicate_hands"], 1)


if __name__ == "__main__":
    unittest.main()

import json
import threading
import unittest
from urllib.request import Request, urlopen

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


if __name__ == "__main__":
    unittest.main()

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
        self.assertIn("new120", [profile["id"] for profile in state["rule_profiles"]])

    def test_debug_state_explicitly_reveals_ai_hands(self):
        with urlopen(f"{self.base_url}/api/game?debug=1") as response:
            state = json.load(response)
        self.assertTrue(all(player["hand"] for player in state["players"][1:]))

    def test_static_page_is_served(self):
        with urlopen(f"{self.base_url}/") as response:
            page = response.read().decode("utf-8")
        self.assertIn("厦门麻将", page)
        self.assertIn("隐藏 AI 手牌（调试）", page)
        self.assertIn("进入 120 张新厦麻", page)
        with urlopen(f"{self.base_url}/app.js") as response:
            script = response.read().decode("utf-8")
        self.assertIn("let debugAiHands = true;", script)

    def test_new120_game_page_and_api_are_independent(self):
        with urlopen(f"{self.base_url}/play-120.html") as response:
            page = response.read().decode("utf-8")
        self.assertIn("120 张新厦麻", page)
        self.assertIn('data-api-base="/api/game-120"', page)
        self.assertIn("切换 144 张老厦麻", page)
        with urlopen(f"{self.base_url}/play-120.css") as response:
            stylesheet = response.read().decode("utf-8")
        self.assertIn(".new120-summary", stylesheet)
        with urlopen(f"{self.base_url}/api/game-120?debug=1") as response:
            state = json.load(response)
        self.assertEqual(state["rules"]["profile"], "new120")
        self.assertEqual(state["rules"]["tile_count"], 120)
        self.assertTrue(all(player["hand"] for player in state["players"]))
        present = {
            tile["id"]
            for player in state["players"]
            for tile in [*(player["hand"] or []), *player["flowers"]]
        }
        self.assertFalse(any(27 <= tile < 33 for tile in present))

        request = Request(
            f"{self.base_url}/api/game-120/new",
            data=json.dumps({"reset_match": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            new_state = json.load(response)
        self.assertEqual(new_state["rules"]["profile"], "new120")
        with urlopen(f"{self.base_url}/api/game") as response:
            classic_state = json.load(response)
        self.assertEqual(classic_state["rules"]["profile"], "classic")

    def test_120_tile_game_lobby_exposes_both_game_modes(self):
        with urlopen(f"{self.base_url}/games-120.html") as response:
            page = response.read().decode("utf-8")
        self.assertIn("120 张厦门麻将 · 游戏大厅", page)
        self.assertIn("1 人对 3 个机器人", page)
        self.assertIn("1v1v1v1 四人验牌", page)
        self.assertIn('href="/play-120.html"', page)
        self.assertIn('href="/play-120-four.html"', page)
        with urlopen(f"{self.base_url}/games-120.css") as response:
            stylesheet = response.read().decode("utf-8")
        self.assertIn(".game-mode-grid", stylesheet)

    def test_four_player_frontend_rule_lab_is_served(self):
        with urlopen(f"{self.base_url}/play-120-four.html") as response:
            page = response.read().decode("utf-8")
        self.assertIn("1v1v1v1", page)
        self.assertIn("四人验牌桌", page)
        self.assertIn("四家明牌", page)
        self.assertIn("发牌、摸牌、出牌、吃碰杠已可操作", page)
        self.assertIn("胡 ＞ 杠／碰 ＞ 吃", page)
        self.assertIn("平胡 2", page)
        self.assertIn('src="/four-player-lab.js"', page)
        with urlopen(f"{self.base_url}/four-player-lab.js") as response:
            script = response.read().decode("utf-8")
        self.assertIn("function build120Wall()", script)
        self.assertIn("function collectResponseActions", script)
        self.assertIn("function applyClaim", script)
        self.assertIn("function markManualWin", script)
        with urlopen(f"{self.base_url}/play-120-four.css") as response:
            stylesheet = response.read().decode("utf-8")
        self.assertIn(".four-mahjong-table", stylesheet)

    def test_rules_guide_page_is_served(self):
        with urlopen(f"{self.base_url}/guide.html") as response:
            page = response.read().decode("utf-8")
        self.assertIn("120 张新厦麻与 144 张老厦麻", page)
        self.assertIn("120 张牌是怎么组成的？", page)
        self.assertIn("新厦麻 · 120 张无大字", page)
        self.assertIn("游金、明暗游、双游与三游", page)
        self.assertIn("17 张牌", page)
        self.assertIn("平胡、自摸与抢金怎样区分？", page)
        self.assertIn("固定分值与水钱怎样算？", page)
        self.assertIn("当前 Demo 不是这张分值表", page)
        with urlopen(f"{self.base_url}/guide.js") as response:
            script = response.read().decode("utf-8")
        self.assertIn("updateGuideNavigation", script)

    def test_120_tile_rules_page_is_served(self):
        with urlopen(f"{self.base_url}/rules-120.html") as response:
            page = response.read().decode("utf-8")
        self.assertIn("120 张厦门麻将玩法规则", page)
        self.assertIn("从零学会", page)
        self.assertIn("108 张数牌＋4 张白板＋8 张花", page)
        self.assertIn("五分钟记住七条主规则", page)
        self.assertIn("真金万能，白板固定代牌", page)
        self.assertIn("游金、明暗游、双游与三游", page)
        self.assertIn("17 张怎样组成五组加一对", page)
        self.assertIn("完整胡牌示例", page)
        self.assertIn("平胡、自摸、抢金与三金倒", page)
        self.assertIn("固定分值与水钱怎样算", page)
        self.assertIn("三金倒</span><i>优先于</i><span>抢金", page)
        self.assertIn("抢杠是房规，不是本页默认", page)
        self.assertIn("严格房规：尾八熟张、尾四禁游", page)
        self.assertIn("游金圈里的花与杠", page)
        self.assertIn("游金花是否另计水", page)
        self.assertIn("最终核验结论", page)
        self.assertIn("四家起手都无花", page)
        self.assertIn("过胡与多家同时胡", page)
        self.assertIn("开局前必须确认的十项规则", page)
        self.assertIn("程序建议默认", page)
        with urlopen(f"{self.base_url}/rules-120.css") as response:
            stylesheet = response.read().decode("utf-8")
        self.assertIn(".quick-rule-grid", stylesheet)
        self.assertIn(".tour-process", stylesheet)

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

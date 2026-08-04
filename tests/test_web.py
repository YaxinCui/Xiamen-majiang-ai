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

    def test_static_page_is_served(self):
        with urlopen(f"{self.base_url}/") as response:
            page = response.read().decode("utf-8")
        self.assertIn("厦门麻将", page)

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


if __name__ == "__main__":
    unittest.main()

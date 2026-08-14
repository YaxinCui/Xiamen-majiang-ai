"""Dependency-free local HTTP server for the single-hand browser game."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from mimetypes import guess_type
from pathlib import Path
import threading
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .game import GameError, XiamenMahjongGame
from .rules import XiamenRules

ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = ROOT / "web_game_static"


class GameStore:
    def __init__(self, default_profile: str = "classic") -> None:
        self.lock = threading.Lock()
        self.rules_profile = default_profile
        self.game = XiamenMahjongGame(rules=XiamenRules.from_profile(self.rules_profile))

    def _public_state(self, *, reveal_ai_hands: bool = False) -> dict[str, Any]:
        state = self.game.public_state(reveal_ai_hands=reveal_ai_hands)
        state["rule_profiles"] = XiamenRules.available_profiles()
        return state

    def state(self, *, reveal_ai_hands: bool = False) -> dict[str, Any]:
        with self.lock:
            return self._public_state(reveal_ai_hands=reveal_ai_hands)

    def new_game(
        self,
        seed: int | None = None,
        rules_profile: str | None = None,
        *,
        reset_match: bool = False,
    ) -> dict[str, Any]:
        with self.lock:
            profile = rules_profile or self.rules_profile
            try:
                rules = XiamenRules.from_profile(profile)
            except ValueError as error:
                raise GameError(str(error)) from error
            dealer = None
            dealer_streak = 0
            scores = None
            hand_number = 1
            same_match = (
                not reset_match
                and profile == self.rules_profile
                and rules.enable_dealer_continuation
            )
            if same_match:
                previous = self.game
                scores = [player.score for player in previous.players]
                dealer = previous.dealer
                dealer_streak = previous.dealer_streak
                hand_number = previous.hand_number + 1
                if previous.phase == "over":
                    if previous.winner == previous.dealer or previous.win_type == "draw":
                        dealer_streak += 1
                    else:
                        dealer = (previous.dealer + 1) % rules.player_count
                        dealer_streak = 0
            self.rules_profile = profile
            self.game = XiamenMahjongGame(
                seed=seed,
                rules=rules,
                dealer=dealer,
                dealer_streak=dealer_streak,
                scores=scores,
                hand_number=hand_number,
            )
            return self._public_state()

    def action(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.game.apply_human_action(payload)
            return self._public_state()


def make_handler(store: GameStore, new120_store: GameStore | None = None):
    new120_store = new120_store or GameStore("new120")

    class GameHandler(BaseHTTPRequestHandler):
        server_version = "XiamenMahjong/0.1"

        def do_GET(self) -> None:  # noqa: N802
            request_url = urlsplit(self.path)
            if request_url.path == "/api/game":
                # This server is intentionally local-only.  AI hands are still
                # hidden by default and are revealed solely after the explicit
                # in-page debug toggle requests this view.
                reveal_ai_hands = parse_qs(request_url.query).get("debug") == ["1"]
                self._send_json(HTTPStatus.OK, store.state(reveal_ai_hands=reveal_ai_hands))
                return
            if request_url.path == "/api/game-120":
                reveal_ai_hands = parse_qs(request_url.query).get("debug") == ["1"]
                self._send_json(
                    HTTPStatus.OK,
                    new120_store.state(reveal_ai_hands=reveal_ai_hands),
                )
                return
            self._serve_static()

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_json()
                if self.path == "/api/game/new":
                    seed = payload.get("seed")
                    if seed is not None and not isinstance(seed, int):
                        raise GameError("seed 必须是整数")
                    rules_profile = payload.get("rules_profile")
                    if rules_profile is not None and not isinstance(rules_profile, str):
                        raise GameError("rules_profile 必须是字符串")
                    reset_match = payload.get("reset_match", False)
                    if not isinstance(reset_match, bool):
                        raise GameError("reset_match 必须是布尔值")
                    self._send_json(
                        HTTPStatus.OK,
                        store.new_game(seed, rules_profile, reset_match=reset_match),
                    )
                    return
                if self.path == "/api/game-120/new":
                    seed = payload.get("seed")
                    if seed is not None and not isinstance(seed, int):
                        raise GameError("seed 必须是整数")
                    reset_match = payload.get("reset_match", False)
                    if not isinstance(reset_match, bool):
                        raise GameError("reset_match 必须是布尔值")
                    self._send_json(
                        HTTPStatus.OK,
                        new120_store.new_game(seed, "new120", reset_match=reset_match),
                    )
                    return
                if self.path == "/api/game/action":
                    self._send_json(HTTPStatus.OK, store.action(payload))
                    return
                if self.path == "/api/game-120/action":
                    self._send_json(HTTPStatus.OK, new120_store.action(payload))
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "接口不存在"})
            except GameError as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except json.JSONDecodeError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "请求必须是 JSON"})

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 16_384:
                raise GameError("请求过大")
            raw = self.rfile.read(length) if length else b"{}"
            parsed = json.loads(raw.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise GameError("请求 JSON 必须是对象")
            return parsed

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_static(self) -> None:
            requested = self.path.split("?", 1)[0]
            relative = "index.html" if requested in {"/", ""} else requested.lstrip("/")
            target = (STATIC_ROOT / relative).resolve()
            if STATIC_ROOT not in target.parents and target != STATIC_ROOT:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content = target.read_bytes()
            content_type = guess_type(target.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
                content_type += "; charset=utf-8"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    return GameHandler


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    store = GameStore()
    server = ThreadingHTTPServer((host, port), make_handler(store))
    print(f"厦门麻将已启动：http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

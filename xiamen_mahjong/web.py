"""Dependency-free local HTTP server for the single-hand browser game."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from mimetypes import guess_type
from pathlib import Path
import threading
from typing import Any

from .game import GameError, XiamenMahjongGame

ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = ROOT / "web_game_static"


class GameStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.game = XiamenMahjongGame()

    def state(self) -> dict[str, Any]:
        with self.lock:
            return self.game.public_state()

    def new_game(self, seed: int | None = None) -> dict[str, Any]:
        with self.lock:
            self.game = XiamenMahjongGame(seed=seed)
            return self.game.public_state()

    def action(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.game.apply_human_action(payload)
            return self.game.public_state()


def make_handler(store: GameStore):
    class GameHandler(BaseHTTPRequestHandler):
        server_version = "XiamenMahjong/0.1"

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/game":
                self._send_json(HTTPStatus.OK, store.state())
                return
            self._serve_static()

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_json()
                if self.path == "/api/game/new":
                    seed = payload.get("seed")
                    if seed is not None and not isinstance(seed, int):
                        raise GameError("seed 必须是整数")
                    self._send_json(HTTPStatus.OK, store.new_game(seed))
                    return
                if self.path == "/api/game/action":
                    self._send_json(HTTPStatus.OK, store.action(payload))
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

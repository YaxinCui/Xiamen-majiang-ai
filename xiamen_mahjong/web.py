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
from uuid import uuid4

from .agents import GameAction
from .game import GameError, XiamenMahjongGame
from .rules import XiamenRules
from .training import (
    TeacherDecision,
    _perspective_state,
    _trajectory_from_game,
)

ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = ROOT / "web_game_static"


class GameStore:
    """Own one browser table and an optional local-only human data recorder.

    Recording is deliberately disabled unless the command-line caller supplies
    a path.  The writer only appends a completed hand after the human has made
    at least one decision, and the safe trajectory payload excludes replay
    seeds, wall order, and opponents' concealed hands.
    """

    def __init__(
        self,
        *,
        human_log: str | Path | None = None,
        human_recording_purpose: str | None = None,
        ai_agent: Any | None = None,
        ai_profile: str = "heuristic_teacher",
        ai_identity: str | None = None,
    ) -> None:
        valid_purposes = {"training", "evaluation"}
        if human_log is None and human_recording_purpose is not None:
            raise ValueError("未启用 --human-log 时不能指定人类记录用途")
        if human_log is not None and human_recording_purpose not in valid_purposes:
            raise ValueError(
                "启用人类记录时必须明确指定用途：training 或 evaluation"
            )
        self.lock = threading.Lock()
        self.rules_profile = "classic"
        self._ai_agent = ai_agent
        self._ai_profile = ai_profile
        self._ai_identity = ai_identity or ai_profile
        self._human_log = Path(human_log) if human_log is not None else None
        self._human_recording_purpose = human_recording_purpose
        # A random, local-only identifier marks one server-run recording
        # session without asking for a player name, account, device ID, or
        # network identifier.  Evaluation auditing treats sessions—not hands—
        # as its independent statistical units.
        self._human_recording_session_id = (
            uuid4().hex if self._human_log is not None else None
        )
        self._human_decisions: list[TeacherDecision] = []
        self._human_hand_written = False
        self._human_log_error = False
        self.game = XiamenMahjongGame(
            rules=XiamenRules.from_profile(self.rules_profile),
            agents=self._ai_agents(),
        )
        self._human_score_start = tuple(player.score for player in self.game.players)

    def _ai_agents(self) -> dict[int, Any] | None:
        if self._ai_agent is None:
            return None
        return {
            seat: self._ai_agent
            for seat in range(XiamenRules().player_count)
            if seat != 0
        }

    def _public_state(self, *, reveal_ai_hands: bool = False) -> dict[str, Any]:
        state = self.game.public_state(reveal_ai_hands=reveal_ai_hands)
        state["rule_profiles"] = XiamenRules.available_profiles()
        state["ai_profile"] = self._ai_profile
        state["local_human_recording"] = {
            "enabled": self._human_log is not None,
            "purpose": self._human_recording_purpose,
            "pending_decisions": len(self._human_decisions),
            "completed_hand_written": self._human_hand_written,
            "write_failed": self._human_log_error,
            "scope": (
                "completed_hand_actor_visible_only"
                if self._human_log is not None
                else "disabled"
            ),
        }
        return state

    def _capture_human_decision(self, payload: dict[str, Any]) -> TeacherDecision:
        """Create an actor-visible snapshot before the engine mutates state."""

        legal = tuple(
            GameAction(
                str(item["kind"]),
                item.get("tile"),
                tuple(item.get("tiles", [])),
            )
            for item in self.game.human_actions()
        )
        action = GameAction(
            str(payload.get("kind", "")),
            payload.get("tile"),
            tuple(payload.get("tiles", [])),
        )
        if action not in legal:
            raise GameError("该动作不是当前可执行的操作")
        action_index = legal.index(action)
        return TeacherDecision(
            profile=self.game.rules.profile,
            seed=None,
            seat=self.game.human_seat,
            state=_perspective_state(self.game, self.game.human_seat),
            legal_actions=legal,
            # For an opt-in human record, the chosen and executed action are
            # intentionally identical.  There is no synthetic Teacher label.
            chosen_index=action_index,
            executed_index=action_index,
            executed_probability=None,
        )

    def _write_completed_human_hand(self) -> None:
        """Append one safe, complete opt-in human trajectory exactly once."""

        if (
            self._human_log is None
            or self._human_hand_written
            or self.game.phase != "over"
            or not self._human_decisions
        ):
            return
        trajectory = _trajectory_from_game(
            self.game,
            self._human_decisions,
            agent_profiles=(
                "local_human_opt_in",
                *(self._ai_profile for _ in range(self.game.rules.player_count - 1)),
            ),
            source_metadata={
                "collector": "local_human_opt_in",
                "recording_scope": "actor_visible_state_and_public_outcome_only",
                "behavior_label": "executed_human_action",
                "training_default": "excluded_until_separate_quality_review",
                # This purpose is chosen when recording starts, not inferred
                # later from a filename.  It prevents evaluation hands from
                # entering the behavioral-imitation path and lets the human
                # benchmark require an explicit independent source.
                "recording_purpose": self._human_recording_purpose,
                "recording_session_id": self._human_recording_session_id,
                "opponent_policy": self._ai_identity,
            },
        )
        payload = trajectory.payload()
        # A browser match may carry scores across hands because of dealer
        # continuation. Training outcomes must instead retain the reward of
        # this hand alone, never an earlier hand's accumulated result.
        payload["outcome"]["scores"] = [
            player.score - self._human_score_start[index]
            for index, player in enumerate(self.game.players)
        ]
        payload["outcome"]["score_semantics"] = "single_hand_delta"
        try:
            self._human_log.parent.mkdir(parents=True, exist_ok=True)
            with self._human_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        except OSError:
            # A completed game remains playable/reviewable even if its optional
            # local export path becomes unavailable.  Expose only a boolean to
            # the browser, never a local filesystem path or OS error detail.
            self._human_log_error = True
            return
        self._human_hand_written = True

    def _reset_human_recorder(self) -> None:
        self._human_decisions = []
        self._human_hand_written = False
        self._human_log_error = False
        self._human_score_start = tuple(player.score for player in self.game.players)

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
                agents=self._ai_agents(),
            )
            self._reset_human_recorder()
            return self._public_state()

    def action(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            decision = (
                self._capture_human_decision(payload)
                if self._human_log is not None
                else None
            )
            self.game.apply_human_action(payload)
            if decision is not None:
                self._human_decisions.append(decision)
            self._write_completed_human_hand()
            return self._public_state()


def make_handler(store: GameStore):
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


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    human_log: str | Path | None = None,
    human_recording_purpose: str | None = None,
    ai_agent: Any | None = None,
    ai_profile: str = "heuristic_teacher",
    ai_identity: str | None = None,
) -> None:
    store = GameStore(
        human_log=human_log,
        human_recording_purpose=human_recording_purpose,
        ai_agent=ai_agent,
        ai_profile=ai_profile,
        ai_identity=ai_identity,
    )
    server = ThreadingHTTPServer((host, port), make_handler(store))
    print(f"厦门麻将已启动：http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

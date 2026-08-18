"""Local-only browser service for independent expert discard corrections."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from mimetypes import guess_type
from pathlib import Path
import threading
from typing import Any
from urllib.parse import urlsplit

from .human_review import (
    append_review_label,
    make_review_label,
    public_review_item,
    read_review_priority,
    read_review_labels,
    read_review_queue,
    validate_review_label_against_item,
    validate_review_priority_against_queue,
)


ROOT = Path(__file__).resolve().parents[1]
REVIEW_STATIC_ROOT = ROOT / "review_game_static"


class ReviewError(ValueError):
    """Safe, user-facing review workflow error."""


class ReviewStore:
    """Own an immutable queue and append-only local human labels."""

    def __init__(
        self,
        queue_path: str | Path,
        output_path: str | Path,
        priority_path: str | Path | None = None,
    ) -> None:
        self.lock = threading.Lock()
        self._items = read_review_queue(queue_path)
        if not self._items:
            raise ValueError("review queue 不能为空")
        self._items_by_id = {str(item["item_id"]): item for item in self._items}
        self._priority_by_id: dict[str, dict[str, Any]] = {}
        if priority_path is not None:
            priority = read_review_priority(priority_path)
            issues = validate_review_priority_against_queue(priority, self._items)
            if issues:
                raise ValueError("review priority 与 queue 不一致：" + ",".join(issues))
            self._priority_by_id = {
                str(record["item_id"]): record for record in priority
            }
            self._items.sort(
                key=lambda item: int(
                    self._priority_by_id[str(item["item_id"])]["priority_rank"]
                )
            )
        self._output_path = Path(output_path)
        self._completed: set[str] = set()
        self._session_skipped: set[str] = set()
        if self._output_path.exists():
            for record in read_review_labels(self._output_path):
                item_id = str(record.get("item_id", ""))
                item = self._items_by_id.get(item_id)
                if item is None:
                    raise ValueError("已有 review label 不属于当前 queue")
                issues = validate_review_label_against_item(record, item)
                if issues:
                    raise ValueError("已有 review label 与 queue 不一致：" + ",".join(issues))
                if item_id in self._completed:
                    raise ValueError("已有 review label 包含重复 item_id")
                self._completed.add(item_id)

    def _next_item(self) -> dict[str, Any] | None:
        return next(
            (
                item
                for item in self._items
                if item["item_id"] not in self._completed
                and item["item_id"] not in self._session_skipped
            ),
            None,
        )

    def _state_unlocked(self) -> dict[str, Any]:
        item = self._next_item()
        if item is None:
            unfinished = len(self._items) - len(self._completed)
            return {
                "status": "complete" if unfinished == 0 else "session_exhausted",
                "progress": {
                    "completed": len(self._completed),
                    "total": len(self._items),
                    "skipped_this_session": len(self._session_skipped),
                    "unfinished": unfinished,
                },
                "message": (
                    "全部题目已完成"
                    if unfinished == 0
                    else "本次已浏览所有未完成题目；刷新页面可重新查看跳过项"
                ),
            }
        payload = public_review_item(
            item,
            completed=len(self._completed),
            total=len(self._items),
        )
        payload["progress"]["skipped_this_session"] = len(self._session_skipped)
        return payload

    def state(self) -> dict[str, Any]:
        with self.lock:
            return self._state_unlocked()

    def label(
        self, *, item_id: str, chosen_index: int, confidence: str
    ) -> dict[str, Any]:
        with self.lock:
            current = self._next_item()
            if current is None:
                raise ReviewError("当前没有可标注题目")
            if item_id != current["item_id"]:
                raise ReviewError("题目已变化，请刷新后重新选择")
            label = make_review_label(
                current,
                chosen_index=chosen_index,
                confidence=confidence,
            )
            try:
                append_review_label(self._output_path, label)
            except OSError as error:
                raise ReviewError("本地标注写入失败，未进入下一题") from error
            self._completed.add(item_id)
            reference_index = int(current["reference_teacher_index"])
            slow_expert = self._priority_by_id.get(item_id)
            slow_expert_index = (
                int(slow_expert["slow_expert_index"])
                if slow_expert is not None
                else None
            )
            return {
                "status": "saved",
                "feedback": {
                    "confidence": confidence,
                    "agrees_with_teacher": chosen_index == reference_index,
                    "human_index": chosen_index,
                    "reference_teacher_index": reference_index,
                    "human_action": current["legal_actions"][chosen_index],
                    "reference_teacher_action": current["legal_actions"][reference_index],
                    "slow_expert_version": (
                        slow_expert.get("slow_expert_version")
                        if slow_expert is not None
                        else None
                    ),
                    "slow_expert_label": (
                        "两摸 SlowExpert（已拒绝）"
                        if slow_expert is not None
                        else None
                    ),
                    "slow_expert_index": slow_expert_index,
                    "slow_expert_action": (
                        current["legal_actions"][slow_expert_index]
                        if slow_expert_index is not None
                        else None
                    ),
                    "agrees_with_slow_expert": (
                        chosen_index == slow_expert_index
                        if slow_expert_index is not None
                        else None
                    ),
                },
                "next": self._state_unlocked(),
            }

    def skip(self, *, item_id: str) -> dict[str, Any]:
        with self.lock:
            current = self._next_item()
            if current is None:
                raise ReviewError("当前没有可跳过题目")
            if item_id != current["item_id"]:
                raise ReviewError("题目已变化，请刷新后重试")
            self._session_skipped.add(item_id)
            return self._state_unlocked()


def make_review_handler(store: ReviewStore):
    class ReviewHandler(BaseHTTPRequestHandler):
        server_version = "XiamenMahjongReview/0.1"

        def do_GET(self) -> None:  # noqa: N802
            if urlsplit(self.path).path == "/api/review":
                self._send_json(HTTPStatus.OK, store.state())
                return
            self._serve_static()

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_json()
                if self.path == "/api/review/label":
                    item_id = payload.get("item_id")
                    chosen_index = payload.get("chosen_index")
                    confidence = payload.get("confidence")
                    if not isinstance(item_id, str):
                        raise ReviewError("item_id 必须是字符串")
                    if isinstance(chosen_index, bool) or not isinstance(chosen_index, int):
                        raise ReviewError("chosen_index 必须是整数")
                    if not isinstance(confidence, str):
                        raise ReviewError("confidence 必须是字符串")
                    self._send_json(
                        HTTPStatus.OK,
                        store.label(
                            item_id=item_id,
                            chosen_index=chosen_index,
                            confidence=confidence,
                        ),
                    )
                    return
                if self.path == "/api/review/skip":
                    item_id = payload.get("item_id")
                    if not isinstance(item_id, str):
                        raise ReviewError("item_id 必须是字符串")
                    self._send_json(HTTPStatus.OK, store.skip(item_id=item_id))
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "接口不存在"})
            except (ReviewError, ValueError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "请求必须是 JSON"})

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 16_384:
                raise ReviewError("请求过大")
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ReviewError("请求 JSON 必须是对象")
            return payload

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_static(self) -> None:
            requested = urlsplit(self.path).path
            relative = "index.html" if requested in {"", "/"} else requested.lstrip("/")
            target = (REVIEW_STATIC_ROOT / relative).resolve()
            if REVIEW_STATIC_ROOT not in target.parents and target != REVIEW_STATIC_ROOT:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content = target.read_bytes()
            content_type = guess_type(target.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type in {
                "application/javascript",
                "application/json",
            }:
                content_type += "; charset=utf-8"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    return ReviewHandler


def serve_review(
    host: str,
    port: int,
    *,
    queue_path: str | Path,
    output_path: str | Path,
    priority_path: str | Path | None = None,
) -> None:
    store = ReviewStore(queue_path, output_path, priority_path)
    server = ThreadingHTTPServer((host, port), make_review_handler(store))
    print(f"厦门麻将专家纠错审阅已启动：http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

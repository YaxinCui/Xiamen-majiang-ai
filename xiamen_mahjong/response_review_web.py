"""Local append-only web store for human pass/chi/pong review."""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

from .response_review import (
    RESPONSE_SLOW_EXPERT_VERSION,
    append_response_review_label,
    make_response_review_label,
    public_response_review_item,
    read_response_review_labels,
    read_response_review_queue,
    response_slow_expert_index,
    validate_response_review_label_against_item,
)
from .review_web import ReviewError


class ResponseReviewStore:
    def __init__(self, queue_path: str | Path, output_path: str | Path) -> None:
        self.lock = threading.Lock()
        self._items = read_response_review_queue(queue_path)
        if not self._items:
            raise ValueError("response review queue 不能为空")
        self._items_by_id = {str(item["item_id"]): item for item in self._items}
        self._output_path = Path(output_path)
        self._completed: set[str] = set()
        self._session_skipped: set[str] = set()
        if self._output_path.exists():
            for record in read_response_review_labels(self._output_path):
                item_id = str(record.get("item_id", ""))
                item = self._items_by_id.get(item_id)
                if item is None:
                    raise ValueError("已有 response review label 不属于当前 queue")
                issues = validate_response_review_label_against_item(record, item)
                if issues:
                    raise ValueError(
                        "已有 response review label 与 queue 不一致："
                        + ",".join(issues)
                    )
                if item_id in self._completed:
                    raise ValueError("已有 response review label 包含重复 item_id")
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
                    "全部响应题已完成"
                    if unfinished == 0
                    else "本次已浏览所有未完成响应题；刷新可重看跳过项"
                ),
            }
        payload = public_response_review_item(
            item, completed=len(self._completed), total=len(self._items)
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
                raise ReviewError("当前没有可标注响应题")
            if item_id != current["item_id"]:
                raise ReviewError("题目已变化，请刷新后重新选择")
            label = make_response_review_label(
                current, chosen_index=chosen_index, confidence=confidence
            )
            try:
                append_response_review_label(self._output_path, label)
            except OSError as error:
                raise ReviewError("本地响应标注写入失败，未进入下一题") from error
            self._completed.add(item_id)
            reference_index = int(current["reference_teacher_index"])
            slow_index = response_slow_expert_index(current)
            return {
                "status": "saved",
                "feedback": {
                    "confidence": confidence,
                    "agrees_with_teacher": chosen_index == reference_index,
                    "human_index": chosen_index,
                    "reference_teacher_index": reference_index,
                    "human_action": current["legal_actions"][chosen_index],
                    "reference_teacher_action": current["legal_actions"][reference_index],
                    "slow_expert_version": RESPONSE_SLOW_EXPERT_VERSION,
                    "slow_expert_label": "精确向听响应 SlowExpert（已拒绝）",
                    "slow_expert_index": slow_index,
                    "slow_expert_action": current["legal_actions"][slow_index],
                    "agrees_with_slow_expert": chosen_index == slow_index,
                },
                "next": self._state_unlocked(),
            }

    def skip(self, *, item_id: str) -> dict[str, Any]:
        with self.lock:
            current = self._next_item()
            if current is None:
                raise ReviewError("当前没有可跳过响应题")
            if item_id != current["item_id"]:
                raise ReviewError("题目已变化，请刷新后重试")
            self._session_skipped.add(item_id)
            return self._state_unlocked()

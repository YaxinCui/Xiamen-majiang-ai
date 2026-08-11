import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen

from tests.test_human_review import review_item
from xiamen_mahjong.human_review import (
    build_review_slow_expert_priority,
    write_review_priority,
    write_review_queue,
)
from tests.test_human_review import priority_review_item
from xiamen_mahjong.review_web import ReviewStore, make_review_handler


class ReviewWebTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.queue = root / "queue.jsonl"
        self.output = root / "labels.jsonl"
        write_review_queue(self.queue, [review_item()])
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_review_handler(ReviewStore(self.queue, self.output)),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def test_api_hides_teacher_until_independent_label(self):
        with urlopen(f"{self.base_url}/api/review") as response:
            state = json.load(response)
        self.assertNotIn("reference_teacher_index", json.dumps(state))
        request = Request(
            f"{self.base_url}/api/review/label",
            data=json.dumps(
                {
                    "item_id": state["item_id"],
                    "chosen_index": 1,
                    "confidence": "confirmed",
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            result = json.load(response)
        self.assertEqual(result["feedback"]["reference_teacher_index"], 0)
        self.assertEqual(result["next"]["status"], "complete")
        self.assertTrue(self.output.exists())

    def test_static_review_page_is_served(self):
        with urlopen(f"{self.base_url}/") as response:
            page = response.read().decode("utf-8")
        self.assertIn("专家纠错审阅", page)
        self.assertIn("提交前不会显示规则 Teacher", page)
        with urlopen(f"{self.base_url}/review.js") as response:
            script = response.read().decode("utf-8")
        self.assertIn("/api/review/label", script)
        self.assertIn("describeAction", script)
        self.assertIn("action-options", page)

    def test_priority_api_hides_slow_expert_until_label(self):
        priority_queue = Path(self.temporary.name) / "priority-queue.jsonl"
        priority_path = Path(self.temporary.name) / "priority.jsonl"
        priority_output = Path(self.temporary.name) / "priority-labels.jsonl"
        item = priority_review_item(item_id="c" * 32, group_id="d" * 32)
        records, _report = build_review_slow_expert_priority([item])
        write_review_queue(priority_queue, [item])
        write_review_priority(priority_path, records)
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_review_handler(
                ReviewStore(priority_queue, priority_output, priority_path)
            ),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(f"{base_url}/api/review") as response:
                state = json.load(response)
            encoded_state = json.dumps(state)
            self.assertNotIn("slow_expert", encoded_state)
            self.assertNotIn("priority", encoded_state)
            request = Request(
                f"{base_url}/api/review/label",
                data=json.dumps(
                    {
                        "item_id": state["item_id"],
                        "chosen_index": 0,
                        "confidence": "confirmed",
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request) as response:
                result = json.load(response)
            self.assertIsNotNone(result["feedback"]["slow_expert_action"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()

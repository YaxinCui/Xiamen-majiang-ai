import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen

from tests.test_pairwise_review import pairwise_queue_item
from xiamen_mahjong.pairwise_review import write_pairwise_queue
from xiamen_mahjong.pairwise_review_web import (
    PairwiseReviewStore,
    make_pairwise_review_handler,
)


class PairwiseReviewWebTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.queue = root / "queue.jsonl"
        self.output = root / "labels.jsonl"
        write_pairwise_queue(self.queue, [pairwise_queue_item()])
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_pairwise_review_handler(
                PairwiseReviewStore(self.queue, self.output)
            ),
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def test_api_hides_both_identities_until_pairwise_choice(self):
        with urlopen(f"{self.base_url}/api/review") as response:
            state = json.load(response)
        encoded = json.dumps(state, sort_keys=True)
        self.assertNotIn("reference_teacher_index", encoded)
        self.assertNotIn("candidate_index", encoded)
        self.assertEqual(len(state["legal_actions"]), 2)
        request = Request(
            f"{self.base_url}/api/review/label",
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
        self.assertIn("reference_teacher_index", result["feedback"])
        self.assertIsNotNone(result["feedback"]["slow_expert_action"])
        self.assertEqual(result["feedback"]["target_semantics"], "pairwise_only")
        self.assertTrue(self.output.exists())

    def test_static_page_explains_anonymous_pairwise_scope(self):
        with urlopen(f"{self.base_url}/") as response:
            page = response.read().decode("utf-8")
        self.assertIn("完全并列二选一", page)
        self.assertIn("提交前不会告诉你哪张是 Teacher", page)
        self.assertIn("不会把未展示动作当负例", page)
        with urlopen(f"{self.base_url}/review.js") as response:
            script = response.read().decode("utf-8")
        self.assertIn("/api/review/label", script)


if __name__ == "__main__":
    unittest.main()

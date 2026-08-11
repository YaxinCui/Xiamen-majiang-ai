import json
from pathlib import Path
import tempfile
import unittest

from xiamen_mahjong.agents import GameAction
from xiamen_mahjong.response_review import (
    RESPONSE_REVIEW_GROUP_VERSION,
    RESPONSE_REVIEW_QUEUE_VERSION,
    audit_response_review_labels,
    audit_response_review_labels_against_queue,
    build_response_review_queue,
    make_response_review_label,
    public_response_review_item,
    read_confirmed_response_review_decisions,
    response_slow_expert_index,
    split_confirmed_response_review_labels,
    validate_response_review_label_against_item,
    validate_response_review_queue_item,
    write_response_review_queue,
)
from xiamen_mahjong.response_review_web import ResponseReviewStore
from xiamen_mahjong.training import (
    TeacherDecision,
    TrainingTrajectory,
    write_trajectory_jsonl,
)


def response_item(*, item_id="1" * 32, group_id="2" * 32):
    hand = [0, 1, 1, 2, 2, 3, 9, 10, 10, 12, 19, 21, 22, 24, 25, 33]
    return {
        "version": RESPONSE_REVIEW_QUEUE_VERSION,
        "item_id": item_id,
        "review_group_id": group_id,
        "review_group_version": RESPONSE_REVIEW_GROUP_VERSION,
        "profile": "classic",
        "rules_version": "xiamen-classic-full-v2",
        "state": {
            "phase": "response",
            "tour": None,
            "gold_locked": False,
            "gold_tile": 13,
            "gold_indicator": 12,
            "hand": hand,
            "drawn_tile": None,
            "wall_remaining": 60,
            "turn_count": 8,
            "dealer_relative": 2,
            "current_player_relative": 3,
            "discarder_relative": 3,
            "last_discard": 2,
            "is_dealer": False,
            "flowers": 0,
            "river_counts": [0] * 34,
            "meld_counts": [0] * 34,
            "opening_wait_relative_seats": [],
            "public_players": [
                {
                    "relative_seat": seat,
                    "hand_count": 16,
                    "flowers": 0,
                    "melds": [],
                    "discards": [2] if seat == 3 else [],
                }
                for seat in range(4)
            ],
            "recent_public_actions": [
                {"kind": "discard", "relative_seat": 3, "tile": 2, "turn": 8}
            ],
        },
        "legal_actions": [
            {"kind": "pass"},
            {"kind": "pong", "tile": 2, "tiles": [2, 2]},
        ],
        "reference_teacher_index": 1,
        "source_metadata": {
            "collector": "teacher_self_play_actor_visible_response_review",
            "source_scope": "ordinary_pass_chi_pong_response",
            "opponent_hand_reveal": "unavailable_by_construction",
            "training_default": "excluded_until_human_review_audit",
        },
    }


class ResponseReviewTests(unittest.TestCase):
    def test_public_response_item_hides_all_automatic_references(self):
        item = response_item()
        self.assertEqual(validate_response_review_queue_item(item), [])
        public = public_response_review_item(item, completed=0, total=1)
        encoded = json.dumps(public, sort_keys=True)
        self.assertNotIn("reference_teacher", encoded)
        self.assertNotIn("slow_expert", encoded)
        self.assertNotIn("review_group", encoded)
        self.assertEqual(public["review_scope"], "pass_chi_pong_response")

    def test_rejected_slow_expert_is_only_revealed_after_binding(self):
        item = response_item()
        self.assertEqual(response_slow_expert_index(item), 0)
        label = make_response_review_label(
            item, chosen_index=0, confidence="confirmed"
        )
        self.assertEqual(
            validate_response_review_label_against_item(label, item), []
        )
        tampered = {**item, "reference_teacher_index": 0}
        self.assertIn(
            "queue_item_digest_mismatch",
            validate_response_review_label_against_item(label, tampered),
        )

    def test_build_queue_reproduces_teacher_and_prioritizes_disagreement(self):
        item = response_item()
        decision = TeacherDecision(
            profile="classic",
            seed=None,
            seat=0,
            state=item["state"],
            legal_actions=(
                GameAction("pass"),
                GameAction("pong", 2, (2, 2)),
            ),
            chosen_index=1,
            executed_index=1,
        )
        trajectory = TrainingTrajectory(
            profile="classic",
            rules_version="xiamen-classic-full-v2",
            rules={},
            seed=None,
            hand_number=1,
            agent_profiles=("heuristic_teacher",) * 4,
            source_metadata={"collector": "teacher_self_play"},
            decisions=(decision,),
            outcome={"scores": [0, 0, 0, 0]},
            public_actions=(),
            trajectory_id="3" * 32,
            split_group_id="4" * 32,
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "teacher.jsonl"
            write_trajectory_jsonl([trajectory], source)
            queue, report = build_response_review_queue(
                [source], maximum_items=1, maximum_items_per_group=1
            )
        self.assertEqual(len(queue), 1)
        self.assertEqual(report["teacher_reproduction_mismatches"], 0)
        self.assertEqual(report["queue_acquisition_disagreements"], 1)

    def test_store_persists_response_label_without_automatic_answer_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            queue_path = Path(directory) / "queue.jsonl"
            output_path = Path(directory) / "labels.jsonl"
            item = response_item()
            write_response_review_queue(queue_path, [item])
            store = ResponseReviewStore(queue_path, output_path)
            state = store.state()
            encoded = json.dumps(state, sort_keys=True)
            self.assertNotIn("reference_teacher", encoded)
            self.assertNotIn("slow_expert", encoded)
            result = store.label(
                item_id=state["item_id"],
                chosen_index=0,
                confidence="confirmed",
            )
            self.assertFalse(result["feedback"]["agrees_with_teacher"])
            self.assertEqual(result["feedback"]["slow_expert_index"], 0)
            saved = output_path.read_text(encoding="utf-8")
            self.assertNotIn("slow_expert", saved)
            report = audit_response_review_labels_against_queue(
                [json.loads(saved)],
                [item],
                minimum_confirmed_labels=1,
                minimum_confirmed_disagreements=1,
                minimum_groups=1,
            )
            self.assertTrue(report["ready_for_manual_training_review"])
            self.assertEqual(report["confirmed_action_counts"], {"pass": 1})

    def test_response_split_never_crosses_physical_group(self):
        records = []
        queue = []
        for index in range(40):
            item = response_item(
                item_id=f"{index + 100:032x}",
                group_id=f"{index // 2 + 500:032x}",
            )
            queue.append(item)
            records.append(
                make_response_review_label(
                    item, chosen_index=index % 2, confidence="confirmed"
                )
            )
        partitions = split_confirmed_response_review_labels(records, queue)
        group_splits = {}
        for split, rows in partitions.items():
            for row in rows:
                previous = group_splits.setdefault(row["review_group_id"], split)
                self.assertEqual(previous, split)

    def test_confirmed_split_loader_rejects_uncertainty_and_duplicates(self):
        item = response_item()
        confirmed = make_response_review_label(
            item, chosen_index=0, confidence="confirmed"
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.response-review.jsonl"
            source.write_text(json.dumps(confirmed) + "\n", encoding="utf-8")
            rows = read_confirmed_response_review_decisions(source)
            self.assertEqual(rows[0][0], item["review_group_id"])
            self.assertEqual(rows[0][1].chosen_action.kind, "pass")
            source.write_text(
                json.dumps(confirmed) + "\n" + json.dumps(confirmed) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "重复 item_id"):
                read_confirmed_response_review_decisions(source)
            uncertain = make_response_review_label(
                item, chosen_index=0, confidence="uncertain"
            )
            source.write_text(json.dumps(uncertain) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "只能包含 confirmed"):
                read_confirmed_response_review_decisions(source)

    def test_standalone_response_audit_rejects_nonbehavior_targets(self):
        item = response_item()
        label = make_response_review_label(
            item, chosen_index=0, confidence="confirmed"
        )
        label["decision"]["action_values"] = [0.0, 1.0]
        report = audit_response_review_labels(
            [label],
            minimum_confirmed_labels=1,
            minimum_confirmed_disagreements=1,
            minimum_groups=1,
        )
        self.assertFalse(report["ready_for_manual_training_review"])
        self.assertIn("response_review_contains_nonbehavior_target", report["issues"])


if __name__ == "__main__":
    unittest.main()

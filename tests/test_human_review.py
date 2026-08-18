import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from xiamen_mahjong.agents import GameAction
from xiamen_mahjong.human_review import (
    REVIEW_GROUP_VERSION,
    REVIEW_QUEUE_VERSION,
    _teacher_discard_scores,
    audit_review_labels,
    audit_review_labels_against_queue,
    build_review_slow_expert_priority,
    build_review_queue,
    make_review_label,
    public_review_item,
    split_confirmed_review_labels,
    validate_review_label_against_item,
    validate_review_priority_against_queue,
    validate_review_queue_item,
    write_review_priority,
    write_review_queue,
)
from xiamen_mahjong.review_web import ReviewStore
from xiamen_mahjong.training import (
    TeacherDecision,
    TrainingTrajectory,
    write_trajectory_jsonl,
)
from scripts.train_policy_value import load_review_examples


def review_item(*, item_id="a" * 32, group_id="b" * 32):
    return {
        "version": REVIEW_QUEUE_VERSION,
        "item_id": item_id,
        "review_group_id": group_id,
        "review_group_version": REVIEW_GROUP_VERSION,
        "profile": "classic",
        "rules_version": "xiamen-classic-full-v2",
        "state": {
            "phase": "discard",
            "tour": None,
            "gold_locked": False,
            "gold_tile": 31,
            "gold_indicator": 30,
            "hand": [0, 0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29, 31, 32, 33, 33],
            "drawn_tile": 33,
            "wall_remaining": 70,
            "turn_count": 3,
            "dealer_relative": 0,
            "current_player_relative": 0,
            "discarder_relative": None,
            "is_dealer": True,
            "flowers": 0,
            "river_counts": [0] * 34,
            "meld_counts": [0] * 34,
            "public_players": [
                {"relative_seat": seat, "hand_count": 17 if seat == 0 else 16, "flowers": 0, "melds": [], "discards": []}
                for seat in range(4)
            ],
            "recent_public_actions": [],
        },
        "legal_actions": [
            {"kind": "discard", "tile": 0},
            {"kind": "discard", "tile": 33},
        ],
        "reference_teacher_index": 0,
        "teacher_score_margin": 1.0,
        "source_metadata": {
            "collector": "teacher_self_play_actor_visible_review_queue",
            "opponent_hand_reveal": "unavailable_by_construction",
        },
    }


def priority_review_item(*, item_id="a" * 32, group_id="b" * 32):
    item = review_item(item_id=item_id, group_id=group_id)
    item["legal_actions"] = [
        {"kind": "discard", "tile": tile}
        for tile in sorted(set(item["state"]["hand"]))
    ]
    item["reference_teacher_index"] = next(
        index
        for index, action in enumerate(item["legal_actions"])
        if action["tile"] == 27
    )
    return item


class HumanReviewTests(unittest.TestCase):
    def test_public_item_hides_reference_group_and_margin(self):
        item = review_item()
        self.assertEqual(validate_review_queue_item(item), [])
        public = public_review_item(item, completed=0, total=1)
        encoded = json.dumps(public, sort_keys=True)
        self.assertNotIn("reference_teacher", encoded)
        self.assertNotIn("teacher_score_margin", encoded)
        self.assertNotIn(item["review_group_id"], encoded)
        self.assertEqual(public["state"]["hand"], item["state"]["hand"])

    def test_confirmed_label_audit_counts_only_confirmed_groups(self):
        confirmed = make_review_label(review_item(), chosen_index=1, confidence="confirmed")
        uncertain = make_review_label(
            review_item(item_id="c" * 32, group_id="d" * 32),
            chosen_index=0,
            confidence="uncertain",
        )
        report = audit_review_labels(
            [confirmed, uncertain],
            minimum_confirmed_labels=1,
            minimum_confirmed_disagreements=1,
            minimum_groups=1,
        )
        self.assertTrue(report["ready_for_manual_training_review"])
        self.assertEqual(report["confirmed_labels"], 1)
        self.assertEqual(report["uncertain_labels"], 1)
        self.assertEqual(report["confirmed_review_groups"], 1)
        self.assertEqual(report["confirmed_teacher_disagreements"], 1)

    def test_label_is_bound_to_immutable_queue_item(self):
        item = review_item()
        label = make_review_label(item, chosen_index=1, confidence="confirmed")
        self.assertEqual(validate_review_label_against_item(label, item), [])
        tampered = {**item, "reference_teacher_index": 1}
        self.assertIn(
            "queue_item_digest_mismatch",
            validate_review_label_against_item(label, tampered),
        )
        report = audit_review_labels_against_queue(
            [label],
            [tampered],
            minimum_confirmed_labels=1,
            minimum_confirmed_disagreements=1,
            minimum_groups=1,
        )
        self.assertFalse(report["ready_for_manual_training_review"])
        self.assertIn("queue_binding_mismatch", report["gate_reasons"])

    def test_split_never_crosses_review_group(self):
        records = []
        for index in range(40):
            group_id = f"{index // 2:032x}"
            records.append(
                make_review_label(
                    review_item(item_id=f"{index + 100:032x}", group_id=group_id),
                    chosen_index=index % 2,
                    confidence="confirmed",
                )
            )
        partitions = split_confirmed_review_labels(records)
        group_splits = {}
        for split, rows in partitions.items():
            for row in rows:
                previous = group_splits.setdefault(row["review_group_id"], split)
                self.assertEqual(previous, split)

    def test_build_queue_reproduces_teacher_scores(self):
        state = review_item()["state"]
        actions = tuple(GameAction("discard", tile) for tile in sorted(set(state["hand"])))
        provisional = TeacherDecision(
            profile="classic",
            seed=None,
            seat=0,
            state=state,
            legal_actions=actions,
            chosen_index=0,
            executed_index=0,
        )
        scores = _teacher_discard_scores(provisional)
        chosen = min(scores, key=lambda index: (-scores[index], actions[index].tile))
        decision = TeacherDecision(
            profile="classic",
            seed=None,
            seat=0,
            state=state,
            legal_actions=actions,
            chosen_index=chosen,
            executed_index=chosen,
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
            trajectory_id="e" * 32,
            split_group_id="f" * 32,
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "teacher.jsonl"
            write_trajectory_jsonl([trajectory], source)
            queue, report = build_review_queue(
                [source], maximum_items=1, maximum_teacher_score_margin=10_000
            )
        self.assertEqual(len(queue), 1)
        self.assertEqual(report["score_reproduction_mismatches"], 0)

    def test_store_persists_resumes_and_skips_without_label(self):
        with tempfile.TemporaryDirectory() as directory:
            queue_path = Path(directory) / "queue.jsonl"
            output_path = Path(directory) / "labels.jsonl"
            items = [
                review_item(),
                review_item(item_id="c" * 32, group_id="d" * 32),
            ]
            write_review_queue(queue_path, items)
            store = ReviewStore(queue_path, output_path)
            first = store.state()
            self.assertNotIn("reference_teacher_index", first)
            saved = store.label(
                item_id=first["item_id"], chosen_index=1, confidence="confirmed"
            )
            self.assertFalse(saved["feedback"]["agrees_with_teacher"])
            self.assertEqual(saved["next"]["progress"]["completed"], 1)
            resumed = ReviewStore(queue_path, output_path)
            self.assertEqual(resumed.state()["item_id"], "c" * 32)
            exhausted = resumed.skip(item_id="c" * 32)
            self.assertEqual(exhausted["status"], "session_exhausted")
            self.assertEqual(len(output_path.read_text(encoding="utf-8").splitlines()), 1)

    def test_slow_expert_priority_is_queue_bound_and_hidden_before_label(self):
        item = priority_review_item()
        records, report = build_review_slow_expert_priority([item])
        self.assertEqual(validate_review_priority_against_queue(records, [item]), [])
        self.assertEqual(report["queue_items"], 1)
        self.assertFalse(report["queue_mutated"])
        self.assertEqual(records[0]["priority_rank"], 0)
        public = public_review_item(item, completed=0, total=1)
        encoded = json.dumps(public, sort_keys=True)
        self.assertNotIn("slow_expert", encoded)
        self.assertNotIn("priority", encoded)

        tampered = {**records[0], "queue_item_digest": "0" * 32}
        self.assertIn(
            "priority_queue_digest_mismatch",
            validate_review_priority_against_queue([tampered], [item]),
        )

    def test_priority_store_reorders_but_does_not_write_priority_into_label(self):
        with tempfile.TemporaryDirectory() as directory:
            queue_path = Path(directory) / "queue.jsonl"
            priority_path = Path(directory) / "priority.jsonl"
            output_path = Path(directory) / "labels.jsonl"
            items = [
                priority_review_item(),
                priority_review_item(item_id="c" * 32, group_id="d" * 32),
            ]
            records, _report = build_review_slow_expert_priority(items)
            reversed_records = [
                {**record, "priority_rank": 1 - int(record["priority_rank"])}
                for record in records
            ]
            write_review_queue(queue_path, items)
            write_review_priority(priority_path, reversed_records)
            store = ReviewStore(queue_path, output_path, priority_path)
            state = store.state()
            first_by_rank = next(
                record["item_id"]
                for record in reversed_records
                if record["priority_rank"] == 0
            )
            self.assertEqual(state["item_id"], first_by_rank)
            encoded_state = json.dumps(state, sort_keys=True)
            self.assertNotIn("slow_expert", encoded_state)
            self.assertNotIn("priority", encoded_state)
            saved = store.label(
                item_id=state["item_id"],
                chosen_index=0,
                confidence="confirmed",
            )
            self.assertIsNotNone(saved["feedback"]["slow_expert_action"])
            encoded_label = output_path.read_text(encoding="utf-8")
            self.assertNotIn("slow_expert", encoded_label)
            self.assertNotIn("priority", encoded_label)

    def test_confirmed_review_encodes_as_behavior_only_weighted_example(self):
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory) / "train.review.jsonl"
            label = make_review_label(
                review_item(), chosen_index=1, confidence="confirmed"
            )
            labels.write_text(json.dumps(label) + "\n", encoding="utf-8")
            examples = load_review_examples(
                labels,
                SimpleNamespace(
                    allow_local_human_review_data=True,
                    feature_version=3,
                    history_window=24,
                    human_review_weight=1.0,
                    human_teacher_disagreement_weight=2.0,
                ),
            )
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].source, "local_human_review_opt_in")
        self.assertEqual(examples[0].sample_weight, 2.0)
        self.assertIsNone(examples[0].value_target)
        self.assertIsNone(examples[0].action_values)


if __name__ == "__main__":
    unittest.main()

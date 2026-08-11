import json
from pathlib import Path
import tempfile
import unittest

from tests.test_human_review import priority_review_item
from xiamen_mahjong.pairwise_review import (
    audit_pairwise_labels_against_queue,
    build_pairwise_review_queue,
    make_pairwise_label,
    pairwise_training_comparisons,
    public_pairwise_item,
    split_confirmed_pairwise_labels,
    validate_pairwise_label_against_item,
    validate_pairwise_queue_item,
    write_pairwise_queue,
    read_pairwise_queue,
)


class _FakeTieCandidate:
    def explain_discard(self, game, player_id):
        del player_id
        reference = 27
        selected = next(tile for tile in sorted(set(game.players[0].hand)) if tile != reference)
        return [{"tile": selected, "selected_by_public_progress_tiebreak": True}]


def pairwise_queue_item():
    source = priority_review_item()
    source["teacher_score_margin"] = 0.0
    queue, report = build_pairwise_review_queue(
        [source], candidate=_FakeTieCandidate()
    )
    if not queue:
        raise AssertionError(report)
    return queue[0]


class PairwiseReviewTests(unittest.TestCase):
    def test_builder_exports_one_anonymous_exact_tie_pair(self):
        item = pairwise_queue_item()
        self.assertEqual(validate_pairwise_queue_item(item), [])
        self.assertEqual(len(item["pair_action_indices"]), 2)
        self.assertEqual(
            set(item["pair_action_indices"]),
            {item["reference_teacher_index"], item["candidate_index"]},
        )

    def test_public_payload_hides_both_identities_and_unshown_actions(self):
        item = pairwise_queue_item()
        public = public_pairwise_item(item, completed=0, total=1)
        encoded = json.dumps(public, ensure_ascii=False, sort_keys=True)
        self.assertNotIn("reference_teacher_index", encoded)
        self.assertNotIn("candidate_index", encoded)
        self.assertNotIn(item["review_group_id"], encoded)
        self.assertEqual(len(public["legal_actions"]), 2)
        self.assertGreater(len(item["legal_actions"]), 2)

    def test_label_maps_display_position_back_to_full_action_pair(self):
        item = pairwise_queue_item()
        candidate_position = item["pair_action_indices"].index(item["candidate_index"])
        label = make_pairwise_label(
            item, chosen_position=candidate_position, confidence="confirmed"
        )
        self.assertEqual(validate_pairwise_label_against_item(label, item), [])
        self.assertEqual(label["decision"]["chosen_index"], item["candidate_index"])
        rows = pairwise_training_comparisons([label])
        self.assertEqual(rows[0][2], tuple(item["pair_action_indices"]))
        self.assertEqual(rows[0][1].chosen_index, item["candidate_index"])

    def test_audit_counts_candidate_preference_without_full_action_claim(self):
        item = pairwise_queue_item()
        position = item["pair_action_indices"].index(item["candidate_index"])
        label = make_pairwise_label(item, chosen_position=position, confidence="confirmed")
        report = audit_pairwise_labels_against_queue(
            [label],
            [item],
            minimum_confirmed_labels=1,
            minimum_candidate_preferences=1,
            minimum_groups=1,
        )
        self.assertTrue(report["ready_for_pairwise_split"])
        self.assertEqual(report["confirmed_candidate_preferences"], 1)
        self.assertEqual(
            report["target_semantics"],
            "pairwise_only_not_full_action_classification",
        )

    def test_pairwise_split_never_crosses_original_hand_group(self):
        records = []
        for index in range(30):
            item = pairwise_queue_item()
            item["item_id"] = f"{index + 1000:032x}"
            item["review_group_id"] = f"{index // 2:032x}"
            records.append(
                make_pairwise_label(item, chosen_position=index % 2, confidence="confirmed")
            )
        partitions = split_confirmed_pairwise_labels(records)
        by_group = {}
        for split, rows in partitions.items():
            for row in rows:
                self.assertEqual(by_group.setdefault(row["review_group_id"], split), split)

    def test_queue_round_trip_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairwise.jsonl"
            item = pairwise_queue_item()
            write_pairwise_queue(path, [item])
            self.assertEqual(read_pairwise_queue(path), [item])
            with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
                write_pairwise_queue(path, [item])


if __name__ == "__main__":
    unittest.main()

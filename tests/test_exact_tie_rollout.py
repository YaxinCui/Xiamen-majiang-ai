import json
from pathlib import Path
import tempfile
import unittest

from xiamen_mahjong.exact_tie_rollout import (
    ExactTieRolloutRecord,
    collect_exact_tie_rollout_records,
    exact_tie_distinct_decision_count,
    exact_tie_future_averaged_pairwise_examples,
    exact_tie_pairwise_examples,
    read_exact_tie_rollout_records,
    split_exact_tie_rollout_records_by_group,
    write_exact_tie_rollout_records,
)


class ExactTieRolloutTests(unittest.TestCase):
    def test_small_collection_is_grouped_paired_and_actor_visible(self):
        records, report = collect_exact_tie_rollout_records(
            seed_count=1,
            seed=202638700,
            samples_per_rotation=1,
            selection_seed=202638701,
        )
        self.assertGreater(len(records), 0)
        self.assertEqual(report["branch_rollouts"], sum(len(row.actions) for row in records))
        self.assertLessEqual(report["wall_groups"], 1)
        for row in records:
            self.assertEqual(row.teacher_index, 0)
            self.assertGreaterEqual(len(row.actions), 2)
            self.assertEqual(len(row.actions), len(row.terminal_scores))
            serialized = json.dumps(row.payload(), ensure_ascii=False)
            self.assertNotIn('"wall"', serialized)
            self.assertNotIn('"seed"', serialized)
            self.assertNotIn('"opponent_hands"', serialized)

    def test_round_trip_and_reject_private_field(self):
        records, _report = collect_exact_tie_rollout_records(
            seed_count=1,
            seed=202638702,
            samples_per_rotation=1,
            selection_seed=202638703,
        )
        self.assertGreater(len(records), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            write_exact_tie_rollout_records(records, path)
            self.assertEqual(read_exact_tie_rollout_records(path), records)

        first = records[0]
        invalid = ExactTieRolloutRecord(
            item_id=first.item_id,
            split_group_id=first.split_group_id,
            profile=first.profile,
            rules_version=first.rules_version,
            state={**first.state, "wall": [1, 2, 3]},
            actions=first.actions,
            teacher_index=first.teacher_index,
            terminal_scores=first.terminal_scores,
        )
        with self.assertRaisesRegex(ValueError, "私有"):
            invalid.validate()

    def test_group_split_and_pairwise_expansion_preserve_contract(self):
        records, _report = collect_exact_tie_rollout_records(
            seed_count=4,
            seed=202638704,
            samples_per_rotation=1,
            selection_seed=202638705,
        )
        partitions = split_exact_tie_rollout_records_by_group(
            records,
            split_salt="unit-test-exact-tie-split",
            train_fraction=0.5,
            validation_fraction=0.25,
        )
        memberships = {}
        for split, rows in partitions.items():
            for row in rows:
                memberships.setdefault(row.split_group_id, set()).add(split)
        self.assertTrue(all(len(splits) == 1 for splits in memberships.values()))
        examples = exact_tie_pairwise_examples(records)
        self.assertGreater(len(examples), 0)
        self.assertTrue(all(target in {0, 1} for _group, _state, _pair, target in examples))
        for group, state, pair, target in examples:
            source = next(
                row for row in records if row.split_group_id == group and row.state == state
            )
            indices = tuple(source.actions.index(action) for action in pair)
            self.assertNotEqual(
                source.terminal_scores[indices[0]], source.terminal_scores[indices[1]]
            )
            expected = int(
                source.terminal_scores[indices[1]] > source.terminal_scores[indices[0]]
            )
            self.assertEqual(target, expected)

    def test_future_permutations_collapse_to_decisions_before_pair_labels(self):
        records, report = collect_exact_tie_rollout_records(
            seed_count=1,
            seed=202638706,
            samples_per_rotation=1,
            selection_seed=202638707,
            future_wall_permutations=3,
        )
        self.assertEqual(len(records), report["selected_decisions"] * 3)
        self.assertEqual(
            exact_tie_distinct_decision_count(records), report["selected_decisions"]
        )
        examples = exact_tie_future_averaged_pairwise_examples(records)
        self.assertTrue(all(target in {0, 1} for *_prefix, target in examples))
        for group in {record.split_group_id for record in records}:
            self.assertTrue(
                all(example_group == group for example_group, *_rest in examples)
            )


if __name__ == "__main__":
    unittest.main()

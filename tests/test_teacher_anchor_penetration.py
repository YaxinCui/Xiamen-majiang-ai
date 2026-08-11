import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_teacher_anchor_penetration_v1 import (
    actor_visible_decisions,
    audit_checkpoint,
)
from scripts.init_fresh_selfplay_policy import create_fresh_policy


class TeacherAnchorPenetrationTests(unittest.TestCase):
    def test_zero_residual_has_zero_gap_and_no_margin_crossing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "visible.jsonl"
            decision = {
                "version": "xiamen-rule-teacher-v1",
                "profile": "core",
                "seat": 0,
                "state": {
                    "phase": "discard",
                    "hand": [0, 1],
                    "river_counts": [0] * 34,
                    "meld_counts": [0] * 34,
                    "gold_tile": 2,
                    "gold_indicator": 2,
                    "wall_remaining": 30,
                    "flowers": 0,
                    "tour_level": 0,
                    "gold_locked": False,
                    "is_dealer": False,
                    "rules_profile": "core",
                    "public_players": [],
                },
                "legal_actions": [
                    {"kind": "discard", "tile": 0, "tiles": []},
                    {"kind": "discard", "tile": 1, "tiles": []},
                ],
                "chosen_index": 0,
            }
            data.write_text(json.dumps({"decisions": [decision]}) + "\n")
            checkpoint = root / "fresh.pt"
            agent = create_fresh_policy(
                seed=1,
                feature_version=3,
                hidden_size=8,
                device="cpu",
                zero_policy_head=True,
            )
            agent.save(checkpoint)
            rows = actor_visible_decisions(data, maximum=1)
            audit = audit_checkpoint(
                checkpoint, rows, margin=5.0, batch_size=1, device="cpu"
            )
        self.assertEqual(audit["residual_best_alternative_gap"]["maximum"], 0.0)
        self.assertEqual(audit["margin_crossing_rates"]["5.0"], 0.0)
        self.assertGreater(audit["mean_non_teacher_probability_at_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()

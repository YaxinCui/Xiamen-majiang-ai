import tempfile
import unittest
from pathlib import Path

from xiamen_mahjong.training import (
    NeuralRulePolicyModel,
    RulePolicyModel,
    collect_teacher_decisions,
    collect_tour_curriculum,
    read_jsonl,
    write_jsonl,
)


class TrainingTests(unittest.TestCase):
    def test_teacher_collection_exports_only_legal_actions(self):
        decisions, summary = collect_teacher_decisions(hands=3, profile="classic", seed=101)
        self.assertGreater(summary.decisions, 20)
        self.assertEqual(summary.decisions, len(decisions))
        self.assertTrue(all(decision.chosen_action in decision.legal_actions for decision in decisions))
        self.assertTrue(all(decision.state["phase"] in {"discard", "response"} for decision in decisions))
        self.assertTrue(all(len(decision.state["hand"]) <= 17 for decision in decisions))

    def test_jsonl_and_model_round_trip(self):
        decisions, _ = collect_teacher_decisions(hands=2, profile="core", seed=211)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "teacher.jsonl"
            self.assertEqual(write_jsonl(decisions, path), len(decisions))
            loaded = read_jsonl(path)
            self.assertEqual(loaded, decisions)

            model = RulePolicyModel()
            model.fit(loaded, epochs=2, learning_rate=0.05, seed=211)
            checkpoint = root / "policy.json"
            model.save(checkpoint)
            restored = RulePolicyModel.load(checkpoint)
            self.assertEqual(restored.predict_index(loaded[0]), model.predict_index(loaded[0]))

    def test_training_improves_teacher_fit_on_seen_data(self):
        decisions, _ = collect_teacher_decisions(hands=4, profile="core", seed=307)
        model = RulePolicyModel()
        before = model.evaluate(decisions)
        model.fit(decisions, epochs=8, learning_rate=0.05, seed=307)
        after = model.evaluate(decisions)
        self.assertGreater(after["accuracy"], before["accuracy"])
        self.assertLess(after["loss"], before["loss"])

    def test_tour_curriculum_uses_engine_legal_advance_actions(self):
        decisions = collect_tour_curriculum(examples=8, seed=401)
        self.assertTrue(all(decision.chosen_action.kind == "advance_tour" for decision in decisions))
        self.assertTrue(
            all({action.kind for action in decision.legal_actions} == {"hu", "advance_tour"} for decision in decisions)
        )
        model = RulePolicyModel()
        model.fit(decisions, epochs=8, learning_rate=0.05, seed=401)
        self.assertEqual(model.evaluate(decisions)["accuracy"], 1.0)

    def test_mlp_learns_legal_tour_ranking_and_round_trips(self):
        decisions = collect_tour_curriculum(examples=16, seed=503)
        model = NeuralRulePolicyModel(hidden_size=8, seed=503)
        before = model.evaluate(decisions)
        model.fit(decisions, epochs=10, learning_rate=0.03, seed=503)
        after = model.evaluate(decisions)
        self.assertGreater(after["accuracy"], before["accuracy"])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "mlp.json"
            model.save(checkpoint)
            restored = NeuralRulePolicyModel.load(checkpoint)
            self.assertEqual(restored.predict_index(decisions[0]), model.predict_index(decisions[0]))


if __name__ == "__main__":
    unittest.main()

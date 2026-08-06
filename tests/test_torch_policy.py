import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch 仅在项目 .venv 中安装")
class CheckpointSelectionTests(unittest.TestCase):
    def test_lowest_validation_policy_loss_wins_then_accuracy_breaks_ties(self):
        from scripts.train_policy_value import better_validation_checkpoint

        first = {"policy_loss": 0.30, "accuracy": 0.90}
        lower_loss = {"policy_loss": 0.20, "accuracy": 0.80}
        tie_higher_accuracy = {"policy_loss": 0.20, "accuracy": 0.85}
        self.assertTrue(better_validation_checkpoint(first, None))
        self.assertTrue(better_validation_checkpoint(lower_loss, first))
        self.assertTrue(better_validation_checkpoint(tie_higher_accuracy, lower_loss))
        self.assertFalse(better_validation_checkpoint(lower_loss, tie_higher_accuracy))

    def test_rollout_soft_preference_masks_padding_and_rewards_better_actions(self):
        from scripts.train_policy_value import policy_preference_loss

        chosen = torch.tensor([0, 1])
        action_values = torch.tensor([[12.0, -12.0, 0.0], [-6.0, 14.0, 0.0]])
        action_value_mask = torch.tensor([True, True])
        legal = torch.tensor([[True, True, False], [True, True, False]])
        preferred_logits = torch.tensor([[4.0, -4.0, -1e30], [-4.0, 4.0, -1e30]])
        reversed_logits = torch.tensor([[-4.0, 4.0, -1e30], [4.0, -4.0, -1e30]])
        preferred = policy_preference_loss(
            preferred_logits,
            chosen=chosen,
            action_values=action_values,
            action_value_mask=action_value_mask,
            action_mask=legal,
            temperature=4.0,
        )
        reversed_loss = policy_preference_loss(
            reversed_logits,
            chosen=chosen,
            action_values=action_values,
            action_value_mask=action_value_mask,
            action_mask=legal,
            temperature=4.0,
        )
        self.assertTrue(torch.isfinite(preferred).all())
        self.assertTrue(torch.isfinite(reversed_loss).all())
        self.assertLess(float(preferred.mean()), float(reversed_loss.mean()))

    def test_direct_action_value_loss_keeps_terminal_score_magnitude(self):
        from scripts.train_policy_value import action_value_regression_loss

        action_values = torch.tensor([[40.0, -40.0, 0.0]])
        action_value_mask = torch.tensor([True])
        legal = torch.tensor([[True, True, False]])
        aligned = torch.tensor([[0.50, -0.50, 99.0]])
        reversed_values = torch.tensor([[-0.50, 0.50, 99.0]])
        aligned_loss, aligned_error, valid = action_value_regression_loss(
            aligned,
            action_values=action_values,
            action_value_mask=action_value_mask,
            action_mask=legal,
            target_scale=80.0,
        )
        reversed_loss, _reversed_error, _valid = action_value_regression_loss(
            reversed_values,
            action_values=action_values,
            action_value_mask=action_value_mask,
            action_mask=legal,
            target_scale=80.0,
        )
        self.assertTrue(torch.isfinite(aligned_loss).all())
        self.assertLess(float(aligned_loss[0]), float(reversed_loss[0]))
        self.assertEqual(valid.tolist(), [[True, True, False]])
        self.assertAlmostEqual(float(aligned_error[0, 0]), 0.0)

    def test_checkpoint_selection_can_target_a_held_out_action_value_source(self):
        from scripts.train_policy_value import (
            better_validation_checkpoint,
            checkpoint_selection_metrics,
        )

        validation = {
            "overall": {"decisions": 100.0, "policy_loss": 0.2, "accuracy": 0.9},
            "by_source": {
                "counterfactual_action_value_rollout": {
                    "decisions": 24.0,
                    "policy_loss": 0.7,
                    "accuracy": 0.5,
                    "action_value_decisions": 24.0,
                    "action_value_huber_loss": 0.7,
                    "action_value_rank_accuracy": 0.0,
                }
            },
        }
        self.assertEqual(
            checkpoint_selection_metrics(
                validation,
                source="counterfactual_action_value_rollout",
                minimum_decisions=20,
            ),
            {"policy_loss": 0.7, "accuracy": 0.5},
        )
        with self.assertRaises(ValueError):
            checkpoint_selection_metrics(
                validation,
                source="counterfactual_action_value_rollout",
                minimum_decisions=25,
            )

        q_first = {
            "action_value_huber_loss": 0.15,
            "action_value_rank_accuracy": 0.50,
        }
        q_better = {
            "action_value_huber_loss": 0.10,
            "action_value_rank_accuracy": 0.40,
        }
        self.assertEqual(
            checkpoint_selection_metrics(
                validation,
                source="counterfactual_action_value_rollout",
                minimum_decisions=20,
                metric="action_value_huber_loss",
            ),
            {
                "action_value_huber_loss": 0.7,
                "action_value_rank_accuracy": 0.0,
            },
        )
        self.assertTrue(
            better_validation_checkpoint(
                q_better,
                q_first,
                metric="action_value_huber_loss",
            )
        )

    def test_held_out_rare_action_uses_neutral_class_weight(self):
        from scripts.train_policy_value import Example, tensors

        example = Example(
            candidates=((0.0,) * 145, (1.0,) * 145),
            public_events=(),
            chosen_index=1,
            action_kind="rare_held_out_action",
            source="test",
            action_values=None,
            action_value_stderrs=None,
            value_target=None,
            sample_weight=0.5,
        )
        result = tensors(
            [example], feature_dim=145, device=torch.device("cpu"), weights={}
        )
        self.assertAlmostEqual(float(result[7][0]), 0.5)


@unittest.skipIf(torch is None, "PyTorch 仅在项目 .venv 中安装")
class TorchPolicyTests(unittest.TestCase):
    def test_candidate_policy_value_forward_and_checkpoint_round_trip(self):
        from xiamen_mahjong.torch_policy import (
            CandidatePolicyValueNetwork,
            TorchPolicyValueAgent,
        )
        from xiamen_mahjong.training import collect_teacher_decisions

        network = CandidatePolicyValueNetwork(feature_dim=145, hidden_size=16)
        candidates = torch.rand((2, 3, 145))
        mask = torch.tensor([[True, True, False], [True, True, True]])
        logits, values = network(candidates, mask)
        _q_logits, _q_values, action_values = network.forward_with_action_values(
            candidates, mask
        )
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(values.shape), (2,))
        self.assertEqual(tuple(action_values.shape), (2, 3))
        self.assertEqual(action_values[0, 2].item(), 0.0)
        self.assertLess(logits[0, 2].item(), -1e30)

        decisions, _ = collect_teacher_decisions(hands=2, profile="core", seed=741)
        agent = TorchPolicyValueAgent(
            feature_version=3, hidden_size=16, device="cpu", network=network
        )
        self.assertEqual(len(agent.scores(decisions[0])), len(decisions[0].legal_actions))
        self.assertEqual(
            len(agent.action_value_scores(decisions[0]) or ()),
            len(decisions[0].legal_actions),
        )
        logits, value = agent.policy_value(decisions[0])
        self.assertEqual(len(logits), len(decisions[0].legal_actions))
        self.assertIsInstance(value, float)
        single = [agent.policy_value(decision) for decision in decisions[:4]]
        batched = agent.policy_values_batch(decisions[:4])
        self.assertEqual(len(batched), len(single))
        for (single_logits, single_value), (batch_logits, batch_value) in zip(
            single, batched
        ):
            self.assertEqual(len(batch_logits), len(single_logits))
            self.assertAlmostEqual(batch_value, single_value, places=6)
            for actual, expected in zip(batch_logits, single_logits):
                self.assertAlmostEqual(actual, expected, places=6)
        single_scores = [agent.scores(decision) for decision in decisions[:4]]
        self.assertEqual(len(agent.scores_batch(decisions[:4])), len(single_scores))
        for expected, actual in zip(single_scores, agent.scores_batch(decisions[:4])):
            for expected_score, actual_score in zip(expected, actual):
                self.assertAlmostEqual(actual_score, expected_score, places=6)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "policy-value.pt"
            agent.save(checkpoint)
            restored = TorchPolicyValueAgent.load(checkpoint, device="cpu")
            self.assertEqual(
                restored.predict_index(decisions[0]), agent.predict_index(decisions[0])
            )

    def test_v2_candidate_checkpoint_loads_with_a_zero_initialized_q_head(self):
        from xiamen_mahjong.torch_policy import (
            ACTION_SELECTION_ACTION_VALUE,
            CandidatePolicyValueNetwork,
            TorchPolicyValueAgent,
        )
        from xiamen_mahjong.training import collect_teacher_decisions

        network = CandidatePolicyValueNetwork(feature_dim=145, hidden_size=16)
        legacy_state = {
            key: value
            for key, value in network.state_dict().items()
            if not key.startswith("action_value_head.")
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy-policy-value.pt"
            torch.save(
                {
                    "version": "xiamen-candidate-policy-value-v2",
                    "model": "candidate_policy_value",
                    "architecture": "candidate_mlp",
                    "feature_version": 3,
                    "feature_dim": 145,
                    "hidden_size": 16,
                    "attention_heads": 4,
                    "state_dict": legacy_state,
                    "metadata": {},
                },
                checkpoint,
            )
            restored = TorchPolicyValueAgent.load(
                checkpoint,
                device="cpu",
                action_selection=ACTION_SELECTION_ACTION_VALUE,
            )
            decisions, _ = collect_teacher_decisions(hands=1, profile="core", seed=771)
            self.assertTrue(
                all(value == 0.0 for value in restored.action_value_scores(decisions[0]) or ())
            )
            self.assertEqual(restored.action_selection, ACTION_SELECTION_ACTION_VALUE)

    def test_public_sequence_transformer_masks_events_and_round_trips(self):
        from xiamen_mahjong.torch_policy import (
            ARCHITECTURE_PUBLIC_SEQUENCE_TRANSFORMER,
            PublicSequencePolicyValueNetwork,
            TorchPolicyValueAgent,
        )
        from xiamen_mahjong.training import (
            PUBLIC_ACTION_SEQUENCE_DIM,
            PUBLIC_ACTION_SEQUENCE_LENGTH,
            collect_teacher_decisions,
        )

        network = PublicSequencePolicyValueNetwork(
            feature_dim=145, hidden_size=16, attention_heads=4
        )
        candidates = torch.rand((2, 3, 145))
        action_mask = torch.tensor([[True, True, False], [True, True, True]])
        events = torch.zeros((2, PUBLIC_ACTION_SEQUENCE_LENGTH, PUBLIC_ACTION_SEQUENCE_DIM))
        event_mask = torch.zeros((2, PUBLIC_ACTION_SEQUENCE_LENGTH), dtype=torch.bool)
        event_mask[1, :2] = True
        logits, values = network(candidates, action_mask, events, event_mask)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(tuple(values.shape), (2,))
        self.assertTrue(torch.isfinite(values).all())
        self.assertLess(logits[0, 2].item(), -1e30)

        decisions, _ = collect_teacher_decisions(hands=2, profile="core", seed=751)
        agent = TorchPolicyValueAgent(
            feature_version=3,
            hidden_size=16,
            architecture=ARCHITECTURE_PUBLIC_SEQUENCE_TRANSFORMER,
            attention_heads=4,
            device="cpu",
            network=network,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "sequence-policy-value.pt"
            agent.save(checkpoint)
            restored = TorchPolicyValueAgent.load(checkpoint, device="cpu")
            self.assertEqual(
                restored.architecture, ARCHITECTURE_PUBLIC_SEQUENCE_TRANSFORMER
            )
            self.assertEqual(
                restored.predict_index(decisions[0]), agent.predict_index(decisions[0])
            )

    def test_residual_sequence_starts_exactly_as_candidate_checkpoint(self):
        from xiamen_mahjong.torch_policy import (
            ARCHITECTURE_PUBLIC_SEQUENCE_RESIDUAL,
            CandidatePolicyValueNetwork,
            ResidualPublicSequencePolicyValueNetwork,
            TorchPolicyValueAgent,
        )
        from xiamen_mahjong.training import (
            PUBLIC_ACTION_SEQUENCE_DIM,
            PUBLIC_ACTION_SEQUENCE_LENGTH,
            collect_teacher_decisions,
        )

        base = CandidatePolicyValueNetwork(feature_dim=145, hidden_size=16)
        residual = ResidualPublicSequencePolicyValueNetwork(
            feature_dim=145, hidden_size=16, attention_heads=4
        )
        residual.initialize_from_candidate(base)
        candidates = torch.rand((2, 3, 145))
        action_mask = torch.tensor([[True, True, False], [True, True, True]])
        events = torch.rand((2, PUBLIC_ACTION_SEQUENCE_LENGTH, PUBLIC_ACTION_SEQUENCE_DIM))
        event_mask = torch.zeros((2, PUBLIC_ACTION_SEQUENCE_LENGTH), dtype=torch.bool)
        event_mask[0, 0] = True
        base_logits, base_values = base(candidates, action_mask)
        logits, values = residual(candidates, action_mask, events, event_mask)
        self.assertTrue(torch.equal(base_logits, logits))
        self.assertTrue(torch.equal(base_values, values))

        decisions, _ = collect_teacher_decisions(hands=2, profile="core", seed=761)
        agent = TorchPolicyValueAgent(
            feature_version=3,
            hidden_size=16,
            architecture=ARCHITECTURE_PUBLIC_SEQUENCE_RESIDUAL,
            attention_heads=4,
            device="cpu",
            network=residual,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "residual-sequence-policy-value.pt"
            agent.save(checkpoint)
            restored = TorchPolicyValueAgent.load(checkpoint, device="cpu")
            self.assertEqual(
                restored.architecture, ARCHITECTURE_PUBLIC_SEQUENCE_RESIDUAL
            )
            self.assertEqual(
                restored.predict_index(decisions[0]), agent.predict_index(decisions[0])
            )


if __name__ == "__main__":
    unittest.main()

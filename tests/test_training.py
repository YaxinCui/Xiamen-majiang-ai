import tempfile
import unittest
from pathlib import Path

from xiamen_mahjong.training import (
    NeuralRulePolicyModel,
    PUBLIC_ACTION_SEQUENCE_DIM,
    RulePolicyModel,
    StateValueBaseline,
    _turn_actions,
    collect_candidate_teacher_dagger_trajectories,
    collect_counterfactual_action_value_trajectories,
    collect_dagger_decisions,
    collect_exploration_trajectories,
    collect_policy_episodes,
    collect_response_pass_curriculum,
    collect_teacher_decisions,
    collect_teacher_trajectories,
    collect_tour_curriculum,
    collect_tour_trajectories,
    read_trajectory_jsonl,
    read_jsonl,
    split_trajectories_by_hand,
    public_action_sequence_features,
    trajectory_manifest,
    write_trajectory_replay_index,
    write_trajectory_jsonl,
    write_jsonl,
)


class _BatchFirstLegalPolicy:
    """Test double whose optional batch interface matches its single path."""

    def __init__(self) -> None:
        self.batch_calls: list[int] = []

    def choose_turn_action(self, game, player_id):
        return _turn_actions(game, player_id)[0]

    def choose_response(self, game, player_id, options):
        return tuple(options)[0]

    def scores_batch(self, decisions):
        self.batch_calls.append(len(decisions))
        return [
            [-float(index) for index in range(len(decision.legal_actions))]
            for decision in decisions
        ]


class TrainingTests(unittest.TestCase):
    def test_teacher_collection_exports_only_legal_actions(self):
        decisions, summary = collect_teacher_decisions(hands=3, profile="classic", seed=101)
        self.assertGreater(summary.decisions, 20)
        self.assertEqual(summary.decisions, len(decisions))
        self.assertTrue(all(decision.chosen_action in decision.legal_actions for decision in decisions))
        self.assertTrue(all(decision.state["phase"] in {"discard", "response"} for decision in decisions))
        self.assertTrue(all(len(decision.state["hand"]) <= 17 for decision in decisions))

    def test_trajectory_export_is_split_by_hand_and_has_no_hidden_state(self):
        trajectories, summary = collect_teacher_trajectories(
            hands=4, profile="core", seed=151
        )
        self.assertEqual(len(trajectories), 4)
        self.assertEqual(sum(len(item.decisions) for item in trajectories), summary.decisions)
        decision = next(item for trajectory in trajectories for item in trajectory.decisions)
        self.assertIn("public_players", decision.state)
        self.assertIn("recent_public_actions", decision.state)
        self.assertNotIn("wall", decision.state)
        self.assertNotIn("opponent_hands", decision.state)
        self.assertEqual(decision.state["public_players"][0]["relative_seat"], 0)
        self.assertEqual(len(decision.state["public_players"]), 4)
        partitions = split_trajectories_by_hand(trajectories)
        partition_seeds = [
            trajectory.seed
            for partition in partitions.values()
            for trajectory in partition
        ]
        self.assertCountEqual(partition_seeds, [151, 152, 153, 154])
        manifest = trajectory_manifest(trajectories)
        self.assertEqual(manifest["hands"], 4)
        self.assertEqual(manifest["decisions"], summary.decisions)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectories.jsonl"
            self.assertEqual(write_trajectory_jsonl(trajectories, path), 4)
            serialized = path.read_text(encoding="utf-8")
            self.assertNotIn('"seed"', serialized)
            self.assertNotIn('"behavior_seed"', serialized)
            loaded = read_trajectory_jsonl(path)
            self.assertEqual([item.trajectory_id for item in loaded], [item.trajectory_id for item in trajectories])
            self.assertTrue(all(item.seed is None for item in loaded))
            self.assertTrue(
                all(
                    decision.seed is None
                    for trajectory in loaded
                    for decision in trajectory.decisions
                )
            )
            replay = Path(directory) / "private-replay.jsonl"
            self.assertEqual(write_trajectory_replay_index(trajectories, replay), 4)
            self.assertIn('"seed"', replay.read_text(encoding="utf-8"))

    def test_exploration_trajectories_are_teacher_labeled_and_replayable(self):
        trajectories, summary = collect_exploration_trajectories(
            hands=3, profile="core", seed=171, behavior_seed=971
        )
        self.assertEqual(len(trajectories), 3)
        self.assertEqual(summary.hands, 3)
        self.assertTrue(all(item.source_metadata["behavior_seed"] >= 971 for item in trajectories))
        self.assertTrue(
            all(
                decision.chosen_action in decision.legal_actions
                for trajectory in trajectories
                for decision in trajectory.decisions
            )
        )

    def test_response_pass_curriculum_uses_a_teacher_pass_label(self):
        trajectories = collect_response_pass_curriculum(
            examples=1, profile="classic", seed=1082
        )
        self.assertEqual(len(trajectories), 1)
        decision = trajectories[0].decisions[0]
        self.assertEqual(decision.chosen_action.kind, "pass")
        self.assertGreater(len(decision.legal_actions), 1)
        self.assertTrue(trajectories[0].outcome["synthetic"])

    def test_tour_trajectories_keep_engine_validated_advance_labels(self):
        trajectories = collect_tour_trajectories(examples=4, seed=411)
        self.assertEqual(len(trajectories), 4)
        self.assertTrue(
            all(
                trajectory.decisions[0].chosen_action.kind == "advance_tour"
                for trajectory in trajectories
            )
        )

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

    def test_dagger_labels_policy_visited_states_with_legal_teacher_actions(self):
        policy = NeuralRulePolicyModel(hidden_size=4, seed=457)
        decisions, summary = collect_dagger_decisions(
            policy, hands=2, profile="classic", seed=457
        )
        self.assertGreater(summary.decisions, 20)
        self.assertEqual(summary.decisions, len(decisions))
        self.assertTrue(
            all(decision.chosen_action in decision.legal_actions for decision in decisions)
        )

    def test_candidate_teacher_dagger_rotates_one_candidate_and_keeps_walls_together(self):
        policy = NeuralRulePolicyModel(hidden_size=4, seed=473)
        trajectories, summary = collect_candidate_teacher_dagger_trajectories(
            policy, seed_count=2, profile="core", seed=473
        )
        self.assertEqual(len(trajectories), 8)
        self.assertEqual(summary.hands, 8)
        self.assertTrue(all(trajectory.decisions for trajectory in trajectories))
        for trajectory in trajectories:
            candidate_seat = trajectory.source_metadata["candidate_seat"]
            self.assertEqual(trajectory.agent_profiles[candidate_seat], "candidate_policy")
            self.assertTrue(
                all(
                    decision.seat == candidate_seat
                    and decision.chosen_action in decision.legal_actions
                    for decision in trajectory.decisions
                )
            )
        partitions = split_trajectories_by_hand(trajectories)
        for seed in {trajectory.seed for trajectory in trajectories}:
            seed_partitions = {
                partition
                for partition, records in partitions.items()
                for trajectory in records
                if trajectory.seed == seed
            }
            self.assertEqual(len(seed_partitions), 1)
            self.assertEqual(
                sum(
                    trajectory.seed == seed
                    for records in partitions.values()
                    for trajectory in records
                ),
                4,
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate-dagger.jsonl"
            write_trajectory_jsonl(trajectories, path)
            safe_partitions = split_trajectories_by_hand(read_trajectory_jsonl(path))
            for group_id in {trajectory.split_group_id for trajectory in trajectories}:
                self.assertEqual(
                    len(
                        {
                            partition
                            for partition, records in safe_partitions.items()
                            for trajectory in records
                            if trajectory.split_group_id == group_id
                        }
                    ),
                    1,
                )

    def test_counterfactual_action_values_are_safe_and_grouped_by_wall(self):
        policy = NeuralRulePolicyModel(hidden_size=4, seed=587)
        trajectories, summary = collect_counterfactual_action_value_trajectories(
            policy,
            seed_count=2,
            profile="core",
            seed=587,
            response_sample_probability=0.0,
            rollouts_per_action=2,
        )
        self.assertEqual(len(trajectories), 8)
        self.assertEqual(summary.hands, 8)
        self.assertGreater(summary.branch_rollouts, summary.decisions)
        self.assertEqual(summary.repeated_decisions, summary.decisions)
        self.assertIsNotNone(summary.mean_action_value_stderr)
        manifest = trajectory_manifest(trajectories)
        self.assertEqual(manifest["action_value_decisions"], summary.decisions)
        self.assertGreater(manifest["action_value_stderr_observations"], 0)
        for trajectory in trajectories:
            decision = trajectory.decisions[0]
            self.assertTrue(trajectory.outcome["synthetic"])
            self.assertEqual(
                len(decision.action_values or ()), len(decision.legal_actions)
            )
            self.assertEqual(
                len(decision.action_value_stderrs or ()), len(decision.legal_actions)
            )
            self.assertEqual(
                decision.chosen_index,
                max(
                    range(len(decision.legal_actions)),
                    key=lambda index: ((decision.action_values or ())[index], -index),
                ),
            )
            self.assertNotIn("wall", decision.state)
            self.assertNotIn("opponent_hands", decision.state)
        partitions = split_trajectories_by_hand(trajectories)
        for seed in {trajectory.seed for trajectory in trajectories}:
            self.assertEqual(
                len(
                    {
                        partition
                        for partition, records in partitions.items()
                        for trajectory in records
                        if trajectory.seed == seed
                    }
                ),
                1,
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counterfactual.jsonl"
            write_trajectory_jsonl(trajectories, path)
            serialized = path.read_text(encoding="utf-8")
            self.assertIn('"action_values"', serialized)
            self.assertIn('"action_value_stderrs"', serialized)
            self.assertNotIn('"seed"', serialized)
            restored = read_trajectory_jsonl(path)
            self.assertTrue(
                all(
                    decision.action_values is not None
                    and decision.action_value_stderrs is not None
                    for trajectory in restored
                    for decision in trajectory.decisions
                )
            )

    def test_belief_resampled_action_values_keep_only_actor_visible_state(self):
        policy = NeuralRulePolicyModel(hidden_size=4, seed=931)
        trajectories, summary = collect_counterfactual_action_value_trajectories(
            policy,
            seed_count=1,
            profile="classic",
            seed=931,
            rollouts_per_action=2,
            response_sample_probability=0.0,
            belief_resample=True,
        )
        self.assertEqual(len(trajectories), 4)
        self.assertEqual(summary.belief_resampled_worlds, 8)
        self.assertEqual(summary.belief_resample_skipped, 0)
        self.assertTrue(
            all(
                trajectory.source_metadata["belief_resample"]
                and "wall" not in trajectory.decisions[0].state
                and "opponent_hands" not in trajectory.decisions[0].state
                for trajectory in trajectories
            )
        )

    def test_batched_counterfactual_branches_match_serial_safe_targets(self):
        serial_policy = _BatchFirstLegalPolicy()
        serial, serial_summary = collect_counterfactual_action_value_trajectories(
            serial_policy,
            seed_count=1,
            profile="core",
            seed=947,
            response_sample_probability=0.0,
            rollouts_per_action=2,
            rollout_batch_size=1,
        )
        batched_policy = _BatchFirstLegalPolicy()
        batched, batched_summary = collect_counterfactual_action_value_trajectories(
            batched_policy,
            seed_count=1,
            profile="core",
            seed=947,
            response_sample_probability=0.0,
            rollouts_per_action=2,
            rollout_batch_size=8,
        )

        def safe_targets(trajectories):
            return [
                (
                    decision.state,
                    decision.legal_actions,
                    decision.chosen_index,
                    decision.action_values,
                    decision.action_value_stderrs,
                    trajectory.public_actions,
                )
                for trajectory in trajectories
                for decision in trajectory.decisions
            ]

        self.assertEqual(safe_targets(batched), safe_targets(serial))
        self.assertEqual(batched_summary.branch_rollouts, serial_summary.branch_rollouts)
        self.assertEqual(
            batched_summary.belief_resampled_worlds,
            serial_summary.belief_resampled_worlds,
        )
        self.assertEqual(serial_summary.batched_inference_calls, 0)
        self.assertGreater(batched_summary.batched_inference_calls, 0)
        self.assertGreater(batched_summary.batched_inference_decisions, 0)
        self.assertGreater(batched_summary.max_batched_inference_decisions, 1)
        self.assertGreater(len(batched_policy.batch_calls), 0)
        self.assertGreater(max(batched_policy.batch_calls), 1)

    def test_reinforce_collects_legal_episodes_and_updates(self):
        policy = NeuralRulePolicyModel(hidden_size=4, seed=613)
        episodes = collect_policy_episodes(
            policy, episodes=4, profile="classic", seed=613
        )
        self.assertEqual(len(episodes), 4)
        self.assertTrue(all(episode.steps for episode in episodes))
        self.assertTrue(
            all(
                0 <= step.action_index < len(step.decision.legal_actions)
                for episode in episodes
                for step in episode.steps
            )
        )
        metrics = policy.reinforce(episodes, learning_rate=0.0005)
        self.assertGreater(metrics["decisions"], 0)

    def test_state_value_baseline_fits_actor_visible_policy_states(self):
        policy = NeuralRulePolicyModel(hidden_size=6, seed=627)
        episodes = collect_policy_episodes(
            policy, episodes=8, profile="classic", seed=627
        )
        critic = StateValueBaseline(policy.feature_dim)
        first_step = next(step for episode in episodes for step in episode.steps)
        features = critic._features(policy, first_step.decision)
        self.assertEqual(len(features), policy.feature_dim)
        before_prediction = critic.predict(policy, first_step.decision)
        metrics = critic.fit(policy, episodes, reward_scale=40.0, epochs=8)
        self.assertLessEqual(metrics["value_mse_after"], metrics["value_mse_before"])
        self.assertNotEqual(before_prediction, critic.predict(policy, first_step.decision))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "value.json"
            critic.save(checkpoint)
            restored = StateValueBaseline.load(checkpoint)
            self.assertEqual(
                restored.predict(policy, first_step.decision),
                critic.predict(policy, first_step.decision),
            )

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

    def test_mlp_v3_accepts_public_trajectory_context_and_class_weights(self):
        trajectories, _ = collect_teacher_trajectories(
            hands=4, profile="core", seed=541
        )
        decisions = [
            decision for trajectory in trajectories for decision in trajectory.decisions
        ]
        model = NeuralRulePolicyModel(hidden_size=8, feature_version=3, seed=541)
        self.assertEqual(len(model._feature_vectors(decisions[0])[0]), 145)
        before = model.evaluate(decisions)
        model.fit(
            decisions,
            epochs=3,
            learning_rate=0.01,
            class_weights={"discard": 1.0, "pong": 2.0},
            sample_weights=[1.0] * len(decisions),
            seed=541,
        )
        after = model.evaluate(decisions)
        self.assertGreater(after["accuracy"], before["accuracy"])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "mlp-v3.json"
            model.save(checkpoint)
            self.assertEqual(
                NeuralRulePolicyModel.load(checkpoint).feature_version, 3
            )

    def test_public_action_sequence_is_ordered_and_redacts_concealed_kong_face(self):
        state = {
            "recent_public_actions": [
                {"kind": "discard", "relative_seat": 2, "tile": 7},
                {
                    "kind": "an_kan",
                    "relative_seat": 1,
                    # A malformed historical record must not turn into a
                    # visible tile feature just because it carries this field.
                    "tile": 11,
                    "tiles": [11, 11, 11, 11],
                },
                {"kind": "chi", "relative_seat": 3, "tile": 8, "tiles": [6, 7]},
            ]
        }
        sequence = public_action_sequence_features(state)
        self.assertEqual(len(sequence), 3)
        self.assertTrue(all(len(token) == PUBLIC_ACTION_SEQUENCE_DIM for token in sequence))
        # The first kind segment differs, proving that temporal positions are
        # preserved instead of collapsing history into counts.
        self.assertNotEqual(sequence[0][:11], sequence[2][:11])
        # Robustness against an accidental malformed an_kan export: its tile
        # slots should be ignored by the sequence encoder.
        concealed = public_action_sequence_features(
            {"recent_public_actions": [{"kind": "an_kan", "relative_seat": 1}]}
        )[0]
        self.assertEqual(sequence[1][:-1], concealed[:-1])

    def test_mlp_v1_checkpoint_remains_loadable(self):
        decisions = collect_tour_curriculum(examples=4, seed=557)
        model = NeuralRulePolicyModel(hidden_size=4, seed=557, feature_version=1)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "mlp-v1.json"
            model.save(checkpoint)
            restored = NeuralRulePolicyModel.load(checkpoint)
            self.assertEqual(restored.feature_version, 1)
            self.assertEqual(restored.predict_index(decisions[0]), model.predict_index(decisions[0]))


if __name__ == "__main__":
    unittest.main()

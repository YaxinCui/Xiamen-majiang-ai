import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch 仅在项目 .venv 中安装")
class TorchPpoTests(unittest.TestCase):
    def test_candidate_teacher_rollout_and_ppo_update_remain_engine_legal(self):
        from scripts.train_torch_ppo import collect_rollouts, ppo_update
        from xiamen_mahjong.torch_policy import TorchPolicyValueAgent

        agent = TorchPolicyValueAgent(
            feature_version=3, hidden_size=16, device="cpu"
        )
        steps, summary = collect_rollouts(
            agent,
            episodes=4,
            profile="core",
            seed=811,
            reward_scale=80.0,
        )
        self.assertEqual(summary.episodes, 4)
        self.assertGreater(summary.decisions, 0)
        self.assertTrue(
            all(
                0 <= step.action_index < len(step.decision.legal_actions)
                for step in steps
            )
        )
        metrics = ppo_update(
            agent.network,
            steps,
            device=agent.device,
            batch_size=32,
            epochs=1,
            learning_rate=0.0001,
            clip_ratio=0.15,
            value_weight=0.25,
            entropy_weight=0.002,
            seed=811,
        )
        self.assertGreater(metrics["updates"], 0)
        self.assertGreater(metrics["entropy"], 0)

    def test_rollout_can_mix_a_frozen_opponent_pool(self):
        from scripts.train_torch_ppo import collect_rollouts
        from xiamen_mahjong.torch_policy import TorchPolicyValueAgent

        policy = TorchPolicyValueAgent(feature_version=3, hidden_size=16, device="cpu")
        _steps, summary = collect_rollouts(
            policy,
            episodes=4,
            profile="core",
            seed=821,
            reward_scale=80.0,
            opponents=(("frozen_candidate", policy),),
            teacher_opponent_probability=0.0,
        )
        self.assertEqual(summary.opponent_profile_counts["frozen_candidate"], 12)

    def test_batched_rollout_keeps_actions_legal_and_accounts_for_all_episodes(self):
        from scripts.train_torch_ppo import collect_rollouts_batched
        from xiamen_mahjong.torch_policy import TorchPolicyValueAgent

        policy = TorchPolicyValueAgent(feature_version=3, hidden_size=16, device="cpu")
        steps, summary = collect_rollouts_batched(
            policy,
            episodes=8,
            profile="core",
            seed=831,
            reward_scale=80.0,
            rollout_batch_size=3,
        )
        self.assertEqual(summary.episodes, 8)
        self.assertEqual(summary.wins + summary.draws, 8)
        self.assertGreater(summary.decisions, 0)
        self.assertTrue(
            all(
                0 <= step.action_index < len(step.decision.legal_actions)
                for step in steps
            )
        )

    def test_batched_rollout_batches_frozen_torch_opponents(self):
        from scripts.train_torch_ppo import collect_rollouts_batched
        from xiamen_mahjong.torch_policy import TorchPolicyValueAgent

        policy = TorchPolicyValueAgent(feature_version=3, hidden_size=16, device="cpu")
        steps, summary = collect_rollouts_batched(
            policy,
            episodes=8,
            profile="core",
            seed=841,
            reward_scale=80.0,
            opponents=(("frozen_candidate", policy),),
            teacher_opponent_probability=0.0,
            rollout_batch_size=4,
        )
        self.assertEqual(summary.opponent_profile_counts["frozen_candidate"], 24)
        self.assertGreater(len(steps), 0)


if __name__ == "__main__":
    unittest.main()

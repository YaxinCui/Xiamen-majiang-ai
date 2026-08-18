import tempfile
import unittest
from pathlib import Path
import copy
from types import SimpleNamespace

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch 仅在项目 .venv 中安装")
class TorchPpoTests(unittest.TestCase):
    def test_training_config_payload_records_every_reproduction_knob(self):
        from scripts.train_torch_ppo import training_config_payload

        args = SimpleNamespace(
            iterations=4,
            episodes_per_iteration=1024,
            rollout_batch_size=64,
            ppo_epochs=2,
            batch_size=512,
            learning_rate=0.00005,
            policy_head_learning_rate_multiplier=10.0,
            reference_kl_weight=0.05,
            clip_ratio=0.15,
            value_weight=0.25,
            entropy_weight=0.002,
            reward_scale=80.0,
            seed=202646000,
            profile="classic",
            device="cuda",
            teacher_opponent_probability=1.0,
            self_play_opponent_probability=0.0,
            teacher_prior_margin=0.0,
            opponent_checkpoint=[],
            privileged_critic=False,
            privileged_critic_hidden_size=128,
            privileged_critic_weight=0.25,
        )
        payload = training_config_payload(args)
        self.assertEqual(payload["episodes_per_iteration"], 1024)
        self.assertEqual(payload["ppo_epochs"], 2)
        self.assertEqual(payload["seed"], 202646000)
        self.assertEqual(payload["opponent_checkpoints"], [])
        self.assertEqual(
            set(payload),
            {
                "iterations",
                "episodes_per_iteration",
                "rollout_batch_size",
                "ppo_epochs",
                "batch_size",
                "learning_rate",
                "policy_head_learning_rate_multiplier",
                "reference_kl_weight",
                "clip_ratio",
                "value_weight",
                "entropy_weight",
                "reward_scale",
                "seed",
                "profile",
                "device",
                "teacher_opponent_probability",
                "self_play_opponent_probability",
                "teacher_prior_margin",
                "opponent_checkpoints",
                "privileged_critic",
                "privileged_critic_hidden_size",
                "privileged_critic_weight",
            },
        )

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
        self.assertNotIn("anchor_rows", metrics)
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
        self.assertLessEqual(summary.wins + summary.draws, 8)
        self.assertGreaterEqual(summary.wins, 0)
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

    def test_batched_rollout_can_use_an_isolated_current_policy_snapshot(self):
        from scripts.train_torch_ppo import (
            collect_rollouts_batched,
            freeze_policy_snapshot,
        )
        from xiamen_mahjong.torch_policy import TorchPolicyValueAgent

        policy = TorchPolicyValueAgent(feature_version=3, hidden_size=16, device="cpu")
        snapshot = freeze_policy_snapshot(policy)
        snapshot_before = {
            name: value.detach().clone()
            for name, value in snapshot.network.state_dict().items()
        }
        with torch.no_grad():
            next(policy.network.parameters()).add_(1.0)
        self.assertTrue(
            all(
                torch.equal(snapshot_before[name], value)
                for name, value in snapshot.network.state_dict().items()
            )
        )
        steps, summary = collect_rollouts_batched(
            policy,
            episodes=8,
            profile="core",
            seed=846,
            reward_scale=80.0,
            teacher_opponent_probability=0.0,
            self_play_snapshot=snapshot,
            self_play_opponent_probability=1.0,
            rollout_batch_size=4,
        )
        self.assertEqual(summary.opponent_profile_counts["current_policy_snapshot"], 24)
        self.assertGreater(len(steps), 0)

    def test_self_play_probability_requires_a_snapshot(self):
        from scripts.train_torch_ppo import collect_rollouts
        from xiamen_mahjong.torch_policy import TorchPolicyValueAgent

        with self.assertRaisesRegex(ValueError, "冻结快照"):
            collect_rollouts(
                TorchPolicyValueAgent(feature_version=3, hidden_size=16, device="cpu"),
                episodes=1,
                profile="core",
                seed=847,
                reward_scale=80.0,
                teacher_opponent_probability=0.0,
                self_play_opponent_probability=1.0,
            )

    def test_teacher_anchored_rollout_preserves_prior_through_ppo_update(self):
        from scripts.init_fresh_selfplay_policy import create_fresh_policy
        from scripts.train_torch_ppo import collect_rollouts_batched, ppo_update

        policy = create_fresh_policy(
            seed=860,
            feature_version=3,
            hidden_size=16,
            device="cpu",
            zero_policy_head=True,
        )
        snapshot = policy
        steps, summary = collect_rollouts_batched(
            policy,
            episodes=8,
            profile="core",
            seed=861,
            reward_scale=80.0,
            teacher_opponent_probability=0.0,
            self_play_snapshot=snapshot,
            self_play_opponent_probability=1.0,
            teacher_prior_margin=5.0,
            rollout_batch_size=4,
        )
        self.assertEqual(summary.episodes, 8)
        self.assertTrue(all(step.teacher_prior_logits is not None for step in steps))
        self.assertTrue(
            all(
                max(step.teacher_prior_logits or ()) == 0.0
                and min(step.teacher_prior_logits or ()) <= -5.0
                for step in steps
                if len(step.teacher_prior_logits or ()) > 1
            )
        )
        metrics = ppo_update(
            policy.network,
            steps,
            device=policy.device,
            batch_size=32,
            epochs=1,
            learning_rate=0.0001,
            clip_ratio=0.15,
            value_weight=0.25,
            entropy_weight=0.002,
            seed=862,
            reference_network=copy.deepcopy(policy.network),
            reference_kl_weight=0.1,
            policy_head_learning_rate_multiplier=5.0,
        )
        self.assertGreater(metrics["updates"], 0)
        self.assertGreater(metrics["anchor_rows"], 0)
        self.assertIn("residual_best_alternative_gap_maximum", metrics)
        self.assertEqual(metrics["anchor_deterministic_override_rate"], 0.0)
        self.assertGreater(metrics["anchor_teacher_probability_mean"], 0.5)
        self.assertGreaterEqual(metrics["reference_kl"], 0.0)
        self.assertEqual(metrics["reference_kl_weight"], 0.1)
        self.assertEqual(metrics["policy_head_learning_rate_multiplier"], 5.0)
        self.assertGreaterEqual(metrics["final_reference_kl_mean"], 0.0)
        self.assertGreaterEqual(
            metrics["final_reference_argmax_disagreement_rate"], 0.0
        )

    def test_training_only_privileged_critic_never_enters_actor_observations(self):
        from scripts.train_torch_ppo import (
            PRIVILEGED_CRITIC_FEATURE_DIM,
            PrivilegedCritic,
            collect_rollouts_batched,
            evaluate_privileged_critic,
            mask_progressive_hiding_features,
            ppo_update,
            train_progressive_hiding_critic,
            train_privileged_critic,
        )
        from xiamen_mahjong.torch_policy import TorchPolicyValueAgent

        policy = TorchPolicyValueAgent(feature_version=3, hidden_size=16, device="cpu")
        critic = PrivilegedCritic(hidden_size=16).to(policy.device)
        steps, summary = collect_rollouts_batched(
            policy,
            episodes=8,
            profile="core",
            seed=851,
            reward_scale=80.0,
            rollout_batch_size=4,
            privileged_critic=critic,
        )
        self.assertEqual(summary.episodes, 8)
        self.assertTrue(
            all(
                step.privileged_features is not None
                and len(step.privileged_features) == PRIVILEGED_CRITIC_FEATURE_DIM
                and "wall" not in step.decision.state
                and "opponent_hands" not in step.decision.state
                for step in steps
            )
        )
        metrics = ppo_update(
            policy.network,
            steps,
            device=policy.device,
            batch_size=32,
            epochs=1,
            learning_rate=0.0001,
            clip_ratio=0.15,
            value_weight=0.25,
            entropy_weight=0.002,
            seed=851,
            privileged_critic=critic,
            privileged_critic_weight=0.25,
        )
        self.assertIn("privileged_critic_loss", metrics)
        self.assertGreater(metrics["updates"], 0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "actor-only.pt"
            policy.save(checkpoint, metadata={"training_only_privileged_critic": True})
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.assertFalse(
            any("privileged" in key or "critic" in key for key in payload["state_dict"])
        )

        actor_before = {
            name: value.detach().clone() for name, value in policy.network.state_dict().items()
        }
        critic_before = {
            name: value.detach().clone() for name, value in critic.state_dict().items()
        }
        critic_metrics = train_privileged_critic(
            critic,
            steps,
            device=policy.device,
            batch_size=32,
            epochs=1,
            learning_rate=0.0001,
            seed=852,
        )
        self.assertGreater(critic_metrics["updates"], 0)
        self.assertTrue(
            all(
                torch.equal(actor_before[name], value)
                for name, value in policy.network.state_dict().items()
            )
        )
        self.assertTrue(
            any(
                not torch.equal(critic_before[name], value)
                for name, value in critic.state_dict().items()
            )
        )
        visible_features = mask_progressive_hiding_features(
            steps[0].privileged_features or (), stage="visible"
        )
        self.assertTrue(all(value == 0.0 for value in visible_features[34:170]))
        curriculum_critic = PrivilegedCritic(hidden_size=16).to(policy.device)
        curriculum = train_progressive_hiding_critic(
            curriculum_critic,
            steps,
            schedule=(("oracle", 1), ("hide_wall", 1), ("visible", 1)),
            device=policy.device,
            batch_size=32,
            learning_rate=0.0001,
            seed=853,
        )
        self.assertEqual(len(curriculum), 3)
        visible_metrics = evaluate_privileged_critic(
            curriculum_critic,
            steps,
            device=policy.device,
            hiding_stage="visible",
        )
        self.assertGreater(visible_metrics["decisions"], 0)
        self.assertGreaterEqual(visible_metrics["mae"], 0.0)

    def test_progressive_hiding_visible_stage_is_invariant_to_hidden_world(self):
        import random

        from scripts.train_torch_ppo import (
            PRIVILEGED_CRITIC_FEATURE_DIM,
            progressive_hiding_features,
        )
        from xiamen_mahjong.game import XiamenMahjongGame
        from xiamen_mahjong.rules import XiamenRules
        from xiamen_mahjong.tiles import BASE_TILE_COUNT
        from xiamen_mahjong.training import _resample_private_world_for_actor

        game = XiamenMahjongGame(
            seed=911,
            rules=XiamenRules.from_profile("core"),
            dealer=0,
            auto_advance=False,
            human_seat=-1,
        )
        resampled = _resample_private_world_for_actor(
            game, actor_seat=0, rng=random.Random(91_101)
        )
        self.assertIsNotNone(resampled)
        assert resampled is not None
        oracle = progressive_hiding_features(game, 0, stage="oracle")
        visible = progressive_hiding_features(game, 0, stage="visible")
        self.assertEqual(len(oracle), PRIVILEGED_CRITIC_FEATURE_DIM)
        self.assertEqual(len(visible), PRIVILEGED_CRITIC_FEATURE_DIM)
        self.assertNotEqual(
            oracle,
            progressive_hiding_features(resampled, 0, stage="oracle"),
        )
        self.assertEqual(
            visible,
            progressive_hiding_features(resampled, 0, stage="visible"),
        )
        # Actor hand occupies the first segment. The three hidden opponent
        # hands and hidden wall composition immediately after it are zero.
        self.assertTrue(
            all(value == 0.0 for value in visible[BASE_TILE_COUNT : BASE_TILE_COUNT * 5])
        )
        with self.assertRaisesRegex(ValueError, "未知 progressive hiding"):
            progressive_hiding_features(game, 0, stage="invalid")


if __name__ == "__main__":
    unittest.main()

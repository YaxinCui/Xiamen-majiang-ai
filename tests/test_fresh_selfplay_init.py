import unittest


try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch 仅在项目 .venv 中安装")
class FreshSelfPlayInitializationTests(unittest.TestCase):
    def test_same_seed_creates_identical_new_actor_without_checkpoint_ancestry(self):
        from scripts.init_fresh_selfplay_policy import create_fresh_policy

        first = create_fresh_policy(
            seed=202611700, feature_version=3, hidden_size=16, device="cpu"
        )
        second = create_fresh_policy(
            seed=202611700, feature_version=3, hidden_size=16, device="cpu"
        )
        self.assertTrue(
            all(
                torch.equal(first.network.state_dict()[name], value)
                for name, value in second.network.state_dict().items()
            )
        )
        self.assertEqual(first.architecture, "candidate_mlp")
        self.assertEqual(first.action_selection, "policy")

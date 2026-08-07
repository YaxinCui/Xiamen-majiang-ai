import tempfile
import unittest
from pathlib import Path

import torch

from xiamen_mahjong.relative_advantage import (
    RelativeAdvantageAgent,
    RelativeAdvantageNetwork,
)


class RelativeAdvantageTests(unittest.TestCase):
    def test_teacher_action_is_exactly_zero_after_centering(self):
        network = RelativeAdvantageNetwork(feature_dim=4, hidden_size=8)
        candidates = torch.randn(2, 3, 4)
        mask = torch.tensor([[True, True, True], [True, True, False]])
        teacher = torch.tensor([1, 0])
        values = network(candidates, mask, teacher)
        self.assertEqual(float(values[0, 1].detach()), 0.0)
        self.assertEqual(float(values[1, 0].detach()), 0.0)
        self.assertEqual(float(values[1, 2].detach()), 0.0)

    def test_checkpoint_round_trip(self):
        agent = RelativeAdvantageAgent(feature_version=3, hidden_size=8, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "relative-advantage.pt"
            agent.save(path, metadata={"status": "test"})
            restored = RelativeAdvantageAgent.load(path, device="cpu")
        self.assertEqual(restored.feature_version, 3)
        self.assertEqual(restored.hidden_size, 8)


if __name__ == "__main__":
    unittest.main()

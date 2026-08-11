import unittest
from types import SimpleNamespace

from xiamen_mahjong.agents import ExactTwoDrawTenpaiReachTeacherAgent


class _ToyExactTwoDrawAgent(ExactTwoDrawTenpaiReachTeacherAgent):
    """Tiny state machine used to verify the probability arithmetic."""

    def _is_complete(self, game, player_id, hand):
        del game, player_id
        return 7 in hand

    def _regular_shanten(self, game, player_id, hand):
        del game, player_id
        return 0 if 2 in hand else 1

    def _allowed_efficiency_discards(self, game, hand, appeared_honors):
        del game, appeared_honors
        return sorted(set(hand))


class ExactTwoDrawTeacherTests(unittest.TestCase):
    def setUp(self):
        self.agent = _ToyExactTwoDrawAgent(
            score_margin=2.0,
            minimum_probability_advantage=0.05,
        )
        self.game = SimpleNamespace(gold_tile=33, gold_proxy_tile=None)

    def test_exact_probability_samples_two_faces_without_replacement(self):
        # Face 2 is the only successful face.  Drawing twice from three
        # single-copy faces reaches it with probability 2/3.
        counts = [0] * 34
        counts[1] = counts[2] = counts[3] = 1

        probability = self.agent._public_tenpai_reach_probability(
            self.game,
            0,
            [10],
            tuple(counts),
            frozenset(),
        )

        self.assertAlmostEqual(probability, 2.0 / 3.0)

    def test_immediate_win_precedes_any_forced_discard(self):
        counts = [0] * 34
        counts[7] = 1

        probability = self.agent._public_tenpai_reach_probability(
            self.game,
            0,
            [10],
            tuple(counts),
            frozenset(),
        )

        self.assertEqual(probability, 1.0)

    def test_solver_rejects_an_unregistered_horizon(self):
        counts = [0] * 34
        counts[1] = 1
        with self.assertRaisesRegex(ValueError, "horizon"):
            self.agent._public_tenpai_reach_probability(
                self.game,
                0,
                [10],
                tuple(counts),
                frozenset(),
                horizon=3,
            )


if __name__ == "__main__":
    unittest.main()

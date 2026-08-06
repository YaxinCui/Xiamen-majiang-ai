import math
import unittest

from xiamen_mahjong.belief import (
    SequentialParticleBelief,
    smoothed_deterministic_likelihood,
    smoothed_policy_likelihood,
)


class ParticleBeliefTests(unittest.TestCase):
    def test_matches_exact_posterior_in_a_small_enumerable_hidden_type_game(self):
        # Two hidden opponent styles are equally likely.  Their public action
        # likelihoods define an exact posterior that can be enumerated by hand.
        particles = ["aggressive"] * 20 + ["defensive"] * 20
        belief = SequentialParticleBelief(
            particles, seed=41, resample_ess_fraction=0.0
        )
        likelihoods = {
            ("aggressive", "claim"): 0.8,
            ("defensive", "claim"): 0.2,
            ("aggressive", "pass"): 0.1,
            ("defensive", "pass"): 0.9,
        }

        def replay(style, public_action, _rng):
            return style, likelihoods[(style, public_action)]

        belief.observe("claim", replay)
        self.assertAlmostEqual(
            belief.expectation(lambda style: style == "aggressive"), 0.8
        )
        diagnostics = belief.observe("pass", replay)
        # Exact: 0.5 * 0.8 * 0.1 / (0.5 * 0.8 * 0.1 + 0.5 * 0.2 * 0.9) = 4/13.
        self.assertAlmostEqual(
            belief.expectation(lambda style: style == "aggressive"), 4.0 / 13.0
        )
        self.assertEqual(diagnostics.observation_count, 2)
        self.assertEqual(diagnostics.resample_count, 0)
        self.assertGreater(diagnostics.effective_sample_size, 0.0)
        self.assertLess(diagnostics.effective_sample_size, diagnostics.particle_count)
        self.assertTrue(math.isfinite(diagnostics.log_evidence))

    def test_resampling_records_pre_resample_degeneracy_but_normalizes_next_state(self):
        belief = SequentialParticleBelief(
            [True] * 20 + [False] * 20,
            seed=43,
            resample_ess_fraction=0.9,
        )

        diagnostics = belief.observe(
            "observed",
            lambda hidden, _event, _rng: (hidden, 0.99 if hidden else 0.01),
        )
        self.assertEqual(diagnostics.resample_count, 1)
        self.assertLess(
            diagnostics.effective_sample_size,
            0.9 * diagnostics.particle_count,
        )
        self.assertEqual(sum(belief.weights), 1.0)
        self.assertTrue(
            all(weight == 1.0 / belief.particle_count for weight in belief.weights)
        )

    def test_impossible_observation_does_not_mutate_the_existing_belief(self):
        belief = SequentialParticleBelief(["a", "b"], seed=47)
        before_particles = belief.particles
        before_weights = belief.weights
        with self.assertRaisesRegex(ValueError, "所有 belief 粒子"):
            belief.observe("impossible", lambda particle, _event, _rng: (particle, 0.0))
        self.assertEqual(belief.particles, before_particles)
        self.assertEqual(belief.weights, before_weights)
        self.assertEqual(belief.diagnostics.observation_count, 0)

    def test_smoothed_behavior_likelihoods_keep_full_support(self):
        soft = smoothed_policy_likelihood(
            [2.0, 0.0], observed_index=1, temperature=1.0, uniform_mixture=0.1
        )
        deterministic = smoothed_deterministic_likelihood(
            selected_index=0,
            observed_index=1,
            action_count=2,
            uniform_mixture=0.1,
        )
        self.assertGreater(soft, 0.0)
        self.assertGreater(deterministic, 0.0)
        self.assertAlmostEqual(deterministic, 0.05)


if __name__ == "__main__":
    unittest.main()

import tempfile
from types import SimpleNamespace
import unittest
from collections import Counter
from itertools import permutations
import math
from pathlib import Path
import random

from xiamen_mahjong.agents import GameAction, HeuristicTeacherAgent
from xiamen_mahjong.game import XiamenMahjongGame
from xiamen_mahjong.rules import XiamenRules
from xiamen_mahjong.tiles import base_wall, gold_indicator_index
from xiamen_mahjong.training import (
    NeuralRulePolicyModel,
    PUBLIC_ACTION_SEQUENCE_DIM,
    RulePolicyModel,
    StateValueBaseline,
    _audit_constraint_repaired_history_prefix,
    _audit_initial_normal_draw_density,
    _audit_opening_initial_response_claim_density,
    _audit_opening_first_opponent_draw_density,
    _audit_initial_setup_response_claim_density,
    _audit_resampled_history_prefix,
    _audit_sequential_history_prefix,
    _frozen_behavior_action_likelihood,
    _history_replay_transition_targets,
    _initial_normal_draw_discard_constraint,
    _initial_setup_response_claim_constraint,
    _initial_setup_response_claim_discard_constraint,
    _linear_policy_discard_tile_factor_weights,
    _sample_multivariate_hand_given_required_tiles,
    _sample_weighted_multivariate_hand_given_required_tiles,
    _sample_latest_discard_conditioned_world,
    _sample_replay_setup_for_actor,
    _sample_replay_setup_given_opening_gold_and_draw,
    _sample_replay_setup_given_opening_and_initial_response_claim,
    _sample_replay_setup_given_opening_initial_claim_and_discard,
    _sample_post_gold_wall_given_indicator,
    _sample_post_gold_wall_given_indicator_and_opening_draw,
    _sample_structured_setup_given_normal_draw_flowers,
    _sample_structured_setup_given_gold_indicator_and_opening_draw,
    _sample_structured_setup_given_opening_gold_and_initial_hand,
    _sample_structured_setup_given_opening_gold_and_first_opponent_draw,
    _sample_wall_given_normal_draw_flowers,
    _replay_snapshot_public_history,
    _run_candidate_base_hand,
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
    split_heldout_trajectories_by_group,
    split_trajectories_by_group_folds,
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


class InterventionCollectorTests(unittest.TestCase):
    def test_teacher_base_is_an_explicit_no_checkpoint_behavior(self):
        from scripts.collect_candidate_teacher_dagger_trajectories import load_base_policy

        policy, identity = load_base_policy(
            SimpleNamespace(teacher_base=True, checkpoint=None, device="cpu")
        )

        self.assertIsInstance(policy, HeuristicTeacherAgent)
        self.assertEqual(identity["kind"], "heuristic_teacher")
        self.assertIsNone(identity["checkpoint"])
        self.assertIsNone(identity["inference_device"])


class _ScoreAwareFallbackPolicy:
    """Scores disagree with the deterministic fallback to test replay use."""

    def scores(self, decision):
        return [-5.0] * (len(decision.legal_actions) - 1) + [5.0]

    def choose_turn_action(self, game, player_id):
        return _turn_actions(game, player_id)[0]

    def choose_response(self, game, player_id, options):
        return tuple(options)[0]


class _KnownFirstLegalPolicy(_BatchFirstLegalPolicy):
    """Deterministic test policy that explicitly exposes its propensity."""

    def action_probability(self, game, player_id, legal, action, *, is_response):
        self.assert_action = action
        return 1.0


class TrainingTests(unittest.TestCase):
    def test_exact_initial_hand_constraint_uses_multivariate_density(self):
        # Four physical unknown tiles: 0, 0, 1, 1. A two-tile hand contains
        # tile 1 with probability 5/6 (only 00 fails). Conditional on that
        # public feasibility event, the 11 hand has probability 1/5.
        samples = 4_000
        double_one_hands = 0
        for index in range(samples):
            sampled = _sample_multivariate_hand_given_required_tiles(
                Counter({0: 2, 1: 2}),
                hand_size=2,
                required_tiles=(1,),
                rng=random.Random(41 + index),
            )
            self.assertIsNotNone(sampled)
            hand, condition_probability = sampled or ([], 0.0)
            self.assertGreaterEqual(hand.count(1), 1)
            self.assertAlmostEqual(condition_probability, 5.0 / 6.0)
            double_one_hands += hand.count(1) == 2
        self.assertAlmostEqual(double_one_hands / samples, 1.0 / 5.0, delta=0.03)

    def test_weighted_initial_hand_proposal_has_exact_partition_correction(self):
        # From [0, 0, 1, 1, 2], a two-card hand containing 1 has seven
        # labelled realizations.  Tile factors weight {0,1}, {1,1}, {1,2}
        # by 1, 1/4 and 3/2 respectively, so the exact partition is 29/4.
        # Importance weighting must recover the ordinary conditional
        # hypergeometric posterior rather than the deliberately tilted draw.
        pool = Counter({0: 2, 1: 2, 2: 1})
        samples = 12_000
        proposal_counts: Counter[tuple[int, ...]] = Counter()
        total_weight = 0.0
        double_one_weight = 0.0
        one_two_weight = 0.0
        for index in range(samples):
            sampled = _sample_weighted_multivariate_hand_given_required_tiles(
                pool,
                hand_size=2,
                required_tiles=(1,),
                tile_weights={0: 2.0, 1: 0.5, 2: 3.0},
                rng=random.Random(7_000 + index),
            )
            self.assertIsNotNone(sampled)
            hand, condition_probability, conditional_prior_over_proposal = (
                sampled or ([], 0.0, 0.0)
            )
            key = tuple(hand)
            proposal_counts[key] += 1
            self.assertAlmostEqual(condition_probability, 7.0 / 10.0)
            weight = condition_probability * conditional_prior_over_proposal
            total_weight += weight
            double_one_weight += weight * (key == (1, 1))
            one_two_weight += weight * (key == (1, 2))
        # q({0,1}), q({1,1}), q({1,2}) = 16/29, 1/29, 12/29.
        self.assertAlmostEqual(
            proposal_counts[(0, 1)] / samples, 16.0 / 29.0, delta=0.025
        )
        self.assertAlmostEqual(
            proposal_counts[(1, 1)] / samples, 1.0 / 29.0, delta=0.012
        )
        self.assertAlmostEqual(
            proposal_counts[(1, 2)] / samples, 12.0 / 29.0, delta=0.025
        )
        # The mean combined p/q is P(hand contains 1) = 7/10. Conditional
        # posterior masses are 1/7 and 2/7 for {1,1} and {1,2}.
        self.assertAlmostEqual(total_weight / samples, 7.0 / 10.0, delta=0.012)
        self.assertAlmostEqual(double_one_weight / total_weight, 1.0 / 7.0, delta=0.02)
        self.assertAlmostEqual(one_two_weight / total_weight, 2.0 / 7.0, delta=0.025)

    def test_linear_discard_energy_maps_only_hand_coefficients_to_factors(self):
        policy = RulePolicyModel()
        discard_tile = 7
        # These are the hand-count coefficients for a fixed discard action;
        # all public/action bias terms intentionally cancel from q(H).
        from xiamen_mahjong import training

        policy.weights[
            training._TARGET_HAND + discard_tile * 34 + 0
        ] = math.log(4.0)
        policy.weights[
            training._TARGET_HAND + discard_tile * 34 + 1
        ] = -math.log(4.0)
        factors = _linear_policy_discard_tile_factor_weights(
            policy,
            discard_tile=discard_tile,
            energy_scale=1.0,
            max_abs_log_factor=1.0,
        )
        self.assertAlmostEqual(factors[0], math.e)
        self.assertAlmostEqual(factors[1], 1.0 / math.e)
        self.assertEqual(factors[2], 1.0)
        with self.assertRaises(ValueError):
            _linear_policy_discard_tile_factor_weights(
                policy,
                discard_tile=34,
                energy_scale=1.0,
            )

    def test_normal_draw_flower_wall_proposal_has_exact_density(self):
        # In [0, 0, 1, 34], observing flower 34 and then a hidden base tile
        # has probability 1/4.  Conditional on that fact, base 0 has mass 2/3
        # and base 1 has mass 1/3; the remaining suffix is unconstrained.
        samples = 3_000
        base_counts: Counter[int] = Counter()
        for index in range(samples):
            sampled = _sample_wall_given_normal_draw_flowers(
                Counter({0: 2, 1: 1, 34: 1}),
                observed_flowers=(34,),
                rng=random.Random(8_000 + index),
            )
            self.assertIsNotNone(sampled)
            wall, condition_probability = sampled or ([], 0.0)
            self.assertEqual(wall[0], 34)
            self.assertIn(wall[1], {0, 1})
            self.assertAlmostEqual(condition_probability, 1.0 / 4.0)
            base_counts[wall[1]] += 1
        self.assertAlmostEqual(base_counts[0] / samples, 2.0 / 3.0, delta=0.035)
        self.assertAlmostEqual(base_counts[1] / samples, 1.0 / 3.0, delta=0.035)
        self.assertIsNone(
            _sample_wall_given_normal_draw_flowers(
                Counter({0: 2, 1: 1, 34: 1}),
                observed_flowers=(35,),
                rng=random.Random(1),
            )
        )

    def test_gold_indicator_wall_proposal_matches_enumerated_posterior(self):
        # Label the two physical 0s to enumerate the actual shuffle measure.
        # The selector starts at index 2, scans backward, then wraps.  Given
        # that it reveals face 0, the sampler must match the resulting
        # post-flip wall distribution, not merely remove an arbitrary 0.
        labeled_faces = (0, 0, 1, 34)
        enumerated_walls: list[tuple[int, ...]] = []
        for ordering in permutations(range(len(labeled_faces))):
            wall = [labeled_faces[index] for index in ordering]
            selected = gold_indicator_index(wall, (1, 1))
            if selected is not None and wall[selected] == 0:
                enumerated_walls.append(tuple([*wall[:selected], *wall[selected + 1 :]]))
        self.assertTrue(enumerated_walls)
        expected_first_flower = sum(wall[0] == 34 for wall in enumerated_walls) / len(
            enumerated_walls
        )
        samples = 4_000
        observed_first_flower = 0
        for index in range(samples):
            sampled = _sample_post_gold_wall_given_indicator(
                Counter({0: 2, 1: 1, 34: 1}),
                indicator=0,
                dice=(1, 1),
                rng=random.Random(20_000 + index),
            )
            self.assertIsNotNone(sampled)
            wall, condition_probability = sampled or ([], 0.0)
            self.assertEqual(Counter([*wall, 0]), Counter({0: 2, 1: 1, 34: 1}))
            self.assertAlmostEqual(condition_probability, 2.0 / 3.0)
            observed_first_flower += wall[0] == 34
        self.assertAlmostEqual(
            observed_first_flower / samples,
            expected_first_flower,
            delta=0.03,
        )

    def test_gold_indicator_and_opening_draw_proposal_has_exact_density(self):
        # The pre-flip wall begins with the dealer's observed [34, 1] draw.
        # With dice 1+1, the flip scan is guaranteed to remain after this
        # prefix. The joint condition is 1/6 * 1/5 * 2/3 = 1/45.
        pool = Counter({0: 2, 1: 1, 2: 1, 34: 1, 35: 1})
        for index in range(500):
            sampled = _sample_post_gold_wall_given_indicator_and_opening_draw(
                pool,
                indicator=0,
                dice=(1, 1),
                opening_flowers=(34,),
                opening_tile=1,
                rng=random.Random(25_000 + index),
            )
            self.assertIsNotNone(sampled)
            wall, condition_probability = sampled or ([], 0.0)
            self.assertEqual(wall[:2], [34, 1])
            self.assertEqual(Counter([*wall, 0]), pool)
            self.assertAlmostEqual(condition_probability, 1.0 / 45.0)

        # Indicator and opening tile may share a face while using different
        # physical copies; rejecting that case would distort real gold draws.
        same_face = _sample_post_gold_wall_given_indicator_and_opening_draw(
            Counter({0: 3, 1: 1, 34: 1, 35: 1}),
            indicator=0,
            dice=(1, 1),
            opening_flowers=(34,),
            opening_tile=0,
            rng=random.Random(25_999),
        )
        self.assertIsNotNone(same_face)
        same_wall, same_probability = same_face or ([], 0.0)
        self.assertEqual(same_wall[:2], [34, 0])
        self.assertAlmostEqual(same_probability, 1.0 / 15.0)

    def test_structured_gold_and_opening_draw_proposal_has_exact_density(self):
        # One concealed base-hand slot and a five-tile pre-flip wall share
        # [0, 0, 1, 2, 34, 35]. The known opening prefix [34, 1] has
        # probability 3/80, and face 0 is then the indicator with probability
        # 2/3, for joint p/q = 1/40. The enumerated labelled deck agrees.
        pool = Counter({0: 2, 1: 1, 2: 1, 34: 1, 35: 1})
        accepted = 0
        valid = 0
        labeled_faces = (0, 0, 1, 2, 34, 35)
        for ordering in permutations(range(len(labeled_faces))):
            hand = labeled_faces[ordering[0]]
            wall = [labeled_faces[index] for index in ordering[1:]]
            if hand >= 34:
                continue
            valid += 1
            selected = gold_indicator_index(wall, (1, 1))
            if (
                wall[:2] == [34, 1]
                and selected is not None
                and wall[selected] == 0
            ):
                accepted += 1
        self.assertEqual(accepted / valid, 1.0 / 40.0)
        for index in range(500):
            sampled = _sample_structured_setup_given_gold_indicator_and_opening_draw(
                pool,
                opponent_hand_sizes=(1,),
                opponent_flower_sizes=(0,),
                pre_flip_wall_size=5,
                indicator=0,
                dice=(1, 1),
                opening_flowers=(34,),
                opening_tile=1,
                rng=random.Random(30_000 + index),
            )
            self.assertIsNotNone(sampled)
            proposal = sampled
            assert proposal is not None
            self.assertEqual(proposal.wall[:2], (34, 1))
            self.assertEqual(Counter([*proposal.wall, 0, *proposal.opponent_hands[0]]), pool)
            self.assertAlmostEqual(proposal.condition_probability, 1.0 / 40.0)

    def test_opening_and_first_opponent_draw_proposal_matches_exact_posterior(self):
        # One opponent base-hand slot and six pre-flip wall slots share the
        # labelled multiset [0, 0, 1, 2, 3, 34, 35]. We observe dealer
        # [34, 1], opponent [35, hidden base], a following discard of 2, and
        # indicator face 0. Enumeration gives a prior event probability 1/450.
        #
        # This deliberately checks more than the event normalizer.  The
        # sampler is allowed to give different valid hidden allocations
        # different p/q values, so a correct proposal must recover the exact
        # posterior *after importance weighting*.  A mere legality test (or a
        # test of an average condition probability) would not catch a missing
        # branch factor in the joint gold/draw/discard proposal.
        pool = Counter({0: 2, 1: 1, 2: 1, 3: 1, 34: 1, 35: 1})
        labeled_faces = (0, 0, 1, 2, 3, 34, 35)
        valid = 0
        event_count = 0
        posterior_hand_is_two = 0
        posterior_post_gold_wall_ends_in_three = 0
        for ordering in permutations(range(len(labeled_faces))):
            hand = labeled_faces[ordering[0]]
            wall = [labeled_faces[index] for index in ordering[1:]]
            if hand >= 34:
                continue
            valid += 1
            selected = gold_indicator_index(wall, (1, 1))
            if (
                wall[:3] == [34, 1, 35]
                and wall[3] < 34
                and 2 in {hand, wall[3]}
                and selected is not None
                and wall[selected] == 0
            ):
                event_count += 1
                posterior_hand_is_two += hand == 2
                post_gold_wall = [*wall[:selected], *wall[selected + 1 :]]
                posterior_post_gold_wall_ends_in_three += post_gold_wall[-1] == 3
        self.assertEqual(event_count / valid, 1.0 / 450.0)
        self.assertEqual(posterior_hand_is_two / event_count, 1.0 / 2.0)
        self.assertEqual(
            posterior_post_gold_wall_ends_in_three / event_count,
            1.0 / 2.0,
        )
        total_weight = 0.0
        hand_is_two_weight = 0.0
        post_gold_wall_ends_in_three_weight = 0.0
        samples = 8_000
        for index in range(samples):
            sampled = _sample_structured_setup_given_opening_gold_and_first_opponent_draw(
                pool,
                opponent_hand_sizes=(1,),
                opponent_flower_sizes=(0,),
                pre_flip_wall_size=6,
                indicator=0,
                dice=(1, 1),
                opening_flowers=(34,),
                opening_tile=1,
                opponent_draw_flowers=(35,),
                drawer_index=0,
                required_drawer_tile=2,
                rng=random.Random(40_000 + index),
            )
            self.assertIsNotNone(sampled)
            proposal = sampled
            assert proposal is not None
            self.assertEqual(proposal.wall[:3], (34, 1, 35))
            self.assertLess(proposal.wall[3], 34)
            self.assertIn(2, {proposal.opponent_hands[0][0], proposal.wall[3]})
            self.assertEqual(
                Counter([*proposal.wall, 0, *proposal.opponent_hands[0]]), pool
            )
            weight = proposal.condition_probability
            total_weight += weight
            hand_is_two_weight += weight * (proposal.opponent_hands[0][0] == 2)
            post_gold_wall_ends_in_three_weight += weight * (proposal.wall[-1] == 3)
        self.assertAlmostEqual(total_weight / samples, 1.0 / 450.0, delta=0.00008)
        self.assertAlmostEqual(
            hand_is_two_weight / total_weight,
            posterior_hand_is_two / event_count,
            delta=0.035,
        )
        self.assertAlmostEqual(
            post_gold_wall_ends_in_three_weight / total_weight,
            posterior_post_gold_wall_ends_in_three / event_count,
            delta=0.035,
        )

    def test_opening_gold_and_initial_claim_hand_matches_exact_posterior(self):
        # The same physical micro-deck has one opponent opening hand slot and
        # six pre-flip wall slots.  After observing [34, 1], require that the
        # opening claimant owns tile 2 (a pong/chi consumed-tile constraint)
        # and that the dice scan exposes gold face 0.  The sampler returns a
        # post-gold wall, so enumerate against that exact representation.
        pool = Counter({0: 2, 1: 1, 2: 1, 3: 1, 34: 1, 35: 1})
        labeled_faces = (0, 0, 1, 2, 3, 34, 35)
        valid = 0
        event_count = 0
        posterior_first_tail_is_zero = 0
        posterior_last_is_three = 0
        for ordering in permutations(range(len(labeled_faces))):
            hand = labeled_faces[ordering[0]]
            wall = [labeled_faces[index] for index in ordering[1:]]
            if hand >= 34:
                continue
            valid += 1
            selected = gold_indicator_index(wall, (1, 1))
            if (
                wall[:2] == [34, 1]
                and hand == 2
                and selected is not None
                and wall[selected] == 0
            ):
                event_count += 1
                post_gold_wall = [*wall[:selected], *wall[selected + 1 :]]
                posterior_first_tail_is_zero += post_gold_wall[2] == 0
                posterior_last_is_three += post_gold_wall[-1] == 3
        self.assertEqual(event_count / valid, 1.0 / 225.0)
        self.assertEqual(posterior_first_tail_is_zero / event_count, 3.0 / 8.0)
        self.assertEqual(posterior_last_is_three / event_count, 3.0 / 8.0)

        total_weight = 0.0
        first_tail_is_zero_weight = 0.0
        last_is_three_weight = 0.0
        samples = 6_000
        for index in range(samples):
            sampled = _sample_structured_setup_given_opening_gold_and_initial_hand(
                pool,
                opponent_hand_sizes=(1,),
                opponent_flower_sizes=(0,),
                pre_flip_wall_size=6,
                indicator=0,
                dice=(1, 1),
                opening_flowers=(34,),
                opening_tile=1,
                claimant_index=0,
                required_claim_tiles=(2,),
                rng=random.Random(45_000 + index),
            )
            self.assertIsNotNone(sampled)
            proposal = sampled
            assert proposal is not None
            self.assertEqual(proposal.opponent_hands, ((2,),))
            self.assertEqual(proposal.wall[:2], (34, 1))
            self.assertEqual(
                Counter([*proposal.wall, 0, *proposal.opponent_hands[0]]), pool
            )
            weight = proposal.condition_probability
            total_weight += weight
            first_tail_is_zero_weight += weight * (proposal.wall[2] == 0)
            last_is_three_weight += weight * (proposal.wall[-1] == 3)
        self.assertAlmostEqual(total_weight / samples, 1.0 / 225.0, delta=0.00008)
        self.assertAlmostEqual(
            first_tail_is_zero_weight / total_weight,
            posterior_first_tail_is_zero / event_count,
            delta=0.035,
        )
        self.assertAlmostEqual(
            last_is_three_weight / total_weight,
            posterior_last_is_three / event_count,
            delta=0.035,
        )

    def test_opening_gold_and_claim_discard_initial_hand_matches_exact_posterior(self):
        # A claim immediately followed by the claimant's discard constrains
        # both initial-hand faces in this two-slot micro-deck.  This checks
        # the joint initial-hand event independently of action likelihoods;
        # it is the structural basis for the longer claim → discard audit.
        pool = Counter({0: 2, 1: 1, 2: 1, 3: 1, 4: 1, 34: 1, 35: 1})
        labeled_faces = (0, 0, 1, 2, 3, 4, 34, 35)
        valid = 0
        event_count = 0
        posterior_first_tail_is_zero = 0
        posterior_last_is_four = 0
        for ordering in permutations(range(len(labeled_faces))):
            hand = [labeled_faces[ordering[0]], labeled_faces[ordering[1]]]
            wall = [labeled_faces[index] for index in ordering[2:]]
            if any(tile >= 34 for tile in hand):
                continue
            valid += 1
            selected = gold_indicator_index(wall, (1, 1))
            if (
                wall[:2] == [34, 1]
                and 2 in hand
                and 3 in hand
                and selected is not None
                and wall[selected] == 0
            ):
                event_count += 1
                post_gold_wall = [*wall[:selected], *wall[selected + 1 :]]
                posterior_first_tail_is_zero += post_gold_wall[2] == 0
                posterior_last_is_four += post_gold_wall[-1] == 4
        self.assertEqual(event_count / valid, 1.0 / 675.0)
        self.assertEqual(posterior_first_tail_is_zero / event_count, 3.0 / 8.0)
        self.assertEqual(posterior_last_is_four / event_count, 3.0 / 8.0)

        total_weight = 0.0
        first_tail_is_zero_weight = 0.0
        last_is_four_weight = 0.0
        samples = 6_000
        for index in range(samples):
            sampled = _sample_structured_setup_given_opening_gold_and_initial_hand(
                pool,
                opponent_hand_sizes=(2,),
                opponent_flower_sizes=(0,),
                pre_flip_wall_size=6,
                indicator=0,
                dice=(1, 1),
                opening_flowers=(34,),
                opening_tile=1,
                claimant_index=0,
                required_claim_tiles=(2, 3),
                rng=random.Random(46_000 + index),
            )
            self.assertIsNotNone(sampled)
            proposal = sampled
            assert proposal is not None
            self.assertEqual(proposal.opponent_hands, ((2, 3),))
            self.assertEqual(proposal.wall[:2], (34, 1))
            self.assertEqual(
                Counter([*proposal.wall, 0, *proposal.opponent_hands[0]]), pool
            )
            weight = proposal.condition_probability
            total_weight += weight
            first_tail_is_zero_weight += weight * (proposal.wall[2] == 0)
            last_is_four_weight += weight * (proposal.wall[-1] == 4)
        self.assertAlmostEqual(total_weight / samples, 1.0 / 675.0, delta=0.00004)
        self.assertAlmostEqual(
            first_tail_is_zero_weight / total_weight,
            posterior_first_tail_is_zero / event_count,
            delta=0.035,
        )
        self.assertAlmostEqual(
            last_is_four_weight / total_weight,
            posterior_last_is_four / event_count,
            delta=0.035,
        )

    def test_opening_gold_setup_reconstruction_preserves_actor_information(self):
        teacher = HeuristicTeacherAgent()
        for seed in (271, 32):
            game = XiamenMahjongGame(
                seed=seed,
                rules=XiamenRules.from_profile("core"),
                dealer=0,
                auto_advance=False,
                human_seat=-1,
            )
            snapshots = _run_candidate_base_hand(
                game,
                candidate_seat=0,
                candidate_policy=teacher,
                opponents={
                    seat: ("heuristic_teacher", teacher) for seat in range(1, 4)
                },
            )
            snapshot = next(
                item
                for item in snapshots
                if not any(
                    draw.after_public_action_count > len(item.initial_game.public_actions)
                    for draw in item.actor_trace.draws
                )
            )
            proposal = _sample_replay_setup_given_opening_gold_and_draw(
                snapshot,
                rng=random.Random(31_001 + seed),
            )
            self.assertIsNotNone(proposal)
            assert proposal is not None
            sampled = proposal.game
            source = snapshot.initial_game
            self.assertEqual(sampled.gold_indicator, source.gold_indicator)
            self.assertEqual(sampled.gold_dice, source.gold_dice)
            self.assertEqual(sampled.public_actions, source.public_actions)
            self.assertEqual(sampled.players[0].hand, source.players[0].hand)
            self.assertEqual(sampled.players[0].flowers, source.players[0].flowers)
            self.assertEqual(len(sampled.wall), len(source.wall))
            self.assertGreater(proposal.condition_probability, 0.0)
            reconstructed = Counter(sampled.wall)
            reconstructed.update([sampled.gold_indicator])
            for player in sampled.players:
                reconstructed.update(player.hand)
                reconstructed.update(player.flowers)
            self.assertEqual(reconstructed, Counter(base_wall()))

    def test_joint_setup_and_normal_draw_flower_proposal_has_exact_density(self):
        # There is one hidden base-hand slot, one hidden flower slot and a
        # three-tile wall.  Among [0, 0, 1, 34, 35], the wall has two bases
        # and one flower.  Observing [34] then a base has exact probability
        # (1/3 wall flower) * (1/2 flower identity) * (2/2 next base) = 1/6.
        samples = 3_000
        base_counts: Counter[int] = Counter()
        for index in range(samples):
            sampled = _sample_structured_setup_given_normal_draw_flowers(
                Counter({0: 2, 1: 1, 34: 1, 35: 1}),
                opponent_hand_sizes=(1,),
                opponent_flower_sizes=(1,),
                wall_size=3,
                observed_flowers=(34,),
                rng=random.Random(12_000 + index),
            )
            self.assertIsNotNone(sampled)
            proposal = sampled
            assert proposal is not None
            self.assertEqual(proposal.wall[0], 34)
            self.assertIn(proposal.wall[1], {0, 1})
            self.assertEqual(proposal.opponent_flowers, ((35,),))
            self.assertAlmostEqual(proposal.condition_probability, 1.0 / 6.0)
            base_counts[proposal.wall[1]] += 1
        self.assertAlmostEqual(base_counts[0] / samples, 2.0 / 3.0, delta=0.035)
        self.assertAlmostEqual(base_counts[1] / samples, 1.0 / 3.0, delta=0.035)

    def test_joint_draw_discard_feasibility_has_exact_density(self):
        # Reuse the preceding micro-deck but require the drawer's one-card
        # hand plus its hidden base draw to contain tile 1.  That base group
        # contains 1 with probability 2/3, so the complete condition has
        # probability (1/6) * (2/3) = 1/9.  Every proposal must consequently
        # make the public discard face legal after the hidden draw.
        for index in range(1_000):
            sampled = _sample_structured_setup_given_normal_draw_flowers(
                Counter({0: 2, 1: 1, 34: 1, 35: 1}),
                opponent_hand_sizes=(1,),
                opponent_flower_sizes=(1,),
                wall_size=3,
                observed_flowers=(34,),
                drawer_index=0,
                required_drawer_tile=1,
                rng=random.Random(16_000 + index),
            )
            self.assertIsNotNone(sampled)
            proposal = sampled
            assert proposal is not None
            self.assertAlmostEqual(proposal.condition_probability, 1.0 / 9.0)
            self.assertIn(1, [*proposal.opponent_hands[0], proposal.wall[1]])

    def test_initial_normal_draw_density_audit_replays_public_flowers(self):
        teacher = HeuristicTeacherAgent()
        game = XiamenMahjongGame(
            seed=271,
            rules=XiamenRules.from_profile("core"),
            dealer=0,
            auto_advance=False,
            human_seat=-1,
        )
        snapshots = _run_candidate_base_hand(
            game,
            candidate_seat=0,
            candidate_policy=teacher,
            opponents={seat: ("heuristic_teacher", teacher) for seat in range(1, 4)},
        )
        snapshot = next(
            item
            for item in snapshots
            if (_initial_normal_draw_discard_constraint(item) or (None, None, (), None, None))[2]
        )
        audit = _audit_initial_normal_draw_density(
            snapshot,
            opponents={seat: ("heuristic_teacher", teacher) for seat in range(1, 4)},
            particle_count=64,
            rng=random.Random(1_917),
        )
        self.assertEqual(audit.initialized_particles, 64)
        self.assertGreater(audit.accepted_particles, 0)
        self.assertGreater(audit.condition_probability or 0.0, 0.0)
        self.assertLess(audit.condition_probability or 1.0, 1.0)
        self.assertGreater(audit.effective_sample_size, 0.0)
        payload = audit.payload()
        self.assertEqual(
            payload["proposal"], "core_initial_normal_draw_post_setup_density_v0"
        )
        self.assertIn("prior_over_proposal", payload["proposal_density_ratio"])
        self.assertIn("not authorized", payload["warning"])

    def test_opening_aware_normal_draw_audit_reports_particle_density_range(self):
        teacher = HeuristicTeacherAgent()
        game = XiamenMahjongGame(
            seed=271,
            rules=XiamenRules.from_profile("core"),
            dealer=0,
            auto_advance=False,
            human_seat=-1,
        )
        opponents = {seat: ("heuristic_teacher", teacher) for seat in range(1, 4)}
        snapshot = next(
            item
            for item in _run_candidate_base_hand(
                game,
                candidate_seat=0,
                candidate_policy=teacher,
                opponents=opponents,
            )
            if _initial_normal_draw_discard_constraint(item) is not None
        )
        audit = _audit_opening_first_opponent_draw_density(
            snapshot,
            opponents=opponents,
            particle_count=64,
            rng=random.Random(41_917),
        )
        self.assertEqual(audit.initialized_particles, 64)
        self.assertEqual(audit.accepted_particles, 64)
        self.assertIsNotNone(audit.minimum_prior_over_proposal)
        self.assertIsNotNone(audit.maximum_prior_over_proposal)
        self.assertLess(
            audit.minimum_prior_over_proposal or 0.0,
            audit.maximum_prior_over_proposal or 0.0,
        )
        self.assertGreater(audit.structural_effective_sample_size, 0.0)
        self.assertLessEqual(
            audit.structural_effective_sample_size,
            audit.accepted_particles,
        )
        self.assertGreater(audit.effective_sample_size, 0.0)
        payload = audit.payload()
        self.assertEqual(
            payload["proposal"], "core_opening_first_opponent_draw_density_v0"
        )
        self.assertEqual(
            payload["proposal_density_ratio"], "per_particle_prior_over_proposal"
        )
        self.assertIn("structural_effective_sample_size", payload)
        self.assertIn("behavior likelihood", payload["effective_sample_size_definition"])
        self.assertIn("outside", payload["warning"])

    def test_exact_density_audit_covers_only_initial_response_claims(self):
        teacher = HeuristicTeacherAgent()
        game = XiamenMahjongGame(
            seed=2,
            rules=XiamenRules.from_profile("core"),
            auto_advance=False,
            human_seat=-1,
        )
        snapshots = _run_candidate_base_hand(
            game,
            candidate_seat=0,
            candidate_policy=teacher,
            opponents={seat: ("heuristic_teacher", teacher) for seat in range(1, 4)},
        )
        snapshot = next(
            item
            for item in snapshots
            if _initial_setup_response_claim_constraint(item) is not None
        )
        target_public_action_count, _claimant, _required_tiles = (
            _initial_setup_response_claim_constraint(snapshot) or (0, 0, ())
        )
        actor_hand_before = tuple(snapshot.initial_game.players[0].hand)
        wall_before = tuple(snapshot.initial_game.wall)
        audit = _audit_initial_setup_response_claim_density(
            snapshot,
            opponents={seat: ("heuristic_teacher", teacher) for seat in range(1, 4)},
            particle_count=64,
            rng=random.Random(917),
        )
        self.assertEqual(audit.initialized_particles, 64)
        self.assertEqual(audit.accepted_particles, 64)
        self.assertGreater(audit.condition_probability or 0.0, 0.0)
        self.assertLess(audit.condition_probability or 1.0, 1.0)
        self.assertGreater(audit.effective_sample_size, 0.0)
        baseline_accepts = 0
        baseline_rng = random.Random(917)
        for _ in range(64):
            setup = _sample_replay_setup_for_actor(snapshot, rng=baseline_rng)
            self.assertIsNotNone(setup)
            result = _replay_snapshot_public_history(
                snapshot,
                opponents={seat: ("heuristic_teacher", teacher) for seat in range(1, 4)},
                initial_game=setup,
                condition_actor_draws=True,
                target_public_action_count=target_public_action_count,
            )
            baseline_accepts += result.accepted
        self.assertLess(baseline_accepts, audit.accepted_particles)
        self.assertEqual(tuple(snapshot.initial_game.players[0].hand), actor_hand_before)
        self.assertEqual(tuple(snapshot.initial_game.wall), wall_before)
        payload = audit.payload()
        self.assertEqual(
            payload["proposal"], "core_initial_response_claim_exact_density_v0"
        )
        self.assertIn("prior_over_proposal", payload["proposal_density_ratio"])
        self.assertIn("not authorized", payload["warning"])
        self.assertNotIn("wall", payload)
        self.assertNotIn("opponent_hands", payload)

    def test_opening_aware_initial_claim_audit_has_exact_density_diagnostics(self):
        teacher = HeuristicTeacherAgent()
        game = XiamenMahjongGame(
            seed=2,
            rules=XiamenRules.from_profile("core"),
            auto_advance=False,
            human_seat=-1,
        )
        opponents = {seat: ("heuristic_teacher", teacher) for seat in range(1, 4)}
        snapshot = next(
            item
            for item in _run_candidate_base_hand(
                game,
                candidate_seat=0,
                candidate_policy=teacher,
                opponents=opponents,
            )
            if _initial_setup_response_claim_constraint(item) is not None
        )
        prepared = _sample_replay_setup_given_opening_and_initial_response_claim(
            snapshot,
            rng=random.Random(51_007),
        )
        self.assertIsNotNone(prepared)
        proposal, target_public_action_count = prepared or (None, 0)
        assert proposal is not None
        self.assertEqual(
            target_public_action_count,
            (_initial_setup_response_claim_constraint(snapshot) or (0, 0, ()))[0],
        )
        reconstructed = Counter(proposal.game.wall)
        reconstructed.update([proposal.game.gold_indicator])
        for player in proposal.game.players:
            reconstructed.update(player.hand)
            reconstructed.update(player.flowers)
        self.assertEqual(reconstructed, Counter(base_wall()))

        audit = _audit_opening_initial_response_claim_density(
            snapshot,
            opponents=opponents,
            particle_count=64,
            rng=random.Random(51_017),
        )
        self.assertEqual(audit.initialized_particles, 64)
        self.assertEqual(audit.accepted_particles, 64)
        self.assertGreater(audit.structural_effective_sample_size, 0.0)
        self.assertLessEqual(
            audit.structural_effective_sample_size,
            audit.accepted_particles,
        )
        self.assertGreater(audit.effective_sample_size, 0.0)
        payload = audit.payload()
        self.assertEqual(
            payload["proposal"], "core_opening_initial_response_claim_density_v0"
        )
        self.assertEqual(
            payload["proposal_density_ratio"], "per_particle_prior_over_proposal"
        )
        self.assertNotIn("wall", payload)
        self.assertNotIn("opponent_hands", payload)
        self.assertIn("outside", payload["warning"])

    def test_opening_aware_claim_discard_audit_stays_a_gated_prefix(self):
        teacher = HeuristicTeacherAgent()
        game = XiamenMahjongGame(
            seed=67,
            rules=XiamenRules.from_profile("core"),
            auto_advance=False,
            human_seat=-1,
        )
        opponents = {seat: ("heuristic_teacher", teacher) for seat in range(1, 4)}
        snapshot = next(
            item
            for item in _run_candidate_base_hand(
                game,
                candidate_seat=0,
                candidate_policy=teacher,
                opponents=opponents,
            )
            if _initial_setup_response_claim_discard_constraint(item) is not None
        )
        prepared = _sample_replay_setup_given_opening_initial_claim_and_discard(
            snapshot,
            rng=random.Random(51_103),
        )
        self.assertIsNotNone(prepared)
        proposal, target_public_action_count = prepared or (None, 0)
        assert proposal is not None
        self.assertEqual(
            target_public_action_count,
            (_initial_setup_response_claim_discard_constraint(snapshot) or (0, 0, ()))[0],
        )
        audit = _audit_opening_initial_response_claim_density(
            snapshot,
            opponents=opponents,
            particle_count=64,
            rng=random.Random(51_113),
            include_claimant_discard=True,
        )
        self.assertGreater(audit.initialized_particles, 0)
        self.assertGreater(audit.accepted_particles, 0)
        self.assertLessEqual(audit.accepted_particles, audit.initialized_particles)
        payload = audit.payload()
        self.assertEqual(
            payload["proposal"],
            "core_opening_initial_response_claim_discard_density_v0",
        )
        self.assertIn("following_claim_discard", payload["proposal_density"])
        self.assertIn("outside", payload["warning"])

        # Seed 2 has the same early claim shape but no actor response to the
        # claimant's discard.  The engine would automatically draw next, so
        # the three-action truncation must be rejected instead of emitting a
        # spurious public_event_mismatch.
        automatic_draw_game = XiamenMahjongGame(
            seed=2,
            rules=XiamenRules.from_profile("core"),
            auto_advance=False,
            human_seat=-1,
        )
        automatic_draw_snapshots = _run_candidate_base_hand(
            automatic_draw_game,
            candidate_seat=0,
            candidate_policy=teacher,
            opponents=opponents,
        )
        self.assertTrue(
            any(
                _initial_setup_response_claim_constraint(item) is not None
                for item in automatic_draw_snapshots
            )
        )
        self.assertFalse(
            any(
                _initial_setup_response_claim_discard_constraint(item) is not None
                for item in automatic_draw_snapshots
            )
        )

    def test_resampled_history_rejects_positioned_opponent_flower_transition_without_density(self):
        teacher = HeuristicTeacherAgent()
        game = XiamenMahjongGame(
            seed=2,
            rules=XiamenRules.from_profile("core"),
            auto_advance=False,
            human_seat=-1,
        )
        snapshots = _run_candidate_base_hand(
            game,
            candidate_seat=0,
            candidate_policy=teacher,
            opponents={seat: ("heuristic_teacher", teacher) for seat in range(1, 4)},
        )
        snapshot = next(
            item
            for item in snapshots
            if any(
                len(item.game.players[seat].flowers)
                != len(item.initial_game.players[seat].flowers)
                for seat in range(1, 4)
            )
        )
        observed_flowers = [
            event
            for event in snapshot.game.public_actions
            if event.get("kind") == "draw"
            and event.get("seat") in {1, 2, 3}
            and event.get("tiles")
        ]
        self.assertTrue(observed_flowers)
        self.assertTrue(
            all(
                all(isinstance(tile, int) and tile >= 34 for tile in event["tiles"])
                for event in observed_flowers
            )
        )
        proposal = _sample_replay_setup_for_actor(snapshot, rng=random.Random(929))
        self.assertIsNotNone(proposal)
        replay = _replay_snapshot_public_history(
            snapshot,
            opponents={seat: ("heuristic_teacher", teacher) for seat in range(1, 4)},
            initial_game=proposal,
            condition_actor_draws=True,
        )
        self.assertFalse(replay.accepted)
        self.assertEqual(
            replay.rejection_reason, "opponent_flower_transition_unsupported"
        )

    def test_history_replay_prefers_available_policy_scores_over_argmax_fallback(self):
        game = XiamenMahjongGame(
            seed=929,
            rules=XiamenRules.from_profile("core"),
            dealer=0,
            auto_advance=False,
            human_seat=-1,
        )
        player_id = game.current_player
        legal = tuple(_turn_actions(game, player_id))
        self.assertGreater(len(legal), 1)
        action, likelihood = _frozen_behavior_action_likelihood(
            _ScoreAwareFallbackPolicy(),
            game,
            player_id=player_id,
            legal=legal,
            compatible=(legal[-1],),
            is_response=False,
            temperature=1.0,
            uniform_mixture=0.02,
        ) or (None, 0.0)
        # ``choose_turn_action`` returns the first legal action, so this only
        # holds if replay uses the exposed logits as its behavior model.
        self.assertEqual(action, legal[-1])
        self.assertGreater(likelihood, 0.9)
        self.assertIsNone(
            _frozen_behavior_action_likelihood(
                _ScoreAwareFallbackPolicy(),
                game,
                player_id=player_id,
                legal=legal,
                compatible=(GameAction("discard", 999),),
                is_response=False,
                temperature=1.0,
                uniform_mixture=0.02,
            )
        )

    def test_latest_discard_sir_is_publicly_reconstructible_and_safe(self):
        teacher = HeuristicTeacherAgent()
        candidate_seat = 0
        opponents = {
            seat: ("heuristic_teacher", teacher)
            for seat in range(4)
            if seat != candidate_seat
        }
        game = XiamenMahjongGame(
            seed=930,
            rules=XiamenRules.from_profile("core"),
            auto_advance=False,
            human_seat=-1,
        )
        snapshots = _run_candidate_base_hand(
            game,
            candidate_seat=candidate_seat,
            candidate_policy=teacher,
            opponents=opponents,
        )
        snapshot = next(
            item
            for item in snapshots
            if item.game.phase == "response"
            and len(item.game.public_actions) >= 2
            and item.game.public_actions[-1]["kind"] == "discard"
            and item.game.public_actions[-2]["kind"] == "draw"
            and item.game.public_actions[-1].get("seat")
            == item.game.public_actions[-2].get("seat")
        )
        world, diagnostics = _sample_latest_discard_conditioned_world(
            snapshot.game,
            actor_seat=candidate_seat,
            expected_legal_actions=snapshot.legal_actions,
            opponents=opponents,
            particle_count=32,
            rng=random.Random(932),
        )
        self.assertIsNotNone(world)
        self.assertEqual(diagnostics.proposed_particles, 32)
        self.assertEqual(diagnostics.consistent_particles, 32)
        self.assertGreater(diagnostics.effective_sample_size, 0.0)
        self.assertLessEqual(
            diagnostics.effective_sample_size, diagnostics.consistent_particles
        )
        self.assertEqual(
            tuple(world.response_options[candidate_seat]), snapshot.legal_actions
        )
        payload = diagnostics.payload()
        self.assertNotIn("wall", payload)
        self.assertNotIn("opponent_hands", payload)

    def test_classic_latest_discard_sir_rejects_special_prefixes_but_keeps_safe_one(self):
        teacher = HeuristicTeacherAgent()
        candidate_seat = 0
        opponents = {
            seat: ("heuristic_teacher", teacher)
            for seat in range(4)
            if seat != candidate_seat
        }
        game = XiamenMahjongGame(
            seed=930,
            rules=XiamenRules.from_profile("classic"),
            auto_advance=False,
            human_seat=-1,
        )
        snapshots = _run_candidate_base_hand(
            game,
            candidate_seat=candidate_seat,
            candidate_policy=teacher,
            opponents=opponents,
        )
        snapshot = next(
            item
            for item in snapshots
            if item.game.phase == "response"
            and item.game.turn_count > item.game.rules.player_count
            and not item.game.tour_state
            and not item.game.opening_wait_seats
            and len(item.game.public_actions) >= 2
            and item.game.public_actions[-1]["kind"] == "discard"
            and item.game.public_actions[-2]["kind"] == "draw"
            and item.game.public_actions[-1].get("seat")
            == item.game.public_actions[-2].get("seat")
        )
        world, diagnostics = _sample_latest_discard_conditioned_world(
            snapshot.game,
            actor_seat=candidate_seat,
            expected_legal_actions=snapshot.legal_actions,
            opponents=opponents,
            particle_count=32,
            rng=random.Random(937),
            likelihood_power=0.1,
        )
        self.assertIsNotNone(world)
        # A reallocated opponent hand can acquire a publicly forced-honor
        # follow tile, making the observed ordinary discard illegal.  Those
        # particles must be rejected rather than treating the source-world
        # legality as an oracle.  The remaining consistent particles are the
        # only ones eligible for the SIR target.
        self.assertGreater(diagnostics.consistent_particles, 0)
        self.assertLessEqual(diagnostics.consistent_particles, 32)
        self.assertIn(
            "observed_discard_illegal", diagnostics.rejection_counts
        )
        self.assertGreaterEqual(
            diagnostics.effective_sample_size / diagnostics.consistent_particles,
            0.2,
        )
        self.assertEqual(
            tuple(world.response_options[candidate_seat]), snapshot.legal_actions
        )

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
            self.assertTrue(
                all(
                    decision.executed_index == decision.chosen_index
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
                    and decision.executed_index is not None
                    and decision.legal_actions[decision.executed_index]
                    in decision.legal_actions
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
            heldout = split_heldout_trajectories_by_group(
                read_trajectory_jsonl(path), selection_fraction=0.5
            )
            for group_id in {trajectory.split_group_id for trajectory in trajectories}:
                self.assertEqual(
                    len(
                        {
                            partition
                            for partition, records in heldout.items()
                            for trajectory in records
                            if trajectory.split_group_id == group_id
                        }
                    ),
                    1,
                )

    def test_candidate_dagger_retains_explicit_behavior_propensity(self):
        trajectories, _summary = collect_candidate_teacher_dagger_trajectories(
            _KnownFirstLegalPolicy(), seed_count=1, profile="core", seed=479
        )
        self.assertTrue(
            all(
                decision.executed_probability == 1.0
                for trajectory in trajectories
                for decision in trajectory.decisions
            )
        )

    def test_cross_fitting_folds_keep_every_wall_group_together(self):
        rows = tuple(
            SimpleNamespace(
                trajectory_id=f"trajectory-{index}",
                split_group_id=f"wall-{index // 4}",
            )
            for index in range(48)
        )
        folds = split_trajectories_by_group_folds(
            rows, fold_count=3, split_salt="cross-fit-test-v1"
        )
        self.assertEqual(len(folds), 3)
        self.assertTrue(all(fold for fold in folds))
        memberships: dict[str, set[int]] = {}
        for fold_index, fold in enumerate(folds):
            for row in fold:
                memberships.setdefault(row.split_group_id, set()).add(fold_index)
        self.assertEqual(set(memberships), {f"wall-{index}" for index in range(12)})
        self.assertTrue(all(len(indices) == 1 for indices in memberships.values()))

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
        self.assertIsNotNone(summary.mean_action_value_gap_stderr)
        manifest = trajectory_manifest(trajectories)
        self.assertEqual(manifest["action_value_decisions"], summary.decisions)
        self.assertGreater(manifest["action_value_stderr_observations"], 0)
        self.assertGreater(manifest["action_value_gap_stderr_observations"], 0)
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
                len(decision.action_value_gap_stderrs or ()), len(decision.legal_actions)
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
            self.assertIn('"action_value_gap_stderrs"', serialized)
            self.assertNotIn('"seed"', serialized)
            restored = read_trajectory_jsonl(path)
            self.assertTrue(
                all(
                    decision.action_values is not None
                    and decision.action_value_stderrs is not None
                    and decision.action_value_gap_stderrs is not None
                    for trajectory in restored
                    for decision in trajectory.decisions
                )
            )

    def test_counterfactual_response_only_collection_never_exports_discard_decisions(self):
        policy = NeuralRulePolicyModel(hidden_size=4, seed=588)
        trajectories, summary = collect_counterfactual_action_value_trajectories(
            policy,
            seed_count=3,
            profile="core",
            seed=588,
            decision_phase="response",
        )
        self.assertGreater(len(trajectories), 0)
        self.assertEqual(summary.response_decisions, len(trajectories))
        self.assertTrue(
            all(
                decision.state["phase"] == "response"
                for trajectory in trajectories
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

    def test_latest_discard_sir_rollouts_are_gated_and_safe_to_export(self):
        policy = NeuralRulePolicyModel(hidden_size=4, seed=933)
        trajectories, summary = collect_counterfactual_action_value_trajectories(
            policy,
            seed_count=1,
            profile="core",
            seed=930,
            rollouts_per_action=1,
            belief_resample=True,
            belief_latest_discard_particles=32,
            belief_latest_discard_likelihood_power=0.25,
            belief_latest_discard_min_ess_fraction=0.5,
        )
        self.assertTrue(trajectories)
        self.assertGreater(summary.belief_conditioned_worlds, 0)
        self.assertGreater(
            summary.mean_belief_conditioning_consistency_rate or 0.0, 0.0
        )
        self.assertGreaterEqual(
            summary.mean_belief_conditioning_ess_fraction or 0.0, 0.5
        )
        self.assertTrue(
            all(
                trajectory.source_metadata["belief_conditioning"]
                == "latest_normal_draw_discard_sir_v1"
                and trajectory.source_metadata["belief_latest_discard"]["particles"]
                == 32
                and "wall" not in trajectory.decisions[0].state
                and "opponent_hands" not in trajectory.decisions[0].state
                for trajectory in trajectories
            )
        )

    def test_classic_latest_discard_sir_rollouts_use_the_explicit_gate(self):
        policy = NeuralRulePolicyModel(hidden_size=4, seed=933)
        trajectories, summary = collect_counterfactual_action_value_trajectories(
            policy,
            seed_count=1,
            profile="classic",
            seed=930,
            rollouts_per_action=1,
            belief_resample=True,
            belief_latest_discard_particles=32,
            belief_latest_discard_likelihood_power=0.1,
            belief_latest_discard_min_ess_fraction=0.2,
        )
        self.assertTrue(trajectories)
        self.assertGreater(summary.belief_conditioned_worlds, 0)
        self.assertGreater(
            summary.mean_belief_conditioning_consistency_rate or 0.0, 0.0
        )
        self.assertGreaterEqual(
            summary.mean_belief_conditioning_ess_fraction or 0.0, 0.2
        )
        self.assertTrue(
            all(
                trajectory.profile == "classic"
                and trajectory.source_metadata["belief_latest_discard"]["likelihood_power"]
                == 0.1
                and trajectory.source_metadata["belief_latest_discard"][
                    "minimum_ess_fraction"
                ]
                == 0.2
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
                    decision.action_value_gap_stderrs,
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

    def test_private_actor_replay_trace_is_positioned_but_never_exported(self):
        policy = _BatchFirstLegalPolicy()
        rules = XiamenRules.from_profile("core")
        game = XiamenMahjongGame(
            seed=953,
            rules=rules,
            dealer=0,
            auto_advance=False,
            human_seat=-1,
        )
        teacher = HeuristicTeacherAgent()
        snapshots = _run_candidate_base_hand(
            game,
            candidate_seat=0,
            candidate_policy=policy,
            opponents={seat: ("heuristic_teacher", teacher) for seat in range(1, 4)},
        )
        self.assertTrue(snapshots)
        trace = snapshots[0].actor_trace
        self.assertEqual(trace.actor_seat, 0)
        self.assertEqual(trace.dealer, 0)
        self.assertEqual(len(trace.initial_hand), rules.initial_hand_size)
        self.assertTrue(trace.draws)
        self.assertEqual(trace.draws[0].public_draw_event_index, 0)
        self.assertEqual(trace.draws[0].after_public_action_count, 1)
        self.assertEqual(trace.public_action_count, len(snapshots[0].game.public_actions))
        self.assertTrue(
            all(
                draw.after_public_action_count <= trace.public_action_count
                for draw in trace.draws
            )
        )
        self.assertTrue(
            all(
                action.before_public_action_count
                <= action.after_public_action_count
                <= trace.public_action_count
                for action in trace.actions
            )
        )
        self.assertTrue(
            any(snapshot.actor_trace.actions for snapshot in snapshots[1:])
        )

        # The source's private world is only a replay oracle, never a target
        # exported to training.  Every retained public prefix must be exactly
        # reproducible from setup plus the actor-private trace.
        replayed = _replay_snapshot_public_history(
            snapshots[-1],
            opponents={seat: ("heuristic_teacher", teacher) for seat in range(1, 4)},
        )
        self.assertTrue(replayed.accepted, replayed.rejection_reason)
        self.assertEqual(
            replayed.public_action_count, snapshots[-1].actor_trace.public_action_count
        )
        self.assertGreater(replayed.likelihood, 0.0)

        # A resampled setup removes the actor's later private draw from the
        # unknown pool and reconstructs it only inside the replay callback.
        # This early prefix has one opponent claim, so many proposed worlds
        # are correctly rejected as illegal; at least one fixed-seed particle
        # must reproduce the actor-known draw without exporting it.
        particle_rng = random.Random(954)
        particle_audit = _audit_resampled_history_prefix(
            snapshots[1],
            opponents={seat: ("heuristic_teacher", teacher) for seat in range(1, 4)},
            particle_count=200,
            rng=particle_rng,
        )
        self.assertGreater(particle_audit.accepted_particles, 0)
        self.assertLess(particle_audit.acceptance_rate, 0.1)
        self.assertNotIn("wall", particle_audit.payload())
        self.assertNotIn("opponent_hands", particle_audit.payload())

        # This separate audit is a constructive proposal only.  It exchanges
        # unknown physical tiles to satisfy recorded opponent discards/claims,
        # which should remove the rejection-sampler collapse without exposing
        # any sampled world or becoming a collection pathway.
        repaired_audit = _audit_constraint_repaired_history_prefix(
            snapshots[1],
            opponents={seat: ("heuristic_teacher", teacher) for seat in range(1, 4)},
            particle_count=100,
            rng=random.Random(955),
        )
        self.assertGreater(repaired_audit.acceptance_rate, 0.8)
        self.assertGreater(repaired_audit.effective_sample_size, 0.0)
        self.assertGreater(repaired_audit.mean_constraint_repairs or 0.0, 0.0)
        repaired_payload = repaired_audit.payload()
        self.assertEqual(
            repaired_payload["proposal"], "core_public_history_constraint_repair_v0"
        )
        self.assertIn("not an exact", repaired_payload["warning"])
        self.assertNotIn("wall", repaired_payload)
        self.assertNotIn("opponent_hands", repaired_payload)
        # Replay works on a deep clone.  Even the source-world oracle must
        # neither need nor receive a hidden-card exchange when it is already
        # legal, and the audit cannot mutate the retained candidate trace.
        source_actor_hand = tuple(snapshots[1].initial_game.players[0].hand)
        source_wall = tuple(snapshots[1].initial_game.wall)
        source_repaired = _replay_snapshot_public_history(
            snapshots[1],
            opponents={seat: ("heuristic_teacher", teacher) for seat in range(1, 4)},
            allow_constraint_repairs=True,
        )
        self.assertTrue(source_repaired.accepted, source_repaired.rejection_reason)
        self.assertEqual(source_repaired.constraint_repairs, 0)
        self.assertEqual(tuple(snapshots[1].initial_game.players[0].hand), source_actor_hand)
        self.assertEqual(tuple(snapshots[1].initial_game.wall), source_wall)

        # Sequential SMC observes stable action transitions rather than
        # stopping halfway through a discard that automatically starts the
        # next draw. It retains the unedited actor-visible setup proposal and
        # must surface low-particle collapse instead of manufacturing a world.
        transition_targets = _history_replay_transition_targets(snapshots[1])
        self.assertEqual(transition_targets, (3, 5, 6, 7, 9))
        replay_prefix = _replay_snapshot_public_history(
            snapshots[1],
            opponents={seat: ("heuristic_teacher", teacher) for seat in range(1, 4)},
            target_public_action_count=transition_targets[0],
        )
        self.assertTrue(replay_prefix.accepted, replay_prefix.rejection_reason)
        self.assertEqual(replay_prefix.public_action_count, transition_targets[0])
        sequential_audit = _audit_sequential_history_prefix(
            snapshots[1],
            opponents={seat: ("heuristic_teacher", teacher) for seat in range(1, 4)},
            particle_count=64,
            rng=random.Random(955),
        )
        self.assertFalse(sequential_audit.completed)
        self.assertLess(
            sequential_audit.conditioned_public_events,
            sequential_audit.requested_public_events,
        )
        self.assertLess(sequential_audit.minimum_ess_fraction, 0.2)
        sequential_payload = sequential_audit.payload()
        self.assertEqual(
            sequential_payload["proposal"], "core_public_history_sequential_smc_v0"
        )
        self.assertEqual(
            sequential_payload["proposal_density_ratio"],
            "prior_over_proposal_equals_1",
        )
        self.assertNotIn("wall", sequential_payload)
        self.assertNotIn("opponent_hands", sequential_payload)

        trajectories, _summary = collect_counterfactual_action_value_trajectories(
            policy,
            seed_count=1,
            profile="core",
            seed=953,
            response_sample_probability=0.0,
        )
        self.assertTrue(
            all(
                "actor_trace" not in trajectory.source_metadata
                and "private_replay_trace" not in trajectory.source_metadata
                and "wall" not in decision.state
                and "opponent_hands" not in decision.state
                for trajectory in trajectories
                for decision in trajectory.decisions
            )
        )

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

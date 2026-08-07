import unittest

from xiamen_mahjong.agents import GameAction
from xiamen_mahjong.off_policy import (
    LoggedIntervention,
    LoggedStochasticIntervention,
    doubly_robust_action_advantages,
    doubly_robust_delta,
    effective_sample_size,
    intervention_estimates,
    ips_delta,
    stochastic_intervention_estimates,
    winsorized_doubly_robust_action_advantages,
)
from xiamen_mahjong.training import TeacherDecision, TrainingTrajectory
from scripts.export_crossfit_teacher_advantages import export_rows
from scripts.audit_teacher_response_intervention_ope import (
    selected_interventions,
    teacher_epsilon_propensities,
)
from scripts.train_afterstate_outcomes import initial_agent_for_outcome_training
from scripts.train_afterstate_outcomes import parse_args as parse_afterstate_args
from scripts.select_teacher_response_override import choose_candidate, gate_summary
from scripts.select_teacher_relative_advantage_override import (
    choose_candidate as choose_relative_advantage_candidate,
)


class OffPolicyTests(unittest.TestCase):
    def test_crossfit_advantage_export_omits_terminal_outcome(self):
        class _FixedDirectModel:
            def afterstate_outcomes(self, decision):
                self.assert_decision = decision
                return ([0.0, 0.2], [0.0, 0.0], [0.0, 0.0])

        decision = TeacherDecision(
            profile="classic",
            seed=None,
            seat=0,
            state={"phase": "discard"},
            legal_actions=(GameAction("discard", 0), GameAction("discard", 1)),
            chosen_index=0,
            executed_index=1,
            executed_probability=0.4,
        )
        trajectory = TrainingTrajectory(
            profile="classic",
            rules_version="test",
            rules={},
            seed=None,
            hand_number=0,
            agent_profiles=("heuristic_teacher",) * 4,
            source_metadata={
                "behavior_policy": "single_intervention_epsilon_uniform",
                "base_policy": "heuristic_teacher",
                "intervention_phase": "discard",
                "uniform_exploration_probability": 0.8,
            },
            decisions=(decision,),
            outcome={"scores": [12, 0, 0, 0]},
            public_actions=(),
            trajectory_id="trajectory-a",
            split_group_id="wall-a",
        )
        rows, values = export_rows(
            (trajectory,),
            agent=_FixedDirectModel(),
            source_sha256="source",
            direct_checkpoint_sha256="checkpoint",
            maximum_abs_correction=20.0,
            phase="discard",
            value_scale=80.0,
        )
        self.assertEqual(len(rows), 1)
        self.assertNotIn("outcome", rows[0])
        self.assertEqual(rows[0]["pseudo_advantages"][0], 0.0)
        self.assertEqual(rows[0]["pseudo_advantages"][1], 6.0)
        self.assertEqual(values, [6.0])

    def test_ips_and_dr_recover_a_two_action_grouped_delta(self):
        # Under a uniform behavior policy, one logged target reward of 10 and
        # one logged baseline reward of 0 estimate a +10 target improvement.
        rows = (
            LoggedIntervention(
                group_id="wall-a",
                logged_index=1,
                propensities=(0.5, 0.5),
                reward=10.0,
                baseline_index=0,
                target_index=1,
                direct_values=(0.0, 10.0),
            ),
            LoggedIntervention(
                group_id="wall-a",
                logged_index=0,
                propensities=(0.5, 0.5),
                reward=0.0,
                baseline_index=0,
                target_index=1,
                direct_values=(0.0, 10.0),
            ),
            LoggedIntervention(
                group_id="wall-b",
                logged_index=1,
                propensities=(0.5, 0.5),
                reward=12.0,
                baseline_index=0,
                target_index=1,
                direct_values=(2.0, 12.0),
            ),
            LoggedIntervention(
                group_id="wall-b",
                logged_index=0,
                propensities=(0.5, 0.5),
                reward=2.0,
                baseline_index=0,
                target_index=1,
                direct_values=(2.0, 12.0),
            ),
        )

        self.assertEqual(ips_delta(rows[0]), 20.0)
        self.assertEqual(doubly_robust_delta(rows[0]), 10.0)
        self.assertEqual(doubly_robust_action_advantages(rows[0]), (0.0, 10.0))
        report = intervention_estimates(rows)
        self.assertEqual(report["ips"]["groups"], 2)
        self.assertAlmostEqual(report["ips"]["mean"], 10.0)
        self.assertAlmostEqual(report["doubly_robust"]["mean"], 10.0)
        self.assertEqual(report["support"]["target_matched_logged_action_count"], 2)
        self.assertEqual(report["support"]["baseline_matched_logged_action_count"], 2)

    def test_stochastic_ips_and_dr_recover_teacher_mixture_delta(self):
        rows = (
            LoggedStochasticIntervention(
                group_id="wall-a",
                logged_index=1,
                propensities=(0.5, 0.5),
                reward=10.0,
                baseline_probabilities=(1.0, 0.0),
                target_probabilities=(0.5, 0.5),
                direct_values=(0.0, 10.0),
            ),
            LoggedStochasticIntervention(
                group_id="wall-a",
                logged_index=0,
                propensities=(0.5, 0.5),
                reward=0.0,
                baseline_probabilities=(1.0, 0.0),
                target_probabilities=(0.5, 0.5),
                direct_values=(0.0, 10.0),
            ),
            LoggedStochasticIntervention(
                group_id="wall-b",
                logged_index=1,
                propensities=(0.5, 0.5),
                reward=12.0,
                baseline_probabilities=(1.0, 0.0),
                target_probabilities=(0.5, 0.5),
                direct_values=(2.0, 12.0),
            ),
            LoggedStochasticIntervention(
                group_id="wall-b",
                logged_index=0,
                propensities=(0.5, 0.5),
                reward=2.0,
                baseline_probabilities=(1.0, 0.0),
                target_probabilities=(0.5, 0.5),
                direct_values=(2.0, 12.0),
            ),
        )
        estimates = stochastic_intervention_estimates(rows)
        self.assertEqual(estimates["ips"]["mean"], 5.0)
        self.assertEqual(estimates["doubly_robust"]["mean"], 5.0)
        self.assertEqual(estimates["support"]["target_effective_sample_size"], 4.0)
        self.assertEqual(estimates["support"]["baseline_effective_sample_size"], 2.0)

    def test_same_target_and_baseline_has_exact_zero_delta(self):
        row = LoggedIntervention(
            group_id="wall-a",
            logged_index=0,
            propensities=(0.75, 0.25),
            reward=-18.0,
            baseline_index=0,
            target_index=0,
            direct_values=(-12.0, 4.0),
        )
        self.assertEqual(ips_delta(row), 0.0)
        self.assertEqual(doubly_robust_delta(row), 0.0)

    def test_dr_action_advantages_apply_each_importance_residual_once(self):
        row = LoggedIntervention(
            group_id="wall-a",
            logged_index=2,
            propensities=(0.2, 0.3, 0.5),
            reward=14.0,
            baseline_index=0,
            target_index=2,
            direct_values=(3.0, 6.0, 10.0),
        )
        # Residual is +4 on action 2; its only correction is 4 / 0.5.
        self.assertEqual(
            doubly_robust_action_advantages(row),
            (0.0, 3.0, 15.0),
        )
        self.assertEqual(doubly_robust_delta(row), 15.0)

        baseline_logged = LoggedIntervention(
            group_id="wall-b",
            logged_index=0,
            propensities=(0.5, 0.5),
            reward=7.0,
            baseline_index=0,
            target_index=1,
            direct_values=(3.0, 8.0),
        )
        # Baseline residual is subtracted from every non-baseline action.
        self.assertEqual(
            doubly_robust_action_advantages(baseline_logged),
            (0.0, -3.0),
        )

    def test_winsorized_dr_advantages_bound_low_propensity_residuals(self):
        row = LoggedIntervention(
            group_id="wall-a",
            logged_index=1,
            propensities=(0.9, 0.1),
            reward=14.0,
            baseline_index=0,
            target_index=1,
            direct_values=(3.0, 10.0),
        )
        # The unshrunk correction is (14 - 10) / 0.1 = 40.
        self.assertEqual(
            winsorized_doubly_robust_action_advantages(
                row, maximum_abs_correction=12.0
            ),
            (0.0, 19.0),
        )
        with self.assertRaisesRegex(ValueError, "有限正数"):
            winsorized_doubly_robust_action_advantages(
                row, maximum_abs_correction=0.0
            )

    def test_rejects_invalid_support_and_reports_zero_weight_ess(self):
        with self.assertRaisesRegex(ValueError, "归一化"):
            LoggedIntervention(
                group_id="wall-a",
                logged_index=0,
                propensities=(0.2, 0.2),
                reward=0.0,
                baseline_index=0,
                target_index=1,
            )
        self.assertEqual(effective_sample_size((0.0, 0.0)), 0.0)

    def test_teacher_mixture_keeps_exact_logged_action_support(self):
        self.assertEqual(
            teacher_epsilon_propensities(
                action_count=3, teacher_index=1, epsilon=0.4
            ),
            (0.4 / 3.0, 0.4 / 3.0 + 0.6, 0.4 / 3.0),
        )
        with self.assertRaisesRegex(ValueError, "随机化"):
            teacher_epsilon_propensities(action_count=2, teacher_index=0, epsilon=0.0)

    def test_single_intervention_audit_rejects_unknown_phase(self):
        with self.assertRaisesRegex(ValueError, "intervention_phase"):
            selected_interventions(
                (),
                (),
                value_scale=80.0,
                score_lcb_z=1.0,
                minimum_lcb_advantage=0.0,
                intervention_phase="all",
            )

    def test_intervention_coverage_exposes_only_phase_count_and_wall_groups(self):
        from types import SimpleNamespace

        from scripts.collect_candidate_teacher_dagger_trajectories import (
            randomized_intervention_coverage,
        )

        def decision(phase, probability):
            return SimpleNamespace(
                state={"phase": phase}, executed_probability=probability
            )

        rows = (
            SimpleNamespace(
                split_group_id="wall-a",
                decisions=(decision("discard", 0.2), decision("response", 0.3)),
            ),
            SimpleNamespace(
                split_group_id="wall-a",
                decisions=(decision("discard", 1.0),),
            ),
            SimpleNamespace(
                split_group_id="wall-b",
                decisions=(decision("discard", 0.4),),
            ),
        )
        self.assertEqual(
            randomized_intervention_coverage(rows, intervention_phase="discard"),
            {"randomized_decisions": 2, "wall_groups": 2},
        )

    def test_fresh_outcome_anchors_are_reproducible_without_a_checkpoint(self):
        from types import SimpleNamespace
        import torch

        args = SimpleNamespace(
            init_checkpoint=None,
            fresh_policy_anchor_seed=81,
            fresh_anchor_hidden_size=16,
            feature_version=3,
        )
        first, first_identity = initial_agent_for_outcome_training(
            args, device=torch.device("cpu")
        )
        second, second_identity = initial_agent_for_outcome_training(
            args, device=torch.device("cpu")
        )
        self.assertEqual(first_identity, second_identity)
        self.assertEqual(first_identity["kind"], "fresh_untrained_policy_anchor")
        for name, value in first.network.state_dict().items():
            self.assertTrue((value == second.network.state_dict()[name]).all(), name)

    def test_selection_prefers_stronger_lower_bound_then_conservative_threshold(self):
        def audit(threshold, ips, dr, target_ess=40.0, base_ess=45.0):
            estimates = {
                "ips": {"95pct_low": ips},
                "doubly_robust": {"95pct_low": dr},
                "support": {
                    "target_effective_sample_size": target_ess,
                    "baseline_effective_sample_size": base_ess,
                },
            }
            return {
                "minimum_lcb_advantage": threshold,
                "gate": gate_summary(estimates, minimum_effective_sample_size=30.0),
            }

        chosen = choose_candidate((audit(0.0, 2.0, 2.0), audit(8.0, 1.0, 1.0)))
        self.assertEqual(chosen["minimum_lcb_advantage"], 0.0)
        tied = choose_candidate((audit(0.0, 1.0, 1.0), audit(8.0, 1.0, 1.0)))
        self.assertEqual(tied["minimum_lcb_advantage"], 8.0)
        self.assertIsNone(choose_candidate((audit(0.0, -1.0, 2.0),)))

    def test_relative_advantage_selection_has_fixed_tie_breaks(self):
        def audit(correction, threshold, lower, passes=True):
            return {
                "maximum_abs_correction": correction,
                "minimum_advantage": threshold,
                "gate": {"minimum_lower_bound": lower, "passes": passes},
            }

        chosen = choose_relative_advantage_candidate(
            (
                audit(20.0, 0.0, 1.0),
                audit(80.0, 8.0, 1.0),
                audit(40.0, 8.0, 1.0),
            )
        )
        self.assertEqual(chosen["minimum_advantage"], 8.0)
        self.assertEqual(chosen["maximum_abs_correction"], 40.0)
        self.assertIsNone(choose_relative_advantage_candidate((audit(20.0, 0.0, 1.0, False),)))

    def test_outcome_trainer_allows_a_blind_terminal_set(self):
        import sys
        from unittest.mock import patch

        with patch.object(
            sys,
            "argv",
            [
                "train_afterstate_outcomes.py",
                "--train",
                "train.jsonl",
                "--validation",
                "validation.jsonl",
                "--skip-test",
                "--fresh-policy-anchor-seed",
                "1",
                "--output-dir",
                "output",
            ],
        ):
            args = parse_afterstate_args()
        self.assertTrue(args.skip_test)
        self.assertIsNone(args.test)

    def test_outcome_trainer_accepts_fixed_epochs_without_validation(self):
        import sys
        from unittest.mock import patch

        with patch.object(
            sys,
            "argv",
            [
                "train_afterstate_outcomes.py",
                "--train",
                "train.jsonl",
                "--skip-test",
                "--fixed-epochs",
                "--fresh-policy-anchor-seed",
                "1",
                "--output-dir",
                "output",
            ],
        ):
            args = parse_afterstate_args()
        self.assertTrue(args.fixed_epochs)
        self.assertIsNone(args.validation)


if __name__ == "__main__":
    unittest.main()

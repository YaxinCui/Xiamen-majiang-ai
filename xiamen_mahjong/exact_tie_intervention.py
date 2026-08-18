"""High-support binary intervention for a frozen exact-tie model ensemble."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
from typing import Any, Iterable, Sequence

from .agents import GameAction, HeuristicTeacherAgent
from .exact_tie_rollout import _top_tied_actions
from .human_review import _ActorVisibleReviewGame
from .training import TeacherDecision, TrainingTrajectory, _perspective_state, _turn_actions


TARGETED_EXACT_TIE_ENSEMBLE_BEHAVIOR_VERSION = (
    "first-exact-tie-v2-ensemble-disagreement-binary-v1"
)


def exact_tie_ensemble_action_from_state(
    state: dict[str, Any],
    legal_actions: Sequence[GameAction],
    teacher_index: int,
    policies: Sequence[Any],
) -> GameAction:
    """Reproduce the unanimous ensemble target from actor-visible state."""

    if len(policies) != 5 or not 0 <= teacher_index < len(legal_actions):
        raise ValueError("exact-tie ensemble 需要五个模型和合法 Teacher index")
    game = _ActorVisibleReviewGame({"state": state})
    frozen = HeuristicTeacherAgent().explain_discard(game, 0)
    if not frozen:
        return legal_actions[teacher_index]
    top_score = float(frozen[0]["score"])
    tied_tiles = {
        int(row["tile"]) for row in frozen if float(row["score"]) == top_score
    }
    tie_indices = tuple(
        index
        for index, action in enumerate(legal_actions)
        if action.kind == "discard" and action.tile in tied_tiles
    )
    if teacher_index not in tie_indices or len(tie_indices) < 2:
        return legal_actions[teacher_index]
    teacher_local = tie_indices.index(teacher_index)
    tie_actions = tuple(legal_actions[index] for index in tie_indices)
    decision = TeacherDecision(
        profile=str(state.get("rules_profile", "classic")),
        seed=None,
        seat=0,
        state=state,
        legal_actions=tie_actions,
        chosen_index=teacher_local,
        reference_teacher_index=teacher_local,
    )
    votes = [policy.predict_index(decision) for policy in policies]
    if len(set(votes)) != 1:
        return legal_actions[teacher_index]
    return tie_actions[votes[0]]


class TargetedExactTieEnsembleInterventionBehavior:
    """Randomize Teacher vs fixed ensemble once, then restore Teacher."""

    def __init__(
        self,
        policies: Sequence[Any],
        *,
        seed: int,
        candidate_probability: float = 0.5,
    ) -> None:
        if len(policies) != 5:
            raise ValueError("targeted exact-tie behavior 需要五个模型")
        if not 0.0 < candidate_probability < 1.0:
            raise ValueError("candidate_probability 必须位于 (0, 1)")
        self.policies = tuple(policies)
        self.teacher = HeuristicTeacherAgent()
        self.rng = random.Random(seed)
        self.candidate_probability = float(candidate_probability)
        self.intervened = False
        self._last_action: GameAction | None = None
        self._last_probability: float | None = None
        self.disagreement_hands = 0
        self.candidate_assignments = 0
        self.teacher_assignments = 0

    def reset_episode(self) -> None:
        self.intervened = False
        self._last_action = None
        self._last_probability = None

    def _remember(self, action: GameAction, probability: float) -> GameAction:
        self._last_action = action
        self._last_probability = float(probability)
        return action

    def choose_turn_action(self, game: Any, player_id: int) -> GameAction:
        legal = tuple(_turn_actions(game, player_id))
        teacher_action = self.teacher.choose_turn_action(game, player_id)
        if teacher_action not in legal:
            raise RuntimeError("exact-tie behavior Teacher 动作非法")
        if self.intervened:
            return self._remember(teacher_action, 1.0)
        tied = _top_tied_actions(game, player_id)
        if len(tied) < 2:
            return self._remember(teacher_action, 1.0)
        state = _perspective_state(game, player_id)
        target = exact_tie_ensemble_action_from_state(
            state, tied, 0, self.policies
        )
        if target == teacher_action:
            return self._remember(teacher_action, 1.0)
        if target not in legal:
            raise RuntimeError("exact-tie ensemble 目标动作非法")
        self.intervened = True
        self.disagreement_hands += 1
        if self.rng.random() < self.candidate_probability:
            self.candidate_assignments += 1
            return self._remember(target, self.candidate_probability)
        self.teacher_assignments += 1
        return self._remember(teacher_action, 1.0 - self.candidate_probability)

    def choose_response(
        self, game: Any, player_id: int, options: Sequence[GameAction]
    ) -> GameAction:
        action = self.teacher.choose_response(game, player_id, list(options))
        if action not in options:
            raise RuntimeError("exact-tie behavior Teacher 响应非法")
        return self._remember(action, 1.0)

    def action_probability(
        self,
        game: Any,
        player_id: int,
        legal: tuple[GameAction, ...],
        action: GameAction,
        *,
        is_response: bool,
    ) -> float:
        del game, player_id, legal, is_response
        if action != self._last_action or self._last_probability is None:
            raise ValueError("action_probability 必须紧随 exact-tie behavior 动作")
        return self._last_probability


def _interval(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"groups": 0, "mean": None, "stderr": None, "95pct_low": None, "95pct_high": None}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"groups": 1, "mean": mean, "stderr": None, "95pct_low": None, "95pct_high": None}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    stderr = math.sqrt(variance / len(values))
    return {
        "groups": len(values),
        "mean": mean,
        "stderr": stderr,
        "95pct_low": mean - 1.96 * stderr,
        "95pct_high": mean + 1.96 * stderr,
    }


def audit_targeted_exact_tie_interventions(
    trajectories: Iterable[TrainingTrajectory],
    *,
    policies: Sequence[Any],
    minimum_interventions: int,
    minimum_wall_groups: int,
) -> dict[str, Any]:
    """Validate binary support and grouped HT candidate-minus-Teacher value."""

    if len(policies) != 5 or minimum_interventions <= 0 or minimum_wall_groups <= 1:
        raise ValueError("targeted exact-tie audit 参数无效")
    rows = tuple(trajectories)
    issues: Counter[str] = Counter()
    group_values: dict[str, list[float]] = defaultdict(list)
    group_seats: dict[str, set[int]] = defaultdict(set)
    interventions = 0
    candidate_assignments = 0
    teacher_assignments = 0
    for trajectory in rows:
        metadata = trajectory.source_metadata
        candidate_seat = metadata.get("candidate_seat")
        group = trajectory.split_group_id
        if (
            trajectory.profile != "classic"
            or metadata.get("behavior_policy")
            != TARGETED_EXACT_TIE_ENSEMBLE_BEHAVIOR_VERSION
            or metadata.get("base_policy") != "heuristic_teacher"
            or metadata.get("candidate_policy") != "exact_tie_structured_ensemble_v2"
            or metadata.get("candidate_action_probability") != 0.5
            or metadata.get("maximum_interventions_per_trajectory") != 1
            or isinstance(candidate_seat, bool)
            or not isinstance(candidate_seat, int)
            or not 0 <= candidate_seat < 4
            or not isinstance(group, str)
            or not group
        ):
            issues["invalid_trajectory_metadata"] += 1
            continue
        randomized = [
            decision
            for decision in trajectory.decisions
            if decision.executed_probability is not None
            and abs(decision.executed_probability - 0.5) <= 1e-12
        ]
        if len(randomized) > 1:
            issues["multiple_interventions"] += 1
            continue
        contribution = 0.0
        if randomized:
            interventions += 1
            decision = randomized[0]
            try:
                target_action = exact_tie_ensemble_action_from_state(
                    decision.state,
                    decision.legal_actions,
                    decision.chosen_index,
                    policies,
                )
                target_index = decision.legal_actions.index(target_action)
            except (ValueError, RuntimeError):
                issues["unreproducible_target"] += 1
                continue
            if target_index == decision.chosen_index:
                issues["randomized_row_has_no_ensemble_disagreement"] += 1
                continue
            if decision.executed_index not in {target_index, decision.chosen_index}:
                issues["executed_action_outside_binary_support"] += 1
                continue
            scores = trajectory.outcome.get("scores")
            if not isinstance(scores, list) or len(scores) != 4:
                issues["invalid_terminal_scores"] += 1
                continue
            score = float(scores[candidate_seat])
            if decision.executed_index == target_index:
                candidate_assignments += 1
                contribution = score / 0.5
            else:
                teacher_assignments += 1
                contribution = -score / 0.5
        group_values[group].append(contribution)
        group_seats[group].add(candidate_seat)
    wall_means = [
        sum(values) / 4.0
        for group, values in group_values.items()
        if len(values) == 4 and group_seats[group] == {0, 1, 2, 3}
    ]
    incomplete = len(group_values) - len(wall_means)
    if incomplete:
        issues["incomplete_wall_groups"] += incomplete
    fraction = candidate_assignments / interventions if interventions else None
    reasons = []
    if issues:
        reasons.append("invalid_or_unreproducible_trajectory")
    if interventions < minimum_interventions:
        reasons.append("insufficient_interventions")
    if len(wall_means) < minimum_wall_groups:
        reasons.append("insufficient_complete_wall_groups")
    if not isinstance(fraction, float) or not 0.4 <= fraction <= 0.6:
        reasons.append("random_assignment_imbalance")
    return {
        "status": "targeted_exact_tie_ensemble_intervention_audit",
        "trajectories": len(rows),
        "complete_wall_groups": len(wall_means),
        "intervention_trajectories": interventions,
        "candidate_assignments": candidate_assignments,
        "teacher_assignments": teacher_assignments,
        "candidate_assignment_fraction": fraction,
        "minimum_interventions": minimum_interventions,
        "minimum_wall_groups": minimum_wall_groups,
        "issues": dict(sorted(issues.items())),
        "structurally_ready": not reasons,
        "gate_reasons": reasons,
        "horvitz_thompson_candidate_minus_teacher_per_wall": _interval(wall_means),
        "estimand": (
            "first exact-tie unanimous ensemble disagreement, one binary assignment, "
            "then frozen Teacher"
        ),
        "privacy": "aggregate_only_no_state_hand_action_face_group_or_terminal_trace",
    }

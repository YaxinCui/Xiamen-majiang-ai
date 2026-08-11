"""High-support binary interventions at TwoDraw/Teacher disagreement states."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
from typing import Any, Iterable, Sequence

from .agents import (
    GameAction,
    HeuristicTeacherAgent,
    TwoDrawTenpaiReachTeacherAgent,
)
from .human_review import _ActorVisibleReviewGame
from .training import TeacherDecision, TrainingTrajectory, _turn_actions


TARGETED_TWO_DRAW_BEHAVIOR_VERSION = (
    "first-two-draw-teacher-disagreement-binary-v1"
)


class TargetedTwoDrawInterventionBehavior:
    """Randomize Teacher vs TwoDraw once, then restore Teacher for the hand.

    Unlike epsilon-uniform exploration over every legal discard, the only two
    supported actions are the frozen Teacher action and the already-fixed
    TwoDraw action.  A 0.5/0.5 assignment therefore gives both policies high
    support while changing at most one ordinary discard in each trajectory.
    """

    def __init__(
        self,
        *,
        seed: int,
        candidate_probability: float = 0.5,
        teacher: Any | None = None,
        candidate: Any | None = None,
    ) -> None:
        if not 0.0 < candidate_probability < 1.0:
            raise ValueError("candidate_probability 必须位于 (0, 1)")
        self.rng = random.Random(seed)
        self.candidate_probability = float(candidate_probability)
        self.teacher = teacher or HeuristicTeacherAgent()
        self.candidate = candidate or TwoDrawTenpaiReachTeacherAgent(
            score_margin=2.0,
            minimum_probability_advantage=0.05,
        )
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
            raise RuntimeError("Teacher 产生规则引擎非法动作")
        if (
            self.intervened
            or game.rules.profile != "classic"
            or game.phase != "discard"
            or game.tour_state is not None
            or game.gold_discard_lock_seat == player_id
            or player_id in game.opening_wait_seats
            or teacher_action.kind != "discard"
            or sum(action.kind == "discard" for action in legal) < 2
        ):
            return self._remember(teacher_action, 1.0)
        candidate_action = self.candidate.choose_turn_action(game, player_id)
        if candidate_action not in legal:
            raise RuntimeError("TwoDraw 产生规则引擎非法动作")
        if candidate_action == teacher_action or candidate_action.kind != "discard":
            return self._remember(teacher_action, 1.0)

        self.intervened = True
        self.disagreement_hands += 1
        if self.rng.random() < self.candidate_probability:
            self.candidate_assignments += 1
            return self._remember(candidate_action, self.candidate_probability)
        self.teacher_assignments += 1
        return self._remember(teacher_action, 1.0 - self.candidate_probability)

    def choose_response(
        self, game: Any, player_id: int, options: Sequence[GameAction]
    ) -> GameAction:
        legal = tuple(options)
        action = self.teacher.choose_response(game, player_id, list(legal))
        if action not in legal:
            raise RuntimeError("Teacher 产生规则引擎非法响应")
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
            raise ValueError("action_probability 必须紧随同一行为动作调用")
        return self._last_probability


def _two_draw_target_index(
    decision: TeacherDecision,
    *,
    candidate: Any,
) -> int:
    """Reproduce the public TwoDraw alternative for one randomized row."""

    if decision.state.get("phase") != "discard":
        raise ValueError("定向干预只能出现在 discard phase")
    teacher_index = decision.chosen_index
    teacher_action = decision.legal_actions[teacher_index]
    if teacher_action.kind != "discard":
        raise ValueError("定向干预的 Teacher 动作必须为 discard")
    game = _ActorVisibleReviewGame({"state": decision.state})
    frozen = HeuristicTeacherAgent().explain_discard(game, 0)
    if not frozen or int(frozen[0]["tile"]) != int(teacher_action.tile):
        raise ValueError("定向干预 Teacher 无法从 actor-visible 状态复现")
    ranked = candidate.explain_discard(game, 0)
    if not ranked:
        raise ValueError("TwoDraw 无法从 actor-visible 状态复现")
    target_tile = int(ranked[0]["tile"])
    indices = [
        index
        for index, action in enumerate(decision.legal_actions)
        if action.kind == "discard" and int(action.tile) == target_tile
    ]
    if len(indices) != 1 or indices[0] == teacher_index:
        raise ValueError("随机行没有唯一 TwoDraw/Teacher 分歧")
    return indices[0]


def _mean_interval(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "groups": 0,
            "mean": None,
            "stderr": None,
            "95pct_low": None,
            "95pct_high": None,
        }
    mean = sum(values) / len(values)
    if len(values) == 1:
        stderr = None
        low = None
        high = None
    else:
        variance = sum((value - mean) ** 2 for value in values) / (
            len(values) - 1
        )
        stderr = math.sqrt(variance / len(values))
        low = mean - 1.96 * stderr
        high = mean + 1.96 * stderr
    return {
        "groups": len(values),
        "mean": mean,
        "stderr": stderr,
        "95pct_low": low,
        "95pct_high": high,
    }


def audit_targeted_two_draw_interventions(
    trajectories: Iterable[TrainingTrajectory],
    *,
    minimum_interventions: int,
    minimum_wall_groups: int,
    candidate: Any | None = None,
) -> dict[str, Any]:
    """Validate exact binary support and estimate per-wall HT score delta."""

    if minimum_interventions <= 0 or minimum_wall_groups <= 1:
        raise ValueError("定向干预审计门槛无效")
    rows = tuple(trajectories)
    slow = candidate or TwoDrawTenpaiReachTeacherAgent(
        score_margin=2.0,
        minimum_probability_advantage=0.05,
    )
    issues: Counter[str] = Counter()
    group_contributions: dict[str, list[float]] = defaultdict(list)
    group_seats: dict[str, set[int]] = defaultdict(set)
    candidate_assignments = 0
    teacher_assignments = 0
    intervention_trajectories = 0
    trajectories_without_disagreement = 0
    for trajectory in rows:
        metadata = trajectory.source_metadata
        if (
            trajectory.profile != "classic"
            or metadata.get("collector") != "candidate_vs_teacher_dagger"
            or metadata.get("behavior_policy")
            != TARGETED_TWO_DRAW_BEHAVIOR_VERSION
            or metadata.get("base_policy") != "heuristic_teacher"
            or metadata.get("candidate_policy")
            != "two_draw_tenpai_reach_v1"
            or metadata.get("candidate_action_probability") != 0.5
            or metadata.get("maximum_interventions_per_trajectory") != 1
        ):
            issues["invalid_trajectory_metadata"] += 1
            continue
        candidate_seat = metadata.get("candidate_seat")
        group_id = trajectory.split_group_id
        if (
            isinstance(candidate_seat, bool)
            or not isinstance(candidate_seat, int)
            or not 0 <= candidate_seat < 4
            or not isinstance(group_id, str)
            or not group_id
        ):
            issues["invalid_candidate_seat_or_group"] += 1
            continue
        randomized: list[tuple[TeacherDecision, int]] = []
        valid = True
        for decision in trajectory.decisions:
            probability = decision.executed_probability
            executed = decision.executed_index
            if probability is None or executed is None:
                issues["missing_behavior_propensity"] += 1
                valid = False
                break
            if abs(probability - 1.0) <= 1e-12:
                if executed != decision.chosen_index:
                    issues["nonrandom_suffix_differs_from_teacher"] += 1
                    valid = False
                    break
                continue
            if abs(probability - 0.5) > 1e-12:
                issues["invalid_binary_propensity"] += 1
                valid = False
                break
            try:
                target_index = _two_draw_target_index(
                    decision, candidate=slow
                )
            except ValueError:
                issues["unreproducible_targeted_disagreement"] += 1
                valid = False
                break
            if executed not in {decision.chosen_index, target_index}:
                issues["executed_action_outside_binary_support"] += 1
                valid = False
                break
            randomized.append((decision, target_index))
        if not valid:
            continue
        if len(randomized) > 1:
            issues["multiple_interventions_in_trajectory"] += 1
            continue
        scores = trajectory.outcome.get("scores")
        if not (
            isinstance(scores, list)
            and len(scores) == 4
            and all(isinstance(score, (int, float)) for score in scores)
        ):
            issues["invalid_terminal_scores"] += 1
            continue
        contribution = 0.0
        if randomized:
            intervention_trajectories += 1
            decision, target_index = randomized[0]
            score = float(scores[candidate_seat])
            if decision.executed_index == target_index:
                candidate_assignments += 1
                contribution = score / 0.5
            else:
                teacher_assignments += 1
                contribution = -score / 0.5
        else:
            trajectories_without_disagreement += 1
        group_contributions[group_id].append(contribution)
        group_seats[group_id].add(candidate_seat)

    incomplete_groups = sum(
        len(values) != 4 or group_seats[group_id] != {0, 1, 2, 3}
        for group_id, values in group_contributions.items()
    )
    if incomplete_groups:
        issues["incomplete_wall_rotation_group"] += incomplete_groups
    complete_group_means = [
        sum(values) / 4.0
        for group_id, values in sorted(group_contributions.items())
        if len(values) == 4 and group_seats[group_id] == {0, 1, 2, 3}
    ]
    candidate_fraction = (
        candidate_assignments / intervention_trajectories
        if intervention_trajectories
        else None
    )
    gate_reasons: list[str] = []
    if issues:
        gate_reasons.append("invalid_or_unreproducible_trajectory")
    if intervention_trajectories < minimum_interventions:
        gate_reasons.append("insufficient_targeted_interventions")
    if len(complete_group_means) < minimum_wall_groups:
        gate_reasons.append("insufficient_complete_wall_groups")
    if not (
        isinstance(candidate_fraction, float)
        and 0.3 <= candidate_fraction <= 0.7
    ):
        gate_reasons.append("random_assignment_imbalance")
    return {
        "status": "targeted_two_draw_intervention_audit",
        "behavior_version": TARGETED_TWO_DRAW_BEHAVIOR_VERSION,
        "trajectories": len(rows),
        "complete_wall_groups": len(complete_group_means),
        "intervention_trajectories": intervention_trajectories,
        "trajectories_without_disagreement": trajectories_without_disagreement,
        "candidate_assignments": candidate_assignments,
        "teacher_assignments": teacher_assignments,
        "candidate_assignment_fraction": candidate_fraction,
        "minimum_interventions": minimum_interventions,
        "minimum_wall_groups": minimum_wall_groups,
        "issues": dict(sorted(issues.items())),
        "structurally_ready": not gate_reasons,
        "gate_reasons": gate_reasons,
        "horvitz_thompson_candidate_minus_teacher_per_wall": (
            _mean_interval(complete_group_means)
        ),
        "estimand": (
            "first ordinary TwoDraw disagreement override followed by Teacher "
            "minus all-Teacher, averaged over four seat rotations per physical wall"
        ),
        "privacy": "aggregate_only_no_state_hand_action_tile_group_id_or_terminal_trace",
        "warning": (
            "This is a one-intervention causal estimand, not a complete multi-override "
            "policy or human-strength result."
        ),
    }

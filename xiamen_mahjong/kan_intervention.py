"""High-support binary intervention at the first frozen-Teacher kan."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import random
from statistics import NormalDist
from typing import Any, Iterable, Sequence

from .agents import GameAction, HeuristicTeacherAgent
from .human_review import _ActorVisibleReviewGame
from .training import TeacherDecision, TrainingTrajectory, _turn_actions


KAN_KINDS = ("an_kan", "add_kan", "ming_kan")
KAN_INTERVENTION_BEHAVIOR_VERSION = "first-teacher-kan-binary-v1"
BONFERRONI_KIND_Z = NormalDist().inv_cdf(1.0 - 0.05 / (2.0 * len(KAN_KINDS)))


class TargetedKanInterventionBehavior:
    """Randomize Teacher kan vs its deterministic non-kan fallback once."""

    def __init__(
        self,
        *,
        seed: int,
        fallback_probability: float = 0.5,
        teacher: Any | None = None,
    ) -> None:
        if not 0.0 < fallback_probability < 1.0:
            raise ValueError("fallback_probability 必须位于 (0, 1)")
        self.rng = random.Random(seed)
        self.fallback_probability = float(fallback_probability)
        self.teacher = teacher or HeuristicTeacherAgent()
        self.intervened = False
        self._last_action: GameAction | None = None
        self._last_probability: float | None = None
        self.interventions = Counter()
        self.fallback_assignments = Counter()
        self.teacher_assignments = Counter()

    def reset_episode(self) -> None:
        self.intervened = False
        self._last_action = None
        self._last_probability = None

    def _remember(self, action: GameAction, probability: float) -> GameAction:
        self._last_action = action
        self._last_probability = float(probability)
        return action

    def _assign(
        self, teacher_action: GameAction, fallback_action: GameAction
    ) -> GameAction:
        if teacher_action.kind not in KAN_KINDS or fallback_action.kind in KAN_KINDS:
            raise ValueError("杠干预的二元动作定义无效")
        self.intervened = True
        kind = teacher_action.kind
        self.interventions[kind] += 1
        if self.rng.random() < self.fallback_probability:
            self.fallback_assignments[kind] += 1
            return self._remember(fallback_action, self.fallback_probability)
        self.teacher_assignments[kind] += 1
        return self._remember(teacher_action, 1.0 - self.fallback_probability)

    def choose_turn_action(self, game: Any, player_id: int) -> GameAction:
        legal = tuple(_turn_actions(game, player_id))
        teacher_action = self.teacher.choose_turn_action(game, player_id)
        if teacher_action not in legal:
            raise RuntimeError("Teacher 产生规则引擎非法动作")
        if self.intervened or teacher_action.kind not in {"an_kan", "add_kan"}:
            return self._remember(teacher_action, 1.0)
        fallback_action = GameAction(
            "discard", self.teacher._best_discard(game, player_id)
        )
        if fallback_action not in legal:
            raise RuntimeError("Teacher 的非杠弃牌 fallback 非法")
        return self._assign(teacher_action, fallback_action)

    def choose_response(
        self, game: Any, player_id: int, options: Sequence[GameAction]
    ) -> GameAction:
        legal = tuple(options)
        teacher_action = self.teacher.choose_response(game, player_id, list(legal))
        if teacher_action not in legal:
            raise RuntimeError("Teacher 产生规则引擎非法响应")
        if self.intervened or teacher_action.kind != "ming_kan":
            return self._remember(teacher_action, 1.0)
        non_kan = [action for action in legal if action.kind != "ming_kan"]
        fallback_action = self.teacher.choose_response(game, player_id, non_kan)
        if fallback_action not in legal or fallback_action.kind in KAN_KINDS:
            raise RuntimeError("Teacher 的非明杠响应 fallback 非法")
        return self._assign(teacher_action, fallback_action)

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


def _non_kan_fallback_index(decision: TeacherDecision) -> int:
    teacher_action = decision.chosen_action
    if teacher_action.kind not in KAN_KINDS:
        raise ValueError("随机行的 Teacher 动作不是杠")
    game = _ActorVisibleReviewGame({"state": decision.state})
    if teacher_action.kind in {"an_kan", "add_kan"}:
        ranked = HeuristicTeacherAgent().explain_discard(game, 0)
        if not ranked:
            raise ValueError("无法从 actor-visible 状态复现非杠弃牌")
        fallback = GameAction("discard", int(ranked[0]["tile"]))
    else:
        non_kan = [
            action for action in decision.legal_actions
            if action.kind != "ming_kan"
        ]
        fallback = HeuristicTeacherAgent().choose_response(game, 0, non_kan)
    indices = [
        index for index, action in enumerate(decision.legal_actions)
        if action == fallback
    ]
    if len(indices) != 1 or indices[0] == decision.chosen_index:
        raise ValueError("随机行没有唯一非杠 fallback")
    return indices[0]


def _mean_interval(
    values: Sequence[float], *, z: float = 1.96
) -> dict[str, float | int | None]:
    if not values:
        return {
            "groups": 0,
            "mean": None,
            "stderr": None,
            "z": z,
            "low": None,
            "high": None,
        }
    mean = sum(values) / len(values)
    if len(values) == 1:
        stderr = low = high = None
    else:
        variance = sum((value - mean) ** 2 for value in values) / (
            len(values) - 1
        )
        stderr = math.sqrt(variance / len(values))
        low = mean - z * stderr
        high = mean + z * stderr
    return {
        "groups": len(values),
        "mean": mean,
        "stderr": stderr,
        "z": z,
        "low": low,
        "high": high,
    }


def audit_targeted_kan_interventions(
    trajectories: Iterable[TrainingTrajectory],
    *,
    minimum_interventions: int,
    minimum_wall_groups: int,
) -> dict[str, Any]:
    """Validate binary support and estimate skip-kan minus Teacher-kan."""

    if minimum_interventions <= 0 or minimum_wall_groups <= 1:
        raise ValueError("杠干预审计门槛无效")
    rows = tuple(trajectories)
    issues: Counter[str] = Counter()
    group_total: dict[str, list[float]] = defaultdict(list)
    group_by_kind: dict[str, dict[str, list[float]]] = {
        kind: defaultdict(list) for kind in KAN_KINDS
    }
    group_seats: dict[str, set[int]] = defaultdict(set)
    interventions = Counter()
    fallback_assignments = Counter()
    teacher_assignments = Counter()
    no_opportunity = 0
    for trajectory in rows:
        metadata = trajectory.source_metadata
        if (
            trajectory.profile != "classic"
            or metadata.get("collector") != "candidate_vs_teacher_dagger"
            or metadata.get("behavior_policy") != KAN_INTERVENTION_BEHAVIOR_VERSION
            or metadata.get("base_policy") != "heuristic_teacher"
            or metadata.get("fallback_policy") != "teacher_non_kan_fallback_v1"
            or metadata.get("fallback_probability") != 0.5
            or metadata.get("maximum_interventions_per_trajectory") != 1
        ):
            issues["invalid_trajectory_metadata"] += 1
            continue
        seat = metadata.get("candidate_seat")
        group_id = trajectory.split_group_id
        if (
            isinstance(seat, bool)
            or not isinstance(seat, int)
            or not 0 <= seat < 4
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
                fallback_index = _non_kan_fallback_index(decision)
            except ValueError:
                issues["unreproducible_kan_fallback"] += 1
                valid = False
                break
            if executed not in {decision.chosen_index, fallback_index}:
                issues["executed_action_outside_binary_support"] += 1
                valid = False
                break
            randomized.append((decision, fallback_index))
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
        kind_contributions = {kind: 0.0 for kind in KAN_KINDS}
        if randomized:
            decision, fallback_index = randomized[0]
            kind = decision.chosen_action.kind
            interventions[kind] += 1
            score = float(scores[seat])
            if decision.executed_index == fallback_index:
                fallback_assignments[kind] += 1
                contribution = score / 0.5
            else:
                teacher_assignments[kind] += 1
                contribution = -score / 0.5
            kind_contributions[kind] = contribution
        else:
            no_opportunity += 1
        group_total[group_id].append(contribution)
        for kind in KAN_KINDS:
            group_by_kind[kind][group_id].append(kind_contributions[kind])
        group_seats[group_id].add(seat)

    complete_ids = [
        group_id for group_id, values in group_total.items()
        if len(values) == 4 and group_seats[group_id] == {0, 1, 2, 3}
    ]
    incomplete = len(group_total) - len(complete_ids)
    if incomplete:
        issues["incomplete_wall_rotation_group"] += incomplete
    total_means = [sum(group_total[group_id]) / 4.0 for group_id in complete_ids]
    kind_reports = {}
    for kind in KAN_KINDS:
        values = [
            sum(group_by_kind[kind][group_id]) / 4.0
            for group_id in complete_ids
        ]
        count = interventions[kind]
        fallback_fraction = (
            fallback_assignments[kind] / count if count else None
        )
        kind_reports[kind] = {
            "interventions": count,
            "fallback_assignments": fallback_assignments[kind],
            "teacher_assignments": teacher_assignments[kind],
            "fallback_assignment_fraction": fallback_fraction,
            "bonferroni_98_333pct_interval_per_wall": _mean_interval(
                values, z=BONFERRONI_KIND_Z
            ),
        }
    total_interventions = sum(interventions.values())
    total_fallbacks = sum(fallback_assignments.values())
    fallback_fraction = (
        total_fallbacks / total_interventions if total_interventions else None
    )
    gate_reasons = []
    if issues:
        gate_reasons.append("invalid_or_unreproducible_trajectory")
    if total_interventions < minimum_interventions:
        gate_reasons.append("insufficient_kan_interventions")
    if len(complete_ids) < minimum_wall_groups:
        gate_reasons.append("insufficient_complete_wall_groups")
    if not (
        isinstance(fallback_fraction, float)
        and 0.3 <= fallback_fraction <= 0.7
    ):
        gate_reasons.append("random_assignment_imbalance")
    return {
        "status": "targeted_kan_intervention_audit",
        "behavior_version": KAN_INTERVENTION_BEHAVIOR_VERSION,
        "trajectories": len(rows),
        "complete_wall_groups": len(complete_ids),
        "intervention_trajectories": total_interventions,
        "trajectories_without_kan_opportunity": no_opportunity,
        "fallback_assignments": total_fallbacks,
        "teacher_assignments": sum(teacher_assignments.values()),
        "fallback_assignment_fraction": fallback_fraction,
        "minimum_interventions": minimum_interventions,
        "minimum_wall_groups": minimum_wall_groups,
        "issues": dict(sorted(issues.items())),
        "structurally_ready": not gate_reasons,
        "gate_reasons": gate_reasons,
        "overall_skip_kan_minus_teacher_95pct_interval_per_wall": (
            _mean_interval(total_means)
        ),
        "by_teacher_kan_kind": kind_reports,
        "kind_familywise_error_control": {
            "hypotheses": len(KAN_KINDS),
            "two_sided_family_alpha": 0.05,
            "bonferroni_z": BONFERRONI_KIND_Z,
        },
        "estimand": (
            "At the first Teacher-kan opportunity only, execute the frozen "
            "non-kan fallback instead of kan, then restore Teacher; average "
            "candidate-seat score difference over four seat rotations per wall."
        ),
        "privacy": "aggregate_only_no_state_hand_action_tile_group_id_or_trace",
    }

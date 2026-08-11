"""Randomized high-support interventions between Teacher top-two discards.

The behavior changes at most one ordinary discard in a candidate trajectory.
It is a data-collection policy, not a deployable gameplay policy.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

from .agents import GameAction, HeuristicTeacherAgent
from .human_review import _ActorVisibleReviewGame
from .training import TeacherDecision, TrainingTrajectory, _turn_actions


LOW_MARGIN_TOP2_BEHAVIOR_VERSION = "first-low-margin-top2-discard-binary-v1"
LOW_MARGIN_TOP2_POLICY_VERSION = "heuristic-teacher-top2-margin-2-v1"
LOW_MARGIN_CAUSAL_RECORD_VERSION = "xiamen-low-margin-top2-causal-v1"


def _teacher_top2(
    game: Any,
    player_id: int,
    *,
    maximum_margin: float,
    teacher: HeuristicTeacherAgent,
) -> tuple[GameAction, GameAction, float] | None:
    """Return the reproducible frozen top-two pair for an eligible state."""

    ranked = teacher.explain_discard(game, player_id)
    if len(ranked) < 2:
        return None
    teacher_action = GameAction("discard", int(ranked[0]["tile"]))
    alternative = GameAction("discard", int(ranked[1]["tile"]))
    margin = float(ranked[0]["score"]) - float(ranked[1]["score"])
    if (
        teacher_action == alternative
        or not math.isfinite(margin)
        or margin < -1e-9
        or margin > maximum_margin + 1e-9
    ):
        return None
    return teacher_action, alternative, margin


class LowMarginTop2InterventionBehavior:
    """Randomize frozen Teacher top-1 versus top-2 once per trajectory."""

    def __init__(
        self,
        *,
        seed: int,
        maximum_margin: float = 2.0,
        alternative_probability: float = 0.5,
        teacher: HeuristicTeacherAgent | None = None,
    ) -> None:
        if not math.isfinite(maximum_margin) or maximum_margin < 0.0:
            raise ValueError("maximum_margin 必须是有限非负数")
        if not 0.0 < alternative_probability < 1.0:
            raise ValueError("alternative_probability 必须位于 (0, 1)")
        self.rng = random.Random(seed)
        self.maximum_margin = float(maximum_margin)
        self.alternative_probability = float(alternative_probability)
        self.teacher = teacher or HeuristicTeacherAgent()
        self.intervened = False
        self._last_action: GameAction | None = None
        self._last_probability: float | None = None
        self.eligible_trajectories = 0
        self.alternative_assignments = 0
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
        frozen = self.teacher.choose_turn_action(game, player_id)
        if frozen not in legal:
            raise RuntimeError("Teacher 产生规则引擎非法动作")
        if (
            self.intervened
            or game.rules.profile != "classic"
            or game.phase != "discard"
            or game.tour_state is not None
            or game.gold_discard_lock_seat == player_id
            or player_id in game.opening_wait_seats
            or frozen.kind != "discard"
        ):
            return self._remember(frozen, 1.0)
        pair = _teacher_top2(
            game,
            player_id,
            maximum_margin=self.maximum_margin,
            teacher=self.teacher,
        )
        if pair is None:
            return self._remember(frozen, 1.0)
        top, alternative, _margin = pair
        if top != frozen:
            raise RuntimeError("Teacher choose/explain 在同一公开状态不一致")
        if alternative not in legal:
            raise RuntimeError("Teacher top-2 不是规则引擎合法动作")

        self.intervened = True
        self.eligible_trajectories += 1
        if self.rng.random() < self.alternative_probability:
            self.alternative_assignments += 1
            return self._remember(alternative, self.alternative_probability)
        self.teacher_assignments += 1
        return self._remember(top, 1.0 - self.alternative_probability)

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


def low_margin_top2_indices(
    decision: TeacherDecision,
    *,
    maximum_margin: float,
) -> tuple[int, int, float]:
    """Reproduce top-1/top-2 indices using only the exported observation."""

    if decision.state.get("phase") != "discard":
        raise ValueError("低分差干预必须位于 discard phase")
    game = _ActorVisibleReviewGame({"state": decision.state})
    pair = _teacher_top2(
        game,
        0,
        maximum_margin=maximum_margin,
        teacher=HeuristicTeacherAgent(),
    )
    if pair is None:
        raise ValueError("actor-visible 状态无法复现低分差 top-2")
    teacher_action, alternative, margin = pair
    indices: list[int] = []
    for action in (teacher_action, alternative):
        matches = [
            index for index, legal in enumerate(decision.legal_actions)
            if legal == action
        ]
        if len(matches) != 1:
            raise ValueError("top-2 动作无法唯一映射到合法动作")
        indices.append(matches[0])
    if indices[0] != decision.chosen_index or indices[0] == indices[1]:
        raise ValueError("导出的 Teacher 标签与 top-2 复算不一致")
    return indices[0], indices[1], margin


def _mean_interval(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "groups": 0,
            "mean": None,
            "stderr": None,
            "95pct_low": None,
            "95pct_high": None,
            "sample_standard_deviation": None,
        }
    mean = sum(values) / len(values)
    if len(values) == 1:
        stderr = None
        low = None
        high = None
        standard_deviation = None
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        standard_deviation = math.sqrt(variance)
        stderr = standard_deviation / math.sqrt(len(values))
        low = mean - 1.96 * stderr
        high = mean + 1.96 * stderr
    return {
        "groups": len(values),
        "mean": mean,
        "stderr": stderr,
        "95pct_low": low,
        "95pct_high": high,
        "sample_standard_deviation": standard_deviation,
    }


def audit_low_margin_top2_interventions(
    trajectories: Iterable[TrainingTrajectory],
    *,
    maximum_margin: float,
    minimum_interventions: int,
    minimum_wall_groups: int,
) -> dict[str, Any]:
    """Fail closed on support, visibility, grouping and terminal scores."""

    if not math.isfinite(maximum_margin) or maximum_margin < 0.0:
        raise ValueError("maximum_margin 必须是有限非负数")
    if minimum_interventions <= 0 or minimum_wall_groups <= 1:
        raise ValueError("低分差干预审计门槛无效")
    rows = tuple(trajectories)
    issues: Counter[str] = Counter()
    group_contributions: dict[str, list[float]] = defaultdict(list)
    group_seats: dict[str, set[int]] = defaultdict(set)
    margins: list[float] = []
    alternative_assignments = 0
    teacher_assignments = 0
    intervention_trajectories = 0
    trajectories_without_eligible_state = 0
    terminal_candidate_scores: list[float] = []

    for trajectory in rows:
        metadata = trajectory.source_metadata
        if (
            trajectory.profile != "classic"
            or trajectory.rules_version != "xiamen-classic-full-v2"
            or metadata.get("collector") != "candidate_vs_teacher_dagger"
            or metadata.get("behavior_policy") != LOW_MARGIN_TOP2_BEHAVIOR_VERSION
            or metadata.get("base_policy") != "heuristic_teacher"
            or metadata.get("candidate_policy") != LOW_MARGIN_TOP2_POLICY_VERSION
            or metadata.get("alternative_action_probability") != 0.5
            or metadata.get("maximum_interventions_per_trajectory") != 1
            or float(metadata.get("maximum_teacher_score_margin", -1.0))
            != float(maximum_margin)
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

        randomized: list[tuple[TeacherDecision, int, float]] = []
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
                    issues["nonrandom_action_differs_from_teacher"] += 1
                    valid = False
                    break
                continue
            if abs(probability - 0.5) > 1e-12:
                issues["invalid_binary_propensity"] += 1
                valid = False
                break
            try:
                teacher_index, alternative_index, margin = low_margin_top2_indices(
                    decision, maximum_margin=maximum_margin
                )
            except ValueError:
                issues["unreproducible_top2_intervention"] += 1
                valid = False
                break
            if executed not in {teacher_index, alternative_index}:
                issues["executed_action_outside_binary_support"] += 1
                valid = False
                break
            randomized.append((decision, alternative_index, margin))
        if not valid:
            continue
        if len(randomized) > 1:
            issues["multiple_interventions_in_trajectory"] += 1
            continue

        scores = trajectory.outcome.get("scores")
        if not (
            isinstance(scores, list)
            and len(scores) == 4
            and all(
                not isinstance(score, bool)
                and isinstance(score, (int, float))
                and math.isfinite(float(score))
                for score in scores
            )
            and abs(sum(float(score) for score in scores)) <= 1e-9
        ):
            issues["invalid_or_non_zero_sum_terminal_scores"] += 1
            continue
        score = float(scores[candidate_seat])
        terminal_candidate_scores.append(score)
        contribution = 0.0
        if randomized:
            intervention_trajectories += 1
            decision, alternative_index, margin = randomized[0]
            margins.append(margin)
            if decision.executed_index == alternative_index:
                alternative_assignments += 1
                contribution = score / 0.5
            else:
                teacher_assignments += 1
                contribution = -score / 0.5
        else:
            trajectories_without_eligible_state += 1
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
    assignment_fraction = (
        alternative_assignments / intervention_trajectories
        if intervention_trajectories
        else None
    )
    if terminal_candidate_scores and len(set(terminal_candidate_scores)) <= 1:
        issues["degenerate_terminal_score_distribution"] += 1

    gate_reasons: list[str] = []
    if issues:
        gate_reasons.append("invalid_or_unreproducible_trajectory")
    if intervention_trajectories < minimum_interventions:
        gate_reasons.append("insufficient_interventions")
    if len(complete_group_means) < minimum_wall_groups:
        gate_reasons.append("insufficient_complete_wall_groups")
    if not (
        isinstance(assignment_fraction, float)
        and 0.3 <= assignment_fraction <= 0.7
    ):
        gate_reasons.append("random_assignment_imbalance")

    margin_bins = Counter(
        "exact_tie" if margin <= 1e-12 else "margin_(0,1]" if margin <= 1.0 else "margin_(1,2]"
        for margin in margins
    )
    return {
        "status": "low_margin_top2_intervention_audit",
        "behavior_version": LOW_MARGIN_TOP2_BEHAVIOR_VERSION,
        "trajectories": len(rows),
        "complete_wall_groups": len(complete_group_means),
        "intervention_trajectories": intervention_trajectories,
        "trajectories_without_eligible_state": trajectories_without_eligible_state,
        "alternative_assignments": alternative_assignments,
        "teacher_assignments": teacher_assignments,
        "alternative_assignment_fraction": assignment_fraction,
        "teacher_margin_bins": dict(sorted(margin_bins.items())),
        "terminal_candidate_score_min": (
            min(terminal_candidate_scores) if terminal_candidate_scores else None
        ),
        "terminal_candidate_score_max": (
            max(terminal_candidate_scores) if terminal_candidate_scores else None
        ),
        "minimum_interventions": minimum_interventions,
        "minimum_wall_groups": minimum_wall_groups,
        "issues": dict(sorted(issues.items())),
        "structurally_ready": not gate_reasons,
        "gate_reasons": gate_reasons,
        "horvitz_thompson_alternative_minus_teacher_per_wall": _mean_interval(
            complete_group_means
        ),
        "estimand": (
            "first eligible top-2 low-margin discard intervention followed by "
            "frozen Teacher, averaged over four seat rotations per physical wall"
        ),
        "privacy": "aggregate_only_no_state_hand_action_tile_group_id_or_terminal_trace",
        "warning": (
            "Pilot direction cannot select a policy; a learned gate needs grouped "
            "train/validation/terminal data and a separate full-deployment evaluation."
        ),
    }


@dataclass(frozen=True)
class LowMarginCausalRecord:
    """One actor-visible randomized decision and its observed terminal score."""

    record_id: str
    group_id: str
    candidate_seat: int
    state: dict[str, Any] | None
    teacher_action: GameAction | None
    alternative_action: GameAction | None
    teacher_margin: float | None
    executed_arm: str
    propensity: float
    terminal_candidate_score: float

    def payload(self) -> dict[str, Any]:
        def action_payload(action: GameAction | None) -> dict[str, Any] | None:
            if action is None:
                return None
            return {
                "kind": action.kind,
                "tile": action.tile,
                "tiles": list(action.tiles),
            }

        return {
            "version": LOW_MARGIN_CAUSAL_RECORD_VERSION,
            "record_id": self.record_id,
            "group_id": self.group_id,
            "profile": "classic",
            "rules_version": "xiamen-classic-full-v2",
            "behavior_version": LOW_MARGIN_TOP2_BEHAVIOR_VERSION,
            "candidate_policy": LOW_MARGIN_TOP2_POLICY_VERSION,
            "candidate_seat": self.candidate_seat,
            "state": self.state,
            "teacher_action": action_payload(self.teacher_action),
            "alternative_action": action_payload(self.alternative_action),
            "teacher_margin": self.teacher_margin,
            "executed_arm": self.executed_arm,
            "propensity": self.propensity,
            "terminal_candidate_score": self.terminal_candidate_score,
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        validate_actor_visible_pair: bool = True,
    ) -> "LowMarginCausalRecord":
        expected = {
            "version",
            "record_id",
            "group_id",
            "profile",
            "rules_version",
            "behavior_version",
            "candidate_policy",
            "candidate_seat",
            "state",
            "teacher_action",
            "alternative_action",
            "teacher_margin",
            "executed_arm",
            "propensity",
            "terminal_candidate_score",
        }
        if set(payload) != expected:
            raise ValueError("低分差因果记录字段集合不符合 v1 契约")
        if (
            payload.get("version") != LOW_MARGIN_CAUSAL_RECORD_VERSION
            or payload.get("profile") != "classic"
            or payload.get("rules_version") != "xiamen-classic-full-v2"
            or payload.get("behavior_version") != LOW_MARGIN_TOP2_BEHAVIOR_VERSION
            or payload.get("candidate_policy") != LOW_MARGIN_TOP2_POLICY_VERSION
        ):
            raise ValueError("低分差因果记录身份不一致")
        record_id = payload.get("record_id")
        group_id = payload.get("group_id")
        candidate_seat = payload.get("candidate_seat")
        if (
            not isinstance(record_id, str)
            or not record_id
            or not isinstance(group_id, str)
            or not group_id
            or isinstance(candidate_seat, bool)
            or not isinstance(candidate_seat, int)
            or not 0 <= candidate_seat < 4
        ):
            raise ValueError("低分差因果记录 ID／座位无效")
        state = payload.get("state")

        def parse_action(raw: Any) -> GameAction:
            if not isinstance(raw, dict) or set(raw) != {"kind", "tile", "tiles"}:
                raise ValueError("低分差因果动作字段无效")
            if raw.get("kind") != "discard":
                raise ValueError("低分差因果动作必须为 discard")
            tile = raw.get("tile")
            tiles = raw.get("tiles")
            if (
                isinstance(tile, bool)
                or not isinstance(tile, int)
                or not isinstance(tiles, list)
                or tiles
            ):
                raise ValueError("低分差因果弃牌内容无效")
            return GameAction("discard", tile)

        margin = payload.get("teacher_margin")
        propensity = payload.get("propensity")
        score = payload.get("terminal_candidate_score")
        arm = payload.get("executed_arm")
        if arm == "none":
            if (
                state is not None
                or payload.get("teacher_action") is not None
                or payload.get("alternative_action") is not None
                or margin is not None
                or isinstance(propensity, bool)
                or not isinstance(propensity, (int, float))
                or abs(float(propensity) - 1.0) > 1e-12
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise ValueError("no_intervention 记录必须是显式空动作和单位 propensity")
            return cls(
                record_id=record_id,
                group_id=group_id,
                candidate_seat=candidate_seat,
                state=None,
                teacher_action=None,
                alternative_action=None,
                teacher_margin=None,
                executed_arm="none",
                propensity=1.0,
                terminal_candidate_score=float(score),
            )
        if not isinstance(state, dict):
            raise ValueError("有干预的低分差因果记录 state 无效")
        teacher_action = parse_action(payload.get("teacher_action"))
        alternative_action = parse_action(payload.get("alternative_action"))
        if (
            teacher_action == alternative_action
            or isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(float(margin))
            or not 0.0 <= float(margin) <= 2.0
            or arm not in {"teacher", "alternative"}
            or isinstance(propensity, bool)
            or not isinstance(propensity, (int, float))
            or abs(float(propensity) - 0.5) > 1e-12
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise ValueError("低分差因果 margin／arm／propensity／score 无效")
        record = cls(
            record_id=record_id,
            group_id=group_id,
            candidate_seat=candidate_seat,
            state=state,
            teacher_action=teacher_action,
            alternative_action=alternative_action,
            teacher_margin=float(margin),
            executed_arm=arm,
            propensity=float(propensity),
            terminal_candidate_score=float(score),
        )
        if validate_actor_visible_pair:
            record.validate_actor_visible_pair(maximum_margin=2.0)
        return record

    def validate_actor_visible_pair(self, *, maximum_margin: float) -> None:
        if self.executed_arm == "none":
            if (
                self.state is not None
                or self.teacher_action is not None
                or self.alternative_action is not None
                or self.teacher_margin is not None
                or abs(self.propensity - 1.0) > 1e-12
            ):
                raise ValueError("no_intervention 记录内容不一致")
            return
        if (
            self.state is None
            or self.teacher_action is None
            or self.alternative_action is None
            or self.teacher_margin is None
        ):
            raise ValueError("有干预的低分差因果记录缺少动作对")
        decision = TeacherDecision(
            profile="classic",
            seed=None,
            seat=0,
            state=self.state,
            legal_actions=(self.teacher_action, self.alternative_action),
            chosen_index=0,
            executed_index=0 if self.executed_arm == "teacher" else 1,
            executed_probability=self.propensity,
        )
        teacher_index, alternative_index, margin = low_margin_top2_indices(
            decision, maximum_margin=maximum_margin
        )
        if teacher_index != 0 or alternative_index != 1:
            raise ValueError("低分差因果动作对顺序不可复现")
        if abs(margin - self.teacher_margin) > 1e-9:
            raise ValueError("低分差因果记录的 Teacher margin 不可复现")


def extract_low_margin_causal_records(
    trajectories: Iterable[TrainingTrajectory],
    *,
    maximum_margin: float = 2.0,
) -> list[LowMarginCausalRecord]:
    """Reduce full safe trajectories to the one randomized actor decision."""

    records: list[LowMarginCausalRecord] = []
    for trajectory in trajectories:
        metadata = trajectory.source_metadata
        if (
            trajectory.profile != "classic"
            or trajectory.rules_version != "xiamen-classic-full-v2"
            or metadata.get("behavior_policy") != LOW_MARGIN_TOP2_BEHAVIOR_VERSION
            or metadata.get("candidate_policy") != LOW_MARGIN_TOP2_POLICY_VERSION
            or metadata.get("alternative_action_probability") != 0.5
            or metadata.get("maximum_interventions_per_trajectory") != 1
        ):
            raise ValueError("低分差因果瘦身输入元数据无效")
        randomized = [
            decision for decision in trajectory.decisions
            if decision.executed_probability is not None
            and abs(decision.executed_probability - 0.5) <= 1e-12
        ]
        if len(randomized) > 1:
            raise ValueError("每条正式低分差轨迹最多一次随机干预")
        if any(
            decision.executed_index is None
            or decision.executed_probability is None
            or (
                abs(decision.executed_probability - 1.0) <= 1e-12
                and decision.executed_index != decision.chosen_index
            )
            for decision in trajectory.decisions
        ):
            raise ValueError("正式低分差轨迹含未知 propensity 或非随机偏离 Teacher")
        candidate_seat = metadata.get("candidate_seat")
        group_id = trajectory.split_group_id
        scores = trajectory.outcome.get("scores")
        if (
            isinstance(candidate_seat, bool)
            or not isinstance(candidate_seat, int)
            or not 0 <= candidate_seat < 4
            or not isinstance(group_id, str)
            or not group_id
            or not isinstance(scores, list)
            or len(scores) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in scores
            )
            or abs(sum(float(value) for value in scores)) > 1e-9
        ):
            raise ValueError("正式低分差轨迹 group／score 无效")
        if randomized:
            decision = randomized[0]
            teacher_index, alternative_index, margin = low_margin_top2_indices(
                decision, maximum_margin=maximum_margin
            )
            if decision.executed_index not in {teacher_index, alternative_index}:
                raise ValueError("正式低分差轨迹执行动作不在二元支持内")
            record = LowMarginCausalRecord(
                record_id=trajectory.trajectory_id,
                group_id=group_id,
                candidate_seat=candidate_seat,
                state=dict(decision.state),
                teacher_action=decision.legal_actions[teacher_index],
                alternative_action=decision.legal_actions[alternative_index],
                teacher_margin=margin,
                executed_arm=(
                    "alternative"
                    if decision.executed_index == alternative_index
                    else "teacher"
                ),
                propensity=0.5,
                terminal_candidate_score=float(scores[candidate_seat]),
            )
        else:
            record = LowMarginCausalRecord(
                record_id=trajectory.trajectory_id,
                group_id=group_id,
                candidate_seat=candidate_seat,
                state=None,
                teacher_action=None,
                alternative_action=None,
                teacher_margin=None,
                executed_arm="none",
                propensity=1.0,
                terminal_candidate_score=float(scores[candidate_seat]),
            )
        record.validate_actor_visible_pair(maximum_margin=maximum_margin)
        records.append(record)
    return records


def write_low_margin_causal_jsonl(
    records: Iterable[LowMarginCausalRecord], path: str | Path
) -> int:
    destination = Path(path)
    if destination.exists():
        raise ValueError("低分差因果 JSONL 已存在，拒绝覆盖")
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.payload(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def read_low_margin_causal_jsonl(
    path: str | Path,
    *,
    validate_actor_visible_pairs: bool = True,
) -> list[LowMarginCausalRecord]:
    records: list[LowMarginCausalRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(
                    LowMarginCausalRecord.from_payload(
                        payload,
                        validate_actor_visible_pair=validate_actor_visible_pairs,
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"第 {line_number} 行低分差因果记录无效：{error}"
                ) from error
    return records


def audit_low_margin_causal_records(
    records: Iterable[LowMarginCausalRecord],
    *,
    expected_wall_groups: int,
    minimum_alternative_fraction: float = 0.45,
    maximum_alternative_fraction: float = 0.55,
    minimum_intervention_fraction: float = 0.95,
    validate_actor_visible_pairs: bool = True,
) -> dict[str, Any]:
    """Audit slim records without exporting any individual state or action."""

    if expected_wall_groups <= 0:
        raise ValueError("expected_wall_groups 必须为正数")
    if not (
        0.0 < minimum_alternative_fraction
        < maximum_alternative_fraction < 1.0
    ):
        raise ValueError("正式随机分配比例区间无效")
    if not 0.0 < minimum_intervention_fraction <= 1.0:
        raise ValueError("正式干预覆盖下界无效")
    rows = tuple(records)
    issues: Counter[str] = Counter()
    record_ids: set[str] = set()
    group_seats: dict[str, set[int]] = defaultdict(set)
    group_counts: Counter[str] = Counter()
    arms: Counter[str] = Counter()
    scores: list[float] = []
    margin_bins: Counter[str] = Counter()
    for record in rows:
        if record.record_id in record_ids:
            issues["duplicate_record_id"] += 1
        record_ids.add(record.record_id)
        group_counts[record.group_id] += 1
        group_seats[record.group_id].add(record.candidate_seat)
        arms[record.executed_arm] += 1
        scores.append(record.terminal_candidate_score)
        if record.teacher_margin is not None:
            margin_bins[
                "exact_tie"
                if record.teacher_margin <= 1e-12
                else "margin_(0,1]"
                if record.teacher_margin <= 1.0
                else "margin_(1,2]"
            ] += 1
        if validate_actor_visible_pairs:
            try:
                record.validate_actor_visible_pair(maximum_margin=2.0)
            except ValueError:
                issues["unreproducible_actor_visible_pair"] += 1
    incomplete = sum(
        group_counts[group_id] != 4 or seats != {0, 1, 2, 3}
        for group_id, seats in group_seats.items()
    )
    if incomplete:
        issues["incomplete_wall_rotation_group"] += incomplete
    if len(group_seats) != expected_wall_groups:
        issues["unexpected_wall_group_count"] += abs(
            len(group_seats) - expected_wall_groups
        ) or 1
    if len(rows) != expected_wall_groups * 4:
        issues["unexpected_record_count"] += abs(
            len(rows) - expected_wall_groups * 4
        ) or 1
    if scores and len(set(scores)) <= 1:
        issues["degenerate_terminal_score_distribution"] += 1
    intervention_count = arms["alternative"] + arms["teacher"]
    intervention_fraction = intervention_count / len(rows) if rows else None
    alternative_fraction = (
        arms["alternative"] / intervention_count if intervention_count else None
    )
    if not (
        isinstance(intervention_fraction, float)
        and intervention_fraction >= minimum_intervention_fraction
    ):
        issues["insufficient_intervention_coverage"] += 1
    if not (
        isinstance(alternative_fraction, float)
        and minimum_alternative_fraction
        <= alternative_fraction
        <= maximum_alternative_fraction
    ):
        issues["formal_random_assignment_imbalance"] += 1
    return {
        "status": "low_margin_top2_causal_record_audit",
        "records": len(rows),
        "wall_groups": len(group_seats),
        "expected_wall_groups": expected_wall_groups,
        "teacher_assignments": arms["teacher"],
        "alternative_assignments": arms["alternative"],
        "no_intervention_trajectories": arms["none"],
        "intervention_fraction": intervention_fraction,
        "alternative_assignment_fraction": alternative_fraction,
        "teacher_margin_bins": dict(sorted(margin_bins.items())),
        "terminal_candidate_score_min": min(scores) if scores else None,
        "terminal_candidate_score_max": max(scores) if scores else None,
        "issues": dict(sorted(issues.items())),
        "ready": not issues,
        "privacy": "aggregate_only_no_record_group_state_action_or_score_row_exported",
    }

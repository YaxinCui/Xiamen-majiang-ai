"""Actor-visible exact-tie records labelled by paired simulator continuations.

The collector deliberately uses a *private* source-world clone only as a
training-time outcome oracle.  Every tied action is forced in that same world
and all players then return to the frozen rule Teacher.  Exported records keep
only the acting player's information set and terminal score targets; walls,
opponent hands, RNG state and physical seeds never leave the collector.

One source world is not an information-set value estimate.  It is one Monte
Carlo draw from the hidden-state distribution reached by Teacher self-play.
Consequently records must be split by physical-wall group and used only by a
small baseline-bootstrapped policy that can act inside the exact Teacher tie.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from .agents import GameAction, HeuristicTeacherAgent
from .game import XiamenMahjongGame
from .rules import XiamenRules
from .training import (
    _continue_counterfactual_rollout,
    _perspective_state,
    _run_candidate_base_hand,
    _turn_actions,
)


EXACT_TIE_ROLLOUT_VERSION = "xiamen-exact-tie-source-world-paired-rollout-v1"
EXACT_TIE_ROLLOUT_TARGET = (
    "paired_terminal_net_score_one_natural_private_world_teacher_suffix"
)


def _action_payload(action: GameAction) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": action.kind}
    if action.tile is not None:
        payload["tile"] = int(action.tile)
    if action.tiles:
        payload["tiles"] = [int(tile) for tile in action.tiles]
    return payload


def _action_from_payload(payload: Mapping[str, Any]) -> GameAction:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("kind"), str):
        raise ValueError("exact-tie action 格式无效")
    tile = payload.get("tile")
    tiles = payload.get("tiles", ())
    if tile is not None and (isinstance(tile, bool) or not isinstance(tile, int)):
        raise ValueError("exact-tie action tile 无效")
    if not isinstance(tiles, (list, tuple)) or not all(
        not isinstance(value, bool) and isinstance(value, int) for value in tiles
    ):
        raise ValueError("exact-tie action tiles 无效")
    return GameAction(str(payload["kind"]), tile, tuple(int(value) for value in tiles))


def _contains_forbidden_private_key(value: Any) -> bool:
    forbidden = {
        "wall",
        "seed",
        "rng",
        "rng_state",
        "opponent_hands",
        "private_world",
        "source_world",
    }
    if isinstance(value, Mapping):
        return any(
            str(key) in forbidden or _contains_forbidden_private_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_private_key(item) for item in value)
    return False


@dataclass(frozen=True)
class ExactTieRolloutRecord:
    """One exact Teacher tie and paired terminal returns for its tied actions."""

    item_id: str
    split_group_id: str
    profile: str
    rules_version: str
    state: dict[str, Any]
    actions: tuple[GameAction, ...]
    teacher_index: int
    terminal_scores: tuple[int, ...]

    def validate(self) -> None:
        if not self.item_id or not self.split_group_id:
            raise ValueError("exact-tie item/group id 不能为空")
        if self.profile != "classic":
            raise ValueError("exact-tie v1 只接受 classic")
        if not isinstance(self.state, dict) or self.state.get("phase") != "discard":
            raise ValueError("exact-tie state 必须是 actor-visible discard")
        if _contains_forbidden_private_key(self.state):
            raise ValueError("exact-tie state 含私有模拟字段")
        if len(self.actions) < 2 or len(self.actions) != len(self.terminal_scores):
            raise ValueError("exact-tie 动作/终局分数数量无效")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("exact-tie 动作必须互不重复")
        if any(action.kind != "discard" or action.tile is None for action in self.actions):
            raise ValueError("exact-tie v1 只接受弃牌动作")
        if not 0 <= self.teacher_index < len(self.actions):
            raise ValueError("exact-tie Teacher index 越界")
        if any(isinstance(score, bool) or not isinstance(score, int) for score in self.terminal_scores):
            raise ValueError("exact-tie terminal score 必须是整数")

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "version": EXACT_TIE_ROLLOUT_VERSION,
            "target_semantics": EXACT_TIE_ROLLOUT_TARGET,
            "item_id": self.item_id,
            "split_group_id": self.split_group_id,
            "profile": self.profile,
            "rules_version": self.rules_version,
            "state": self.state,
            "actions": [_action_payload(action) for action in self.actions],
            "teacher_index": self.teacher_index,
            "terminal_scores": list(self.terminal_scores),
            "private_simulator_state_exported": False,
            "source_worlds_per_record": 1,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ExactTieRolloutRecord":
        if payload.get("version") != EXACT_TIE_ROLLOUT_VERSION:
            raise ValueError("不支持的 exact-tie rollout 版本")
        if payload.get("target_semantics") != EXACT_TIE_ROLLOUT_TARGET:
            raise ValueError("exact-tie rollout target semantics 不匹配")
        if payload.get("private_simulator_state_exported") is not False:
            raise ValueError("exact-tie rollout 私有状态契约无效")
        if payload.get("source_worlds_per_record") != 1:
            raise ValueError("exact-tie rollout v1 必须明确单一自然 source world")
        state = payload.get("state")
        actions = payload.get("actions")
        scores = payload.get("terminal_scores")
        if not isinstance(state, dict) or not isinstance(actions, list) or not isinstance(scores, list):
            raise ValueError("exact-tie rollout payload 缺少 state/actions/scores")
        record = cls(
            item_id=str(payload.get("item_id", "")),
            split_group_id=str(payload.get("split_group_id", "")),
            profile=str(payload.get("profile", "")),
            rules_version=str(payload.get("rules_version", "")),
            state=state,
            actions=tuple(_action_from_payload(action) for action in actions),
            teacher_index=int(payload.get("teacher_index", -1)),
            terminal_scores=tuple(scores),
        )
        record.validate()
        return record


def _stable_identifier(*parts: object) -> str:
    serialized = "|".join(str(part) for part in parts)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:32]


def _top_tied_actions(snapshot: XiamenMahjongGame, seat: int) -> tuple[GameAction, ...]:
    """Return only engine-legal ordinary discards sharing Teacher's top score."""

    teacher = HeuristicTeacherAgent()
    if (
        snapshot.phase != "discard"
        or snapshot.current_player != seat
        or snapshot.tour_state is not None
        or snapshot.gold_discard_lock_seat == seat
        or seat in snapshot.opening_wait_seats
    ):
        return ()
    teacher_action = teacher.choose_turn_action(snapshot, seat)
    if teacher_action.kind != "discard":
        return ()
    ranked = teacher.explain_discard(snapshot, seat)
    if len(ranked) < 2:
        return ()
    top_score = float(ranked[0]["score"])
    tied_tiles = {
        int(row["tile"]) for row in ranked if float(row["score"]) == top_score
    }
    if len(tied_tiles) < 2:
        return ()
    legal = tuple(_turn_actions(snapshot, seat))
    tied = tuple(
        action
        for action in legal
        if action.kind == "discard" and int(action.tile) in tied_tiles
    )
    if teacher_action not in tied or len(tied) < 2:
        raise RuntimeError("Teacher 完全并列动作无法映射到引擎合法动作")
    # Keep the baseline first.  Remaining order is engine-stable but has no
    # training significance because every action is encoded independently.
    return (teacher_action, *(action for action in tied if action != teacher_action))


def collect_exact_tie_rollout_records(
    *,
    seed_count: int,
    seed: int,
    samples_per_rotation: int = 1,
    selection_seed: int = 202638700,
    future_wall_permutations: int = 1,
) -> tuple[list[ExactTieRolloutRecord], dict[str, Any]]:
    """Collect paired exact-tie returns from fresh Teacher self-play walls."""

    if seed_count <= 0 or samples_per_rotation <= 0 or future_wall_permutations <= 0:
        raise ValueError(
            "seed_count、samples_per_rotation 和 future_wall_permutations 必须为正数"
        )
    rules = XiamenRules.from_profile("classic")
    teacher = HeuristicTeacherAgent()
    rng = random.Random(selection_seed)
    future_rng = random.Random(selection_seed ^ 0x5A17E5)
    records: list[ExactTieRolloutRecord] = []
    eligible_snapshots = 0
    branch_rollouts = 0
    rotations_without_eligible_tie = 0
    selected_decisions = 0
    tie_sizes: dict[int, int] = {}
    delta_counts = {"teacher_best": 0, "alternative_best": 0, "all_equal": 0}
    for wall_offset in range(seed_count):
        hand_seed = seed + wall_offset
        group_id = _stable_identifier(EXACT_TIE_ROLLOUT_VERSION, "wall", hand_seed)
        for candidate_seat in range(rules.player_count):
            opponents = {
                seat: ("heuristic_teacher", teacher)
                for seat in range(rules.player_count)
                if seat != candidate_seat
            }
            game = XiamenMahjongGame(
                seed=hand_seed,
                rules=rules,
                auto_advance=False,
                human_seat=-1,
            )
            snapshots = _run_candidate_base_hand(
                game,
                candidate_seat=candidate_seat,
                candidate_policy=teacher,
                opponents=opponents,
            )
            eligible = [
                (snapshot, actions)
                for snapshot in snapshots
                if (actions := _top_tied_actions(snapshot.game, candidate_seat))
            ]
            eligible_snapshots += len(eligible)
            if not eligible:
                rotations_without_eligible_tie += 1
                continue
            selected = rng.sample(eligible, min(samples_per_rotation, len(eligible)))
            for ordinal, (selected_snapshot, actions) in enumerate(selected):
                snapshot = selected_snapshot.game
                selected_decisions += 1
                public_state = _perspective_state(snapshot, candidate_seat)
                for permutation in range(future_wall_permutations):
                    rollout_world = copy.deepcopy(snapshot)
                    if future_wall_permutations > 1:
                        future_rng.shuffle(rollout_world.wall)
                    scores = tuple(
                        _continue_counterfactual_rollout(
                            copy.deepcopy(rollout_world),
                            candidate_seat=candidate_seat,
                            candidate_policy=teacher,
                            opponents=opponents,
                            forced_action=action,
                        )
                        for action in actions
                    )
                    branch_rollouts += len(actions)
                    tie_sizes[len(actions)] = tie_sizes.get(len(actions), 0) + 1
                    best = max(scores)
                    if len(set(scores)) == 1:
                        delta_counts["all_equal"] += 1
                    elif scores[0] == best:
                        delta_counts["teacher_best"] += 1
                    else:
                        delta_counts["alternative_best"] += 1
                    item_id = _stable_identifier(
                        EXACT_TIE_ROLLOUT_VERSION,
                        group_id,
                        candidate_seat,
                        len(snapshot.public_actions),
                        ordinal,
                        permutation,
                        *(_action_payload(action) for action in actions),
                    )
                    record = ExactTieRolloutRecord(
                        item_id=item_id,
                        split_group_id=group_id,
                        profile=rules.profile,
                        rules_version=rules.version,
                        state=public_state,
                        actions=actions,
                        teacher_index=0,
                        terminal_scores=scores,
                    )
                    record.validate()
                    records.append(record)
    report = {
        "status": "exact_tie_source_world_paired_rollout_collected",
        "version": EXACT_TIE_ROLLOUT_VERSION,
        "target_semantics": EXACT_TIE_ROLLOUT_TARGET,
        "protocol": {
            "profile": "classic",
            "seed_count": seed_count,
            "seat_rotations_per_wall": rules.player_count,
            "samples_per_rotation": samples_per_rotation,
            "selection_seed": selection_seed,
            "continuation": "frozen_heuristic_teacher_all_seats",
            "split_unit": "physical_wall_group",
            "source_worlds_per_record": 1,
            "future_wall_permutations_per_selected_decision": (
                future_wall_permutations
            ),
            "opponent_hidden_hand_allocations_per_selected_decision": 1,
        },
        "records": len(records),
        "selected_decisions": selected_decisions,
        "wall_groups": len({record.split_group_id for record in records}),
        "eligible_snapshots": eligible_snapshots,
        "rotations_without_eligible_tie": rotations_without_eligible_tie,
        "branch_rollouts": branch_rollouts,
        "tie_size_counts": dict(sorted(tie_sizes.items())),
        "paired_outcome_counts": delta_counts,
        "privacy": {
            "export": "actor_visible_state_plus_tied_actions_and_terminal_targets",
            "private_simulator_state_exported": False,
            "seed_exported_per_record": False,
            "interpretation": (
                "each row is one future-wall draw conditional on one naturally reached "
                "hidden-hand allocation; repeated rows are grouped before averaging and "
                "are not independent belief worlds"
            ),
        },
    }
    return records, report


def write_exact_tie_rollout_records(
    records: Iterable[ExactTieRolloutRecord], path: Path
) -> int:
    rows = tuple(records)
    if len({record.item_id for record in rows}) != len(rows):
        raise ValueError("exact-tie rollout item_id 重复")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(record.payload(), ensure_ascii=False) + "\n")
    return len(rows)


def read_exact_tie_rollout_records(path: Path) -> list[ExactTieRolloutRecord]:
    records: list[ExactTieRolloutRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                record = ExactTieRolloutRecord.from_payload(payload)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"exact-tie rollout 第 {line_number} 行无效：{error}") from error
            if record.item_id in seen:
                raise ValueError("exact-tie rollout item_id 重复")
            seen.add(record.item_id)
            records.append(record)
    return records


def split_exact_tie_rollout_records_by_group(
    records: Sequence[ExactTieRolloutRecord],
    *,
    split_salt: str,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, list[ExactTieRolloutRecord]]:
    """Deterministically split records without separating one physical wall."""

    if not split_salt:
        raise ValueError("exact-tie split_salt 不能为空")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("exact-tie split 比例必须位于 (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("exact-tie train + validation 比例必须小于 1")
    group_splits: dict[str, str] = {}
    partitions = {"train": [], "validation": [], "test": []}
    for record in records:
        record.validate()
        if record.split_group_id not in group_splits:
            digest = hashlib.sha256(
                f"{split_salt}|{record.split_group_id}".encode("utf-8")
            ).digest()
            fraction = int.from_bytes(digest[:8], "big") / float(1 << 64)
            group_splits[record.split_group_id] = (
                "train"
                if fraction < train_fraction
                else "validation"
                if fraction < train_fraction + validation_fraction
                else "test"
            )
        partitions[group_splits[record.split_group_id]].append(record)
    return partitions


def exact_tie_pairwise_examples(
    records: Sequence[ExactTieRolloutRecord],
) -> list[tuple[str, dict[str, Any], tuple[GameAction, GameAction], int]]:
    """Expand only unequal paired outcomes into public pairwise preferences.

    The returned target is 0 or 1 relative to the action pair.  Equal-return
    pairs carry no ordering information and are omitted rather than being
    broken by tile id or Teacher identity.
    """

    examples: list[tuple[str, dict[str, Any], tuple[GameAction, GameAction], int]] = []
    for record in records:
        record.validate()
        for left in range(len(record.actions)):
            for right in range(left + 1, len(record.actions)):
                left_score = record.terminal_scores[left]
                right_score = record.terminal_scores[right]
                if left_score == right_score:
                    continue
                examples.append(
                    (
                        record.split_group_id,
                        record.state,
                        (record.actions[left], record.actions[right]),
                        0 if left_score > right_score else 1,
                    )
                )
    return examples


def exact_tie_future_averaged_pairwise_examples(
    records: Sequence[ExactTieRolloutRecord],
) -> list[tuple[str, dict[str, Any], tuple[GameAction, GameAction], int]]:
    """Average repeated future-wall returns before constructing preferences.

    Records with the same wall group, public state and tied action set share a
    naturally reached opponent-hand allocation.  Averaging their terminal
    returns removes future wall-order noise without pretending that the
    repeated permutations are independent hidden-hand samples.
    """

    buckets: dict[
        tuple[str, str, tuple[GameAction, ...], int],
        tuple[dict[str, Any], list[list[int]]],
    ] = {}
    for record in records:
        record.validate()
        state_key = json.dumps(record.state, ensure_ascii=False, sort_keys=True)
        key = (
            record.split_group_id,
            state_key,
            record.actions,
            record.teacher_index,
        )
        if key not in buckets:
            buckets[key] = (
                record.state,
                [[] for _ in record.actions],
            )
        _state, samples = buckets[key]
        for index, score in enumerate(record.terminal_scores):
            samples[index].append(score)

    examples = []
    for (group, _state_key, actions, _teacher), (state, samples) in buckets.items():
        means = [sum(values) / len(values) for values in samples]
        for left in range(len(actions)):
            for right in range(left + 1, len(actions)):
                if means[left] == means[right]:
                    continue
                examples.append(
                    (
                        group,
                        state,
                        (actions[left], actions[right]),
                        0 if means[left] > means[right] else 1,
                    )
                )
    return examples


def exact_tie_distinct_decision_count(
    records: Sequence[ExactTieRolloutRecord],
) -> int:
    """Count public decision identities after collapsing future permutations."""

    return len(
        {
            (
                record.split_group_id,
                json.dumps(record.state, ensure_ascii=False, sort_keys=True),
                record.actions,
                record.teacher_index,
            )
            for record in records
        }
    )

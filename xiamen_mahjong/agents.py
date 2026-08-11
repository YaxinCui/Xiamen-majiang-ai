"""Explainable rule Teacher used by the local browser game and data export."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from types import SimpleNamespace

from .hand import (
    hand_quality,
    hand_quality_components,
    is_winning_hand,
    one_draw_tenpai_routes,
    public_draw_improvement_profile,
    standard_hand_shanten,
    wait_tiles,
)
from .scoring import classic_score
from .tiles import BASE_TILE_COUNT, is_base_tile, is_honor, tile_name


@dataclass(frozen=True)
class GameAction:
    kind: str
    tile: int | None = None
    tiles: tuple[int, ...] = ()


@dataclass(frozen=True)
class DiscardShapeWeights:
    """Linear discard weights; defaults exactly reconstruct frozen Teacher."""

    fixed_meld: float = 28.0
    gold_tile: float = 12.0
    triplet_group: float = 14.0
    pair_remainder: float = 5.0
    adjacent_overlap: float = 2.5
    gap_overlap: float = 1.25
    sequence_overlap: float = 5.0
    wait_face: float = 18.0
    gold_discard_penalty: float = 7.0

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in self.__dict__.values()
        ):
            raise ValueError("弃牌评分权重必须是有限非负数")


class HeuristicTeacherAgent:
    """Small public-information Teacher; deliberately safe and deterministic."""

    def choose_turn_action(self, game, player_id: int) -> GameAction:
        player = game.players[player_id]
        meld_count = len(player.melds)
        if game._tour_resolution_level(player_id):
            if game._can_advance_tour(player_id):
                return GameAction("advance_tour", game.gold_tile)
            return GameAction("hu")
        if game._in_locked_tour_cycle(player_id):
            if game._can_win(player_id):
                return GameAction("hu")
            return GameAction("discard", self._best_discard(game, player_id))
        if is_winning_hand(
            player.hand,
            game.gold_tile,
            meld_count=meld_count,
            melds_required=game.rules.melds_required,
            allow_seven_pairs=game.rules.allow_seven_pairs,
            wildcard_tiles=game.wildcard_tiles,
            proxy_tile=game.gold_proxy_tile,
            proxy_as=game.gold_tile,
        ):
            return GameAction("hu")

        # The classic profile's honor-follow rule constrains the entire turn,
        # including otherwise legal kong choices.  Resolve it before trying a
        # kong so Teacher self-play cannot propose an engine-illegal action.
        if game._forced_follow_tiles(player_id):
            return GameAction("discard", self._best_discard(game, player_id))

        if len(game.wall) > game.rules.dead_wall_tiles:
            for tile, count in sorted(Counter(player.hand).items()):
                if count == 4 and tile != game.gold_tile:
                    return GameAction("an_kan", tile)
            for meld in player.melds:
                if meld["kind"] == "pong" and len(set(meld["tiles"])) == 1:
                    tile = meld["tiles"][0]
                    if tile in player.hand and tile != game.gold_tile:
                        return GameAction("add_kan", tile)

        return GameAction("discard", self._best_discard(game, player_id))

    def choose_response(self, game, player_id: int, options: list[GameAction]) -> GameAction:
        kinds = {option.kind: option for option in options}
        if "hu" in kinds:
            return kinds["hu"]
        if "ming_kan" in kinds and len(game.wall) > 18:
            return kinds["ming_kan"]
        if "pong" in kinds:
            player = game.players[player_id]
            before = hand_quality(
                player.hand,
                game.gold_tile,
                meld_count=len(player.melds),
                melds_required=game.rules.melds_required,
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
            )
            after_hand = list(player.hand)
            for tile in kinds["pong"].tiles:
                after_hand.remove(tile)
            after = hand_quality(
                after_hand,
                game.gold_tile,
                meld_count=len(player.melds) + 1,
                melds_required=game.rules.melds_required,
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
            )
            if after >= before - 1.0:
                return kinds["pong"]
        chi_options = [option for option in options if option.kind == "chi"]
        if chi_options:
            player = game.players[player_id]
            before = hand_quality(
                player.hand,
                game.gold_tile,
                meld_count=len(player.melds),
                melds_required=game.rules.melds_required,
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
            )
            scored = []
            for option in chi_options:
                after_hand = list(player.hand)
                for tile in option.tiles:
                    after_hand.remove(tile)
                scored.append(
                    (
                        hand_quality(
                            after_hand,
                            game.gold_tile,
                            meld_count=len(player.melds) + 1,
                            melds_required=game.rules.melds_required,
                            wildcard_tiles=game.wildcard_tiles,
                            proxy_tile=game.gold_proxy_tile,
                            proxy_as=game.gold_tile,
                        ),
                        option,
                    )
                )
            best_score, best_option = max(scored, key=lambda item: (item[0], tuple(item[1].tiles)))
            if best_score >= before - 1.0:
                return best_option
        return kinds["pass"]

    def explain_discard(self, game, player_id: int) -> list[dict[str, object]]:
        player = game.players[player_id]
        candidates = []
        allowed = set(game._forced_follow_tiles(player_id))
        for tile in sorted(set(player.hand)):
            if allowed and tile not in allowed:
                continue
            candidate = list(player.hand)
            candidate.remove(tile)
            waits = wait_tiles(
                candidate,
                game.gold_tile,
                meld_count=len(player.melds),
                melds_required=game.rules.melds_required,
                allow_seven_pairs=game.rules.allow_seven_pairs,
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
            )
            quality = hand_quality(
                candidate,
                game.gold_tile,
                meld_count=len(player.melds),
                melds_required=game.rules.melds_required,
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
            )
            gold_penalty = 7.0 if tile == game.gold_tile else 0.0
            score = quality + len(waits) * 18 - gold_penalty
            candidates.append(
                {
                    "tile": tile,
                    "name": tile_name(tile),
                    "score": round(score, 3),
                    "waits": [tile_name(wait) for wait in waits],
                }
            )
        return sorted(candidates, key=lambda item: (-float(item["score"]), int(item["tile"])))

    def _best_discard(self, game, player_id: int) -> int:
        ranked = self.explain_discard(game, player_id)
        if not ranked:
            raise RuntimeError("Teacher was asked to discard from an empty hand")
        return int(ranked[0]["tile"])


class ParametricDiscardTeacherAgent(HeuristicTeacherAgent):
    """Frozen Teacher with only its ordinary discard linear weights exposed.

    Turn/response legality, wins, gold tours, kong and claim decisions remain
    inherited.  This makes outcome-driven weight search a small intermediate
    rule model rather than a new end-to-end policy.
    """

    def __init__(self, weights: DiscardShapeWeights | None = None):
        self.weights = weights or DiscardShapeWeights()

    def explain_discard(self, game, player_id: int) -> list[dict[str, object]]:
        player = game.players[player_id]
        candidates = []
        allowed = set(game._forced_follow_tiles(player_id))
        for tile in sorted(set(player.hand)):
            if allowed and tile not in allowed:
                continue
            candidate = list(player.hand)
            candidate.remove(tile)
            waits = wait_tiles(
                candidate,
                game.gold_tile,
                meld_count=len(player.melds),
                melds_required=game.rules.melds_required,
                allow_seven_pairs=game.rules.allow_seven_pairs,
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
            )
            parts = hand_quality_components(
                candidate,
                game.gold_tile,
                meld_count=len(player.melds),
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
            )
            weights = self.weights
            score = (
                parts.fixed_melds * weights.fixed_meld
                + parts.gold_tiles * weights.gold_tile
                + parts.triplet_groups * weights.triplet_group
                + parts.pair_remainders * weights.pair_remainder
                + parts.adjacent_overlap * weights.adjacent_overlap
                + parts.gap_overlap * weights.gap_overlap
                + parts.sequence_overlap * weights.sequence_overlap
                + len(waits) * weights.wait_face
                - (weights.gold_discard_penalty if tile == game.gold_tile else 0.0)
            )
            candidates.append(
                {
                    "tile": tile,
                    "name": tile_name(tile),
                    "score": round(score, 3),
                    "waits": [tile_name(wait) for wait in waits],
                }
            )
        return sorted(
            candidates, key=lambda item: (-float(item["score"]), int(item["tile"]))
        )


class MeldContinuationTeacherAgent(HeuristicTeacherAgent):
    """Response-only Teacher that evaluates the compulsory discard after a call.

    The frozen Teacher compares a chi/pong hand immediately after consuming two
    tiles.  In the actual rules that player must immediately discard one more
    concealed tile, so the comparison is at a different hand size from the
    pass branch.  This candidate changes *only* chi and pong responses: it
    scores the best legal post-call discard with the same deterministic hand
    shape and wait scoring used by the frozen Teacher, then requires a fixed
    improvement over declining the call.  It reads only the actor's hand,
    public discards/melds and public rules; it never samples or inspects the
    wall or opponents' concealed hands.

    Ming-kong, self-draw/discard win, gold-tour and ordinary turn decisions
    remain exactly frozen.  The narrow scope makes this a directly testable
    expert-rule correction rather than another global discard heuristic.
    """

    def __init__(self, *, minimum_claim_gain: float = 0.0):
        if minimum_claim_gain < 0:
            raise ValueError("minimum_claim_gain 不能为负数")
        self.minimum_claim_gain = float(minimum_claim_gain)

    @staticmethod
    def _shape_score(game, player_id: int, hand: list[int], meld_count: int) -> float:
        return hand_quality(
            hand,
            game.gold_tile,
            meld_count=meld_count,
            melds_required=game.rules.melds_required,
            wildcard_tiles=game.wildcard_tiles,
            proxy_tile=game.gold_proxy_tile,
            proxy_as=game.gold_tile,
        )

    @staticmethod
    def _forced_follow_for_hand(game, hand: list[int]) -> list[int]:
        """Apply the public honor-follow rule to a hypothetical own hand."""

        if not game.rules.enable_forced_honor_follow:
            return []
        appeared_honors = {
            tile
            for player in game.players
            for tile in player.discards
            if is_honor(tile) and tile not in {game.gold_tile, game.gold_proxy_tile}
        }
        return sorted(
            tile
            for tile in set(hand)
            if is_honor(tile) and hand.count(tile) == 1 and tile in appeared_honors
        )

    def _post_claim_score(
        self, game, player_id: int, action: GameAction
    ) -> tuple[float, int]:
        """Score the best *legal forced discard* after chi or pong.

        ``action.tiles`` are the two concealed tiles consumed by the call;
        the claimed discard becomes part of the exposed meld and therefore
        does not enter the actor's concealed hand.  The returned hand after
        discard has the exact size on which its next draw may complete it.
        """

        player = game.players[player_id]
        after_claim = list(player.hand)
        for tile in action.tiles:
            after_claim.remove(tile)
        meld_count = len(player.melds) + 1
        forced = set(self._forced_follow_for_hand(game, after_claim))
        candidates: list[tuple[float, int]] = []
        for discarded in sorted(set(after_claim)):
            if forced and discarded not in forced:
                continue
            after_discard = list(after_claim)
            after_discard.remove(discarded)
            waits = wait_tiles(
                after_discard,
                game.gold_tile,
                meld_count=meld_count,
                melds_required=game.rules.melds_required,
                allow_seven_pairs=game.rules.allow_seven_pairs,
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
            )
            score = self._shape_score(game, player_id, after_discard, meld_count)
            score += len(waits) * 18.0
            if discarded == game.gold_tile:
                score -= 7.0
            candidates.append((score, discarded))
        if not candidates:
            raise RuntimeError("响应后没有合法弃牌")
        return max(candidates, key=lambda item: (item[0], -item[1]))

    def explain_response(
        self, game, player_id: int, options: list[GameAction]
    ) -> list[dict[str, object]]:
        """Expose a compact, human-readable chi/pong continuation ranking."""

        player = game.players[player_id]
        before = self._shape_score(game, player_id, player.hand, len(player.melds))
        rows = []
        for action in options:
            if action.kind not in {"chi", "pong"}:
                continue
            after_score, forced_discard = self._post_claim_score(game, player_id, action)
            gain = after_score - before
            rows.append(
                {
                    "kind": action.kind,
                    "tiles": [tile_name(tile) for tile in action.tiles],
                    "post_call_discard": tile_name(forced_discard),
                    "post_call_score": round(after_score, 3),
                    "gain_over_pass_shape": round(gain, 3),
                    "meets_minimum_gain": gain >= self.minimum_claim_gain,
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                -float(row["post_call_score"]),
                str(row["kind"]),
                tuple(str(tile) for tile in row["tiles"]),
            ),
        )

    def choose_response(self, game, player_id: int, options: list[GameAction]) -> GameAction:
        kinds = {option.kind: option for option in options}
        if "hu" in kinds:
            return kinds["hu"]
        # Preserve the frozen Teacher's unconditional high-priority ming-kong
        # behavior.  This candidate isolates chi/pong continuation quality.
        if "ming_kan" in kinds and len(game.wall) > 18:
            return kinds["ming_kan"]
        player = game.players[player_id]
        before = self._shape_score(game, player_id, player.hand, len(player.melds))
        ranked: list[tuple[float, int, tuple[int, ...], GameAction]] = []
        for action in options:
            if action.kind not in {"chi", "pong"}:
                continue
            after_score, _forced_discard = self._post_claim_score(game, player_id, action)
            gain = after_score - before
            if gain >= self.minimum_claim_gain:
                # Keep the frozen Teacher's pong-before-chi preference only
                # as a deterministic tie-breaker, never as a score bonus.
                priority = 1 if action.kind == "pong" else 0
                ranked.append((after_score, priority, tuple(action.tiles), action))
        if ranked:
            return max(ranked, key=lambda item: (item[0], item[1], item[2]))[3]
        return kinds["pass"]


class DeficiencyMeldTeacherAgent(HeuristicTeacherAgent):
    """Response-only correction led by exact classic-hand deficiency.

    The frozen response rule compares the actor's current concealed hand with
    an intermediate chi/pong hand that has consumed two tiles but has not yet
    made the mandatory discard.  ``MeldContinuationTeacherAgent`` corrected
    the timing but still used the same approximate shape score globally.  This
    candidate instead compares pass and fully resolved claim branches in a
    strict lexicographic order:

    1. exact five-meld regular-hand shanten;
    2. direct wait-face count when already in tenpai;
    3. the frozen explainable hand-shape score.

    It changes a response only when that tuple strictly improves the frozen
    action.  Hu, ming-kong, tours, gold locks, opening-wait states, turn
    actions and non-classic profiles remain frozen.  The calculation uses only
    the actor hand and public rivers/melds; no wall or opponent hand is read.
    """

    @staticmethod
    def _appeared_honors_after_claim(game) -> frozenset[int]:
        counts = Counter(
            tile
            for player in game.players
            for tile in player.discards
            if is_honor(tile) and tile not in {game.gold_tile, game.gold_proxy_tile}
        )
        # At response time the latest discard is still in its owner's river.
        # A successful claim removes that physical tile before the mandatory
        # discard, so it must not by itself trigger honor-follow.
        if (
            game.last_discard is not None
            and is_honor(game.last_discard)
            and game.last_discard not in {game.gold_tile, game.gold_proxy_tile}
        ):
            counts[game.last_discard] -= 1
        return frozenset(tile for tile, count in counts.items() if count > 0)

    @staticmethod
    def _forced_follow_for_hand(
        game, hand: list[int], appeared_honors: frozenset[int]
    ) -> list[int]:
        if not game.rules.enable_forced_honor_follow:
            return []
        return sorted(
            tile
            for tile in set(hand)
            if is_honor(tile)
            and tile not in {game.gold_tile, game.gold_proxy_tile}
            and hand.count(tile) == 1
            and tile in appeared_honors
        )

    @staticmethod
    def _profile_key(
        game,
        hand: list[int],
        *,
        meld_count: int,
        immediate_discard: int | None = None,
    ) -> tuple[tuple[int, int, float], dict[str, object]]:
        shanten = standard_hand_shanten(
            hand,
            game.gold_tile,
            meld_count=meld_count,
            melds_required=game.rules.melds_required,
            wildcard_tiles=game.wildcard_tiles,
            proxy_tile=game.gold_proxy_tile,
            proxy_as=game.gold_tile,
        )
        waits = (
            wait_tiles(
                hand,
                game.gold_tile,
                meld_count=meld_count,
                melds_required=game.rules.melds_required,
                allow_seven_pairs=game.rules.allow_seven_pairs,
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
            )
            if shanten == 0
            else []
        )
        shape_score = hand_quality(
            hand,
            game.gold_tile,
            meld_count=meld_count,
            melds_required=game.rules.melds_required,
            wildcard_tiles=game.wildcard_tiles,
            proxy_tile=game.gold_proxy_tile,
            proxy_as=game.gold_tile,
        )
        if immediate_discard == game.gold_tile:
            shape_score -= 7.0
        key = (shanten, -len(waits), -shape_score)
        return key, {
            "regular_hand_shanten": shanten,
            "direct_wait_faces": len(waits),
            "shape_score": round(shape_score, 3),
            "waits": [tile_name(tile) for tile in waits],
        }

    def _post_claim_profile(
        self, game, player_id: int, action: GameAction
    ) -> tuple[tuple[int, int, float], int, dict[str, object]]:
        player = game.players[player_id]
        after_claim = list(player.hand)
        for tile in action.tiles:
            after_claim.remove(tile)
        appeared = self._appeared_honors_after_claim(game)
        forced = self._forced_follow_for_hand(game, after_claim, appeared)
        candidates = []
        for discarded in forced or sorted(set(after_claim)):
            after_discard = list(after_claim)
            after_discard.remove(discarded)
            key, payload = self._profile_key(
                game,
                after_discard,
                meld_count=len(player.melds) + 1,
                immediate_discard=discarded,
            )
            candidates.append((key, discarded, payload))
        if not candidates:
            raise RuntimeError("响应后没有合法弃牌")
        return min(candidates, key=lambda row: (row[0], row[1]))

    def explain_response(
        self, game, player_id: int, options: list[GameAction]
    ) -> list[dict[str, object]]:
        player = game.players[player_id]
        pass_key, pass_payload = self._profile_key(
            game, player.hand, meld_count=len(player.melds)
        )
        rows: list[dict[str, object]] = [
            {"kind": "pass", **pass_payload, "profile_key": list(pass_key)}
        ]
        for action in options:
            if action.kind not in {"chi", "pong"}:
                continue
            key, discarded, payload = self._post_claim_profile(
                game, player_id, action
            )
            rows.append(
                {
                    "kind": action.kind,
                    "tiles": [tile_name(tile) for tile in action.tiles],
                    "post_call_discard": tile_name(discarded),
                    **payload,
                    "profile_key": list(key),
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                tuple(row["profile_key"]),
                0 if row["kind"] == "pong" else 1 if row["kind"] == "chi" else 2,
                tuple(str(tile) for tile in row.get("tiles", [])),
            ),
        )

    def choose_response(
        self, game, player_id: int, options: list[GameAction]
    ) -> GameAction:
        frozen = HeuristicTeacherAgent.choose_response(
            self, game, player_id, options
        )
        player = game.players[player_id]
        if (
            game.rules.profile != "classic"
            or frozen.kind in {"hu", "ming_kan"}
            or game.tour_state is not None
            or game.gold_discard_lock_seat == player_id
            or player_id in game.opening_wait_seats
        ):
            return frozen
        pass_action = next(
            (action for action in options if action.kind == "pass"), None
        )
        claim_actions = [
            action for action in options if action.kind in {"chi", "pong"}
        ]
        if pass_action is None or not claim_actions or frozen.kind not in {
            "pass",
            "chi",
            "pong",
        }:
            return frozen

        pass_key, _payload = self._profile_key(
            game, player.hand, meld_count=len(player.melds)
        )
        ranked: list[
            tuple[tuple[int, int, float], int, tuple[int, ...], GameAction]
        ] = [(pass_key, 2, (), pass_action)]
        for action in claim_actions:
            key, _discarded, _claim_payload = self._post_claim_profile(
                game, player_id, action
            )
            priority = 0 if action.kind == "pong" else 1
            ranked.append((key, priority, tuple(action.tiles), action))
        frozen_row = next(row for row in ranked if row[3] == frozen)
        best = min(ranked, key=lambda row: (row[0], row[1], row[2]))
        return best[3] if best[0] < frozen_row[0] else frozen


class AvailabilityTeacherAgent(HeuristicTeacherAgent):
    """Teacher that weights waits by remaining publicly possible copies.

    The original Teacher treats a one-tile wait with all four copies already
    visible as equal to a one-tile wait with four live copies.  This policy
    preserves the same explainable hand-shape heuristic but replaces that
    coarse wait count with availability derived only from the actor's hand,
    rivers, exposed melds and the visible gold indicator.
    """

    def __init__(self, *, wait_copy_value: float = 4.5):
        if wait_copy_value < 0:
            raise ValueError("wait_copy_value 不能为负数")
        self.wait_copy_value = float(wait_copy_value)

    def explain_discard(self, game, player_id: int) -> list[dict[str, object]]:
        player = game.players[player_id]
        visible = Counter()
        for other in game.players:
            visible.update(tile for tile in other.discards if is_base_tile(tile))
            visible.update(
                tile
                for meld in other.melds
                for tile in meld["tiles"]
                if is_base_tile(tile)
            )
        if game.gold_indicator is not None and is_base_tile(game.gold_indicator):
            visible[game.gold_indicator] += 1
        visible.update(tile for tile in player.hand if is_base_tile(tile))

        candidates = []
        allowed = set(game._forced_follow_tiles(player_id))
        for tile in sorted(set(player.hand)):
            if allowed and tile not in allowed:
                continue
            candidate = list(player.hand)
            candidate.remove(tile)
            wait_visible = Counter(visible)
            if is_base_tile(tile):
                wait_visible[tile] -= 1
            waits = wait_tiles(
                candidate,
                game.gold_tile,
                meld_count=len(player.melds),
                melds_required=game.rules.melds_required,
                allow_seven_pairs=game.rules.allow_seven_pairs,
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
            )
            availability = sum(max(0, 4 - wait_visible[wait]) for wait in waits)
            quality = hand_quality(
                candidate,
                game.gold_tile,
                meld_count=len(player.melds),
                melds_required=game.rules.melds_required,
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
            )
            gold_penalty = 7.0 if tile == game.gold_tile else 0.0
            # 4.5 preserves the original Teacher's nominal one-wait bonus
            # (four live copies * 4.5 == 18); evaluation tunes this public
            # heuristic separately from the final held-out benchmark.
            score = quality + availability * self.wait_copy_value - gold_penalty
            candidates.append(
                {
                    "tile": tile,
                    "name": tile_name(tile),
                    "score": round(score, 3),
                    "waits": [tile_name(wait) for wait in waits],
                    "wait_availability": availability,
                }
            )
        return sorted(candidates, key=lambda item: (-float(item["score"]), int(item["tile"])))


class RiskAwareTeacherAgent(AvailabilityTeacherAgent):
    """Public-information defensive variant of the availability Teacher.

    It does not estimate an opponent's concealed hand.  Instead it applies a
    small, explainable penalty to tiles that have many non-public copies while
    opponents show public signs of commitment (open melds or a later hand).
    A tile previously discarded by an opponent is discounted, but never
    declared fully safe: Xiamen rules and future draws need not imply a
    universal furiten-style guarantee.
    """

    def __init__(self, *, risk_weight: float, wait_copy_value: float = 4.5):
        super().__init__(wait_copy_value=wait_copy_value)
        if risk_weight < 0:
            raise ValueError("risk_weight 不能为负数")
        self.risk_weight = float(risk_weight)

    def _public_visible_counts(self, game, player_id: int) -> Counter[int]:
        """Count only actor-known and public base tiles for danger features."""

        visible = Counter(
            tile for tile in game.players[player_id].hand if is_base_tile(tile)
        )
        for player in game.players:
            visible.update(tile for tile in player.discards if is_base_tile(tile))
            for meld in player.melds:
                # trajectory-v4 and the public event contract deliberately
                # redact another player's concealed-kong face.  A public-only
                # candidate must know that four tiles left play without
                # learning which face they are.
                if meld["kind"] == "an_kan" and player.seat != player_id:
                    continue
                visible.update(
                    tile for tile in meld["tiles"] if is_base_tile(tile)
                )
        if game.gold_indicator is not None and is_base_tile(game.gold_indicator):
            visible[game.gold_indicator] += 1
        return visible

    def _public_danger(self, game, player_id: int, tile: int) -> float:
        if not is_base_tile(tile):
            return 0.0
        visible = self._public_visible_counts(game, player_id)
        unseen_fraction = max(0, 4 - visible[tile]) / 4.0
        # public_actions is a shared event log. Its length is public and a
        # monotone, rule-independent proxy for how late the hand has become.
        late_hand = min(1.0, len(game.public_actions) / 40.0)
        danger = 0.0
        for opponent in game.players:
            if opponent.seat == player_id:
                continue
            threat = 1.0 + 0.75 * len(opponent.melds) + 0.5 * late_hand
            previously_discarded = tile in opponent.discards
            safety_discount = 0.2 if previously_discarded else 1.0
            danger += unseen_fraction * threat * safety_discount
        return danger

    def explain_discard(self, game, player_id: int) -> list[dict[str, object]]:
        candidates = []
        for base in super().explain_discard(game, player_id):
            tile = int(base["tile"])
            danger = self._public_danger(game, player_id, tile)
            score = float(base["score"]) - self.risk_weight * danger
            candidates.append(
                {
                    **base,
                    "public_danger": round(danger, 3),
                    "risk_penalty": round(self.risk_weight * danger, 3),
                    "score": round(score, 3),
                }
            )
        return sorted(candidates, key=lambda item: (-float(item["score"]), int(item["tile"])))


class OnePlyLookaheadTeacherAgent(HeuristicTeacherAgent):
    """Experimental public-information one-ply discard Teacher.

    Unlike :class:`HeuristicTeacherAgent`, which ranks the hand immediately
    after a discard, this candidate enumerates the actor's *next* possible
    playable draw.  Each face is weighted by the number of copies not already
    visible in the actor's hand, rivers, exposed melds, or gold indicator.
    It then chooses the best legal follow-up discard by cheap hand-shape
    quality, with an exact engine-compatible self-draw settlement value when
    the next draw completes the hand.

    This is intentionally a small, deterministic screening candidate.  It
    never reads ``game.wall`` or an opponent's ``hand``; it is not promoted to
    the browser or training data unless it beats the frozen Teacher on a
    prespecified independent evaluation.
    """

    def _public_visible_counts(self, game, player_id: int) -> Counter[int]:
        """Count base-tile faces known unavailable from public information.

        The actor's own concealed hand is known to that actor.  For every
        other player we deliberately inspect only exposed rivers and melds,
        never their concealed hand, flowers' replacement draws, or the wall.
        """

        visible = Counter(
            tile for tile in game.players[player_id].hand if is_base_tile(tile)
        )
        for player in game.players:
            visible.update(tile for tile in player.discards if is_base_tile(tile))
            visible.update(
                tile
                for meld in player.melds
                for tile in meld["tiles"]
                if is_base_tile(tile)
            )
        if game.gold_indicator is not None and is_base_tile(game.gold_indicator):
            visible[game.gold_indicator] += 1
        return visible

    def _shape_score(self, game, player_id: int, hand: list[int], discarded: int) -> float:
        """Return the cheap post-discard shape value used at the leaf."""

        player = game.players[player_id]
        quality = hand_quality(
            hand,
            game.gold_tile,
            meld_count=len(player.melds),
            melds_required=game.rules.melds_required,
            wildcard_tiles=game.wildcard_tiles,
            proxy_tile=game.gold_proxy_tile,
            proxy_as=game.gold_tile,
        )
        return quality - (7.0 if discarded == game.gold_tile else 0.0)

    def _allowed_discards(self, game, player_id: int, hand: list[int]) -> list[int]:
        """Apply the public honor-follow restriction to a hypothetical hand."""

        if not game.rules.enable_forced_honor_follow:
            return sorted(set(hand))
        appeared_honors = {
            tile
            for player in game.players
            for tile in player.discards
            if 27 <= tile < BASE_TILE_COUNT
            and tile not in {game.gold_tile, game.gold_proxy_tile}
        }
        forced = sorted(
            tile
            for tile in set(hand)
            if 27 <= tile < BASE_TILE_COUNT
            and hand.count(tile) == 1
            and tile in appeared_honors
        )
        return forced or sorted(set(hand))

    def _is_winning_hand(self, game, player_id: int, hand: list[int]) -> bool:
        player = game.players[player_id]
        return is_winning_hand(
            hand,
            game.gold_tile,
            meld_count=len(player.melds),
            melds_required=game.rules.melds_required,
            allow_seven_pairs=game.rules.allow_seven_pairs,
            wildcard_tiles=game.wildcard_tiles,
            proxy_tile=game.gold_proxy_tile,
            proxy_as=game.gold_tile,
        )

    def _self_draw_value(self, game, player_id: int, hand: list[int]) -> float:
        """Use the same settlement scale as the engine for an immediate win."""

        player = game.players[player_id]
        if game.rules.enable_complex_water_scoring:
            winner = SimpleNamespace(
                hand=hand,
                flowers=player.flowers,
                melds=player.melds,
            )
            return float(
                classic_score(
                    winner,
                    is_dealer=player_id == game.dealer,
                    wildcard_tiles=game.wildcard_tiles,
                    gold_tile=game.gold_tile,
                    win_type=game._self_draw_win_type(player_id),
                    rules=game.rules,
                    dealer_streak=game.dealer_streak,
                    proxy_tile=game.gold_proxy_tile,
                    proxy_as=game.gold_tile,
                ).total
            )
        multiplier = 1 + len(player.flowers) + hand.count(game.gold_tile)
        return float(game.rules.base_score * multiplier * 3)

    def _next_draw_value(
        self,
        game,
        player_id: int,
        hand_after_discard: list[int],
        visible: Counter[int],
    ) -> float:
        weighted_total = 0.0
        remaining_total = 0
        for drawn_tile in range(BASE_TILE_COUNT):
            remaining = max(0, 4 - visible[drawn_tile])
            if not remaining:
                continue
            next_hand = [*hand_after_discard, drawn_tile]
            if self._is_winning_hand(game, player_id, next_hand):
                value = self._self_draw_value(game, player_id, next_hand)
            else:
                value = max(
                    self._shape_score(
                        game,
                        player_id,
                        self._remove_one(next_hand, discarded),
                        discarded,
                    )
                    for discarded in self._allowed_discards(game, player_id, next_hand)
                )
            weighted_total += remaining * value
            remaining_total += remaining
        if not remaining_total:
            raise RuntimeError("公开剩余张数为空，无法进行一步前瞻")
        return weighted_total / remaining_total

    @staticmethod
    def _remove_one(hand: list[int], tile: int) -> list[int]:
        result = list(hand)
        result.remove(tile)
        return result

    def explain_discard(self, game, player_id: int) -> list[dict[str, object]]:
        player = game.players[player_id]
        visible = self._public_visible_counts(game, player_id)
        allowed = set(self._allowed_discards(game, player_id, player.hand))
        candidates = []
        for tile in sorted(set(player.hand)):
            if tile not in allowed:
                continue
            candidate = self._remove_one(player.hand, tile)
            immediate_shape = self._shape_score(game, player_id, candidate, tile)
            next_draw_score = self._next_draw_value(
                game, player_id, candidate, visible
            )
            candidates.append(
                {
                    "tile": tile,
                    "name": tile_name(tile),
                    "score": round(next_draw_score, 3),
                    "immediate_shape_score": round(immediate_shape, 3),
                    "next_draw_score": round(next_draw_score, 3),
                }
            )
        return sorted(candidates, key=lambda item: (-float(item["score"]), int(item["tile"])))


class ExactOneDrawTenpaiTieBreakTeacherAgent(HeuristicTeacherAgent):
    """A strictly gated exact-tenpai tiebreaker for the frozen Teacher.

    The previously screened :class:`OnePlyLookaheadTeacherAgent` reranked all
    discards using a cheap future shape value and was decisively rejected.
    This candidate is deliberately not another global lookahead: it preserves
    the frozen order unless (1) the frozen choice and an alternative are both
    non-tenpai, (2) their frozen scores differ by at most ``score_margin``,
    and (3) an exact legal one-draw-tenpai calculation gives an alternative a
    strictly larger public live-route potential.

    It consumes only the actor's hand, all exposed rivers/melds, the gold
    indicator, and the public honor-follow rule.  It never reads a wall or an
    opponent's concealed hand.  It is an experimental evaluation candidate;
    passing a prespecified independent gate is required before it may affect
    the browser Teacher or training labels.
    """

    def __init__(self, *, score_margin: float = 2.0, minimum_live_advantage: int = 1):
        if score_margin < 0:
            raise ValueError("score_margin 不能为负数")
        if minimum_live_advantage < 1:
            raise ValueError("minimum_live_advantage 至少为 1")
        self.score_margin = float(score_margin)
        self.minimum_live_advantage = int(minimum_live_advantage)

    @staticmethod
    def _public_visible_counts(game, player_id: int) -> Counter[int]:
        """Count only the actor-known and publicly exposed base-tile faces."""

        visible = Counter(
            tile for tile in game.players[player_id].hand if is_base_tile(tile)
        )
        for player in game.players:
            visible.update(tile for tile in player.discards if is_base_tile(tile))
            for meld in player.melds:
                if meld["kind"] == "an_kan" and player.seat != player_id:
                    continue
                visible.update(
                    tile for tile in meld["tiles"] if is_base_tile(tile)
                )
        if game.gold_indicator is not None and is_base_tile(game.gold_indicator):
            visible[game.gold_indicator] += 1
        return visible

    @staticmethod
    def _allowed_discards(game, hand: list[int]) -> list[int]:
        """Reapply classic's public honor-follow constraint after a draw."""

        if not game.rules.enable_forced_honor_follow:
            return sorted(set(hand))
        appeared_honors = {
            tile
            for player in game.players
            for tile in player.discards
            if is_honor(tile) and tile not in {game.gold_tile, game.gold_proxy_tile}
        }
        forced = sorted(
            tile
            for tile in set(hand)
            if is_honor(tile) and hand.count(tile) == 1 and tile in appeared_honors
        )
        return forced or sorted(set(hand))

    def _live_route_potential(
        self,
        game,
        player_id: int,
        hand_after_discard: list[int],
        visible: Counter[int],
    ) -> int:
        """Return public draw-copy × follow-up live-wait-copy potential.

        For each still-publicly-possible draw face, the exact oracle enumerates
        every legal immediate discard and picks the continuation with the most
        live waits *after that draw*.  This is only a tiebreak statistic, not
        a win probability or a substitute for the frozen hand-shape score.
        """

        player = game.players[player_id]
        routes = one_draw_tenpai_routes(
            hand_after_discard,
            game.gold_tile,
            meld_count=len(player.melds),
            melds_required=game.rules.melds_required,
            allow_seven_pairs=game.rules.allow_seven_pairs,
            wildcard_tiles=game.wildcard_tiles,
            proxy_tile=game.gold_proxy_tile,
            proxy_as=game.gold_tile,
            legal_discards=lambda after_draw: self._allowed_discards(game, after_draw),
        )
        potential = 0
        for drawn, choices in routes.items():
            draw_copies = max(0, 4 - visible[drawn])
            if not draw_copies:
                continue
            best_live_waits = max(
                sum(
                    max(0, 4 - visible[wait] - (1 if wait == drawn else 0))
                    for wait in waits
                )
                for _discarded, waits in choices
            )
            potential += draw_copies * best_live_waits
        return potential

    def explain_discard(self, game, player_id: int) -> list[dict[str, object]]:
        frozen = super().explain_discard(game, player_id)
        if not frozen or frozen[0]["waits"]:
            return frozen

        top_score = float(frozen[0]["score"])
        eligible = [
            row
            for row in frozen
            if not row["waits"]
            and top_score - float(row["score"]) <= self.score_margin
        ]
        if len(eligible) < 2:
            return frozen

        player = game.players[player_id]
        visible = self._public_visible_counts(game, player_id)
        potentials: dict[int, int] = {}
        for row in eligible:
            tile = int(row["tile"])
            after_discard = list(player.hand)
            after_discard.remove(tile)
            potentials[tile] = self._live_route_potential(
                game, player_id, after_discard, visible
            )

        frozen_tile = int(frozen[0]["tile"])
        frozen_potential = potentials[frozen_tile]
        selected = max(
            eligible,
            key=lambda row: (
                potentials[int(row["tile"])],
                float(row["score"]),
                -int(row["tile"]),
            ),
        )
        selected_tile = int(selected["tile"])
        if (
            selected_tile == frozen_tile
            or potentials[selected_tile] - frozen_potential
            < self.minimum_live_advantage
        ):
            return frozen

        enriched = []
        for row in frozen:
            tile = int(row["tile"])
            if tile in potentials:
                enriched.append(
                    {
                        **row,
                        "teacher_score": row["score"],
                        "one_draw_live_potential": potentials[tile],
                        "selected_by_exact_one_draw_tiebreak": tile == selected_tile,
                    }
                )
            else:
                enriched.append(row)
        # `_best_discard` intentionally takes the first explanation row.  In
        # the exceptional gated state, selection order is the explicitly
        # labelled exact tiebreak; every non-selected row keeps frozen order.
        return [
            next(row for row in enriched if int(row["tile"]) == selected_tile),
            *(row for row in enriched if int(row["tile"]) != selected_tile),
        ]


class KnowledgeAwareDeficiencyTeacherAgent(HeuristicTeacherAgent):
    """Narrow low-margin correction using exact regular-hand shanten.

    The global one-ply candidate was rejected because its cheap leaf score
    replaced the frozen policy everywhere.  This candidate instead preserves
    every frozen decision unless two non-tenpai ordinary discards are within a
    fixed score margin and one has a *strictly lower* exact five-meld shanten.
    Publicly live improvement copies break ties only among equally low-shanten
    alternatives; ukeire alone can never override the Teacher.

    The calculation uses the actor's hand, public rivers/melds and gold
    indicator.  It ignores opponent concealed hands, concealed-kong faces and
    the physical wall.  It is an experimental SlowExpertTeacher candidate,
    not a deployed default.
    """

    def __init__(self, *, score_margin: float = 2.0):
        if score_margin < 0:
            raise ValueError("score_margin 不能为负数")
        self.score_margin = float(score_margin)

    def _requires_frozen_improvement_profile(self) -> bool:
        return False

    def _accept_deficiency_override(
        self, selected_profile, frozen_profile
    ) -> bool:
        del selected_profile, frozen_profile
        return True

    @staticmethod
    def _public_visible_counts(game, player_id: int) -> Counter[int]:
        visible = Counter(
            tile for tile in game.players[player_id].hand if is_base_tile(tile)
        )
        for player in game.players:
            visible.update(tile for tile in player.discards if is_base_tile(tile))
            for meld in player.melds:
                if meld["kind"] == "an_kan" and player.seat != player_id:
                    continue
                visible.update(tile for tile in meld["tiles"] if is_base_tile(tile))
        if game.gold_indicator is not None and is_base_tile(game.gold_indicator):
            visible[game.gold_indicator] += 1
        return visible

    @staticmethod
    def _appeared_honors(game, *, extra_discard: int | None = None) -> frozenset[int]:
        appeared = {
            tile
            for player in game.players
            for tile in player.discards
            if is_honor(tile) and tile not in {game.gold_tile, game.gold_proxy_tile}
        }
        if (
            extra_discard is not None
            and is_honor(extra_discard)
            and extra_discard not in {game.gold_tile, game.gold_proxy_tile}
        ):
            appeared.add(extra_discard)
        return frozenset(appeared)

    @staticmethod
    def _allowed_efficiency_discards(
        game, hand: list[int], appeared_honors: frozenset[int]
    ) -> list[int]:
        choices = sorted(set(hand))
        if game.rules.enable_forced_honor_follow:
            forced = [
                tile
                for tile in choices
                if is_honor(tile)
                and tile not in {game.gold_tile, game.gold_proxy_tile}
                and hand.count(tile) == 1
                and tile in appeared_honors
            ]
            if forced:
                choices = forced
        # A hypothetical gold discard starts a self-draw-only lock and is not
        # faithfully represented by a pure hand-efficiency continuation.
        # Keep it out whenever another legal face exists.
        non_gold = [tile for tile in choices if tile != game.gold_tile]
        return non_gold or choices

    def explain_discard(self, game, player_id: int) -> list[dict[str, object]]:
        frozen = super().explain_discard(game, player_id)
        if (
            not frozen
            or frozen[0]["waits"]
            or int(frozen[0]["tile"]) == game.gold_tile
            or game.tour_state is not None
            or game.gold_discard_lock_seat == player_id
            or game.rules.profile != "classic"
        ):
            return frozen

        top_score = float(frozen[0]["score"])
        eligible = [
            row
            for row in frozen
            if not row["waits"]
            and int(row["tile"]) != game.gold_tile
            and top_score - float(row["score"]) <= self.score_margin + 1e-9
        ]
        if len(eligible) < 2:
            return frozen

        player = game.players[player_id]
        shantens: dict[int, int] = {}
        for row in eligible:
            tile = int(row["tile"])
            after_discard = list(player.hand)
            after_discard.remove(tile)
            shantens[tile] = standard_hand_shanten(
                after_discard,
                game.gold_tile,
                meld_count=len(player.melds),
                melds_required=game.rules.melds_required,
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
            )

        frozen_tile = int(frozen[0]["tile"])
        frozen_shanten = shantens[frozen_tile]
        best_shanten = min(shantens.values())
        if best_shanten >= frozen_shanten:
            return frozen

        # Full public-ukeire enumeration is much more expensive than one
        # shanten call.  It can only affect the tie among candidates that have
        # already established a strict structural improvement over frozen.
        best_rows = [
            row for row in eligible if shantens[int(row["tile"])] == best_shanten
        ]
        visible = self._public_visible_counts(game, player_id)
        live_counts = tuple(
            max(0, 4 - visible[tile]) for tile in range(BASE_TILE_COUNT)
        )
        profile_rows = list(best_rows)
        if self._requires_frozen_improvement_profile():
            profile_rows.append(
                next(row for row in eligible if int(row["tile"]) == frozen_tile)
            )
        profiles = {}
        for row in profile_rows:
            tile = int(row["tile"])
            after_discard = list(player.hand)
            after_discard.remove(tile)
            appeared = self._appeared_honors(game, extra_discard=tile)
            profiles[tile] = public_draw_improvement_profile(
                after_discard,
                live_counts,
                game.gold_tile,
                meld_count=len(player.melds),
                melds_required=game.rules.melds_required,
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
                legal_discards=lambda after_draw, appeared=appeared: (
                    self._allowed_efficiency_discards(game, after_draw, appeared)
                ),
            )

        selected = min(
            best_rows,
            key=lambda row: (
                -profiles[int(row["tile"])].improving_live_copies,
                -float(row["score"]),
                int(row["tile"]),
            ),
        )
        selected_tile = int(selected["tile"])
        if not self._accept_deficiency_override(
            profiles[selected_tile], profiles.get(frozen_tile)
        ):
            return frozen

        enriched = []
        for row in frozen:
            tile = int(row["tile"])
            profile = profiles.get(tile)
            shanten = shantens.get(tile)
            if shanten is None:
                enriched.append(row)
                continue
            details = {
                **row,
                "teacher_score": row["score"],
                "regular_hand_shanten": shanten,
                "draws_to_win": shanten + 1,
                "selected_by_knowledge_aware_deficiency": tile == selected_tile,
            }
            if profile is not None:
                details.update(
                    {
                        "public_improving_faces": len(profile.improving_faces),
                        "public_improving_live_copies": (
                            profile.improving_live_copies
                        ),
                    }
                )
            enriched.append(details)
        return [
            next(row for row in enriched if int(row["tile"]) == selected_tile),
            *(row for row in enriched if int(row["tile"]) != selected_tile),
        ]


class PublicParetoDeficiencyTeacherAgent(KnowledgeAwareDeficiencyTeacherAgent):
    """Require shanten and public next-draw reachability to Pareto-dominate.

    The v1 deficiency candidate sometimes preferred a structurally closer hand
    even when very few publicly possible next draws could continue improving
    it.  This independent v2 keeps v1's strict shanten gate and additionally
    requires the selected hand's publicly live improvement-copy count to be no
    lower than frozen Teacher's.  Since shanten is already strictly lower,
    equality in ukeire still constitutes a strict Pareto improvement.
    """

    def _requires_frozen_improvement_profile(self) -> bool:
        return True

    def _accept_deficiency_override(
        self, selected_profile, frozen_profile
    ) -> bool:
        if frozen_profile is None:
            raise AssertionError("Pareto 门控缺少 frozen 有效进张")
        return (
            selected_profile.improving_live_copies
            >= frozen_profile.improving_live_copies
        )


class PublicProgressTieBreakTeacherAgent(KnowledgeAwareDeficiencyTeacherAgent):
    """Pareto tiebreak only among exact top-scoring frozen discards.

    The frozen Teacher resolves an exact score tie by tile order.  This agent
    changes that arbitrary final ordering only when another exactly tied
    discard is no worse in both regular-hand shanten and publicly live
    improvement copies, and strictly better in at least one.  It never opens
    the wider near-score override family used by the rejected deficiency v1.
    """

    def explain_discard(self, game, player_id: int) -> list[dict[str, object]]:
        frozen = HeuristicTeacherAgent.explain_discard(self, game, player_id)
        if (
            not frozen
            or game.rules.profile != "classic"
            or game.tour_state is not None
            or game.gold_discard_lock_seat == player_id
            or int(frozen[0]["tile"]) == game.gold_tile
        ):
            return frozen

        top_score = float(frozen[0]["score"])
        tied = [
            row
            for row in frozen
            if int(row["tile"]) != game.gold_tile
            and abs(float(row["score"]) - top_score) <= 1e-9
        ]
        if len(tied) < 2:
            return frozen

        player = game.players[player_id]
        visible = self._public_visible_counts(game, player_id)
        live_counts = tuple(
            max(0, 4 - visible[tile]) for tile in range(BASE_TILE_COUNT)
        )
        profiles = {}
        for row in tied:
            tile = int(row["tile"])
            after_discard = list(player.hand)
            after_discard.remove(tile)
            appeared = self._appeared_honors(game, extra_discard=tile)
            profiles[tile] = public_draw_improvement_profile(
                after_discard,
                live_counts,
                game.gold_tile,
                meld_count=len(player.melds),
                melds_required=game.rules.melds_required,
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
                legal_discards=lambda after_draw, appeared=appeared: (
                    self._allowed_efficiency_discards(game, after_draw, appeared)
                ),
            )

        frozen_tile = int(frozen[0]["tile"])
        frozen_profile = profiles[frozen_tile]
        dominating = [
            row
            for row in tied
            if (
                profiles[int(row["tile"])].shanten <= frozen_profile.shanten
                and profiles[int(row["tile"])].improving_live_copies
                >= frozen_profile.improving_live_copies
                and (
                    profiles[int(row["tile"])].shanten < frozen_profile.shanten
                    or profiles[int(row["tile"])].improving_live_copies
                    > frozen_profile.improving_live_copies
                )
            )
        ]
        if not dominating:
            return frozen

        selected = min(
            dominating,
            key=lambda row: (
                profiles[int(row["tile"])].shanten,
                -profiles[int(row["tile"])].improving_live_copies,
                -profiles[int(row["tile"])].immediate_winning_live_copies,
                int(row["tile"]),
            ),
        )
        selected_tile = int(selected["tile"])
        enriched = []
        for row in frozen:
            tile = int(row["tile"])
            profile = profiles.get(tile)
            if profile is None:
                enriched.append(row)
                continue
            enriched.append(
                {
                    **row,
                    "teacher_score": row["score"],
                    "regular_hand_shanten": profile.shanten,
                    "public_improving_live_copies": (
                        profile.improving_live_copies
                    ),
                    "public_immediate_winning_live_copies": (
                        profile.immediate_winning_live_copies
                    ),
                    "selected_by_public_progress_tiebreak": (
                        tile == selected_tile
                    ),
                }
            )
        return [
            next(row for row in enriched if int(row["tile"]) == selected_tile),
            *(row for row in enriched if int(row["tile"]) != selected_tile),
        ]


class TwoDrawTenpaiReachTeacherAgent(KnowledgeAwareDeficiencyTeacherAgent):
    """Narrow override by paired two-self-draw public tenpai reachability.

    For each near-Teacher non-tenpai discard, this SlowExpert evaluates 32
    deterministic stratified scenarios from publicly unseen base-tile copies
    and chooses the best legal structural discard after each hypothetical
    draw.  Every candidate shares the same draw quantiles (common random
    numbers); the live wall and opponent hands are never read.  A candidate
    must preserve or improve structural shanten and increase the estimated
    chance of reaching tenpai within two own draws by a fixed absolute margin.
    This is a local speed model, not a complete game value.
    """

    STRATIFIED_SCENARIOS = 32

    def __init__(
        self,
        *,
        score_margin: float = 2.0,
        minimum_probability_advantage: float = 0.05,
    ):
        super().__init__(score_margin=score_margin)
        if not 0 < minimum_probability_advantage <= 1:
            raise ValueError("两摸听牌概率优势必须位于 (0, 1]")
        self.minimum_probability_advantage = float(minimum_probability_advantage)

    def _public_tenpai_reach_probability(
        self,
        game,
        player_id: int,
        hand_after_discard: list[int],
        live_counts: tuple[int, ...],
        appeared_honors: frozenset[int],
        *,
        horizon: int = 2,
    ) -> float:
        if horizon != 2:
            raise ValueError("v1 只支持固定两摸 horizon")
        player = game.players[player_id]
        meld_count = len(player.melds)

        def shanten(hand: list[int]) -> int:
            return standard_hand_shanten(
                hand,
                game.gold_tile,
                meld_count=meld_count,
                melds_required=game.rules.melds_required,
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
            )

        def draw_at_quantile(counts: list[int], quantile: float) -> int | None:
            total = sum(counts)
            if total <= 0:
                return None
            target = quantile * total
            cumulative = 0
            for tile, copies in enumerate(counts):
                cumulative += copies
                if target < cumulative:
                    return tile
            return next(
                (tile for tile in range(BASE_TILE_COUNT - 1, -1, -1) if counts[tile]),
                None,
            )

        def best_follow(
            hand_after_draw: list[int], appeared: frozenset[int]
        ) -> tuple[list[int], frozenset[int], int]:
            ranked = []
            for discarded in self._allowed_efficiency_discards(
                game, hand_after_draw, appeared
            ):
                after_follow = list(hand_after_draw)
                after_follow.remove(discarded)
                distance = shanten(after_follow)
                quality = hand_quality(
                    after_follow,
                    game.gold_tile,
                    meld_count=meld_count,
                    melds_required=game.rules.melds_required,
                    wildcard_tiles=game.wildcard_tiles,
                    proxy_tile=game.gold_proxy_tile,
                    proxy_as=game.gold_tile,
                )
                ranked.append((distance, -quality, discarded, after_follow))
            distance, _negative_quality, discarded, after_follow = min(ranked)
            next_appeared = appeared
            if (
                is_honor(discarded)
                and discarded not in {game.gold_tile, game.gold_proxy_tile}
            ):
                next_appeared = frozenset((*appeared, discarded))
            return after_follow, next_appeared, distance

        initial_distance = shanten(hand_after_discard)
        if initial_distance <= 0:
            return 1.0
        if initial_distance > horizon:
            return 0.0
        successes = 0
        scenario_count = self.STRATIFIED_SCENARIOS
        for scenario in range(scenario_count):
            counts = list(live_counts)
            hand = list(hand_after_discard)
            appeared = appeared_honors
            reached = False
            quantiles = (
                (scenario + 0.5) / scenario_count,
                ((scenario * 37) % scenario_count + 0.5) / scenario_count,
            )
            for quantile in quantiles:
                drawn = draw_at_quantile(counts, quantile)
                if drawn is None:
                    break
                counts[drawn] -= 1
                hand.append(drawn)
                hand, appeared, distance = best_follow(hand, appeared)
                if distance <= 0:
                    reached = True
                    break
            successes += reached
        return successes / scenario_count

    def explain_discard(self, game, player_id: int) -> list[dict[str, object]]:
        frozen = HeuristicTeacherAgent.explain_discard(self, game, player_id)
        if (
            not frozen
            or frozen[0]["waits"]
            or int(frozen[0]["tile"]) == game.gold_tile
            or game.tour_state is not None
            or game.gold_discard_lock_seat == player_id
            or game.rules.profile != "classic"
        ):
            return frozen

        top_score = float(frozen[0]["score"])
        eligible = [
            row
            for row in frozen
            if not row["waits"]
            and int(row["tile"]) != game.gold_tile
            and top_score - float(row["score"]) <= self.score_margin + 1e-9
        ]
        if len(eligible) < 2:
            return frozen

        player = game.players[player_id]
        after_hands: dict[int, list[int]] = {}
        shantens: dict[int, int] = {}
        for row in eligible:
            tile = int(row["tile"])
            after_hand = list(player.hand)
            after_hand.remove(tile)
            after_hands[tile] = after_hand
            shantens[tile] = standard_hand_shanten(
                after_hand,
                game.gold_tile,
                meld_count=len(player.melds),
                melds_required=game.rules.melds_required,
                wildcard_tiles=game.wildcard_tiles,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
            )

        frozen_tile = int(frozen[0]["tile"])
        frozen_shanten = shantens[frozen_tile]
        plausible = [
            row
            for row in eligible
            if shantens[int(row["tile"])] <= frozen_shanten
        ]
        if len(plausible) < 2:
            return frozen

        visible = self._public_visible_counts(game, player_id)
        live_counts = tuple(
            max(0, 4 - visible[tile]) for tile in range(BASE_TILE_COUNT)
        )
        probabilities = {}
        for row in plausible:
            tile = int(row["tile"])
            probabilities[tile] = self._public_tenpai_reach_probability(
                game,
                player_id,
                after_hands[tile],
                live_counts,
                self._appeared_honors(game, extra_discard=tile),
            )

        selected = max(
            plausible,
            key=lambda row: (
                probabilities[int(row["tile"])],
                -shantens[int(row["tile"])],
                float(row["score"]),
                -int(row["tile"]),
            ),
        )
        selected_tile = int(selected["tile"])
        advantage = probabilities[selected_tile] - probabilities[frozen_tile]
        if (
            selected_tile == frozen_tile
            or advantage + 1e-12 < self.minimum_probability_advantage
        ):
            return frozen

        enriched = []
        for row in frozen:
            tile = int(row["tile"])
            if tile not in probabilities:
                enriched.append(row)
                continue
            enriched.append(
                {
                    **row,
                    "teacher_score": row["score"],
                    "regular_hand_shanten": shantens[tile],
                    "two_draw_tenpai_probability": round(probabilities[tile], 8),
                    "selected_by_two_draw_tenpai_reach": tile == selected_tile,
                }
            )
        return [
            next(row for row in enriched if int(row["tile"]) == selected_tile),
            *(row for row in enriched if int(row["tile"]) != selected_tile),
        ]


class ExactTwoDrawTenpaiReachTeacherAgent(TwoDrawTenpaiReachTeacherAgent):
    """Exact finite-horizon confirmation gate for the v1 stratified proxy.

    The v1 SlowExpert approximates two future own draws with 32 deterministic
    quantiles and greedily keeps the lowest-shanten first continuation.  This
    variant first lets the frozen v1 rule propose one override, then solves
    that proposed action and the frozen Teacher action exactly:

    * the first and second draw are sampled without replacement from the
      actor-visible remaining face counts;
    * after the first draw, every public-rule-legal discard is compared by its
      exact probability of winning or reaching tenpai on the second draw;
    * an immediate winning draw counts as success before honor-follow is
      applied, matching the engine's win-before-forced-discard order.

    Restricting the expensive dynamic program to v1's proposal support makes
    this a conservative confirmation layer rather than a newly tuned global
    search. It is still a local tile-efficiency model, not a wall oracle or a
    complete Mahjong value function. Opponent concealed hands, the live wall
    and RNG are never inspected. Independent paired evaluation is required
    before this experimental policy can affect the default Teacher.
    """

    SOLVER_VERSION = "public-two-draw-exact-dp-v2"

    def __init__(
        self,
        *,
        score_margin: float = 2.0,
        minimum_probability_advantage: float = 0.05,
    ) -> None:
        super().__init__(
            score_margin=score_margin,
            minimum_probability_advantage=minimum_probability_advantage,
        )
        self._stratified_proposal = TwoDrawTenpaiReachTeacherAgent(
            score_margin=score_margin,
            minimum_probability_advantage=minimum_probability_advantage,
        )
        self.last_exact_diagnostics: dict[str, object] = {
            "stratified_proposal": False,
            "exact_confirmed": False,
            "exact_probability_advantage": None,
        }

    def _is_complete(self, game, player_id: int, hand: list[int]) -> bool:
        player = game.players[player_id]
        return is_winning_hand(
            hand,
            game.gold_tile,
            meld_count=len(player.melds),
            melds_required=game.rules.melds_required,
            allow_seven_pairs=game.rules.allow_seven_pairs,
            wildcard_tiles=game.wildcard_tiles,
            proxy_tile=game.gold_proxy_tile,
            proxy_as=game.gold_tile,
        )

    def _regular_shanten(self, game, player_id: int, hand: list[int]) -> int:
        player = game.players[player_id]
        return standard_hand_shanten(
            hand,
            game.gold_tile,
            meld_count=len(player.melds),
            melds_required=game.rules.melds_required,
            wildcard_tiles=game.wildcard_tiles,
            proxy_tile=game.gold_proxy_tile,
            proxy_as=game.gold_tile,
        )

    def _second_draw_success_probability(
        self,
        game,
        player_id: int,
        hand_after_first_discard: list[int],
        live_counts: tuple[int, ...],
        appeared_honors: frozenset[int],
    ) -> float:
        """Return exact next-own-draw win-or-tenpai probability."""

        total = sum(live_counts)
        if total <= 0:
            return 0.0
        successful_copies = 0
        for drawn, copies in enumerate(live_counts):
            if copies <= 0:
                continue
            after_draw = [*hand_after_first_discard, drawn]
            if self._is_complete(game, player_id, after_draw):
                successful_copies += copies
                continue
            for discarded in self._allowed_efficiency_discards(
                game, after_draw, appeared_honors
            ):
                after_discard = list(after_draw)
                after_discard.remove(discarded)
                if self._regular_shanten(game, player_id, after_discard) <= 0:
                    successful_copies += copies
                    break
        return successful_copies / total

    def _public_tenpai_reach_probability(
        self,
        game,
        player_id: int,
        hand_after_discard: list[int],
        live_counts: tuple[int, ...],
        appeared_honors: frozenset[int],
        *,
        horizon: int = 2,
    ) -> float:
        if horizon != 2:
            raise ValueError("v2 只支持固定两摸 horizon")
        initial_distance = self._regular_shanten(
            game, player_id, hand_after_discard
        )
        if initial_distance <= 0:
            return 1.0
        if initial_distance > horizon:
            return 0.0
        total = sum(live_counts)
        if total <= 0:
            return 0.0

        # ``explain_discard`` installs one decision-local cache shared by all
        # initial discard candidates.  Different first branches frequently
        # converge on the same concealed hand and remaining multiset; sharing
        # those exact second-draw results is a substantial CPU saving without
        # changing the probability calculation.
        second_cache = getattr(self, "_exact_second_draw_cache", None)
        if second_cache is None:
            second_cache = {}
        weighted_success = 0.0
        for drawn, copies in enumerate(live_counts):
            if copies <= 0:
                continue
            remaining = list(live_counts)
            remaining[drawn] -= 1
            remaining_key = tuple(remaining)
            after_draw = [*hand_after_discard, drawn]
            if self._is_complete(game, player_id, after_draw):
                branch_probability = 1.0
            else:
                branch_probability = 0.0
                for discarded in self._allowed_efficiency_discards(
                    game, after_draw, appeared_honors
                ):
                    after_first_discard = list(after_draw)
                    after_first_discard.remove(discarded)
                    distance = self._regular_shanten(
                        game, player_id, after_first_discard
                    )
                    if distance <= 0:
                        continuation_probability = 1.0
                    elif distance > 1:
                        continuation_probability = 0.0
                    else:
                        next_appeared = appeared_honors
                        if (
                            is_honor(discarded)
                            and discarded
                            not in {game.gold_tile, game.gold_proxy_tile}
                        ):
                            next_appeared = frozenset(
                                (*appeared_honors, discarded)
                            )
                        cache_key = (
                            tuple(sorted(after_first_discard)),
                            remaining_key,
                            next_appeared,
                        )
                        if cache_key in second_cache:
                            continuation_probability = second_cache[cache_key]
                        else:
                            continuation_probability = (
                                self._second_draw_success_probability(
                                    game,
                                    player_id,
                                    after_first_discard,
                                    remaining_key,
                                    next_appeared,
                                )
                            )
                            second_cache[cache_key] = continuation_probability
                    branch_probability = max(
                        branch_probability, continuation_probability
                    )
                    if branch_probability >= 1.0:
                        break
            weighted_success += copies * branch_probability
        return weighted_success / total

    def explain_discard(self, game, player_id: int) -> list[dict[str, object]]:
        frozen = HeuristicTeacherAgent.explain_discard(self, game, player_id)
        proposal = self._stratified_proposal.explain_discard(game, player_id)
        proposed = bool(
            proposal
            and proposal[0].get("selected_by_two_draw_tenpai_reach")
        )
        self.last_exact_diagnostics = {
            "stratified_proposal": proposed,
            "exact_confirmed": False,
            "exact_probability_advantage": None,
        }
        if not proposed:
            return frozen

        frozen_tile = int(frozen[0]["tile"])
        proposed_tile = int(proposal[0]["tile"])
        if proposed_tile == frozen_tile:
            raise AssertionError("v1 标记了未改变 Teacher 的 proposal")

        player = game.players[player_id]
        visible = self._public_visible_counts(game, player_id)
        live_counts = tuple(
            max(0, 4 - visible[tile]) for tile in range(BASE_TILE_COUNT)
        )
        self._exact_second_draw_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...], frozenset[int]], float
        ] = {}
        try:
            probabilities = {}
            shantens = {}
            for tile in (frozen_tile, proposed_tile):
                after_hand = list(player.hand)
                after_hand.remove(tile)
                shantens[tile] = self._regular_shanten(
                    game, player_id, after_hand
                )
                probabilities[tile] = self._public_tenpai_reach_probability(
                    game,
                    player_id,
                    after_hand,
                    live_counts,
                    self._appeared_honors(game, extra_discard=tile),
                )
        finally:
            del self._exact_second_draw_cache

        advantage = probabilities[proposed_tile] - probabilities[frozen_tile]
        confirmed = (
            shantens[proposed_tile] <= shantens[frozen_tile]
            and advantage + 1e-12 >= self.minimum_probability_advantage
        )
        self.last_exact_diagnostics = {
            "stratified_proposal": True,
            "exact_confirmed": confirmed,
            "exact_probability_advantage": advantage,
        }
        if not confirmed:
            return frozen

        proposal_rows = {int(row["tile"]): row for row in proposal}
        enriched = []
        for row in frozen:
            tile = int(row["tile"])
            if tile not in probabilities:
                enriched.append(row)
                continue
            proposal_probability = proposal_rows[tile].get(
                "two_draw_tenpai_probability"
            )
            enriched.append(
                {
                    **row,
                    "teacher_score": row["score"],
                    "regular_hand_shanten": shantens[tile],
                    "stratified_two_draw_tenpai_probability": (
                        proposal_probability
                    ),
                    "exact_two_draw_tenpai_probability": round(
                        probabilities[tile], 8
                    ),
                    "selected_by_exact_two_draw_tenpai_reach": (
                        tile == proposed_tile
                    ),
                    "two_draw_solver": self.SOLVER_VERSION,
                }
            )
        return [
            next(row for row in enriched if int(row["tile"]) == proposed_tile),
            *(row for row in enriched if int(row["tile"]) != proposed_tile),
        ]


class PublicTenpaiValueTeacherAgent(HeuristicTeacherAgent):
    """Correct one narrow direct-tenpai blind spot of the frozen Teacher.

    The frozen Teacher rewards the number of distinct wait faces, but does not
    distinguish waits by publicly possible copies or by their visible
    immediate self-draw settlement.  This candidate preserves every frozen
    decision except an ordinary discard where:

    * the frozen choice already leaves direct tenpai;
    * another direct-tenpai discard is within ``score_margin``; and
    * that alternative strictly improves
      ``sum(public_live_copies * immediate_self_draw_score)``.

    Only the actor's concealed hand and table-public information are read.
    Opponent concealed hands, the wall and concealed-kong faces are excluded.
    The statistic is a deliberately small tiebreak proxy rather than a full
    action-value estimate.  Independent paired evaluation is required before
    this experimental candidate can replace the frozen Teacher.
    """

    def __init__(
        self,
        *,
        score_margin: float = 2.0,
        minimum_weighted_value_advantage: float = 1.0,
    ):
        if score_margin < 0:
            raise ValueError("score_margin 不能为负数")
        if minimum_weighted_value_advantage <= 0:
            raise ValueError("minimum_weighted_value_advantage 必须为正数")
        self.score_margin = float(score_margin)
        self.minimum_weighted_value_advantage = float(
            minimum_weighted_value_advantage
        )

    @staticmethod
    def _public_visible_counts(game, player_id: int) -> Counter[int]:
        visible = Counter(
            tile for tile in game.players[player_id].hand if is_base_tile(tile)
        )
        for player in game.players:
            visible.update(tile for tile in player.discards if is_base_tile(tile))
            for meld in player.melds:
                if meld["kind"] == "an_kan" and player.seat != player_id:
                    continue
                visible.update(tile for tile in meld["tiles"] if is_base_tile(tile))
        if game.gold_indicator is not None and is_base_tile(game.gold_indicator):
            visible[game.gold_indicator] += 1
        return visible

    @staticmethod
    def _direct_tenpai_value(
        game,
        player_id: int,
        hand_after_discard: list[int],
        visible: Counter[int],
    ) -> dict[str, float | int]:
        player = game.players[player_id]
        waits = wait_tiles(
            hand_after_discard,
            game.gold_tile,
            meld_count=len(player.melds),
            melds_required=game.rules.melds_required,
            allow_seven_pairs=game.rules.allow_seven_pairs,
            wildcard_tiles=game.wildcard_tiles,
            proxy_tile=game.gold_proxy_tile,
            proxy_as=game.gold_tile,
        )
        live_copies = 0
        weighted_value = 0.0
        for wait in waits:
            remaining = max(0, 4 - visible[wait])
            if not remaining:
                continue
            winner = SimpleNamespace(
                hand=sorted([*hand_after_discard, wait]),
                flowers=list(player.flowers),
                melds=player.melds,
            )
            score = classic_score(
                winner,
                is_dealer=player_id == game.dealer,
                wildcard_tiles=game.wildcard_tiles,
                gold_tile=game.gold_tile,
                win_type="self_draw",
                rules=game.rules,
                dealer_streak=game.dealer_streak,
                proxy_tile=game.gold_proxy_tile,
                proxy_as=game.gold_tile,
            )
            live_copies += remaining
            weighted_value += remaining * float(score.total)
        return {
            "wait_faces": len(waits),
            "live_copies": live_copies,
            "weighted_value": weighted_value,
            "mean_visible_score": (
                weighted_value / live_copies if live_copies else 0.0
            ),
        }

    def explain_discard(self, game, player_id: int) -> list[dict[str, object]]:
        frozen = super().explain_discard(game, player_id)
        if (
            not frozen
            or not frozen[0]["waits"]
            or game.tour_state is not None
            or game.gold_discard_lock_seat == player_id
            or not game.rules.enable_complex_water_scoring
        ):
            return frozen

        top_score = float(frozen[0]["score"])
        eligible = [
            row
            for row in frozen
            if row["waits"]
            and top_score - float(row["score"]) <= self.score_margin + 1e-9
        ]
        if len(eligible) < 2:
            return frozen

        player = game.players[player_id]
        visible = self._public_visible_counts(game, player_id)
        values: dict[int, dict[str, float | int]] = {}
        for row in eligible:
            tile = int(row["tile"])
            after_discard = list(player.hand)
            after_discard.remove(tile)
            values[tile] = self._direct_tenpai_value(
                game, player_id, after_discard, visible
            )

        frozen_tile = int(frozen[0]["tile"])
        selected = max(
            eligible,
            key=lambda row: (
                float(values[int(row["tile"])]["weighted_value"]),
                int(values[int(row["tile"])]["live_copies"]),
                float(values[int(row["tile"])]["mean_visible_score"]),
                float(row["score"]),
                -int(row["tile"]),
            ),
        )
        selected_tile = int(selected["tile"])
        improvement = float(values[selected_tile]["weighted_value"]) - float(
            values[frozen_tile]["weighted_value"]
        )
        if (
            selected_tile == frozen_tile
            or improvement < self.minimum_weighted_value_advantage
        ):
            return frozen

        enriched = []
        for row in frozen:
            tile = int(row["tile"])
            if tile not in values:
                enriched.append(row)
                continue
            enriched.append(
                {
                    **row,
                    "teacher_score": row["score"],
                    "public_live_copies": values[tile]["live_copies"],
                    "public_weighted_tenpai_value": values[tile]["weighted_value"],
                    "visible_self_draw_score_mean": values[tile][
                        "mean_visible_score"
                    ],
                    "selected_by_public_tenpai_value": tile == selected_tile,
                }
            )
        return [
            next(row for row in enriched if int(row["tile"]) == selected_tile),
            *(row for row in enriched if int(row["tile"]) != selected_tile),
        ]

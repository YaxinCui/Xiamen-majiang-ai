"""Explainable rule Teacher used by the local browser game and data export."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

from .hand import hand_quality, is_winning_hand, one_draw_tenpai_routes, wait_tiles
from .scoring import classic_score
from .tiles import BASE_TILE_COUNT, is_base_tile, is_honor, tile_name


@dataclass(frozen=True)
class GameAction:
    kind: str
    tile: int | None = None
    tiles: tuple[int, ...] = ()


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
            visible.update(
                tile
                for meld in player.melds
                for tile in meld["tiles"]
                if is_base_tile(tile)
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
            visible.update(
                tile
                for meld in player.melds
                for tile in meld["tiles"]
                if is_base_tile(tile)
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

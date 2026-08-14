"""Explainable rule Teacher used by the local browser game and data export."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .hand import hand_quality, wait_tiles
from .tiles import BASE_TILE_COUNT, tile_name


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
        if game._can_win(player_id):
            return GameAction("hu")

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

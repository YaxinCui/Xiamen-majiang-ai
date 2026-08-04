"""Explainable rule Teacher used by the local browser game and data export."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .hand import hand_quality, is_winning_hand, wait_tiles
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
        if is_winning_hand(
            player.hand,
            game.gold_tile,
            meld_count=meld_count,
            allow_seven_pairs=game.rules.allow_seven_pairs,
        ):
            return GameAction("hu")

        if len(game.wall) > 16:
            for tile, count in sorted(Counter(player.hand).items()):
                if count == 4 and tile != game.gold_tile:
                    return GameAction("an_kan", tile)
            for meld in player.melds:
                if meld["kind"] == "pong":
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
            )
            after_hand = list(player.hand)
            for _ in range(2):
                after_hand.remove(kinds["pong"].tile)
            after = hand_quality(after_hand, game.gold_tile, meld_count=len(player.melds) + 1)
            if after >= before - 1.0:
                return kinds["pong"]
        chi_options = [option for option in options if option.kind == "chi"]
        if chi_options:
            player = game.players[player_id]
            before = hand_quality(player.hand, game.gold_tile, meld_count=len(player.melds))
            scored = []
            for option in chi_options:
                after_hand = list(player.hand)
                for tile in option.tiles:
                    after_hand.remove(tile)
                scored.append(
                    (
                        hand_quality(after_hand, game.gold_tile, meld_count=len(player.melds) + 1),
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
        for tile in sorted(set(player.hand)):
            candidate = list(player.hand)
            candidate.remove(tile)
            waits = wait_tiles(candidate, game.gold_tile, meld_count=len(player.melds))
            quality = hand_quality(candidate, game.gold_tile, meld_count=len(player.melds))
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

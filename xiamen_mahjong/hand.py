"""Winning-hand and inexpensive structure analysis with a gold wildcard."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

from .tiles import BASE_TILE_COUNT, is_suited


def is_winning_hand(
    tiles: list[int],
    gold_tile: int | None,
    *,
    meld_count: int = 0,
    allow_seven_pairs: bool = True,
) -> bool:
    """Check a 14-tile hand using the public gold tile as a wildcard.

    Existing exposed melds are supplied as ``meld_count``.  Seven pairs is
    legal only for a closed hand, which matches the local v1 table rules.
    """

    expected = 14 - meld_count * 3
    if len(tiles) != expected:
        return False
    non_gold = [tile for tile in tiles if tile != gold_tile]
    jokers = len(tiles) - len(non_gold) if gold_tile is not None else 0
    counts = [0] * BASE_TILE_COUNT
    for tile in non_gold:
        if not 0 <= tile < BASE_TILE_COUNT:
            return False
        counts[tile] += 1
    if allow_seven_pairs and meld_count == 0 and _is_seven_pairs(counts, jokers):
        return True
    return _is_standard(tuple(counts), jokers, meld_count)


def winning_pattern(
    tiles: list[int], gold_tile: int | None, *, meld_count: int = 0
) -> str | None:
    if not is_winning_hand(tiles, gold_tile, meld_count=meld_count):
        return None
    if meld_count == 0:
        counts = _counts_without_gold(tiles, gold_tile)
        jokers = len(tiles) - sum(counts)
        if _is_seven_pairs(counts, jokers):
            return "七对"
    return "标准和"


def wait_tiles(tiles: list[int], gold_tile: int | None, *, meld_count: int = 0) -> list[int]:
    """Return tile identities that complete a 13-tile (or post-meld) hand."""

    waits = []
    for tile in range(BASE_TILE_COUNT):
        if is_winning_hand(
            [*tiles, tile],
            gold_tile,
            meld_count=meld_count,
        ):
            waits.append(tile)
    return waits


def hand_quality(tiles: list[int], gold_tile: int | None, *, meld_count: int = 0) -> float:
    """A deterministic structure score for the Teacher's discard lookahead."""

    counter = Counter(tile for tile in tiles if tile != gold_tile)
    gold_count = len(tiles) - sum(counter.values())
    score = meld_count * 28 + gold_count * 12
    score += sum((count // 3) * 14 + (count % 3 == 2) * 5 for count in counter.values())
    for base in (0, 9, 18):
        suit_counts = [counter[base + offset] for offset in range(9)]
        for index in range(7):
            score += min(suit_counts[index], suit_counts[index + 1]) * 2.5
            score += min(suit_counts[index], suit_counts[index + 2]) * 1.25
        score += sum(min(suit_counts[index : index + 3]) * 5 for index in range(7))
    return float(score)


def _counts_without_gold(tiles: list[int], gold_tile: int | None) -> list[int]:
    result = [0] * BASE_TILE_COUNT
    for tile in tiles:
        if tile != gold_tile:
            result[tile] += 1
    return result


def _is_seven_pairs(counts: list[int], jokers: int) -> bool:
    natural_pairs = sum(count // 2 for count in counts)
    singles = sum(count % 2 for count in counts)
    if jokers < singles:
        return False
    return natural_pairs + singles + (jokers - singles) // 2 >= 7


def _is_standard(counts: tuple[int, ...], jokers: int, meld_count: int) -> bool:
    required_melds = 4 - meld_count
    if required_melds < 0:
        return False
    for pair_tile in range(BASE_TILE_COUNT):
        mutable = list(counts)
        if mutable[pair_tile] >= 2:
            mutable[pair_tile] -= 2
            if _can_form_melds(tuple(mutable), jokers, required_melds):
                return True
        if mutable[pair_tile] >= 1 and jokers >= 1:
            mutable[pair_tile] -= 1
            if _can_form_melds(tuple(mutable), jokers - 1, required_melds):
                return True
    if jokers >= 2 and _can_form_melds(counts, jokers - 2, required_melds):
        return True
    return False


@lru_cache(maxsize=250_000)
def _can_form_melds(counts: tuple[int, ...], jokers: int, required_melds: int) -> bool:
    remaining = sum(counts)
    if remaining == 0:
        return jokers == required_melds * 3
    if required_melds <= 0:
        return False
    tile = next(index for index, count in enumerate(counts) if count)
    count = counts[tile]

    # Triplet, filling any missing copies with jokers.
    triplet_need = max(0, 3 - count)
    if triplet_need <= jokers:
        mutable = list(counts)
        mutable[tile] -= min(3, count)
        if _can_form_melds(tuple(mutable), jokers - triplet_need, required_melds - 1):
            return True

    # Sequence using this first tile.  Honor tiles cannot be sequenced.
    if is_suited(tile) and tile % 9 <= 6:
        mutable = list(counts)
        needed = 0
        for sequence_tile in (tile, tile + 1, tile + 2):
            if mutable[sequence_tile]:
                mutable[sequence_tile] -= 1
            else:
                needed += 1
        if needed <= jokers and _can_form_melds(
            tuple(mutable), jokers - needed, required_melds - 1
        ):
            return True
    return False

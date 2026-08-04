"""Winning-hand and inexpensive structure analysis with a gold wildcard."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Iterable

from .tiles import BASE_TILE_COUNT, is_suited


def is_winning_hand(
    tiles: list[int],
    gold_tile: int | None,
    *,
    meld_count: int = 0,
    melds_required: int = 4,
    allow_seven_pairs: bool = True,
    wildcard_tiles: Iterable[int] | None = None,
    proxy_tile: int | None = None,
    proxy_as: int | None = None,
) -> bool:
    """Check a complete concealed remainder using the public gold wildcard.

    ``melds_required`` is four for the 13/14-tile teaching profile and five
    for classical Xiamen's 16/17-tile game.  A white dragon proxy is mapped to
    the *face value* of the gold via ``proxy_tile``/``proxy_as``; it is not a
    second universal wildcard.
    """

    expected = melds_required * 3 + 2 - meld_count * 3
    if len(tiles) != expected:
        return False
    wildcards = _wildcard_set(gold_tile, wildcard_tiles)
    non_gold = [
        _proxy_value(tile, proxy_tile, proxy_as)
        for tile in tiles
        if tile not in wildcards
    ]
    jokers = len(tiles) - len(non_gold)
    counts = [0] * BASE_TILE_COUNT
    for tile in non_gold:
        if not 0 <= tile < BASE_TILE_COUNT:
            return False
        counts[tile] += 1
    if (
        allow_seven_pairs
        and melds_required == 4
        and meld_count == 0
        and _is_seven_pairs(counts, jokers)
    ):
        return True
    return _is_standard(tuple(counts), jokers, meld_count, melds_required)


def winning_pattern(
    tiles: list[int],
    gold_tile: int | None,
    *,
    meld_count: int = 0,
    melds_required: int = 4,
    allow_seven_pairs: bool = True,
    wildcard_tiles: Iterable[int] | None = None,
    proxy_tile: int | None = None,
    proxy_as: int | None = None,
) -> str | None:
    if not is_winning_hand(
        tiles,
        gold_tile,
        meld_count=meld_count,
        melds_required=melds_required,
        allow_seven_pairs=allow_seven_pairs,
        wildcard_tiles=wildcard_tiles,
        proxy_tile=proxy_tile,
        proxy_as=proxy_as,
    ):
        return None
    if allow_seven_pairs and melds_required == 4 and meld_count == 0:
        wildcards = _wildcard_set(gold_tile, wildcard_tiles)
        counts = _counts_without_gold(tiles, wildcards, proxy_tile, proxy_as)
        jokers = len(tiles) - sum(counts)
        if _is_seven_pairs(counts, jokers):
            return "七对"
    return "标准和"


def wait_tiles(
    tiles: list[int],
    gold_tile: int | None,
    *,
    meld_count: int = 0,
    melds_required: int = 4,
    allow_seven_pairs: bool = True,
    wildcard_tiles: Iterable[int] | None = None,
    proxy_tile: int | None = None,
    proxy_as: int | None = None,
) -> list[int]:
    """Return physical tile identities that complete the current hand."""

    waits = []
    for tile in range(BASE_TILE_COUNT):
        if is_winning_hand(
            [*tiles, tile],
            gold_tile,
            meld_count=meld_count,
            melds_required=melds_required,
            allow_seven_pairs=allow_seven_pairs,
            wildcard_tiles=wildcard_tiles,
            proxy_tile=proxy_tile,
            proxy_as=proxy_as,
        ):
            waits.append(tile)
    return waits


def hand_quality(
    tiles: list[int],
    gold_tile: int | None,
    *,
    meld_count: int = 0,
    melds_required: int = 4,
    wildcard_tiles: Iterable[int] | None = None,
    proxy_tile: int | None = None,
    proxy_as: int | None = None,
) -> float:
    """A deterministic structure score for the Teacher's discard lookahead."""

    wildcards = _wildcard_set(gold_tile, wildcard_tiles)
    counter = Counter(
        _proxy_value(tile, proxy_tile, proxy_as)
        for tile in tiles
        if tile not in wildcards
    )
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


def is_travelling_ready(
    tiles: list[int],
    gold_tile: int | None,
    *,
    meld_count: int = 0,
    melds_required: int = 5,
    proxy_tile: int | None = None,
    proxy_as: int | None = None,
) -> bool:
    """Return whether ``tiles`` are five melds plus a singleton actual gold.

    This is the declaration shape after the natural mate of the gold has been
    discarded.  One actual gold is reserved as the touring singleton; any
    additional actual gold remains available as a wildcard inside the melds.
    """

    expected = melds_required * 3 + 1 - meld_count * 3
    if gold_tile is None or len(tiles) != expected or gold_tile not in tiles:
        return False
    remaining = list(tiles)
    remaining.remove(gold_tile)
    wildcards = remaining.count(gold_tile)
    natural = [
        _proxy_value(tile, proxy_tile, proxy_as)
        for tile in remaining
        if tile != gold_tile
    ]
    counts = [0] * BASE_TILE_COUNT
    for tile in natural:
        if not 0 <= tile < BASE_TILE_COUNT:
            return False
        counts[tile] += 1
    return _can_form_melds(tuple(counts), wildcards, melds_required - meld_count)


def _counts_without_gold(
    tiles: list[int],
    wildcard_tiles: set[int],
    proxy_tile: int | None = None,
    proxy_as: int | None = None,
) -> list[int]:
    result = [0] * BASE_TILE_COUNT
    for tile in tiles:
        if tile not in wildcard_tiles:
            result[_proxy_value(tile, proxy_tile, proxy_as)] += 1
    return result


def _wildcard_set(gold_tile: int | None, wildcard_tiles: Iterable[int] | None) -> set[int]:
    result = set(wildcard_tiles or ())
    if gold_tile is not None:
        result.add(gold_tile)
    return result


def _proxy_value(tile: int, proxy_tile: int | None, proxy_as: int | None) -> int:
    if proxy_tile is not None and proxy_as is not None and tile == proxy_tile:
        return proxy_as
    return tile


def _is_seven_pairs(counts: list[int], jokers: int) -> bool:
    natural_pairs = sum(count // 2 for count in counts)
    singles = sum(count % 2 for count in counts)
    if jokers < singles:
        return False
    return natural_pairs + singles + (jokers - singles) // 2 >= 7


def _is_standard(
    counts: tuple[int, ...], jokers: int, meld_count: int, melds_required: int
) -> bool:
    required_melds = melds_required - meld_count
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

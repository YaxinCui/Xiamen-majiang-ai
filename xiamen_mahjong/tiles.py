"""Tile constants and presentation helpers for the local Xiamen ruleset."""

from __future__ import annotations

from collections import Counter

SUIT_NAMES = ("万", "筒", "条")
WINDS = ("东", "南", "西", "北")
DRAGONS = ("中", "发", "白")
FLOWERS = ("梅", "兰", "菊", "竹", "春", "夏", "秋", "冬")

BASE_TILE_COUNT = 34
FLOWER_TILE_BASE = BASE_TILE_COUNT
TOTAL_TILE_COUNT = BASE_TILE_COUNT + len(FLOWERS)
WHITE_DRAGON = 33


def is_base_tile(tile: int) -> bool:
    return 0 <= tile < BASE_TILE_COUNT


def is_flower(tile: int) -> bool:
    return FLOWER_TILE_BASE <= tile < TOTAL_TILE_COUNT


def is_suited(tile: int) -> bool:
    return 0 <= tile < 27


def is_honor(tile: int) -> bool:
    return 27 <= tile < BASE_TILE_COUNT


def suit_index(tile: int) -> int | None:
    return tile // 9 if is_suited(tile) else None


def rank(tile: int) -> int | None:
    return tile % 9 + 1 if is_suited(tile) else None


def tile_name(tile: int) -> str:
    if 0 <= tile < 27:
        return f"{tile % 9 + 1}{SUIT_NAMES[tile // 9]}"
    if 27 <= tile < 31:
        return WINDS[tile - 27]
    if 31 <= tile < BASE_TILE_COUNT:
        return DRAGONS[tile - 31]
    if is_flower(tile):
        return FLOWERS[tile - FLOWER_TILE_BASE]
    raise ValueError(f"unknown tile: {tile}")


def tile_payload(tile: int) -> dict[str, object]:
    return {
        "id": tile,
        "name": tile_name(tile),
        "flower": is_flower(tile),
    }


def next_gold_tile(indicator: int) -> int:
    """Return the public gold tile derived from a base-tile indicator.

    Suits wrap 9 -> 1; winds and dragons each wrap within their own group.
    Flower indicators are deliberately never selected by the game.
    """

    if not is_base_tile(indicator):
        raise ValueError("a gold indicator must be a base tile")
    if is_suited(indicator):
        base = indicator // 9 * 9
        return base + (indicator + 1 - base) % 9
    if 27 <= indicator < 31:
        return 27 + (indicator - 27 + 1) % 4
    return 31 + (indicator - 31 + 1) % 3


def base_wall(*, include_honors: bool = True) -> list[int]:
    """Build a physical wall for the selected Xiamen ruleset.

    The 144-tile game contains all 34 base identities.  The current 120-tile
    game removes winds plus red/green dragons, but deliberately keeps the four
    white dragons because they are the fixed face-value proxy for gold.
    """

    tiles: list[int] = []
    base_tiles = (
        range(BASE_TILE_COUNT)
        if include_honors
        else (*range(27), WHITE_DRAGON)
    )
    for tile in base_tiles:
        tiles.extend([tile] * 4)
    tiles.extend(range(FLOWER_TILE_BASE, TOTAL_TILE_COUNT))
    return tiles


def counts(tiles: list[int]) -> Counter[int]:
    return Counter(tile for tile in tiles if is_base_tile(tile))

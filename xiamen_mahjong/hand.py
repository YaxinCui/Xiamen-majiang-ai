"""Winning-hand and inexpensive structure analysis with a gold wildcard."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from collections.abc import Callable
from typing import Iterable, Sequence

from .tiles import BASE_TILE_COUNT, is_suited


@dataclass(frozen=True)
class DrawImprovementProfile:
    """Exact structural shanten plus public live improvement coverage."""

    shanten: int
    draws_to_win: int
    improving_faces: tuple[int, ...]
    improving_live_copies: int
    immediate_winning_faces: tuple[int, ...]
    immediate_winning_live_copies: int


@dataclass(frozen=True)
class HandQualityComponents:
    """Auditable linear components used by the rule Teacher scorer."""

    fixed_melds: int
    gold_tiles: int
    triplet_groups: int
    pair_remainders: int
    adjacent_overlap: int
    gap_overlap: int
    sequence_overlap: int


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

    wildcard_key = tuple(sorted(set(wildcard_tiles or ())))
    return list(
        _wait_tiles_cached(
            tuple(sorted(tiles)),
            gold_tile,
            meld_count,
            melds_required,
            allow_seven_pairs,
            wildcard_key,
            proxy_tile,
            proxy_as,
        )
    )


@lru_cache(maxsize=200_000)
def _wait_tiles_cached(
    tiles: tuple[int, ...],
    gold_tile: int | None,
    meld_count: int,
    melds_required: int,
    allow_seven_pairs: bool,
    wildcard_tiles: tuple[int, ...],
    proxy_tile: int | None,
    proxy_as: int | None,
) -> tuple[int, ...]:
    """Memoize semantic wait queries shared by exact one-draw routes.

    Tile order is immaterial to hand completion.  Canonicalizing it at the
    public boundary lets adjacent hypothetical discard routes reuse the same
    exact answer without changing the rules API or exposing state.
    """

    waits: list[int] = []
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
    return tuple(waits)


def one_draw_tenpai_profile(
    tiles: list[int],
    gold_tile: int | None,
    *,
    meld_count: int = 0,
    melds_required: int = 4,
    allow_seven_pairs: bool = True,
    wildcard_tiles: Iterable[int] | None = None,
    proxy_tile: int | None = None,
    proxy_as: int | None = None,
    legal_discards: Callable[[list[int]], Iterable[int]] | None = None,
) -> dict[int, tuple[int, ...]]:
    """Return exact one-draw routes that can leave the hand in tenpai.

    ``tiles`` is the concealed hand *after a normal discard*.  For every
    physical base-tile face that might be drawn, this function enumerates the
    following discard and retains the largest resulting set of winning waits.
    It is a pure hand-structure oracle: it intentionally knows nothing about
    the wall, opponents, claim priority or live-copy counts.  A caller may
    provide ``legal_discards`` for a public local forced-discard rule; all
    other public-rule and availability handling remains the caller's job.

    This is not a terminal rollout or an estimated win probability.  It gives
    an exact, auditable answer to the limited question: "which next draw can
    create a tenpai hand after one legal-shaped discard?"
    """

    routes = one_draw_tenpai_routes(
        tiles,
        gold_tile,
        meld_count=meld_count,
        melds_required=melds_required,
        allow_seven_pairs=allow_seven_pairs,
        wildcard_tiles=wildcard_tiles,
        proxy_tile=proxy_tile,
        proxy_as=proxy_as,
        legal_discards=legal_discards,
    )
    profile: dict[int, tuple[int, ...]] = {}
    for drawn, choices in routes.items():
        _discarded, best_waits = min(
            choices,
            key=lambda item: (-len(item[1]), item[1], item[0]),
        )
        profile[drawn] = best_waits
    return profile


def one_draw_tenpai_routes(
    tiles: list[int],
    gold_tile: int | None,
    *,
    meld_count: int = 0,
    melds_required: int = 4,
    allow_seven_pairs: bool = True,
    wildcard_tiles: Iterable[int] | None = None,
    proxy_tile: int | None = None,
    proxy_as: int | None = None,
    legal_discards: Callable[[list[int]], Iterable[int]] | None = None,
) -> dict[int, tuple[tuple[int, tuple[int, ...]], ...]]:
    """Return every legal one-draw route from a post-discard hand.

    Each mapping entry is ``drawn_face -> ((follow_discard, waits), ...)``.
    A caller may supply ``legal_discards`` to apply a public local rule to
    the hypothetical hand after the draw, such as Xiamen classic's honor
    follow rule.  The callback must return faces present in that hand; an
    invalid face raises ``ValueError`` rather than silently evaluating an
    impossible route.

    This remains a hand-structure routine: draw availability, the value of a
    wait, wall state and opponents are deliberately left to the caller.
    """

    routes: dict[int, tuple[tuple[int, tuple[int, ...]], ...]] = {}
    for drawn in range(BASE_TILE_COUNT):
        after_draw = [*tiles, drawn]
        discard_faces = (
            legal_discards(list(after_draw))
            if legal_discards is not None
            else set(after_draw)
        )
        choices: list[tuple[int, tuple[int, ...]]] = []
        for discarded in sorted(set(discard_faces)):
            if discarded not in after_draw:
                raise ValueError("legal_discards 返回了不在手牌中的牌")
            after_discard = list(after_draw)
            after_discard.remove(discarded)
            waits = tuple(
                wait_tiles(
                    after_discard,
                    gold_tile,
                    meld_count=meld_count,
                    melds_required=melds_required,
                    allow_seven_pairs=allow_seven_pairs,
                    wildcard_tiles=wildcard_tiles,
                    proxy_tile=proxy_tile,
                    proxy_as=proxy_as,
                )
            )
            if waits:
                choices.append((discarded, waits))
        if choices:
            routes[drawn] = tuple(choices)
    return routes


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

    parts = hand_quality_components(
        tiles,
        gold_tile,
        meld_count=meld_count,
        wildcard_tiles=wildcard_tiles,
        proxy_tile=proxy_tile,
        proxy_as=proxy_as,
    )
    return float(
        parts.fixed_melds * 28
        + parts.gold_tiles * 12
        + parts.triplet_groups * 14
        + parts.pair_remainders * 5
        + parts.adjacent_overlap * 2.5
        + parts.gap_overlap * 1.25
        + parts.sequence_overlap * 5
    )


def hand_quality_components(
    tiles: Sequence[int],
    gold_tile: int | None,
    *,
    meld_count: int = 0,
    wildcard_tiles: Iterable[int] | None = None,
    proxy_tile: int | None = None,
    proxy_as: int | None = None,
) -> HandQualityComponents:
    """Return the frozen Teacher's structure score as integer features."""

    wildcards = _wildcard_set(gold_tile, wildcard_tiles)
    counter = Counter(
        _proxy_value(tile, proxy_tile, proxy_as)
        for tile in tiles
        if tile not in wildcards
    )
    gold_count = len(tiles) - sum(counter.values())
    triplet_groups = sum(count // 3 for count in counter.values())
    pair_remainders = sum(count % 3 == 2 for count in counter.values())
    adjacent_overlap = 0
    gap_overlap = 0
    sequence_overlap = 0
    for base in (0, 9, 18):
        suit_counts = [counter[base + offset] for offset in range(9)]
        for index in range(7):
            adjacent_overlap += min(suit_counts[index], suit_counts[index + 1])
            gap_overlap += min(suit_counts[index], suit_counts[index + 2])
        sequence_overlap += sum(
            min(suit_counts[index : index + 3]) for index in range(7)
        )
    return HandQualityComponents(
        fixed_melds=meld_count,
        gold_tiles=gold_count,
        triplet_groups=triplet_groups,
        pair_remainders=pair_remainders,
        adjacent_overlap=adjacent_overlap,
        gap_overlap=gap_overlap,
        sequence_overlap=sequence_overlap,
    )


def standard_hand_shanten(
    tiles: Sequence[int],
    gold_tile: int | None,
    *,
    meld_count: int = 0,
    melds_required: int = 4,
    wildcard_tiles: Iterable[int] | None = None,
    proxy_tile: int | None = None,
    proxy_as: int | None = None,
) -> int:
    """Return exact regular-hand shanten for four/five-meld variants.

    ``-1`` is a complete hand and ``0`` is tenpai.  Actual gold tiles are
    enumerated as flexible logical faces; a white-dragon proxy is first mapped
    to the gold's fixed face value and never becomes an extra wildcard.

    This is deliberately the regular meld-plus-pair distance used by classic
    Xiamen.  Seven pairs is not mixed into this primitive because the classic
    profile disables it and its hand-size contract differs from five melds.
    """

    if not 0 <= meld_count <= melds_required:
        raise ValueError("meld_count 必须位于 0 和 melds_required 之间")
    wildcards = _wildcard_set(gold_tile, wildcard_tiles)
    counts = [0] * BASE_TILE_COUNT
    jokers = 0
    for tile in tiles:
        if tile in wildcards:
            jokers += 1
            continue
        logical = _proxy_value(tile, proxy_tile, proxy_as)
        if not 0 <= logical < BASE_TILE_COUNT:
            raise ValueError("向听计算只接受基础牌")
        counts[logical] += 1
    return _standard_shanten_with_jokers(
        tuple(counts), jokers, meld_count, melds_required
    )


@lru_cache(maxsize=250_000)
def _standard_shanten_with_jokers(
    counts: tuple[int, ...],
    jokers: int,
    meld_count: int,
    melds_required: int,
) -> int:
    if jokers < 0:
        raise ValueError("jokers 不能为负数")
    natural = _standard_shanten_natural(counts, meld_count, melds_required)
    # Regular-hand shanten is the minimum number of useful tile additions
    # needed to reach tenpai (and -1 for complete).  A wildcard can realize
    # any one such missing face, so each existing joker removes exactly one
    # unit until completion.  This is equivalent to enumerating every logical
    # assignment but avoids 34**j repeated decompositions at every ukeire leaf.
    return max(-1, natural - jokers)


@lru_cache(maxsize=500_000)
def _standard_shanten_natural(
    counts: tuple[int, ...],
    fixed_melds: int,
    melds_required: int,
) -> int:
    """Enumerate meld/head/incomplete-block decompositions exactly."""

    @lru_cache(maxsize=None)
    def search(
        state: tuple[int, ...],
        melds: int,
        incomplete: int,
        has_pair: bool,
    ) -> int:
        total_melds = fixed_melds + melds
        if total_melds > melds_required:
            return 2 * melds_required
        incomplete = min(incomplete, melds_required - total_melds)
        try:
            tile = next(index for index, count in enumerate(state) if count)
        except StopIteration:
            usable_incomplete = min(incomplete, melds_required - total_melds)
            return (
                2 * melds_required
                - 2 * total_melds
                - usable_incomplete
                - int(has_pair)
            )

        candidates: list[int] = []

        def branch(removals: Sequence[int], *, meld: int = 0, block: int = 0, pair: bool = has_pair) -> None:
            mutable = list(state)
            for removed in removals:
                mutable[removed] -= 1
            candidates.append(
                search(tuple(mutable), melds + meld, incomplete + block, pair)
            )

        count = state[tile]
        if count >= 3:
            branch((tile, tile, tile), meld=1)
        if is_suited(tile) and tile % 9 <= 6 and all(
            state[value] for value in (tile, tile + 1, tile + 2)
        ):
            branch((tile, tile + 1, tile + 2), meld=1)
        if count >= 2:
            if not has_pair:
                branch((tile, tile), pair=True)
            branch((tile, tile), block=1)
        if is_suited(tile):
            for other in (tile + 1, tile + 2):
                if other // 9 == tile // 9 and other < BASE_TILE_COUNT and state[other]:
                    branch((tile, other), block=1)

        # Leave this face unused by the selected decomposition.
        branch((tile,))
        return min(candidates)

    return search(counts, 0, 0, False)


def public_draw_improvement_profile(
    tiles: Sequence[int],
    live_counts: Sequence[int],
    gold_tile: int | None,
    *,
    meld_count: int = 0,
    melds_required: int = 4,
    wildcard_tiles: Iterable[int] | None = None,
    proxy_tile: int | None = None,
    proxy_as: int | None = None,
    legal_discards: Callable[[list[int]], Iterable[int]] | None = None,
) -> DrawImprovementProfile:
    """Count publicly live draws that strictly reduce regular-hand shanten.

    ``tiles`` is the concealed post-discard hand.  For each publicly unseen
    base-tile face, the routine draws it and chooses the best legal follow-up
    discard.  It does not inspect a wall or opponent hand and makes no claim
    that public remaining copies are in the live wall; the result is an
    actor-visible tile-efficiency statistic, not a win probability.
    """

    if len(live_counts) != BASE_TILE_COUNT or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in live_counts
    ):
        raise ValueError("live_counts 必须是 34 个非负整数")
    shanten = standard_hand_shanten(
        tiles,
        gold_tile,
        meld_count=meld_count,
        melds_required=melds_required,
        wildcard_tiles=wildcard_tiles,
        proxy_tile=proxy_tile,
        proxy_as=proxy_as,
    )
    improving: list[int] = []
    immediate: list[int] = []
    for drawn, copies in enumerate(live_counts):
        if not copies:
            continue
        after_draw = [*tiles, drawn]
        if is_winning_hand(
            after_draw,
            gold_tile,
            meld_count=meld_count,
            melds_required=melds_required,
            allow_seven_pairs=False,
            wildcard_tiles=wildcard_tiles,
            proxy_tile=proxy_tile,
            proxy_as=proxy_as,
        ):
            immediate.append(drawn)
            improving.append(drawn)
            continue
        discard_faces = (
            legal_discards(list(after_draw))
            if legal_discards is not None
            else set(after_draw)
        )
        best_after = None
        for discarded in sorted(set(discard_faces)):
            if discarded not in after_draw:
                raise ValueError("legal_discards 返回了不在手牌中的牌")
            after_discard = list(after_draw)
            after_discard.remove(discarded)
            candidate = standard_hand_shanten(
                after_discard,
                gold_tile,
                meld_count=meld_count,
                melds_required=melds_required,
                wildcard_tiles=wildcard_tiles,
                proxy_tile=proxy_tile,
                proxy_as=proxy_as,
            )
            best_after = candidate if best_after is None else min(best_after, candidate)
        if best_after is not None and best_after < shanten:
            improving.append(drawn)
    return DrawImprovementProfile(
        shanten=shanten,
        draws_to_win=shanten + 1,
        improving_faces=tuple(improving),
        improving_live_copies=sum(live_counts[tile] for tile in improving),
        immediate_winning_faces=tuple(immediate),
        immediate_winning_live_copies=sum(live_counts[tile] for tile in immediate),
    )


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

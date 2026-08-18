"""Explainable settlement helpers for the classic Xiamen profile.

The public rule descriptions consistently define the payment multipliers and
the common water items, but may differ on optional tour and flower bonuses.
This module therefore scores only the shared base items and returns every
component to the UI.  It is intentionally separate from hand validation so
local variants can replace it without changing legal-play logic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .tiles import is_honor, tile_name


@dataclass(frozen=True)
class ScoreBreakdown:
    base: int
    water: int
    unit: int
    multiplier: int
    per_payer: int
    total: int
    items: tuple[dict[str, Any], ...]

    def payload(self) -> dict[str, Any]:
        return {
            "mode": "classic_water",
            "base": self.base,
            "water": self.water,
            "unit": self.unit,
            "multiplier": self.multiplier,
            "per_payer": self.per_payer,
            "total": self.total,
            "items": list(self.items),
        }


def classic_score(
    player,
    *,
    is_dealer: bool,
    wildcard_tiles: Iterable[int],
    gold_tile: int | None,
    win_type: str,
    rules,
    dealer_streak: int = 0,
    proxy_tile: int | None = None,
    proxy_as: int | None = None,
) -> ScoreBreakdown:
    """Calculate documented base/water settlement for one completed hand.

    The returned ``total`` is the winner's gain.  In the classical profile all
    three opponents pay the same amount for both discard wins and self draws;
    the win type determines that amount's multiplier.
    """

    items: list[dict[str, Any]] = []

    def add(label: str, amount: int, detail: str | None = None) -> None:
        if amount:
            entry: dict[str, Any] = {"label": label, "water": amount}
            if detail:
                entry["detail"] = detail
            items.append(entry)

    # Only actual gold scores water.  White is merely the fixed face-value
    # substitute and must not be counted as a second gold.
    wildcards = {gold_tile} if gold_tile is not None else set()
    gold_count = player.hand.count(gold_tile) if gold_tile is not None else 0
    if win_type == "opening_gold":
        gold_count += 1
    add("真金", gold_count, "每张仍在手中的真金 1 水")

    flower_count = len(player.flowers)
    add("花牌", flower_count, "每张花牌 1 水")
    flower_ids = set(player.flowers)
    # 梅兰菊竹 are ids 34-37 and 春夏秋冬 are 38-41 in this project.
    for label, family in (("四君子齐", set(range(34, 38))), ("四季齐", set(range(38, 42)))):
        if family.issubset(flower_ids):
            add(label, 4, "四张原有 4 水，再加 4 水，共 8 水")
    for meld in player.melds:
        kind = meld["kind"]
        representative = meld.get("value", meld["tiles"][0])
        honor = is_honor(representative)
        if kind == "pong" and honor:
            add(f"{tile_name(representative)}碰", 1)
        elif kind in {"ming_kan", "add_kan"}:
            add(f"{tile_name(representative)}明杠", 3 if honor else 2)
        elif kind == "an_kan":
            add(f"{tile_name(representative)}暗杠", 4 if honor else 3)

    # A closed natural triplet is a documented water item.  We count only
    # natural triplets here (not a wildcard-imputed decomposition), keeping
    # the number stable and auditable even when gold can form several shapes.
    concealed_counts = Counter(
        proxy_as if proxy_tile is not None and tile == proxy_tile else tile
        for tile in player.hand
        if tile not in wildcards
    )
    for tile, count in sorted(concealed_counts.items()):
        if count < 3:
            continue
        add(f"{tile_name(tile)}暗刻", 2 if is_honor(tile) else 1)

    water = sum(int(item["water"]) for item in items)
    base = (
        rules.dealer_base_score * (2 ** max(0, dealer_streak))
        if is_dealer
        else rules.base_score
    )
    unit = base + water
    multipliers = {
        "discard": 1,
        "self_draw": 2,
        "travelling_gold": rules.travelling_gold_multiplier,
        "double_travelling": rules.double_travelling_multiplier,
        "triple_travelling": rules.triple_travelling_multiplier,
        "three_gold_open": rules.opening_three_gold_multiplier,
        "three_gold": rules.later_three_gold_multiplier,
        "opening_wait": rules.opening_wait_multiplier,
        "heaven": rules.heavenly_win_multiplier,
        "opening_gold": rules.opening_wait_multiplier,
    }
    multiplier = multipliers.get(win_type, 1)
    per_payer = unit * multiplier
    return ScoreBreakdown(
        base=base,
        water=water,
        unit=unit,
        multiplier=multiplier,
        per_payer=per_payer,
        total=per_payer * 3,
        items=tuple(items),
    )

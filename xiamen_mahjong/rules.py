"""Versioned, intentionally conservative Xiamen Mahjong rule configuration.

The local v1 uses the common 144-tile, four-player core: flowers, eating,
pong/kong, a public flip-gold wildcard, and ordinary/ seven-pairs wins.
High-variance regional options are modelled as switches and remain off until a
local rules source is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class XiamenRules:
    name: str = "厦门麻将核心金牌版"
    version: str = "xiamen-core-gold-v1"
    player_count: int = 4
    base_score: int = 8
    allow_seven_pairs: bool = True
    gold_is_wildcard: bool = True
    gold_discard_cannot_be_claimed: bool = True
    allow_chi: bool = True
    allow_concealed_kong: bool = True
    allow_added_kong: bool = True
    enable_travelling_gold: bool = False
    enable_three_gold_instant_win: bool = False
    enable_forced_honor_follow: bool = False
    enable_complex_water_scoring: bool = False

    def public_summary(self) -> list[str]:
        return [
            "4 人、144 张牌（含 8 张花牌）",
            "可吃、碰、明杠、暗杠、补杠；花牌自动补牌",
            "翻金：金牌作为万能牌；打出的金牌不可被吃、碰、杠或胡",
            "支持标准和与七对；游金、三金倒、跟打和底/水制默认关闭",
            "简化结算：底分 8；自摸三家各付，点炮由放铳者支付三份",
        ]

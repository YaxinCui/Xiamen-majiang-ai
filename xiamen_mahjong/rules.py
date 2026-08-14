"""Versioned Xiamen Mahjong rule profiles.

Rules at physical Xiamen tables vary by neighbourhood and app.  The profiles
below deliberately name that choice instead of hiding it behind one vague
"Xiamen Mahjong" switch.  ``core`` keeps the original lightweight teaching
table; ``classic`` enables the common rules corroborated by several public
Xiamen rule references.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class XiamenRules:
    profile: str = "core"
    name: str = "厦门麻将核心金牌版"
    version: str = "xiamen-core-gold-v1"
    player_count: int = 4
    tile_count: int = 144
    include_honors: bool = True
    rules_page: str = "/guide.html"
    scoring_mode: str = "core"
    base_score: int = 8
    dealer_base_score: int = 16
    initial_hand_size: int = 13
    melds_required: int = 4
    dead_wall_tiles: int = 0
    allow_seven_pairs: bool = True
    gold_is_wildcard: bool = True
    gold_from_indicator_next: bool = True
    gold_discard_cannot_be_claimed: bool = True
    white_dragon_is_gold_proxy: bool = False
    white_dragon_can_form_melds: bool = False
    gold_in_hand_blocks_discard_win: bool = False
    gold_discard_self_draw_only: bool = False
    allow_chi: bool = True
    allow_concealed_kong: bool = True
    allow_added_kong: bool = True
    enable_travelling_gold: bool = False
    enable_three_gold_instant_win: bool = False
    three_gold_opening_only: bool = False
    enable_opening_gold_capture: bool = False
    enable_opening_wait: bool = False
    enable_heavenly_win: bool = False
    enable_dealer_continuation: bool = False
    all_players_pay_discard_win: bool = False
    travelling_gold_multiplier: int = 4
    double_travelling_multiplier: int = 8
    triple_travelling_multiplier: int = 12
    opening_three_gold_multiplier: int = 4
    later_three_gold_multiplier: int = 3
    opening_wait_multiplier: int = 4
    heavenly_win_multiplier: int = 4
    enable_forced_honor_follow: bool = False
    enable_complex_water_scoring: bool = False
    fixed_discard_score: int = 2
    fixed_self_draw_score: int = 4
    fixed_touring_score: int = 10
    fixed_double_touring_score: int = 20
    fixed_triple_touring_score: int = 80
    fixed_three_gold_score: int = 10

    @classmethod
    def classic(cls) -> "XiamenRules":
        """Common local rules suitable for the browser's default table.

        This profile follows the documented 16-tile / 17-tile classical game.
        Where published room rules conflict, the profile uses the conservative
        defaults documented in ``RULES_RESEARCH.md`` (notably triple tour x12).
        """

        return cls(
            profile="classic",
            name="厦门麻将经典金牌版",
            version="xiamen-classic-full-v2",
            initial_hand_size=16,
            melds_required=5,
            dead_wall_tiles=16,
            allow_seven_pairs=False,
            gold_from_indicator_next=False,
            white_dragon_is_gold_proxy=True,
            white_dragon_can_form_melds=True,
            gold_in_hand_blocks_discard_win=True,
            gold_discard_self_draw_only=True,
            enable_travelling_gold=True,
            enable_three_gold_instant_win=True,
            enable_opening_gold_capture=True,
            enable_opening_wait=True,
            enable_heavenly_win=True,
            enable_dealer_continuation=True,
            all_players_pay_discard_win=True,
            enable_forced_honor_follow=True,
            enable_complex_water_scoring=True,
        )

    @classmethod
    def new120(cls) -> "XiamenRules":
        """Current no-honors, flower-bearing 120-tile Xiamen profile.

        Optional room rules such as robbing an added kong, tail-eight safe
        discards and instant follow bonuses remain disabled.  The fixed score
        defaults match the multi-source rule table documented in
        ``RULES_RESEARCH.md``.
        """

        return cls(
            profile="new120",
            name="120 张新厦麻",
            version="xiamen-new-120-v1",
            tile_count=120,
            include_honors=False,
            rules_page="/rules-120.html",
            scoring_mode="new120_fixed",
            base_score=2,
            dealer_base_score=2,
            initial_hand_size=16,
            melds_required=5,
            dead_wall_tiles=0,
            allow_seven_pairs=False,
            gold_from_indicator_next=False,
            white_dragon_is_gold_proxy=True,
            white_dragon_can_form_melds=True,
            gold_in_hand_blocks_discard_win=True,
            gold_discard_self_draw_only=True,
            enable_travelling_gold=True,
            enable_three_gold_instant_win=True,
            three_gold_opening_only=True,
            enable_opening_gold_capture=True,
            enable_dealer_continuation=True,
            all_players_pay_discard_win=True,
            travelling_gold_multiplier=10,
            double_travelling_multiplier=20,
            triple_travelling_multiplier=80,
            opening_three_gold_multiplier=10,
        )

    @classmethod
    def from_profile(cls, profile: str | None) -> "XiamenRules":
        if profile in {None, "core"}:
            return cls()
        if profile == "new120":
            return cls.new120()
        if profile == "classic":
            return cls.classic()
        raise ValueError(f"未知规则档位：{profile}")

    @classmethod
    def available_profiles(cls) -> list[dict[str, str]]:
        return [
            {
                "id": "new120",
                "name": "120 张新厦麻",
                "description": "当前常见无大字玩法；16/17 张、白板替金、固定分值与游金",
                "rules_page": "/rules-120.html",
            },
            {
                "id": "classic",
                "name": "144 张老厦麻",
                "description": "有大字；16/17 张、跟打、游金/三金倒与传统底水计分",
                "rules_page": "/guide.html",
            },
            {
                "id": "core",
                "name": "核心教学",
                "description": "轻量金牌规则；关闭地方差异项，便于学习和测试",
                "rules_page": "/guide.html",
            },
        ]

    def public_summary(self) -> list[str]:
        summary = [
            (
                "4 人、120 张牌：108 张万筒条＋4 张白板＋8 张花；无东南西北中发"
                if self.tile_count == 120
                else "4 人、144 张牌（含 8 张花牌）"
            ),
            "可吃、碰、明杠、暗杠、补杠；花牌自动补牌",
            "翻金：真金作为万能牌；打出的真金不可被吃、碰、杠或胡",
        ]
        if self.melds_required == 5:
            summary.append("经典 16 张手牌：胡牌为 17 张，组成五组牌加一对将")
        else:
            summary.append("教学 13 张手牌：支持四组一对与七对")
        if self.white_dragon_is_gold_proxy:
            summary.append("白板只按金牌原牌面使用，不是万能牌；可按该牌面参与组牌")
        if self.gold_from_indicator_next:
            summary.append("教学桌使用翻出牌的顺位作为金牌")
        else:
            summary.append("本桌翻出的牌即为金牌（不是顺位下一张）")
        if self.dead_wall_tiles:
            summary.append(f"牌墙剩 {self.dead_wall_tiles} 张时荒庄，不计分")
        if self.gold_in_hand_blocks_discard_win:
            summary.append("手持金牌不可点炮胡，只能自摸")
        if self.gold_discard_self_draw_only:
            summary.append("打出金牌后，本圈打金者只能自摸；其余玩家不可响应该金")
        if self.enable_travelling_gold:
            if self.scoring_mode == "new120_fixed":
                summary.append("游金 10、双游 20、三游 80；双金必须游，三金必须双游")
            else:
                summary.append("支持游金 ×4、双游 ×8、三游 ×12，以及抢金 / 天听 ×4")
        if self.enable_three_gold_instant_win:
            summary.append(
                "三金倒只检查起手，常见 10 分；牌中三金进入双游、三游"
                if self.three_gold_opening_only
                else "三金倒：开局 ×4，牌中摸牌前 ×3；天胡 ×4"
            )
        if self.enable_forced_honor_follow:
            summary.append("跟打：已出现的风牌或箭牌，手中仅一张时须优先打出")
        if self.enable_complex_water_scoring:
            summary.append("经典结算：三家共同付款；闲底 8、庄底 16，连庄翻倍并累计水数")
        elif self.scoring_mode == "new120_fixed":
            summary.append("固定分值：平胡 2、自摸 4；花/明杠 +1，暗杠 +2；默认三家共同结算")
        else:
            summary.append("简化结算：底分 8；自摸三家各付，点炮由放铳者支付三份")
        return summary

"""Server-authoritative single-hand Xiamen Mahjong game state."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

from .agents import GameAction, HeuristicTeacherAgent
from .hand import is_travelling_ready, is_winning_hand, wait_tiles, winning_pattern
from .rules import XiamenRules
from .scoring import classic_score
from .tiles import (
    BASE_TILE_COUNT,
    WHITE_DRAGON,
    base_wall,
    is_honor,
    is_base_tile,
    is_suited,
    next_gold_tile,
    tile_name,
    tile_payload,
)


class GameError(ValueError):
    pass


@dataclass
class Player:
    seat: int
    hand: list[int] = field(default_factory=list)
    discards: list[int] = field(default_factory=list)
    melds: list[dict[str, Any]] = field(default_factory=list)
    flowers: list[int] = field(default_factory=list)
    score: int = 0


class XiamenMahjongGame:
    human_seat = 0

    def __init__(
        self,
        *,
        seed: int | None = None,
        rules: XiamenRules | None = None,
        dealer: int | None = None,
        dealer_streak: int = 0,
        scores: list[int] | None = None,
        hand_number: int = 1,
        auto_advance: bool = True,
    ):
        self.rules = rules or XiamenRules()
        self.auto_advance = auto_advance
        self.random = random.Random(seed)
        self.seed = seed
        self.teacher = HeuristicTeacherAgent()
        self.players = [
            Player(seat=index, score=(scores[index] if scores else 0))
            for index in range(self.rules.player_count)
        ]
        self.wall: list[int] = []
        self.gold_indicator: int | None = None
        self.gold_tile: int | None = None
        self.gold_dice: tuple[int, int] | None = None
        self.dealer = (
            dealer if dealer is not None else self.random.randrange(self.rules.player_count)
        )
        self.dealer_streak = max(0, dealer_streak)
        self.hand_number = max(1, hand_number)
        self.current_player = 0
        self.phase = "setup"
        self.last_discard: int | None = None
        self.discarder: int | None = None
        self.latest_discard: int | None = None
        self.latest_discard_seat: int | None = None
        self.response_options: dict[int, list[GameAction]] = {}
        self.response_choices: dict[int, GameAction] = {}
        self.winner: int | None = None
        self.win_type: str | None = None
        self.win_pattern: str | None = None
        self.score_breakdown: dict[str, Any] | None = None
        self.gold_discard_lock_seat: int | None = None
        self.last_drawn_tiles: list[int | None] = [None] * self.rules.player_count
        self.first_turn_pending: set[int] = set(range(self.rules.player_count))
        self.opening_wait_seats: set[int] = set()
        self.pending_tour_seat: int | None = None
        self.tour_state: dict[str, Any] | None = None
        self.turn_count = 0
        self.events: list[dict[str, Any]] = []
        self.message = "准备开始"
        self._setup()

    def _setup(self) -> None:
        self.wall = base_wall()
        self.random.shuffle(self.wall)
        self.current_player = self.dealer
        for _ in range(self.rules.initial_hand_size):
            for player in self.players:
                self._draw_for_player(player)
        for player in self.players:
            player.hand.sort()
        self._select_gold_indicator()
        self._mark_opening_waits()
        dice_note = ""
        if self.gold_dice:
            dice_note = f"（骰子 {self.gold_dice[0]} + {self.gold_dice[1]}）"
        self._event(
            "开局",
            f"第 {self.hand_number} 局，{self._seat_name(self.dealer)}坐庄"
            f"（连庄 {self.dealer_streak}），补花后翻出{tile_name(self.gold_indicator)}，"
            f"金牌为{tile_name(self.gold_tile)}{dice_note}",
        )
        opening_gold_winner = self._opening_gold_winner()
        if opening_gold_winner is not None:
            self._finish_win(opening_gold_winner, "opening_gold")
            return
        self._start_turn(self.dealer)
        if self.auto_advance:
            self.advance_ais()

    def _select_gold_indicator(self) -> None:
        self.gold_dice = (self.random.randint(1, 6), self.random.randint(1, 6))
        start = len(self.wall) - sum(self.gold_dice)
        indices = list(range(max(start, 0), -1, -1)) + list(
            range(len(self.wall) - 1, max(start, 0), -1)
        )
        for index in indices:
            candidate = self.wall[index]
            if is_base_tile(candidate):
                self.gold_indicator = self.wall.pop(index)
                self.gold_tile = (
                    next_gold_tile(self.gold_indicator)
                    if self.rules.gold_from_indicator_next
                    else self.gold_indicator
                )
                return
        raise RuntimeError("wall has no base tile for the gold indicator")

    def _draw_for_player(self, player: Player) -> int | None:
        while self.wall:
            tile = self.wall.pop(0)
            if tile >= BASE_TILE_COUNT:
                player.flowers.append(tile)
                self._event("补花", f"{self._seat_name(player.seat)}补到花牌")
                continue
            player.hand.append(tile)
            player.hand.sort()
            return tile
        return None

    @property
    def wildcard_tiles(self) -> frozenset[int]:
        # Only the three remaining copies of the flipped tile are universal
        # wildcards.  White is a fixed face-value proxy and is normalized
        # separately by the hand solver.
        wildcards = (
            {self.gold_tile}
            if self.rules.gold_is_wildcard and self.gold_tile is not None
            else set()
        )
        return frozenset(wildcards)

    @property
    def white_dragon_is_proxy(self) -> bool:
        return self.gold_proxy_tile is not None

    @property
    def gold_proxy_tile(self) -> int | None:
        if (
            self.rules.white_dragon_is_gold_proxy
            and self.gold_tile is not None
            and self.gold_tile != WHITE_DRAGON
        ):
            return WHITE_DRAGON
        return None

    def _mark_opening_waits(self) -> None:
        if not self.rules.enable_opening_wait:
            return
        for player in self.players:
            waits = wait_tiles(
                player.hand,
                self.gold_tile,
                meld_count=0,
                melds_required=self.rules.melds_required,
                allow_seven_pairs=self.rules.allow_seven_pairs,
                wildcard_tiles=self.wildcard_tiles,
                proxy_tile=self.gold_proxy_tile,
                proxy_as=self.gold_tile,
            )
            if waits:
                self.opening_wait_seats.add(player.seat)
        if self.opening_wait_seats:
            names = "、".join(self._seat_name(seat) for seat in sorted(self.opening_wait_seats))
            self._event("天听", f"{names}开局听牌；保持原牌形可按天听结算")

    def _opening_gold_winner(self) -> int | None:
        if not self.rules.enable_opening_wait or self.gold_indicator is None:
            return None
        priority = [
            (self.dealer + offset) % self.rules.player_count
            for offset in range(1, self.rules.player_count)
        ] + [self.dealer]
        return next(
            (
                player_id
                for player_id in priority
                if self._can_win(player_id, self.gold_indicator)
            ),
            None,
        )

    def _start_turn(self, player_id: int) -> None:
        if len(self.wall) <= self.rules.dead_wall_tiles:
            self._finish_draw()
            return
        player = self.players[player_id]
        if (
            self.rules.enable_three_gold_instant_win
            and self.gold_tile is not None
            and player.hand.count(self.gold_tile) >= 3
        ):
            opening = player_id in self.first_turn_pending
            self._finish_win(player_id, "three_gold_open" if opening else "three_gold")
            return
        self.current_player = player_id
        tile = self._draw_for_player(player)
        if tile is None:
            self._finish_draw()
            return
        self.last_discard = None
        self.discarder = None
        self.phase = "discard"
        self.last_drawn_tiles[player_id] = tile
        self.turn_count += 1
        if self._tour_resolution_level(player_id):
            labels = {1: "游金", 2: "双游", 3: "三游"}
            self.message = f"{self._seat_name(player_id)}进入{labels[self.tour_state['level']]}决胜摸牌"
        else:
            self.message = f"{self._seat_name(player_id)}摸牌"

    def human_actions(self) -> list[dict[str, Any]]:
        if self.phase == "over":
            return []
        if self.phase == "discard" and self.current_player == self.human_seat:
            return self._turn_actions(self.human_seat)
        if self.phase == "response" and self.human_seat in self.response_options:
            return [self._action_payload(action) for action in self.response_options[self.human_seat]]
        return []

    def _turn_actions(self, player_id: int) -> list[dict[str, Any]]:
        player = self.players[player_id]
        actions: list[dict[str, Any]] = []
        tour_level = self._tour_resolution_level(player_id)
        if tour_level:
            labels = {1: "游金胡", 2: "双游胡", 3: "三游胡"}
            actions.append({"kind": "hu", "label": labels[tour_level]})
            if self._can_advance_tour(player_id):
                next_label = "双游" if tour_level == 1 else "三游"
                actions.append(
                    {
                        "kind": "advance_tour",
                        "tile": self.gold_tile,
                        "label": f"{next_label}（打出一张金）",
                    }
                )
            return actions
        if self._can_win(player_id):
            win_type = self._self_draw_win_type(player_id)
            labels = {"heaven": "天胡", "opening_wait": "天听自摸"}
            actions.append({"kind": "hu", "label": labels.get(win_type, "自摸胡")})
        forced_discards = self._forced_follow_tiles(player_id)
        discard_tiles = forced_discards or sorted(set(player.hand))
        for tile in discard_tiles:
            verb = "跟打" if forced_discards else "打出"
            actions.append({"kind": "discard", "tile": tile, "label": f"{verb} {tile_name(tile)}"})
        if forced_discards or self._in_locked_tour_cycle(player_id):
            return actions
        if self.rules.allow_concealed_kong:
            for tile in sorted(set(player.hand)):
                if player.hand.count(tile) == 4 and tile != self.gold_tile:
                    actions.append({"kind": "an_kan", "tile": tile, "label": f"暗杠 {tile_name(tile)}"})
        if self.rules.allow_added_kong:
            # A white-dragon gold substitute can complete a pong, but that
            # exposed set is not four copies of the same physical tile.  Do
            # not turn it into a misleading added kong later.
            pong_tiles = {
                meld["tiles"][0]
                for meld in player.melds
                if meld["kind"] == "pong" and len(set(meld["tiles"])) == 1
            }
            for tile in sorted(pong_tiles):
                if tile in player.hand and tile != self.gold_tile:
                    actions.append({"kind": "add_kan", "tile": tile, "label": f"补杠 {tile_name(tile)}"})
        return actions

    def apply_human_action(self, payload: dict[str, Any]) -> None:
        if self.phase == "over":
            raise GameError("本局已经结束，请开始新的一局")
        kind = str(payload.get("kind", ""))
        tile = payload.get("tile")
        if tile is not None and not isinstance(tile, int):
            raise GameError("牌参数无效")
        tiles = payload.get("tiles", [])
        if not isinstance(tiles, list) or not all(isinstance(item, int) for item in tiles):
            raise GameError("吃牌参数无效")
        action = GameAction(kind, tile, tuple(tiles))
        if self.phase == "discard" and self.current_player == self.human_seat:
            self._apply_turn_action(self.human_seat, action)
        elif self.phase == "response" and self.human_seat in self.response_options:
            valid = self.response_options[self.human_seat]
            if not self._is_valid_response_action(action, valid):
                raise GameError("该响应不是当前可执行的动作")
            self.response_choices[self.human_seat] = action
            self._resolve_responses()
        else:
            raise GameError("现在不是你的操作回合")
        self.advance_ais()

    def advance_ais(self) -> None:
        """Play automatic seats until a human decision is required."""

        safety = 0
        while self.phase != "over":
            safety += 1
            if safety > 500:
                raise RuntimeError("automatic game loop exceeded its safety limit")
            if self.phase == "discard":
                if self.current_player == self.human_seat:
                    self.message = "轮到你出牌"
                    return
                action = self.teacher.choose_turn_action(self, self.current_player)
                self._apply_turn_action(self.current_player, action)
                continue
            if self.phase == "response":
                if self.human_seat in self.response_options:
                    self.message = "你可以响应上一张弃牌"
                    return
                self._resolve_responses()
                continue
            raise RuntimeError(f"unknown game phase: {self.phase}")

    def _apply_turn_action(self, player_id: int, action: GameAction) -> None:
        if self.phase != "discard" or player_id != self.current_player:
            raise GameError("当前不能执行摸牌后的动作")
        player = self.players[player_id]
        if action.kind == "hu":
            if not self._tour_resolution_level(player_id) and not self._can_win(player_id):
                raise GameError("当前手牌不能自摸胡")
            self._finish_win(player_id, self._self_draw_win_type(player_id))
            return
        if action.kind == "advance_tour":
            if not self._can_advance_tour(player_id) or self.gold_tile is None:
                raise GameError("当前牌形不能升级游金")
            player.hand.remove(self.gold_tile)
            player.discards.append(self.gold_tile)
            self.last_discard = self.gold_tile
            self.discarder = player_id
            self.latest_discard = self.gold_tile
            self.latest_discard_seat = player_id
            self.last_drawn_tiles[player_id] = None
            assert self.tour_state is not None
            self.tour_state["level"] += 1
            label = "双游" if self.tour_state["level"] == 2 else "三游"
            self._event(label, f"{self._seat_name(player_id)}打出金牌，进入{label}封闭摸牌圈")
            self.first_turn_pending.discard(player_id)
            self._begin_locked_tour_cycle()
            return
        if action.kind == "discard":
            if action.tile not in player.hand:
                raise GameError("手中没有这张牌")
            forced_discards = self._forced_follow_tiles(player_id)
            if forced_discards and action.tile not in forced_discards:
                names = "、".join(tile_name(tile) for tile in forced_discards)
                raise GameError(f"当前需跟打：{names}")
            had_gold_lock = self.gold_discard_lock_seat == player_id
            player.hand.remove(action.tile)
            player.discards.append(action.tile)
            self.last_discard = action.tile
            self.discarder = player_id
            self.latest_discard = action.tile
            self.latest_discard_seat = player_id
            if (
                player_id in self.opening_wait_seats
                and action.tile != self.last_drawn_tiles[player_id]
            ):
                self.opening_wait_seats.discard(player_id)
                self._event("天听解除", f"{self._seat_name(player_id)}改变了开局牌形")
            self.first_turn_pending.discard(player_id)
            self.last_drawn_tiles[player_id] = None
            self._event("弃牌", f"{self._seat_name(player_id)}打出{tile_name(action.tile)}")
            if action.tile == self.gold_tile and self.rules.gold_discard_self_draw_only:
                self.gold_discard_lock_seat = player_id
            elif had_gold_lock:
                self.gold_discard_lock_seat = None
            self.pending_tour_seat = None
            if (
                self.rules.enable_travelling_gold
                and action.tile != self.gold_tile
                and is_travelling_ready(
                    player.hand,
                    self.gold_tile,
                    meld_count=len(player.melds),
                    melds_required=self.rules.melds_required,
                    proxy_tile=self.gold_proxy_tile,
                    proxy_as=self.gold_tile,
                )
            ):
                self.pending_tour_seat = player_id
            if self._in_locked_tour_cycle(player_id):
                self._continue_locked_tour_cycle()
                return
            self._open_responses()
            return
        if action.kind == "an_kan":
            if self._in_locked_tour_cycle(player_id):
                raise GameError("双游或三游封闭圈中不能杠牌")
            if not self.rules.allow_concealed_kong or action.tile is None:
                raise GameError("当前规则不允许暗杠")
            if action.tile == self.gold_tile or player.hand.count(action.tile) < 4:
                raise GameError("这张牌不能暗杠")
            for _ in range(4):
                player.hand.remove(action.tile)
            player.melds.append(
                {
                    "kind": "an_kan",
                    "tiles": [action.tile] * 4,
                    "value": self._tile_value(action.tile),
                }
            )
            self.opening_wait_seats.discard(player_id)
            self.first_turn_pending.discard(player_id)
            self._event("暗杠", f"{self._seat_name(player_id)}暗杠")
            self._replacement_draw(player_id)
            return
        if action.kind == "add_kan":
            if self._in_locked_tour_cycle(player_id):
                raise GameError("双游或三游封闭圈中不能杠牌")
            if not self.rules.allow_added_kong or action.tile is None:
                raise GameError("当前规则不允许补杠")
            if action.tile == self.gold_tile or action.tile not in player.hand:
                raise GameError("这张牌不能补杠")
            target = next(
                (
                    meld
                    for meld in player.melds
                    if meld["kind"] == "pong"
                    and len(set(meld["tiles"])) == 1
                    and meld["tiles"][0] == action.tile
                ),
                None,
            )
            if target is None:
                raise GameError("没有可补杠的碰牌")
            player.hand.remove(action.tile)
            target["kind"] = "add_kan"
            target["tiles"] = [action.tile] * 4
            target["value"] = self._tile_value(action.tile)
            self.opening_wait_seats.discard(player_id)
            self.first_turn_pending.discard(player_id)
            self._event("补杠", f"{self._seat_name(player_id)}补杠{tile_name(action.tile)}")
            self._replacement_draw(player_id)
            return
        raise GameError("未知操作")

    def _replacement_draw(self, player_id: int) -> None:
        if len(self.wall) <= self.rules.dead_wall_tiles:
            self._finish_draw()
            return
        tile = self._draw_for_player(self.players[player_id])
        if tile is None:
            self._finish_draw()
            return
        self.current_player = player_id
        self.phase = "discard"
        self.last_drawn_tiles[player_id] = tile

    def _tour_resolution_level(self, player_id: int) -> int:
        if (
            self.tour_state
            and self.tour_state["owner"] == player_id
            and self.current_player == player_id
            and self.phase == "discard"
        ):
            return int(self.tour_state["level"])
        return 0

    def _in_locked_tour_cycle(self, player_id: int) -> bool:
        return bool(
            self.tour_state
            and self.tour_state.get("locked")
            and self.tour_state["owner"] != player_id
        )

    def _can_advance_tour(self, player_id: int) -> bool:
        level = self._tour_resolution_level(player_id)
        if not level or level >= 3 or self.gold_tile is None:
            return False
        player = self.players[player_id]
        if player.hand.count(self.gold_tile) < 2:
            return False
        remainder = list(player.hand)
        remainder.remove(self.gold_tile)
        return is_travelling_ready(
            remainder,
            self.gold_tile,
            meld_count=len(player.melds),
            melds_required=self.rules.melds_required,
            proxy_tile=self.gold_proxy_tile,
            proxy_as=self.gold_tile,
        )

    def _activate_pending_tour(self) -> None:
        if self.pending_tour_seat is None:
            return
        owner = self.pending_tour_seat
        self.pending_tour_seat = None
        self.tour_state = {"owner": owner, "level": 1, "locked": False, "remaining": []}
        self._event("游金", f"{self._seat_name(owner)}进入游金，下一次摸牌可胡")

    def _begin_locked_tour_cycle(self) -> None:
        assert self.tour_state is not None
        owner = int(self.tour_state["owner"])
        self.tour_state["locked"] = True
        self.tour_state["remaining"] = [
            (owner + offset) % self.rules.player_count
            for offset in range(1, self.rules.player_count)
        ]
        self._start_next_locked_tour_turn()

    def _continue_locked_tour_cycle(self) -> None:
        self._start_next_locked_tour_turn()

    def _start_next_locked_tour_turn(self) -> None:
        assert self.tour_state is not None
        remaining = self.tour_state["remaining"]
        if remaining:
            next_player = remaining.pop(0)
            self._start_turn(next_player)
            return
        owner = int(self.tour_state["owner"])
        self.tour_state["locked"] = False
        self._start_turn(owner)

    def _open_responses(self) -> None:
        assert self.discarder is not None and self.last_discard is not None
        if self.rules.gold_discard_cannot_be_claimed and self.last_discard == self.gold_tile:
            self._event("金牌", "金牌弃置，其他玩家不可吃、碰、杠或胡")
            self._start_turn(self._next_player(self.discarder))
            return
        self.response_options = {}
        self.response_choices = {}
        for player_id in range(self.rules.player_count):
            if player_id == self.discarder:
                continue
            options = self._response_actions(player_id)
            if options:
                self.response_options[player_id] = options
        if not self.response_options:
            self._activate_pending_tour()
            self._start_turn(self._next_player(self.discarder))
            return
        self.phase = "response"

    def _response_actions(self, player_id: int) -> list[GameAction]:
        assert self.last_discard is not None and self.discarder is not None
        player = self.players[player_id]
        tile = self.last_discard
        effective_tile = self._tile_value(tile)
        actions: list[GameAction] = []
        if self._can_win(player_id, tile) and self._can_discard_win(player_id):
            actions.append(GameAction("hu"))
        if tile != self.gold_tile:
            for consumed in self._claim_consumptions(player.hand, effective_tile, 3):
                if len(self.wall) > self.rules.dead_wall_tiles:
                    actions.append(GameAction("ming_kan", tile, tuple(consumed)))
            for consumed in self._claim_consumptions(player.hand, effective_tile, 2):
                actions.append(GameAction("pong", tile, tuple(consumed)))
        if self.rules.allow_chi and player_id == self._next_player(self.discarder):
            for consumed in self._chi_requirements(player.hand, tile):
                actions.append(GameAction("chi", tile, tuple(consumed)))
        return [GameAction("pass"), *actions] if actions else []

    def _chi_requirements(self, hand: list[int], tile: int) -> list[list[int]]:
        effective_tile = self._tile_value(tile)
        if not is_suited(effective_tile) or tile == self.gold_tile:
            return []
        base = effective_tile // 9 * 9
        rank = effective_tile % 9
        candidates: list[list[int]] = []
        for offsets in ((-2, -1), (-1, 1), (1, 2)):
            positions = [rank + offset for offset in offsets]
            if not all(0 <= position < 9 for position in positions):
                continue
            required = [base + position for position in positions]
            candidates.extend(self._claim_consumptions(hand, required, 2))
        return _unique_tile_lists(candidates)

    def _claim_consumptions(
        self, hand: list[int], required: int | list[int], size: int
    ) -> list[list[int]]:
        """Return legal hand tiles to expose for a claim.

        In the classic profile, a white dragon is a public gold substitute and
        may fill a missing exposed tile.  Actual gold tiles never enter a
        claim: a discarded gold is handled before this method is called.
        """

        required_tiles = [required] * size if isinstance(required, int) else list(required)
        if len(required_tiles) != size:
            return []
        choices: list[list[int]] = []

        def visit(index: int, remaining: list[int], consumed: list[int]) -> None:
            if index == len(required_tiles):
                choices.append(sorted(consumed))
                return
            target = required_tiles[index]
            candidates = [] if target == self.gold_tile else [target]
            if (
                target == self.gold_tile
                and self.rules.white_dragon_can_form_melds
                and self.white_dragon_is_proxy
            ):
                candidates.append(WHITE_DRAGON)
            for candidate in sorted(set(candidates)):
                if candidate not in remaining:
                    continue
                next_remaining = list(remaining)
                next_remaining.remove(candidate)
                visit(index + 1, next_remaining, [*consumed, candidate])

        visit(0, list(hand), [])
        return _unique_tile_lists(choices)

    def _tile_value(self, tile: int) -> int:
        if self.gold_proxy_tile is not None and tile == self.gold_proxy_tile:
            assert self.gold_tile is not None
            return self.gold_tile
        return tile

    def _can_discard_win(self, player_id: int) -> bool:
        player = self.players[player_id]
        if self.rules.gold_in_hand_blocks_discard_win and self.gold_tile in player.hand:
            return False
        return not (
            self.rules.gold_discard_self_draw_only
            and self.gold_discard_lock_seat == player_id
        )

    def _forced_follow_tiles(self, player_id: int) -> list[int]:
        if not self.rules.enable_forced_honor_follow:
            return []
        appeared_honors = {
            tile
            for player in self.players
            for tile in player.discards
            if is_honor(tile) and tile not in {self.gold_tile, self.gold_proxy_tile}
        }
        hand = self.players[player_id].hand
        return sorted(
            tile
            for tile in set(hand)
            if is_honor(tile) and hand.count(tile) == 1 and tile in appeared_honors
        )

    def _resolve_responses(self) -> None:
        assert self.discarder is not None
        for player_id, options in self.response_options.items():
            if player_id not in self.response_choices:
                self.response_choices[player_id] = self.teacher.choose_response(self, player_id, options)
        choices = list(self.response_choices.items())
        hu_claims = [(player_id, action) for player_id, action in choices if action.kind == "hu"]
        if hu_claims:
            self.pending_tour_seat = None
            player_id, _ = self._nearest_claim(hu_claims)
            self._finish_win(player_id, "discard")
            return
        for kind in ("ming_kan", "pong", "chi"):
            claims = [(player_id, action) for player_id, action in choices if action.kind == kind]
            if claims:
                self.pending_tour_seat = None
                player_id, action = self._nearest_claim(claims)
                self._apply_claim(player_id, action)
                return
        self._activate_pending_tour()
        self._start_turn(self._next_player(self.discarder))

    def _nearest_claim(self, claims: list[tuple[int, GameAction]]) -> tuple[int, GameAction]:
        assert self.discarder is not None
        return min(
            claims,
            key=lambda item: ((item[0] - self.discarder) % self.rules.player_count, item[0]),
        )

    def _apply_claim(self, player_id: int, action: GameAction) -> None:
        assert self.last_discard is not None and self.discarder is not None
        player = self.players[player_id]
        tile = self.last_discard
        discard_pile = self.players[self.discarder].discards
        if discard_pile and discard_pile[-1] == tile:
            discard_pile.pop()
        self.latest_discard = None
        self.latest_discard_seat = None
        self.last_drawn_tiles[player_id] = None
        if self.tour_state and self.tour_state["owner"] == player_id:
            self._event("游金解除", f"{self._seat_name(player_id)}响应他家弃牌，游金解除")
            self.tour_state = None
        self.opening_wait_seats.discard(player_id)
        self.first_turn_pending.discard(player_id)
        if action.kind == "pong":
            if len(action.tiles) != 2 or not _hand_contains(player.hand, action.tiles):
                raise GameError("碰牌组合无效")
            for required in action.tiles:
                player.hand.remove(required)
            player.melds.append(
                {
                    "kind": "pong",
                    "tiles": [tile, *action.tiles],
                    "value": self._tile_value(tile),
                }
            )
            self._event("碰", f"{self._seat_name(player_id)}碰{tile_name(tile)}")
        elif action.kind == "ming_kan":
            if len(action.tiles) != 3 or not _hand_contains(player.hand, action.tiles):
                raise GameError("明杠组合无效")
            for required in action.tiles:
                player.hand.remove(required)
            player.melds.append(
                {
                    "kind": "ming_kan",
                    "tiles": [tile, *action.tiles],
                    "value": self._tile_value(tile),
                }
            )
            self._event("明杠", f"{self._seat_name(player_id)}明杠{tile_name(tile)}")
            self._replacement_draw(player_id)
            return
        elif action.kind == "chi":
            if len(action.tiles) != 2 or not _hand_contains(player.hand, action.tiles):
                raise GameError("吃牌组合无效")
            for required in action.tiles:
                player.hand.remove(required)
            player.melds.append(
                {
                    "kind": "chi",
                    "tiles": sorted([tile, *action.tiles]),
                    "value": self._tile_value(tile),
                }
            )
            self._event("吃", f"{self._seat_name(player_id)}吃{tile_name(tile)}")
        else:
            raise RuntimeError(f"unsupported claim: {action.kind}")
        self.current_player = player_id
        self.phase = "discard"
        self.last_discard = None
        self.discarder = None
        self.response_options = {}
        self.response_choices = {}
        self.pending_tour_seat = None

    def _can_win(self, player_id: int, claimed_tile: int | None = None) -> bool:
        player = self.players[player_id]
        tiles = [*player.hand, *([claimed_tile] if claimed_tile is not None else [])]
        return is_winning_hand(
            tiles,
            self.gold_tile if self.rules.gold_is_wildcard else None,
            meld_count=len(player.melds),
            melds_required=self.rules.melds_required,
            allow_seven_pairs=self.rules.allow_seven_pairs,
            wildcard_tiles=self.wildcard_tiles,
            proxy_tile=self.gold_proxy_tile,
            proxy_as=self.gold_tile,
        )

    def _self_draw_win_type(self, player_id: int) -> str:
        tour_level = self._tour_resolution_level(player_id)
        if tour_level == 1:
            return "travelling_gold"
        if tour_level == 2:
            return "double_travelling"
        if tour_level == 3:
            return "triple_travelling"
        if (
            self.rules.enable_heavenly_win
            and player_id == self.dealer
            and player_id in self.first_turn_pending
            and not any(player.discards for player in self.players)
        ):
            return "heaven"
        if self.rules.enable_opening_wait and player_id in self.opening_wait_seats:
            return "opening_wait"
        return "self_draw"

    def _finish_win(self, winner: int, win_type: str) -> None:
        winner_player = self.players[winner]
        tiles = list(winner_player.hand)
        if win_type == "discard" and self.last_discard is not None:
            tiles.append(self.last_discard)
        self.winner = winner
        self.win_type = win_type
        special_patterns = {
            "travelling_gold": "游金",
            "double_travelling": "双游",
            "triple_travelling": "三游",
            "three_gold_open": "三金倒（开局）",
            "three_gold": "三金倒",
            "opening_wait": "天听",
            "heaven": "天胡",
            "opening_gold": "抢金",
        }
        self.win_pattern = special_patterns.get(win_type) or winning_pattern(
            tiles,
            self.gold_tile,
            meld_count=len(winner_player.melds),
            melds_required=self.rules.melds_required,
            allow_seven_pairs=self.rules.allow_seven_pairs,
            wildcard_tiles=self.wildcard_tiles,
            proxy_tile=self.gold_proxy_tile,
            proxy_as=self.gold_tile,
        )
        if self.rules.enable_complex_water_scoring:
            breakdown = classic_score(
                winner_player,
                is_dealer=winner == self.dealer,
                wildcard_tiles=self.wildcard_tiles,
                gold_tile=self.gold_tile,
                win_type=win_type,
                rules=self.rules,
                dealer_streak=self.dealer_streak,
                proxy_tile=self.gold_proxy_tile,
                proxy_as=self.gold_tile,
            )
            amount = breakdown.per_payer
            self.score_breakdown = breakdown.payload()
            self.score_breakdown["payment_mode"] = (
                "all_pay" if self.rules.all_players_pay_discard_win else "discarder_pays"
            )
        else:
            multiplier = 1 + len(winner_player.flowers) + winner_player.hand.count(self.gold_tile)
            amount = self.rules.base_score * multiplier
            self.score_breakdown = {
                "mode": "core",
                "base": self.rules.base_score,
                "water": multiplier - 1,
                "unit": self.rules.base_score,
                "multiplier": multiplier,
                "per_payer": amount,
                "total": amount * 3,
                "items": [],
                "payment_mode": "discarder_pays" if win_type == "discard" else "all_pay",
            }
        if win_type != "discard" or self.rules.all_players_pay_discard_win:
            for player in self.players:
                if player.seat != winner:
                    player.score -= amount
                    winner_player.score += amount
        else:
            assert self.discarder is not None
            payment = amount * 3
            self.players[self.discarder].score -= payment
            winner_player.score += payment
        win_labels = {
            "discard": "点炮胡",
            "self_draw": "自摸",
            "travelling_gold": "游金",
            "double_travelling": "双游",
            "triple_travelling": "三游",
            "three_gold_open": "开局三金倒",
            "three_gold": "三金倒",
            "opening_wait": "天听自摸",
            "heaven": "天胡",
            "opening_gold": "抢金",
        }
        win_label = win_labels.get(win_type, win_type)
        self._event(
            "胡牌",
            f"{self._seat_name(winner)}{win_label}，{self.win_pattern}，"
            f"结算 {self.score_breakdown['unit']} × {self.score_breakdown['multiplier']}",
        )
        self.phase = "over"
        self.message = f"{self._seat_name(winner)}{win_label}：{self.win_pattern}"
        self.response_options = {}
        self.response_choices = {}
        self.pending_tour_seat = None
        self.tour_state = None

    def _finish_draw(self) -> None:
        self.phase = "over"
        self.winner = None
        self.win_type = "draw"
        self.win_pattern = None
        self.score_breakdown = None
        self.message = "牌墙耗尽，本局流局"
        self._event("流局", self.message)
        self.response_options = {}
        self.response_choices = {}

    def _is_valid_response_action(self, action: GameAction, options: list[GameAction]) -> bool:
        return any(
            option.kind == action.kind
            and option.tile == action.tile
            and option.tiles == action.tiles
            for option in options
        )

    def _action_payload(self, action: GameAction) -> dict[str, Any]:
        labels = {
            "pass": "过",
            "hu": "胡",
            "pong": "碰",
            "ming_kan": "明杠",
        }
        if action.kind == "chi":
            names = " ".join(tile_name(tile) for tile in action.tiles)
            label = f"吃（用 {names}）"
        elif action.kind in {"pong", "ming_kan"} and action.tiles:
            names = " ".join(tile_name(tile) for tile in action.tiles)
            label = f"{labels[action.kind]}（用 {names}）"
        else:
            label = labels.get(action.kind, action.kind)
        payload: dict[str, Any] = {"kind": action.kind, "label": label}
        if action.tile is not None:
            payload["tile"] = action.tile
        if action.tiles:
            payload["tiles"] = list(action.tiles)
        return payload

    def public_state(self, *, reveal_ai_hands: bool = False) -> dict[str, Any]:
        human = self.players[self.human_seat]
        response_tile = tile_payload(self.last_discard) if self.last_discard is not None else None
        return {
            "rules": {
                "profile": self.rules.profile,
                "name": self.rules.name,
                "version": self.rules.version,
                "summary": self.rules.public_summary(),
            },
            "seed": self.seed,
            "phase": self.phase,
            "message": self.message,
            "turn_count": self.turn_count,
            "hand_number": self.hand_number,
            "wall_remaining": len(self.wall),
            "dealer": self.dealer,
            "dealer_streak": self.dealer_streak,
            "current_player": self.current_player,
            "gold_indicator": tile_payload(self.gold_indicator) if self.gold_indicator is not None else None,
            "gold_indicator_label": "指示牌" if self.rules.gold_from_indicator_next else "翻出牌",
            "gold_tile": tile_payload(self.gold_tile) if self.gold_tile is not None else None,
            "gold_dice": list(self.gold_dice) if self.gold_dice else None,
            "last_discard": response_tile,
            "latest_discard": tile_payload(self.latest_discard)
            if self.latest_discard is not None
            else None,
            "latest_discard_seat": self.latest_discard_seat,
            "forced_discards": [
                tile_payload(tile)
                for tile in self._forced_follow_tiles(self.human_seat)
            ]
            if self.phase == "discard" and self.current_player == self.human_seat
            else [],
            "winner": self.winner,
            "win_type": self.win_type,
            "win_pattern": self.win_pattern,
            "score_breakdown": self.score_breakdown,
            "opening_wait_seats": sorted(self.opening_wait_seats),
            "tour_state": dict(self.tour_state) if self.tour_state else None,
            "players": [
                {
                    "seat": player.seat,
                    "name": self._seat_name(player.seat),
                    "score": player.score,
                    "discards": [tile_payload(tile) for tile in player.discards],
                    "melds": [
                        {
                            "kind": meld["kind"],
                            "tiles": [tile_payload(tile) for tile in meld["tiles"]],
                        }
                        for meld in player.melds
                    ],
                    "flowers": [tile_payload(tile) for tile in player.flowers],
                    "hand_count": len(player.hand),
                    "status": self._player_status(player.seat),
                    "drawn_tile": tile_payload(self.last_drawn_tiles[player.seat])
                    if (
                        self.phase == "discard"
                        and self.current_player == player.seat
                        and self.last_drawn_tiles[player.seat] in player.hand
                        and (player.seat == self.human_seat or reveal_ai_hands)
                    )
                    else None,
                    "hand": [tile_payload(tile) for tile in player.hand]
                    if player.seat == self.human_seat or reveal_ai_hands
                    else None,
                }
                for player in self.players
            ],
            "actions": self.human_actions(),
            "events": list(reversed(self.events[-12:])),
        }

    def _event(self, kind: str, text: str) -> None:
        self.events.append({"turn": self.turn_count, "kind": kind, "text": text})

    def _next_player(self, player_id: int) -> int:
        return (player_id + 1) % self.rules.player_count

    def _player_status(self, player_id: int) -> list[str]:
        status: list[str] = []
        if player_id in self.opening_wait_seats:
            status.append("天听")
        if self.tour_state and self.tour_state["owner"] == player_id:
            status.append({1: "游金", 2: "双游", 3: "三游"}[self.tour_state["level"]])
        if self.gold_discard_lock_seat == player_id:
            status.append("仅自摸")
        return status

    @staticmethod
    def _seat_name(player_id: int) -> str:
        return ("你", "右家 AI", "对家 AI", "左家 AI")[player_id]


def _hand_contains(hand: list[int], required: tuple[int, ...]) -> bool:
    remaining = list(hand)
    for tile in required:
        if tile not in remaining:
            return False
        remaining.remove(tile)
    return True


def _unique_tile_lists(values: list[list[int]]) -> list[list[int]]:
    return [list(value) for value in sorted({tuple(value) for value in values})]

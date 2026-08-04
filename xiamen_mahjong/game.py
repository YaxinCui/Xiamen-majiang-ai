"""Server-authoritative single-hand Xiamen Mahjong game state."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

from .agents import GameAction, HeuristicTeacherAgent
from .hand import is_winning_hand, winning_pattern
from .rules import XiamenRules
from .tiles import (
    BASE_TILE_COUNT,
    base_wall,
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

    def __init__(self, *, seed: int | None = None, rules: XiamenRules | None = None):
        self.rules = rules or XiamenRules()
        self.random = random.Random(seed)
        self.seed = seed
        self.teacher = HeuristicTeacherAgent()
        self.players = [Player(seat=index) for index in range(self.rules.player_count)]
        self.wall: list[int] = []
        self.gold_indicator: int | None = None
        self.gold_tile: int | None = None
        self.dealer = 0
        self.current_player = 0
        self.phase = "setup"
        self.last_discard: int | None = None
        self.discarder: int | None = None
        self.response_options: dict[int, list[GameAction]] = {}
        self.response_choices: dict[int, GameAction] = {}
        self.winner: int | None = None
        self.win_type: str | None = None
        self.win_pattern: str | None = None
        self.turn_count = 0
        self.events: list[dict[str, Any]] = []
        self.message = "准备开始"
        self._setup()

    def _setup(self) -> None:
        self.wall = base_wall()
        self.random.shuffle(self.wall)
        self.dealer = self.random.randrange(self.rules.player_count)
        self.current_player = self.dealer
        self._select_gold_indicator()
        for _ in range(13):
            for player in self.players:
                self._draw_for_player(player)
        for player in self.players:
            player.hand.sort()
        self._event("开局", f"{self._seat_name(self.dealer)}坐庄，翻出{tile_name(self.gold_indicator)}，金牌为{tile_name(self.gold_tile)}")
        self._start_turn(self.dealer)
        self.advance_ais()

    def _select_gold_indicator(self) -> None:
        for index in range(len(self.wall) - 1, -1, -1):
            candidate = self.wall[index]
            if is_base_tile(candidate):
                self.gold_indicator = self.wall.pop(index)
                self.gold_tile = next_gold_tile(self.gold_indicator)
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

    def _start_turn(self, player_id: int) -> None:
        if not self.wall:
            self._finish_draw()
            return
        self.current_player = player_id
        tile = self._draw_for_player(self.players[player_id])
        if tile is None:
            self._finish_draw()
            return
        self.last_discard = None
        self.discarder = None
        self.phase = "discard"
        self.turn_count += 1
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
        if self._can_win(player_id):
            actions.append({"kind": "hu", "label": "自摸胡"})
        for tile in sorted(set(player.hand)):
            actions.append({"kind": "discard", "tile": tile, "label": f"打出 {tile_name(tile)}"})
        if self.rules.allow_concealed_kong:
            for tile in sorted(set(player.hand)):
                if player.hand.count(tile) == 4 and tile != self.gold_tile:
                    actions.append({"kind": "an_kan", "tile": tile, "label": f"暗杠 {tile_name(tile)}"})
        if self.rules.allow_added_kong:
            pong_tiles = {meld["tiles"][0] for meld in player.melds if meld["kind"] == "pong"}
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
            if not self._can_win(player_id):
                raise GameError("当前手牌不能自摸胡")
            self._finish_win(player_id, "self_draw")
            return
        if action.kind == "discard":
            if action.tile not in player.hand:
                raise GameError("手中没有这张牌")
            player.hand.remove(action.tile)
            player.discards.append(action.tile)
            self.last_discard = action.tile
            self.discarder = player_id
            self._event("弃牌", f"{self._seat_name(player_id)}打出{tile_name(action.tile)}")
            self._open_responses()
            return
        if action.kind == "an_kan":
            if not self.rules.allow_concealed_kong or action.tile is None:
                raise GameError("当前规则不允许暗杠")
            if action.tile == self.gold_tile or player.hand.count(action.tile) < 4:
                raise GameError("这张牌不能暗杠")
            for _ in range(4):
                player.hand.remove(action.tile)
            player.melds.append({"kind": "an_kan", "tiles": [action.tile] * 4})
            self._event("暗杠", f"{self._seat_name(player_id)}暗杠")
            self._replacement_draw(player_id)
            return
        if action.kind == "add_kan":
            if not self.rules.allow_added_kong or action.tile is None:
                raise GameError("当前规则不允许补杠")
            if action.tile == self.gold_tile or action.tile not in player.hand:
                raise GameError("这张牌不能补杠")
            target = next(
                (meld for meld in player.melds if meld["kind"] == "pong" and meld["tiles"][0] == action.tile),
                None,
            )
            if target is None:
                raise GameError("没有可补杠的碰牌")
            player.hand.remove(action.tile)
            target["kind"] = "add_kan"
            target["tiles"] = [action.tile] * 4
            self._event("补杠", f"{self._seat_name(player_id)}补杠{tile_name(action.tile)}")
            self._replacement_draw(player_id)
            return
        raise GameError("未知操作")

    def _replacement_draw(self, player_id: int) -> None:
        tile = self._draw_for_player(self.players[player_id])
        if tile is None:
            self._finish_draw()
            return
        self.current_player = player_id
        self.phase = "discard"

    def _open_responses(self) -> None:
        assert self.discarder is not None and self.last_discard is not None
        if self.rules.gold_discard_cannot_be_claimed and self.last_discard == self.gold_tile:
            self._event("金牌", "金牌弃置，其他玩家不可响应")
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
            self._start_turn(self._next_player(self.discarder))
            return
        self.phase = "response"

    def _response_actions(self, player_id: int) -> list[GameAction]:
        assert self.last_discard is not None and self.discarder is not None
        player = self.players[player_id]
        tile = self.last_discard
        actions = [GameAction("pass")]
        if self._can_win(player_id, tile):
            actions.append(GameAction("hu"))
        if tile != self.gold_tile and player.hand.count(tile) >= 3 and len(self.wall) > 0:
            actions.append(GameAction("ming_kan", tile))
        if tile != self.gold_tile and player.hand.count(tile) >= 2:
            actions.append(GameAction("pong", tile))
        if self.rules.allow_chi and player_id == self._next_player(self.discarder):
            for required in self._chi_requirements(player.hand, tile):
                actions.append(GameAction("chi", tile, tuple(required)))
        return actions

    def _chi_requirements(self, hand: list[int], tile: int) -> list[list[int]]:
        if not is_suited(tile) or tile == self.gold_tile:
            return []
        base = tile // 9 * 9
        rank = tile % 9
        candidates: list[list[int]] = []
        for offsets in ((-2, -1), (-1, 1), (1, 2)):
            positions = [rank + offset for offset in offsets]
            if not all(0 <= position < 9 for position in positions):
                continue
            required = [base + position for position in positions]
            if self.gold_tile in required:
                continue
            if all(hand.count(required_tile) >= required.count(required_tile) for required_tile in set(required)):
                candidates.append(required)
        return candidates

    def _resolve_responses(self) -> None:
        assert self.discarder is not None
        for player_id, options in self.response_options.items():
            if player_id not in self.response_choices:
                self.response_choices[player_id] = self.teacher.choose_response(self, player_id, options)
        choices = list(self.response_choices.items())
        hu_claims = [(player_id, action) for player_id, action in choices if action.kind == "hu"]
        if hu_claims:
            player_id, _ = self._nearest_claim(hu_claims)
            self._finish_win(player_id, "discard")
            return
        for kind in ("ming_kan", "pong", "chi"):
            claims = [(player_id, action) for player_id, action in choices if action.kind == kind]
            if claims:
                player_id, action = self._nearest_claim(claims)
                self._apply_claim(player_id, action)
                return
        self._start_turn(self._next_player(self.discarder))

    def _nearest_claim(self, claims: list[tuple[int, GameAction]]) -> tuple[int, GameAction]:
        assert self.discarder is not None
        return min(
            claims,
            key=lambda item: ((item[0] - self.discarder) % self.rules.player_count, item[0]),
        )

    def _apply_claim(self, player_id: int, action: GameAction) -> None:
        assert self.last_discard is not None
        player = self.players[player_id]
        tile = self.last_discard
        if action.kind == "pong":
            for _ in range(2):
                player.hand.remove(tile)
            player.melds.append({"kind": "pong", "tiles": [tile] * 3})
            self._event("碰", f"{self._seat_name(player_id)}碰{tile_name(tile)}")
        elif action.kind == "ming_kan":
            for _ in range(3):
                player.hand.remove(tile)
            player.melds.append({"kind": "ming_kan", "tiles": [tile] * 4})
            self._event("明杠", f"{self._seat_name(player_id)}明杠{tile_name(tile)}")
            self._replacement_draw(player_id)
            return
        elif action.kind == "chi":
            if len(action.tiles) != 2 or any(required not in player.hand for required in action.tiles):
                raise GameError("吃牌组合无效")
            for required in action.tiles:
                player.hand.remove(required)
            player.melds.append({"kind": "chi", "tiles": sorted([tile, *action.tiles])})
            self._event("吃", f"{self._seat_name(player_id)}吃{tile_name(tile)}")
        else:
            raise RuntimeError(f"unsupported claim: {action.kind}")
        self.current_player = player_id
        self.phase = "discard"
        self.last_discard = None
        self.discarder = None
        self.response_options = {}
        self.response_choices = {}

    def _can_win(self, player_id: int, claimed_tile: int | None = None) -> bool:
        player = self.players[player_id]
        tiles = [*player.hand, *([claimed_tile] if claimed_tile is not None else [])]
        return is_winning_hand(
            tiles,
            self.gold_tile if self.rules.gold_is_wildcard else None,
            meld_count=len(player.melds),
            allow_seven_pairs=self.rules.allow_seven_pairs,
        )

    def _finish_win(self, winner: int, win_type: str) -> None:
        winner_player = self.players[winner]
        tiles = list(winner_player.hand)
        if win_type == "discard" and self.last_discard is not None:
            tiles.append(self.last_discard)
        self.winner = winner
        self.win_type = win_type
        self.win_pattern = winning_pattern(tiles, self.gold_tile, meld_count=len(winner_player.melds))
        multiplier = 1 + len(winner_player.flowers) + winner_player.hand.count(self.gold_tile)
        amount = self.rules.base_score * multiplier
        if win_type == "self_draw":
            for player in self.players:
                if player.seat != winner:
                    player.score -= amount
                    winner_player.score += amount
        else:
            assert self.discarder is not None
            payment = amount * 3
            self.players[self.discarder].score -= payment
            winner_player.score += payment
        win_label = "自摸" if win_type == "self_draw" else "点炮胡"
        self._event("胡牌", f"{self._seat_name(winner)}{win_label}，{self.win_pattern}，结算 {amount} × {multiplier}")
        self.phase = "over"
        self.message = f"{self._seat_name(winner)}{win_label}：{self.win_pattern}"
        self.response_options = {}
        self.response_choices = {}

    def _finish_draw(self) -> None:
        self.phase = "over"
        self.winner = None
        self.win_type = "draw"
        self.win_pattern = None
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
        else:
            label = labels.get(action.kind, action.kind)
        payload: dict[str, Any] = {"kind": action.kind, "label": label}
        if action.tile is not None:
            payload["tile"] = action.tile
        if action.tiles:
            payload["tiles"] = list(action.tiles)
        return payload

    def public_state(self) -> dict[str, Any]:
        human = self.players[self.human_seat]
        response_tile = tile_payload(self.last_discard) if self.last_discard is not None else None
        return {
            "rules": {
                "name": self.rules.name,
                "version": self.rules.version,
                "summary": self.rules.public_summary(),
            },
            "seed": self.seed,
            "phase": self.phase,
            "message": self.message,
            "turn_count": self.turn_count,
            "wall_remaining": len(self.wall),
            "dealer": self.dealer,
            "current_player": self.current_player,
            "gold_indicator": tile_payload(self.gold_indicator) if self.gold_indicator is not None else None,
            "gold_tile": tile_payload(self.gold_tile) if self.gold_tile is not None else None,
            "last_discard": response_tile,
            "winner": self.winner,
            "win_type": self.win_type,
            "win_pattern": self.win_pattern,
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
                    "hand": [tile_payload(tile) for tile in player.hand]
                    if player.seat == self.human_seat
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

    @staticmethod
    def _seat_name(player_id: int) -> str:
        return ("你", "右家 AI", "对家 AI", "左家 AI")[player_id]

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from mjai_bot.bot import AkagiBot
from mjai_bot.logger import logger

from .engine import MahjongHelperEngine, DiscardAnalysis
from .defense import DefenseContext, tile_danger

MASK_UNICODE_4P = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
    "5mr", "5pr", "5sr",
    "reach", "chi_low", "chi_mid", "chi_high", "pon", "kan_select", "hora", "ryukyoku", "none",
]

MASK_UNICODE_3P = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
    "5mr", "5pr", "5sr",
    "reach", "pon", "kan_select", "nukidora", "hora", "ryukyoku", "none",
]

HONOR_DORA_NEXT = {
    "E": "S",
    "S": "W",
    "W": "N",
    "N": "E",
    "P": "F",
    "F": "C",
    "C": "P",
}


@dataclass
class CallOption:
    call_type: str
    consumed: list[str]
    analysis: DiscardAnalysis
    shanten: Optional[int]
    rank: float
    score: float


class Bot(AkagiBot):
    NAME = "mahjong-helper"
    DESCRIPTION = "External CLI engine (EndlessCheng/mahjong-helper) powered recommendations"

    def __init__(self) -> None:
        super().__init__()
        self._engine = MahjongHelperEngine(timeout_ms=_timeout_ms())
        self._meld_events: list[dict] = []
        self._dora_indicators: list[str] = []
        self._discards_by_player: dict[int, list[str]] = {}
        self._riichi_players: set[int] = set()
        self._open_melds_by_player: dict[int, int] = {}
        self._last_self_draw: Optional[str] = None
        self._awaiting_discard = False

    def react(self, events: str) -> str:
        try:
            event_list = json.loads(events)
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to parse events: {events}, {exc}")
            return self._action({"type": "none"})

        self._track_state(event_list)
        return super().react(input_list=event_list)

    def think(self) -> str:
        try:
            if self.can_agari:
                return self._action({"type": "hora"})
            if self._should_nukidora():
                return self._action({
                    "type": "nukidora",
                    "actor": self.player_id,
                    "pai": "N",
                })
            if self._should_riichi():
                return self.action_riichi()
            if self.can_ryukyoku:
                return self._action({"type": "ryukyoku"})
            if self.can_pass and (self.can_chi or self.can_pon or self.can_daiminkan):
                return self._think_call()
            if self.can_discard:
                return self._think_discard()
            if self.can_chi or self.can_pon or self.can_daiminkan:
                return self._think_call()
        except Exception as exc:
            logger.error(f"mahjong-helper bot error: {exc}")
            if self.can_discard:
                return self._fallback_discard_action()

        return self._action({"type": "none"})

    def _think_discard(self) -> str:
        hand_tiles = self._current_hand_tiles()
        melds = self._build_helper_melds()
        dora_tiles = _dora_tiles_from_indicators(self._dora_indicators)
        analysis = self._engine.analyze_discard(
            hand_tiles,
            melds=melds,
            dora_tiles=dora_tiles,
        )
        if analysis.raw_stdout:
            logger.debug(f"mahjong-helper stdout:\n{analysis.raw_stdout}")

        discard_tile = self._select_discard_tile(analysis, hand_tiles)
        if discard_tile is None:
            return self._action({"type": "none"})

        tsumogiri = False
        if self._awaiting_discard and self._last_self_draw not in (None, "?"):
            tsumogiri = discard_tile == self._last_self_draw
        self._awaiting_discard = False

        action = {
            "type": "dahai",
            "actor": self.player_id,
            "pai": discard_tile,
            "tsumogiri": tsumogiri,
        }

        meta = self._build_meta(analysis, hand_tiles)
        if meta:
            action["meta"] = meta

        return self._action(action)

    def _think_call(self) -> str:
        called_tile = getattr(self, "last_kawa_tile", None)
        if not called_tile or called_tile == "?":
            return self._action({"type": "none"})

        hand_tiles = self._current_hand_tiles(include_tsumo=False)
        melds = self._build_helper_melds()
        dora_tiles = _dora_tiles_from_indicators(self._dora_indicators)
        analysis = self._engine.analyze_call(
            hand_tiles=hand_tiles,
            called_tile=called_tile,
            melds=melds,
            dora_tiles=dora_tiles,
        )
        if analysis.raw_stdout:
            logger.debug(f"mahjong-helper call stdout:\n{analysis.raw_stdout}")

        options = self._build_call_options(
            called_tile=called_tile,
            hand_tiles=hand_tiles,
            melds=melds,
            dora_tiles=dora_tiles,
        )
        if not options:
            return self._action({"type": "none"})

        base_shanten = self.shanten
        options.sort(
            key=lambda option: self._call_option_key(
                option,
                base_shanten=base_shanten,
                preferred_call_type=analysis.call_type,
            )
        )
        threat_levels = self._threat_levels()
        has_riichi_threat = any(level >= 2.0 for level in threat_levels.values())
        call_type_has_yaku = _call_type_has_yaku(analysis.candidates)

        best_option = options[0]
        improvement = 0
        if best_option.shanten is not None:
            improvement = base_shanten - best_option.shanten
        no_yaku = call_type_has_yaku.get(best_option.call_type) is False
        yakuhai_call = best_option.call_type in ("pon", "daiminkan") and self.is_yakuhai(called_tile)

        should_call = False
        if improvement > 0:
            should_call = True
        elif improvement == 0:
            if yakuhai_call:
                should_call = True
            elif analysis.should_call and not no_yaku:
                should_call = True

        if has_riichi_threat and improvement <= 0:
            should_call = False
        if best_option.call_type == "daiminkan" and improvement < 1:
            should_call = False
        if not should_call:
            return self._action({"type": "none"})

        if best_option.call_type == "chi" and self.can_chi:
            return self._action({
                "type": "chi",
                "actor": self.player_id,
                "target": self.target_actor,
                "pai": called_tile,
                "consumed": best_option.consumed,
            })
        if best_option.call_type == "pon" and self.can_pon:
            return self._action({
                "type": "pon",
                "actor": self.player_id,
                "target": self.target_actor,
                "pai": called_tile,
                "consumed": best_option.consumed,
            })
        if best_option.call_type == "daiminkan" and self.can_daiminkan:
            return self._action({
                "type": "daiminkan",
                "actor": self.player_id,
                "target": self.target_actor,
                "pai": called_tile,
                "consumed": best_option.consumed,
            })

        return self._action({"type": "none"})

    def _fallback_discard_action(self) -> str:
        hand_tiles = self._current_hand_tiles()
        tile = self._last_self_draw if self._last_self_draw not in (None, "?") else None
        if tile is None and hand_tiles:
            tile = hand_tiles[0]
        if tile is None:
            return self._action({"type": "none"})
        tsumogiri = False
        if self._awaiting_discard and self._last_self_draw not in (None, "?"):
            tsumogiri = tile == self._last_self_draw
        self._awaiting_discard = False
        return self._action({
            "type": "dahai",
            "actor": self.player_id,
            "pai": tile,
            "tsumogiri": tsumogiri,
        })

    def _current_hand_tiles(self, include_tsumo: bool = True) -> list[str]:
        tiles = [t for t in self.tehai_mjai if t != "?"]
        if (
            not include_tsumo
            and self._awaiting_discard
            and self._last_self_draw not in (None, "?")
            and self._last_self_draw in tiles
        ):
            tiles.remove(self._last_self_draw)
        return tiles

    def _select_discard_tile(
        self,
        analysis: DiscardAnalysis,
        hand_tiles: list[str],
    ) -> Optional[str]:
        discardable = self._discardable_tiles()
        if discardable:
            hand_tiles = [tile for tile in hand_tiles if tile in discardable]

        candidates: list[str] = []
        for cand in analysis.candidates:
            tile = _select_tile_from_hand(cand.tile, hand_tiles)
            if not tile:
                continue
            if discardable and tile not in discardable:
                continue
            if tile not in candidates:
                candidates.append(tile)

        if not candidates:
            candidates = hand_tiles

        if not candidates:
            if self._last_self_draw not in (None, "?"):
                return self._last_self_draw
            return None

        threat_levels = self._threat_levels()
        if not threat_levels:
            return candidates[0]

        dora_tiles = set(_dora_tiles_from_indicators(self._dora_indicators))
        ctx = DefenseContext(
            threat_levels=threat_levels,
            discards_by_player=self._discards_by_player,
            tiles_seen=self.tiles_seen,
            dora_tiles=dora_tiles,
        )

        danger_scores = {tile: tile_danger(tile, ctx) for tile in candidates}
        safe_candidates = [tile for tile in candidates if danger_scores[tile] == 0.0]
        if safe_candidates:
            return safe_candidates[0]

        top_n = min(5, len(candidates))
        best_tile = candidates[0]
        best_score = danger_scores[best_tile]
        for tile in candidates[:top_n]:
            score = danger_scores[tile]
            if score < best_score:
                best_score = score
                best_tile = tile

        logger.debug(
            "Defense pick: %s (danger=%.2f) from %s",
            best_tile,
            best_score,
            ", ".join(candidates[:top_n]),
        )
        return best_tile

    def _build_meta(
        self,
        analysis: DiscardAnalysis,
        hand_tiles: list[str],
    ) -> Optional[dict]:
        discardable = set(self._discardable_tiles())
        defense_ctx = None
        defense_weight = 0.0
        threat_levels = self._threat_levels()
        if threat_levels:
            max_threat = max(threat_levels.values())
            defense_weight = 20.0 * max_threat + 5.0 * (len(threat_levels) - 1)
            defense_ctx = DefenseContext(
                threat_levels=threat_levels,
                discards_by_player=self._discards_by_player,
                tiles_seen=self.tiles_seen,
                dora_tiles=set(_dora_tiles_from_indicators(self._dora_indicators)),
            )
        candidates = {}
        for cand in analysis.candidates:
            actual = _select_tile_from_hand(cand.tile, hand_tiles)
            if not actual:
                continue
            if discardable and actual not in discardable:
                continue
            score = cand.score
            if score is None and cand.rank is not None:
                score = float(cand.rank)
            if score is None:
                score = 0.0
            if defense_ctx is not None:
                score -= tile_danger(actual, defense_ctx) * defense_weight
            if actual not in candidates or score > candidates[actual]:
                candidates[actual] = score

        if not candidates:
            return None

        mask_unicode = MASK_UNICODE_3P if self.is_3p else MASK_UNICODE_4P
        mask_bits = 0
        q_values = []
        for idx, key in enumerate(mask_unicode):
            if key in candidates:
                mask_bits |= (1 << idx)
                q_values.append(candidates[key])

        meta = {
            "q_values": q_values,
            "mask_bits": mask_bits,
        }
        if analysis.analysis_text:
            meta["analysis"] = analysis.analysis_text
        return meta

    def _build_helper_melds(self) -> list[str]:
        melds: list[str] = []
        for event in self._meld_events:
            tiles, concealed = _meld_event_to_tiles(event)
            if not tiles:
                continue
            meld = MahjongHelperEngine.meld_to_helper(tiles, concealed)
            if meld:
                melds.append(meld)
        return melds

    def _build_call_options(
        self,
        called_tile: str,
        hand_tiles: list[str],
        melds: list[str],
        dora_tiles: list[str],
    ) -> list[CallOption]:
        options: list[CallOption] = []

        def add_option(call_type: str, consumed: list[str]) -> None:
            remaining = _remove_tiles(hand_tiles, consumed)
            if remaining is None:
                return
            meld_tiles = consumed + [called_tile]
            meld = MahjongHelperEngine.meld_to_helper(meld_tiles, concealed=False)
            if not meld:
                return
            analysis = self._engine.analyze_discard(
                remaining,
                melds=melds + [meld],
                dora_tiles=dora_tiles,
            )
            if analysis.best_tile is None:
                return
            rank, score = _best_candidate_metrics(analysis)
            options.append(
                CallOption(
                    call_type=call_type,
                    consumed=consumed,
                    analysis=analysis,
                    shanten=analysis.best_shanten,
                    rank=rank,
                    score=score,
                )
            )

        if self.can_pon:
            for consumed in self.find_pon_consume_simple():
                add_option("pon", consumed)
        if self.can_chi:
            for consumed in self.find_chi_consume_simple():
                add_option("chi", consumed)
        if self.can_daiminkan:
            consumed = self._find_daiminkan_consume(called_tile, hand_tiles)
            if consumed:
                add_option("daiminkan", consumed)

        return options

    def _call_option_key(
        self,
        option: CallOption,
        base_shanten: int,
        preferred_call_type: Optional[str],
    ) -> tuple[int, int, float, float]:
        shanten = option.shanten if option.shanten is not None else base_shanten + 2
        prefer = 1 if preferred_call_type and option.call_type == preferred_call_type else 0
        return (shanten, -prefer, -option.rank, -option.score)

    def _track_state(self, events: list[dict]) -> None:
        for event in events:
            if event["type"] == "start_game":
                self.player_id = event.get("id", self.player_id)
                self._meld_events = []
                self._dora_indicators = []
                self._discards_by_player = {}
                self._riichi_players = set()
                self._open_melds_by_player = {}
                self._last_self_draw = None
                self._awaiting_discard = False
                continue
            if event["type"] in ("start_kyoku", "end_game"):
                self._meld_events = []
                self._dora_indicators = []
                self._discards_by_player = {}
                self._riichi_players = set()
                self._open_melds_by_player = {}
                self._last_self_draw = None
                self._awaiting_discard = False
                if event["type"] == "start_kyoku" and "dora_marker" in event:
                    self._dora_indicators.append(event["dora_marker"])
                continue
            if event["type"] == "end_kyoku":
                self._meld_events = []
                self._dora_indicators = []
                self._discards_by_player = {}
                self._riichi_players = set()
                self._open_melds_by_player = {}
                self._last_self_draw = None
                self._awaiting_discard = False
                continue
            if event["type"] == "dora" and "dora_marker" in event:
                self._dora_indicators.append(event["dora_marker"])
            if event["type"] == "tsumo" and event.get("actor") == self.player_id:
                self._last_self_draw = event.get("pai")
                self._awaiting_discard = True
            if event["type"] in ("chi", "pon", "daiminkan", "kakan", "ankan"):
                actor = event.get("actor")
                if actor is not None and event["type"] in ("chi", "pon", "daiminkan", "kakan"):
                    self._open_melds_by_player[actor] = self._open_melds_by_player.get(actor, 0) + 1
                if event.get("actor") != self.player_id:
                    continue
                if event["type"] == "kakan":
                    replaced = False
                    for meld in self._meld_events:
                        if meld["type"] == "pon" and meld.get("pai", "")[:2] == event.get("pai", "")[:2]:
                            meld.update(event)
                            replaced = True
                            break
                    if not replaced:
                        self._meld_events.append(event)
                else:
                    self._meld_events.append(event)
                self._awaiting_discard = False
            if event["type"] == "dahai":
                actor = event.get("actor")
                if actor is not None:
                    pai = event.get("pai", "?")
                    if pai != "?":
                        self._discards_by_player.setdefault(actor, []).append(pai)
                if actor == self.player_id:
                    self._awaiting_discard = False
            if event["type"] == "nukidora":
                actor = event.get("actor")
                if actor is not None:
                    self._discards_by_player.setdefault(actor, []).append("N")
            if event["type"] in ("reach", "reach_accepted"):
                actor = event.get("actor")
                if actor is not None:
                    self._riichi_players.add(actor)

    def _threat_levels(self) -> dict[int, float]:
        levels: dict[int, float] = {}
        for player, count in self._open_melds_by_player.items():
            if player == self.player_id:
                continue
            if count > 0:
                levels[player] = 1.2
        for player in self._riichi_players:
            if player == self.player_id:
                continue
            levels[player] = 2.0
        return levels

    def _find_daiminkan_consume(self, called_tile: str, hand_tiles: list[str]) -> Optional[list[str]]:
        target_kind = _tile_kind(called_tile)
        matches = [t for t in hand_tiles if _tile_kind(t) == target_kind]
        if len(matches) >= 3:
            return matches[:3]
        return None

    def _discardable_tiles(self) -> list[str]:
        tiles = []
        forbidden = self.forbidden_tiles
        for tile in self.tehai_mjai:
            key = tile[:-1] if tile.endswith("r") else tile
            if forbidden.get(key, False):
                continue
            tiles.append(tile)
        return tiles

    def _should_riichi(self) -> bool:
        return self.can_riichi and self.shanten == 0

    def _should_nukidora(self) -> bool:
        if not self.is_3p:
            return False
        if self.self_riichi_accepted:
            return False
        if not self.can_discard:
            return False
        return "N" in self.tehai_mjai

    def _action(self, data: dict) -> str:
        return json.dumps(data, separators=(",", ":"))


def _timeout_ms(default: int = 1500) -> int:
    value = os.environ.get("MAHJONG_HELPER_TIMEOUT_MS")
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid MAHJONG_HELPER_TIMEOUT_MS: {value}")
        return default


def _tile_kind(tile: str) -> str:
    if tile in ("E", "S", "W", "N", "P", "F", "C"):
        return tile
    if len(tile) >= 2 and tile[1] in ("m", "p", "s"):
        return tile[0] + tile[1]
    return tile


def _select_tile_from_hand(tile: str, hand_tiles: list[str]) -> Optional[str]:
    if tile in hand_tiles:
        return tile
    kind = _tile_kind(tile)
    if tile.endswith("r"):
        for cand in hand_tiles:
            if cand.endswith("r") and _tile_kind(cand) == kind:
                return cand
    else:
        for cand in hand_tiles:
            if not cand.endswith("r") and _tile_kind(cand) == kind:
                return cand
    for cand in hand_tiles:
        if _tile_kind(cand) == kind:
            return cand
    return None


def _meld_event_to_tiles(event: dict) -> tuple[Optional[list[str]], bool]:
    meld_type = event.get("type")
    if meld_type == "ankan":
        return event.get("consumed"), True
    if meld_type in ("chi", "pon", "daiminkan"):
        tiles = list(event.get("consumed", []))
        pai = event.get("pai")
        if pai:
            tiles.append(pai)
        return tiles, False
    if meld_type == "kakan":
        tiles = list(event.get("consumed", []))
        pai = event.get("pai")
        if pai:
            tiles.append(pai)
        return tiles, False
    return None, False


def _dora_tiles_from_indicators(indicators: list[str]) -> list[str]:
    tiles = []
    for indicator in indicators:
        dora = _dora_from_indicator(indicator)
        if dora:
            tiles.append(dora)
    return tiles


def _dora_from_indicator(indicator: str) -> Optional[str]:
    if not indicator or indicator == "?":
        return None
    tile = indicator.replace("r", "")
    if len(tile) >= 2 and tile[1] in ("m", "p", "s"):
        num = tile[0]
        suit = tile[1]
        if num == "0":
            num = "5"
        if not num.isdigit():
            return None
        value = int(num)
        value = 1 if value == 9 else value + 1
        return f"{value}{suit}"
    if tile in HONOR_DORA_NEXT:
        return HONOR_DORA_NEXT[tile]
    return None


def _remove_tiles(hand_tiles: list[str], consumed: list[str]) -> Optional[list[str]]:
    remaining = hand_tiles[:]
    for tile in consumed:
        if tile in remaining:
            remaining.remove(tile)
            continue
        kind = _tile_kind(tile)
        for idx, cand in enumerate(remaining):
            if _tile_kind(cand) == kind:
                remaining.pop(idx)
                break
        else:
            return None
    return remaining


def _best_candidate_metrics(analysis: DiscardAnalysis) -> tuple[float, float]:
    if not analysis.candidates:
        return 0.0, 0.0
    cand = analysis.candidates[0]
    rank = float(cand.rank) if cand.rank is not None else 0.0
    score = cand.score if cand.score is not None else 0.0
    return rank, score


def _call_type_has_yaku(candidates: list) -> dict[str, bool]:
    has_yaku: dict[str, bool] = {}
    for cand in candidates:
        if not cand.call_type:
            continue
        if cand.call_type not in has_yaku:
            has_yaku[cand.call_type] = False
        if not cand.no_yaku:
            has_yaku[cand.call_type] = True
    return has_yaku

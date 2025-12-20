from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from mjai_bot.bot import AkagiBot
from mjai_bot.logger import logger

from .engine import MahjongHelperEngine, DiscardAnalysis, CallAnalysis, DiscardCandidate
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

YAKUHAI_CALL_BONUS = 80.0
NO_YAKU_PENALTY = 60.0
YAKUHAI_ONLY_PENALTY = 25.0
EARLY_CALL_TURN_LIMIT = 5
RIICHI_DORA_THRESHOLD = 2
LOW_VALUE_OPEN_YAKU = {"断幺", "混全", "纯全", "混老头"}
NO_YAKU_TENPAI_PENALTY = 200.0
SHANTEN_PENALTY = 40.0


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
        self._analysis_cache_key: Optional[tuple] = None
        self._analysis_cache: Optional[DiscardAnalysis] = None

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
                return self._action_with_meta(
                    {"type": "hora"},
                    self._single_action_meta("hora"),
                )
            if self._should_nukidora():
                return self._action_with_meta(
                    {
                        "type": "nukidora",
                        "actor": self.player_id,
                        "pai": "N",
                    },
                    self._single_action_meta("nukidora"),
                )
            if self._should_riichi():
                riichi_meta = self._build_riichi_meta()
                return self._action_with_meta(
                    {
                        "type": "reach",
                        "actor": self.player_id,
                    },
                    riichi_meta or self._single_action_meta("reach"),
                )
            if self.can_ryukyoku:
                return self._action_with_meta(
                    {"type": "ryukyoku"},
                    self._single_action_meta("ryukyoku"),
                )
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

        return self._action_with_meta(
            {"type": "none"},
            self._action_meta({}),
        )

    def _think_discard(self) -> str:
        hand_tiles = self._current_hand_tiles()
        melds = self._build_helper_melds()
        dora_tiles = _dora_tiles_from_indicators(self._dora_indicators)
        analysis = self._get_discard_analysis(hand_tiles, melds, dora_tiles)
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
            return self._action_with_meta(
                {"type": "none"},
                self._action_meta({}),
            )

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

        base_shanten = self._resolve_call_base_shanten(
            analysis=analysis,
            hand_tiles=hand_tiles,
            melds=melds,
            dora_tiles=dora_tiles,
        )
        options = self._build_call_options(
            called_tile=called_tile,
            hand_tiles=hand_tiles,
            melds=melds,
            dora_tiles=dora_tiles,
        )
        options = [
            option for option in options
            if (
                option.call_type != "chi"
                or self._allow_chi_option(
                    option=option,
                    base_shanten=base_shanten,
                    hand_tiles=hand_tiles,
                    called_tile=called_tile,
                    dora_tiles=dora_tiles,
                )
            )
            and (
                option.call_type != "pon"
                or self._allow_pon_option(
                    option=option,
                    base_shanten=base_shanten,
                    hand_tiles=hand_tiles,
                    called_tile=called_tile,
                    dora_tiles=dora_tiles,
                )
            )
        ]
        if not options:
            return self._action_with_meta(
                {"type": "none"},
                self._action_meta({}),
            )

        options.sort(
            key=lambda option: self._call_option_key(
                option,
                base_shanten=base_shanten,
                preferred_call_type=analysis.call_type,
            )
        )
        action_scores = self._call_action_scores(
            options=options,
            base_shanten=base_shanten,
            called_tile=called_tile,
        )
        threat_levels = self._threat_levels()
        has_riichi_threat = any(level >= 2.0 for level in threat_levels.values())
        call_type_has_yaku = _call_type_has_yaku(analysis.candidates)

        best_option = options[0]
        improvement: Optional[int] = None
        if best_option.shanten is not None and base_shanten is not None:
            improvement = base_shanten - best_option.shanten
        no_yaku = call_type_has_yaku.get(best_option.call_type) is False
        yakuhai_call = best_option.call_type in ("pon", "daiminkan") and self.is_yakuhai(called_tile)
        early_turn = self._turn_count() <= EARLY_CALL_TURN_LIMIT
        already_open = self._open_melds_by_player.get(self.player_id, 0) > 0

        should_call = False
        if improvement is None:
            if yakuhai_call and early_turn:
                should_call = True
        elif improvement > 0:
            should_call = True
        elif improvement == 0:
            if yakuhai_call:
                should_call = True
            elif already_open and analysis.should_call and not no_yaku:
                should_call = True
        else:
            if yakuhai_call and improvement >= -1 and (early_turn or already_open):
                should_call = True

        if has_riichi_threat and (improvement is None or improvement <= 0):
            should_call = False
        if best_option.call_type == "daiminkan" and (improvement is None or improvement < 1):
            should_call = False
        if not should_call:
            return self._action_with_meta(
                {"type": "none"},
                self._single_action_meta("none"),
            )

        if best_option.call_type == "chi" and self.can_chi:
            return self._action_with_meta(
                {
                    "type": "chi",
                    "actor": self.player_id,
                    "target": self.target_actor,
                    "pai": called_tile,
                    "consumed": best_option.consumed,
                },
                self._action_meta(action_scores),
            )
        if best_option.call_type == "pon" and self.can_pon:
            return self._action_with_meta(
                {
                    "type": "pon",
                    "actor": self.player_id,
                    "target": self.target_actor,
                    "pai": called_tile,
                    "consumed": best_option.consumed,
                },
                self._action_meta(action_scores),
            )
        if best_option.call_type == "daiminkan" and self.can_daiminkan:
            return self._action_with_meta(
                {
                    "type": "daiminkan",
                    "actor": self.player_id,
                    "target": self.target_actor,
                    "pai": called_tile,
                    "consumed": best_option.consumed,
                },
                self._action_meta(action_scores),
            )

        return self._action_with_meta(
            {"type": "none"},
            self._action_meta({}),
        )

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
        scores, _ = self._evaluate_discard(analysis, hand_tiles)
        if scores:
            return max(scores.items(), key=lambda item: item[1])[0]

        discardable = self._discardable_tiles()
        if discardable:
            return discardable[0]
        if hand_tiles:
            return hand_tiles[0]
        if self._last_self_draw not in (None, "?"):
            return self._last_self_draw
        return None

    def _build_meta(
        self,
        analysis: DiscardAnalysis,
        hand_tiles: list[str],
    ) -> Optional[dict]:
        candidates, analysis_text = self._evaluate_discard(analysis, hand_tiles)
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
        if analysis_text:
            meta["analysis"] = analysis_text
        return meta

    def _evaluate_discard(
        self,
        analysis: DiscardAnalysis,
        hand_tiles: list[str],
    ) -> tuple[dict[str, float], Optional[str]]:
        discardable = set(self._discardable_tiles())
        threat_levels = self._threat_levels()
        yakuhai_pairs = self._yakuhai_pair_tiles(hand_tiles)
        is_open = self._open_melds_by_player.get(self.player_id, 0) > 0

        defense_ctx = None
        defense_weight = 0.0
        if threat_levels:
            max_threat = max(threat_levels.values())
            defense_weight = 20.0 * max_threat + 5.0 * (len(threat_levels) - 1)
            defense_ctx = DefenseContext(
                threat_levels=threat_levels,
                discards_by_player=self._discards_by_player,
                tiles_seen=self.tiles_seen,
                dora_tiles=set(_dora_tiles_from_indicators(self._dora_indicators)),
            )

        candidates: dict[str, float] = {}
        yaku_by_tile: dict[str, int] = {}
        tenpai_tiles: set[str] = set()
        best_shanten = analysis.best_shanten
        has_yaku_tenpai = False
        if is_open and best_shanten == 0:
            for cand in analysis.candidates:
                if cand.shanten != 0:
                    continue
                category = self._yaku_category(cand)
                if category >= 1:
                    has_yaku_tenpai = True
                    break

        use_best_shanten = True
        if is_open and best_shanten == 0 and not has_yaku_tenpai:
            use_best_shanten = False

        filtered = analysis.candidates
        if best_shanten is not None and use_best_shanten:
            filtered_best = [
                cand for cand in analysis.candidates
                if cand.shanten == best_shanten
            ]
            if filtered_best:
                filtered = filtered_best
        elif best_shanten is not None:
            filtered = [
                cand for cand in analysis.candidates
                if cand.shanten is None or cand.shanten <= best_shanten + 1
            ]

        for cand in filtered:
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
            if is_open:
                category = self._yaku_category(cand)
                prev = yaku_by_tile.get(actual, -1)
                if category > prev:
                    yaku_by_tile[actual] = category
                if cand.shanten == 0:
                    tenpai_tiles.add(actual)
                if cand.shanten == 0 and category == 0:
                    score -= NO_YAKU_TENPAI_PENALTY
                elif category == 0:
                    score -= NO_YAKU_PENALTY
                elif category == 1:
                    score -= YAKUHAI_ONLY_PENALTY
                if not use_best_shanten and best_shanten is not None and cand.shanten is not None:
                    score -= (cand.shanten - best_shanten) * SHANTEN_PENALTY
            if defense_ctx is not None:
                score -= tile_danger(actual, defense_ctx) * defense_weight
            if actual not in candidates or score > candidates[actual]:
                candidates[actual] = score

        if not candidates:
            return {}, None

        if is_open and best_shanten == 0 and not has_yaku_tenpai:
            filtered_noyaku = {
                tile: score
                for tile, score in candidates.items()
                if not (tile in tenpai_tiles and yaku_by_tile.get(tile, 0) == 0)
            }
            if filtered_noyaku:
                candidates = filtered_noyaku

        if yakuhai_pairs and not threat_levels:
            non_yakuhai = {
                tile: score
                for tile, score in candidates.items()
                if _tile_kind(tile) not in yakuhai_pairs
            }
            if non_yakuhai:
                candidates = non_yakuhai

        if is_open and yaku_by_tile:
            yaku_by_tile = {tile: yaku_by_tile.get(tile, 0) for tile in candidates}
            max_cat = max(yaku_by_tile.values(), default=0)
            filtered_yaku: dict[str, float] = {}
            if max_cat >= 2:
                filtered_yaku = {
                    tile: score
                    for tile, score in candidates.items()
                    if yaku_by_tile.get(tile, 0) >= 2
                }
            elif max_cat >= 1:
                filtered_yaku = {
                    tile: score
                    for tile, score in candidates.items()
                    if yaku_by_tile.get(tile, 0) >= 1
                }
            if filtered_yaku:
                candidates = filtered_yaku

        if not threat_levels:
            garbage, soft_garbage = self._discard_focus_kinds(hand_tiles)
            filtered = self._filter_scored_by_kinds(candidates, garbage)
            if not filtered:
                filtered = self._filter_scored_by_kinds(candidates, soft_garbage)
            if filtered:
                candidates = filtered

        if defense_ctx is not None:
            safe = {
                tile for tile in candidates
                if tile_danger(tile, defense_ctx) == 0.0
            }
            if safe:
                candidates = {tile: candidates[tile] for tile in safe}

        analysis_text = None
        header = None
        if analysis.analysis_text:
            header = analysis.analysis_text.split("|", 1)[0].strip()
        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        if ranked:
            top = ", ".join(f"{tile} ({score:.2f})" for tile, score in ranked[:3])
            parts = []
            if header:
                parts.append(header)
            if top:
                parts.append("Top discards: " + top)
            analysis_text = " | ".join(parts) if parts else None

        return candidates, analysis_text

    def _build_riichi_meta(self) -> Optional[dict]:
        hand_tiles = self._current_hand_tiles()
        melds = self._build_helper_melds()
        dora_tiles = _dora_tiles_from_indicators(self._dora_indicators)
        analysis = self._get_discard_analysis(hand_tiles, melds, dora_tiles)
        candidates, _ = self._evaluate_discard(analysis, hand_tiles)
        if not candidates:
            return self._single_action_meta("reach")

        riichi_tiles = set(self.discardable_tiles_riichi_declaration)
        if riichi_tiles:
            candidates = {
                tile: score for tile, score in candidates.items()
                if tile in riichi_tiles
            }
        if not candidates:
            return self._single_action_meta("reach")

        header = None
        if analysis.analysis_text:
            header = analysis.analysis_text.split("|", 1)[0].strip()
        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        analysis_text = None
        if ranked:
            top = ", ".join(f"{tile} ({score:.2f})" for tile, score in ranked[:3])
            parts = []
            if header:
                parts.append(header)
            if top:
                parts.append("Riichi discards: " + top)
            analysis_text = " | ".join(parts) if parts else None

        meta = self._action_meta(candidates)
        if analysis_text:
            meta["analysis"] = analysis_text
        return meta

    def _get_discard_analysis(
        self,
        hand_tiles: list[str],
        melds: list[str],
        dora_tiles: list[str],
    ) -> DiscardAnalysis:
        key = (
            tuple(sorted(hand_tiles)),
            tuple(sorted(melds)),
            tuple(sorted(dora_tiles)),
        )
        if self._analysis_cache_key == key and self._analysis_cache is not None:
            return self._analysis_cache
        analysis = self._engine.analyze_discard(
            hand_tiles,
            melds=melds,
            dora_tiles=dora_tiles,
        )
        self._analysis_cache_key = key
        self._analysis_cache = analysis
        return analysis

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
        base_shanten: Optional[int],
        preferred_call_type: Optional[str],
    ) -> tuple[int, int, float, float]:
        base = base_shanten if base_shanten is not None else 8
        shanten = option.shanten if option.shanten is not None else base + 2
        prefer = 1 if preferred_call_type and option.call_type == preferred_call_type else 0
        return (shanten, -prefer, -option.rank, -option.score)

    def _call_action_scores(
        self,
        options: list[CallOption],
        base_shanten: Optional[int],
        called_tile: str,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for option in options:
            action_key = None
            if option.call_type == "pon" and self.can_pon:
                action_key = "pon"
            elif option.call_type == "daiminkan" and self.can_kan:
                action_key = "kan_select"
            elif option.call_type == "chi" and self.can_chi:
                action_key = _chi_action_key(called_tile, option.consumed)
                if action_key == "chi_low" and not self.can_chi_low:
                    action_key = None
                elif action_key == "chi_mid" and not self.can_chi_mid:
                    action_key = None
                elif action_key == "chi_high" and not self.can_chi_high:
                    action_key = None
            if not action_key:
                continue
            if option.score is not None:
                score = option.score
            elif option.rank is not None:
                score = option.rank
            else:
                score = 0.0
            if option.shanten is not None and base_shanten is not None:
                score += (base_shanten - option.shanten) * 100.0
            if action_key in ("pon", "kan_select") and self.is_yakuhai(called_tile):
                score += YAKUHAI_CALL_BONUS
            if action_key not in scores or score > scores[action_key]:
                scores[action_key] = score
        return scores

    def _track_state(self, events: list[dict]) -> None:
        self._analysis_cache_key = None
        self._analysis_cache = None
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
        if not self.can_riichi or self.shanten != 0:
            return False
        threat_levels = self._threat_levels()
        if any(level >= 2.0 for level in threat_levels.values()):
            return True
        if any(
            count >= 3
            for player, count in self._open_melds_by_player.items()
            if player != self.player_id
        ):
            return True

        hand_tiles = self._current_hand_tiles()
        melds = self._build_helper_melds()
        dora_tiles = _dora_tiles_from_indicators(self._dora_indicators)
        analysis = self._get_discard_analysis(hand_tiles, melds, dora_tiles)
        riichi_tiles = set(self.discardable_tiles_riichi_declaration)
        best_category = -1
        for cand in analysis.candidates:
            if cand.shanten != 0:
                continue
            actual = _select_tile_from_hand(cand.tile, hand_tiles)
            if not actual:
                continue
            if riichi_tiles and actual not in riichi_tiles:
                continue
            category = self._riichi_yaku_category(cand)
            if category > best_category:
                best_category = category
            if category >= 2:
                return True

        dora_count = self._count_dora_tiles(hand_tiles, dora_tiles)
        if dora_count >= RIICHI_DORA_THRESHOLD:
            return True
        return False

    def _should_nukidora(self) -> bool:
        if not self.is_3p:
            return False
        if self.self_riichi_accepted:
            return False
        if not self.can_discard:
            return False
        return "N" in self.tehai_mjai

    def _resolve_call_base_shanten(
        self,
        analysis: CallAnalysis,
        hand_tiles: list[str],
        melds: list[str],
        dora_tiles: list[str],
    ) -> Optional[int]:
        if analysis.base_shanten is not None:
            return analysis.base_shanten
        base = self._engine.analyze_discard(
            hand_tiles,
            melds=melds,
            dora_tiles=dora_tiles,
        )
        if base.best_shanten is not None:
            return base.best_shanten
        return self.shanten

    def _yakuhai_tiles(self) -> set[str]:
        tiles = {"P", "F", "C", self.jikaze, self.bakaze}
        return {tile for tile in tiles if tile}

    def _yakuhai_pair_tiles(self, hand_tiles: list[str]) -> set[str]:
        yakuhai = self._yakuhai_tiles()
        counts: dict[str, int] = {}
        for tile in hand_tiles:
            kind = _tile_kind(tile)
            if kind in yakuhai:
                counts[kind] = counts.get(kind, 0) + 1
        return {tile for tile, count in counts.items() if count >= 2}

    def _turn_count(self) -> int:
        return len(self._discards_by_player.get(self.player_id, []))

    def _yaku_category(self, cand: DiscardCandidate) -> int:
        tags = [tag for tag in cand.yaku_tags if tag not in ("宝牌", "无役")]
        if "无役" in cand.yaku_tags or not tags:
            return 0
        if set(tags) == {"役牌"}:
            return 1
        return 2

    def _riichi_yaku_category(self, cand: DiscardCandidate) -> int:
        tags = [
            tag for tag in cand.yaku_tags
            if tag not in ("宝牌", "无役", "立直", "自摸")
        ]
        if "无役" in cand.yaku_tags or not tags:
            return 0
        if set(tags) == {"役牌"}:
            return 1
        return 2

    def _option_yaku_summary(self, option: CallOption) -> tuple[bool, bool]:
        candidates = option.analysis.candidates
        if not candidates:
            return False, False
        best_shanten = option.analysis.best_shanten
        seen = False
        for cand in candidates:
            if best_shanten is not None and cand.shanten != best_shanten:
                continue
            tags = {
                tag for tag in cand.yaku_tags
                if tag not in ("宝牌", "无役", "立直", "自摸", "w立")
            }
            if not tags:
                continue
            seen = True
            if tags - LOW_VALUE_OPEN_YAKU:
                return False, True
        if seen:
            return True, True
        return False, False

    def _chi_low_value_yaku(self, option: CallOption) -> bool:
        low_value_only, has_any = self._option_yaku_summary(option)
        return low_value_only and has_any

    def _allow_pon_option(
        self,
        option: CallOption,
        base_shanten: Optional[int],
        hand_tiles: list[str],
        called_tile: str,
        dora_tiles: list[str],
    ) -> bool:
        if option.call_type != "pon":
            return True
        if self.is_yakuhai(called_tile):
            return True
        improvement = None
        if option.shanten is not None and base_shanten is not None:
            improvement = base_shanten - option.shanten
        if improvement is None:
            return False
        already_open = self._open_melds_by_player.get(self.player_id, 0) > 0
        dora_count = self._count_dora_tiles(hand_tiles + [called_tile], dora_tiles)
        plan_ok = False
        if dora_count >= 1:
            plan_ok = True
        if self._honitsu_potential(hand_tiles, called_tile):
            plan_ok = True
        if self._chinitsu_potential(hand_tiles, called_tile):
            plan_ok = True
        low_value_only, has_any_yaku = self._option_yaku_summary(option)

        if not has_any_yaku:
            return improvement >= 1 and plan_ok
        if low_value_only:
            return improvement >= 1 and dora_count >= 1
        if improvement >= 1:
            return True
        return already_open and plan_ok

    def _allow_chi_option(
        self,
        option: CallOption,
        base_shanten: Optional[int],
        hand_tiles: list[str],
        called_tile: str,
        dora_tiles: list[str],
    ) -> bool:
        if option.call_type != "chi":
            return True
        improvement = None
        if option.shanten is not None and base_shanten is not None:
            improvement = base_shanten - option.shanten
        already_open = self._open_melds_by_player.get(self.player_id, 0) > 0
        dora_count = self._count_dora_tiles(hand_tiles + [called_tile], dora_tiles)
        plan_ok = False
        if dora_count >= 2:
            plan_ok = True
        if self._honitsu_potential(hand_tiles, called_tile):
            plan_ok = True
        if self._chinitsu_potential(hand_tiles, called_tile):
            plan_ok = True
        if self._tanyao_potential(hand_tiles, called_tile):
            plan_ok = True
        low_value = self._chi_low_value_yaku(option)
        if low_value and not plan_ok and not already_open:
            return False
        if already_open:
            return improvement is None or improvement >= 0
        early_turn = self._turn_count() <= EARLY_CALL_TURN_LIMIT
        if not early_turn:
            return improvement is not None and improvement >= 0
        if improvement is not None and improvement > 0:
            if low_value and not plan_ok:
                return False
            return True
        if plan_ok and not low_value:
            return True
        return False

    def _count_dora_tiles(self, tiles: list[str], dora_tiles: list[str]) -> int:
        dora_kinds = {_tile_kind(tile) for tile in dora_tiles if tile}
        count = 0
        for tile in tiles:
            kind = _tile_kind(tile)
            if kind in dora_kinds:
                count += 1
            if tile.endswith("r"):
                count += 1
        return count

    def _honitsu_potential(self, hand_tiles: list[str], called_tile: str) -> bool:
        suit_counts, honor_count = self._suit_counts(hand_tiles + [called_tile])
        max_suit = max(suit_counts.values())
        return max_suit >= 9 and honor_count <= 4

    def _chinitsu_potential(self, hand_tiles: list[str], called_tile: str) -> bool:
        suit_counts, honor_count = self._suit_counts(hand_tiles + [called_tile])
        max_suit = max(suit_counts.values())
        return honor_count == 0 and max_suit >= 10

    def _tanyao_potential(self, hand_tiles: list[str], called_tile: str) -> bool:
        if not self._is_simple_tile(called_tile):
            return False
        tiles = hand_tiles + [called_tile]
        terminals_or_honors = sum(
            1 for tile in tiles if self._is_terminal_or_honor(tile)
        )
        return terminals_or_honors <= 2

    def _suit_counts(self, tiles: list[str]) -> tuple[dict[str, int], int]:
        suit_counts = {"m": 0, "p": 0, "s": 0}
        honor_count = 0
        for tile in tiles:
            kind = _tile_kind(tile)
            if kind in ("E", "S", "W", "N", "P", "F", "C"):
                honor_count += 1
                continue
            if len(kind) >= 2 and kind[1] in suit_counts:
                suit_counts[kind[1]] += 1
        return suit_counts, honor_count

    def _is_terminal_or_honor(self, tile: str) -> bool:
        kind = _tile_kind(tile)
        if kind in ("E", "S", "W", "N", "P", "F", "C"):
            return True
        num = _tile_number(kind)
        return num in (1, 9)

    def _is_simple_tile(self, tile: str) -> bool:
        num = _tile_number(tile)
        return num is not None and 2 <= num <= 8

    def _discard_focus_kinds(self, hand_tiles: list[str]) -> tuple[set[str], set[str]]:
        yakuhai = self._yakuhai_tiles()
        honor_counts: dict[str, int] = {}
        suit_counts: dict[str, dict[int, int]] = {suit: {i: 0 for i in range(1, 10)} for suit in ("m", "p", "s")}

        for tile in hand_tiles:
            kind = _tile_kind(tile)
            if kind in ("E", "S", "W", "N", "P", "F", "C"):
                honor_counts[kind] = honor_counts.get(kind, 0) + 1
                continue
            if len(kind) >= 2 and kind[1] in ("m", "p", "s"):
                num = _tile_number(kind)
                if num is not None:
                    suit_counts[kind[1]][num] += 1

        garbage: set[str] = set()
        soft_garbage: set[str] = set()

        for honor, count in honor_counts.items():
            if honor in yakuhai:
                continue
            if count == 1:
                garbage.add(honor)
            elif count >= 2:
                soft_garbage.add(honor)

        for suit, counts in suit_counts.items():
            for num, count in counts.items():
                if count <= 0:
                    continue
                if not self._is_isolated_number(num, counts):
                    continue
                kind = f"{num}{suit}"
                if num in (1, 9):
                    garbage.add(kind)
                elif num in (2, 8):
                    soft_garbage.add(kind)

        return garbage, soft_garbage

    def _is_isolated_number(self, num: int, counts: dict[int, int]) -> bool:
        if counts.get(num, 0) >= 2:
            return False
        for delta in (1, 2):
            if counts.get(num - delta, 0) > 0:
                return False
            if counts.get(num + delta, 0) > 0:
                return False
        return True

    def _filter_candidates_by_kinds(self, candidates: list[str], kinds: set[str]) -> list[str]:
        if not kinds:
            return []
        return [tile for tile in candidates if _tile_kind(tile) in kinds]

    def _filter_scored_by_kinds(self, candidates: dict[str, float], kinds: set[str]) -> dict[str, float]:
        if not kinds:
            return {}
        return {tile: score for tile, score in candidates.items() if _tile_kind(tile) in kinds}

    def _action(self, data: dict) -> str:
        return json.dumps(data, separators=(",", ":"))

    def _action_with_meta(self, data: dict, meta: Optional[dict]) -> str:
        if meta is not None:
            data["meta"] = meta
        return json.dumps(data, separators=(",", ":"))

    def _single_action_meta(self, action_key: str) -> dict:
        return self._action_meta({action_key: 1.0})

    def _action_meta(self, scores: dict[str, float]) -> dict:
        mask_unicode = MASK_UNICODE_3P if self.is_3p else MASK_UNICODE_4P
        mask_bits = 0
        q_values = []
        for idx, key in enumerate(mask_unicode):
            if key in scores:
                mask_bits |= (1 << idx)
                q_values.append(scores[key])
        return {
            "q_values": q_values,
            "mask_bits": mask_bits,
        }


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


def _tile_number(tile: str) -> Optional[int]:
    if len(tile) >= 2 and tile[1] in ("m", "p", "s"):
        num = tile[0]
        if num == "0":
            num = "5"
        if num.isdigit():
            return int(num)
    return None


def _chi_action_key(called_tile: str, consumed: list[str]) -> Optional[str]:
    if not called_tile or len(called_tile) < 2:
        return None
    if called_tile[1] not in ("m", "p", "s"):
        return None
    base = _tile_number(called_tile)
    if base is None:
        return None
    nums = [_tile_number(tile) for tile in consumed]
    if any(num is None for num in nums):
        return None
    nums_sorted = sorted(nums)
    if nums_sorted == [base + 1, base + 2]:
        return "chi_low"
    if nums_sorted == [base - 1, base + 1]:
        return "chi_mid"
    if nums_sorted == [base - 2, base - 1]:
        return "chi_high"
    return None

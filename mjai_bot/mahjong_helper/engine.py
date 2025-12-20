from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Iterable, Optional

from mjai_bot.logger import logger

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

HONOR_TO_HELPER = {
    "E": "1z",
    "S": "2z",
    "W": "3z",
    "N": "4z",
    "P": "5z",
    "F": "6z",
    "C": "7z",
}
HELPER_TO_HONOR = {v: k for k, v in HONOR_TO_HELPER.items()}

CHINESE_HONOR_TO_HELPER = {
    "东": "1z",
    "南": "2z",
    "西": "3z",
    "北": "4z",
    "白": "5z",
    "发": "6z",
    "中": "7z",
    "東": "1z",
    "發": "6z",
}

CHINESE_SUIT_TO_HELPER = {
    "万": "m",
    "萬": "m",
    "饼": "p",
    "餅": "p",
    "筒": "p",
    "索": "s",
    "条": "s",
    "條": "s",
}

FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

SHANTEN_WORDS = {
    "和了": -1,
    "听牌": 0,
    "一向听": 1,
    "两向听": 2,
    "三向听": 3,
    "四向听": 4,
    "五向听": 5,
    "六向听": 6,
    "七向听": 7,
    "八向听": 8,
}

YAKU_TOKENS = {
    "立直",
    "自摸",
    "七对",
    "平和",
    "两杯口",
    "一杯口",
    "三色",
    "一通",
    "对对",
    "三暗刻",
    "三色同刻",
    "三杠子",
    "断幺",
    "役牌",
    "混全",
    "纯全",
    "混老头",
    "小三元",
    "混一色",
    "清一色",
    "四暗刻",
    "四暗刻单骑",
    "大三元",
    "小四喜",
    "大四喜",
    "字一色",
    "清老头",
    "绿一色",
    "九莲",
    "纯正九莲",
    "四杠子",
    "五门齐",
    "三连刻",
    "一色三顺",
    "十二落抬",
    "大数邻",
    "大车轮",
    "大竹林",
    "大七星",
    "w立",
}

BRACKET_RE = re.compile(r"\[([^\]]+)\]")


@dataclass
class DiscardCandidate:
    tile: str
    score: Optional[float]
    rank: Optional[int]
    raw_line: str
    call_type: Optional[str] = None
    shanten: Optional[int] = None
    no_yaku: bool = False
    yaku_tags: list[str] = field(default_factory=list)


@dataclass
class DiscardAnalysis:
    best_tile: Optional[str]
    candidates: list[DiscardCandidate]
    analysis_text: Optional[str]
    raw_stdout: str
    best_shanten: Optional[int]


@dataclass
class CallAnalysis:
    should_call: bool
    call_type: Optional[str]
    best_discard_tile: Optional[str]
    candidates: list[DiscardCandidate]
    analysis_text: Optional[str]
    raw_stdout: str
    base_shanten: Optional[int]
    best_shanten: Optional[int]


class MahjongHelperEngine:
    _checked_bins: dict[str, bool] = {}

    def __init__(self, bin_path: Optional[str] = None, timeout_ms: int = 1500) -> None:
        self._bin_path = bin_path
        self.timeout_ms = timeout_ms

    def _resolve_bin(self) -> str:
        return self._bin_path or os.environ.get("MAHJONG_HELPER_BIN", "mahjong-helper")

    def ensure_available(self) -> None:
        bin_path = self._resolve_bin()
        if self._checked_bins.get(bin_path):
            return
        try:
            subprocess.run(
                [bin_path, "--help"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=1,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"mahjong-helper binary not found: '{bin_path}'. "
                "Install it or set MAHJONG_HELPER_BIN."
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Failed to execute mahjong-helper: '{bin_path}'."
            ) from exc
        self._checked_bins[bin_path] = True

    def analyze_discard(
        self,
        hand_tiles: Iterable[str],
        melds: Optional[Iterable[str]] = None,
        dora_tiles: Optional[Iterable[str]] = None,
    ) -> DiscardAnalysis:
        args = self._build_args(hand_tiles, melds=melds, dora_tiles=dora_tiles)
        stdout = self._run(args)
        if stdout is None:
            return DiscardAnalysis(None, [], None, "", None)
        return self.parse_discard_output(stdout)

    def analyze_call(
        self,
        hand_tiles: Iterable[str],
        called_tile: str,
        melds: Optional[Iterable[str]] = None,
        dora_tiles: Optional[Iterable[str]] = None,
    ) -> CallAnalysis:
        args = self._build_args(
            hand_tiles,
            melds=melds,
            called_tile=called_tile,
            dora_tiles=dora_tiles,
        )
        stdout = self._run(args)
        if stdout is None:
            return CallAnalysis(False, None, None, [], None, "", None, None)
        return self.parse_call_output(stdout)

    def parse_discard_output(self, stdout: str) -> DiscardAnalysis:
        stripped = ANSI_RE.sub("", stdout)
        lines = [line.rstrip() for line in stripped.splitlines()]
        candidates: list[DiscardCandidate] = []
        current_shanten: Optional[int] = None
        for line in lines:
            header_shanten = self._header_shanten(line)
            if header_shanten is not None:
                current_shanten = header_shanten
                continue
            candidate = self._parse_candidate_line(line)
            if candidate is not None:
                if candidate.shanten is None:
                    candidate.shanten = current_shanten
                candidates.append(candidate)

        best_tile = self._select_best_tile(candidates)
        best_shanten = self._best_shanten_from_candidates(candidates, lines)
        analysis_text = self._format_analysis_text(
            lines,
            candidates,
            best_shanten=best_shanten,
        )
        return DiscardAnalysis(best_tile, candidates, analysis_text, stdout, best_shanten)

    def parse_call_output(self, stdout: str) -> CallAnalysis:
        stripped = ANSI_RE.sub("", stdout)
        lines = [line.rstrip() for line in stripped.splitlines()]
        candidates: list[DiscardCandidate] = []
        current_shanten: Optional[int] = None
        base_shanten: Optional[int] = None
        for line in lines:
            header_shanten = self._header_shanten(line)
            if header_shanten is not None:
                current_shanten = header_shanten
                if "当前" in line:
                    base_shanten = header_shanten
                continue
            candidate = self._parse_candidate_line(line)
            if candidate is not None:
                if candidate.shanten is None:
                    candidate.shanten = current_shanten
                candidates.append(candidate)

        call_candidates = [c for c in candidates if c.call_type]
        best_candidate = call_candidates[0] if call_candidates else None
        best_tile = best_candidate.tile if best_candidate else None
        best_call_type = best_candidate.call_type if best_candidate else None
        best_shanten = best_candidate.shanten if best_candidate else None

        should_call = False
        if base_shanten is not None and best_shanten is not None:
            should_call = best_shanten <= base_shanten
        elif best_candidate:
            should_call = True

        analysis_text = self._format_analysis_text(lines, candidates)
        return CallAnalysis(
            should_call=should_call,
            call_type=best_call_type,
            best_discard_tile=best_tile,
            candidates=candidates,
            analysis_text=analysis_text,
            raw_stdout=stdout,
            base_shanten=base_shanten,
            best_shanten=best_shanten,
        )

    def _run(self, args: list[str]) -> Optional[str]:
        self.ensure_available()
        bin_path = self._resolve_bin()
        cmd = [bin_path, *args]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout_ms / 1000.0,
            )
        except subprocess.TimeoutExpired:
            logger.warning("mahjong-helper timed out")
            return None
        if result.stderr:
            logger.debug(f"mahjong-helper stderr: {result.stderr.strip()}")
        return result.stdout

    def _build_args(
        self,
        hand_tiles: Iterable[str],
        melds: Optional[Iterable[str]] = None,
        called_tile: Optional[str] = None,
        dora_tiles: Optional[Iterable[str]] = None,
    ) -> list[str]:
        grouped = {"m": [], "p": [], "s": [], "z": []}
        for tile in hand_tiles:
            helper_tile = self.mjai_tile_to_helper(tile)
            if helper_tile is None:
                continue
            suit = helper_tile[1]
            if suit not in grouped:
                continue
            grouped[suit].append(int(helper_tile[0]))

        args: list[str] = []
        dora_arg = self._build_dora_arg(dora_tiles)
        if dora_arg:
            args.append(dora_arg)
        for suit in ("m", "p", "s", "z"):
            digits = grouped[suit]
            if not digits:
                continue
            digits.sort(key=lambda d: 5 if d == 0 else d)
            args.append("".join(str(d) for d in digits) + suit)

        if melds:
            args.append("#")
            args.extend(melds)

        if called_tile:
            helper_called = self.mjai_tile_to_helper(called_tile)
            if helper_called:
                args.append("+")
                args.append(helper_called)

        return args

    def _build_dora_arg(self, dora_tiles: Optional[Iterable[str]]) -> Optional[str]:
        if not dora_tiles:
            return None
        grouped = {"m": [], "p": [], "s": [], "z": []}
        for tile in dora_tiles:
            helper_tile = self.mjai_tile_to_helper(tile)
            if helper_tile is None:
                continue
            suit = helper_tile[1]
            if suit not in grouped:
                continue
            grouped[suit].append(int(helper_tile[0]))

        parts = []
        for suit in ("m", "p", "s", "z"):
            digits = grouped[suit]
            if not digits:
                continue
            digits.sort(key=lambda d: 5 if d == 0 else d)
            parts.append("".join(str(d) for d in digits) + suit)
        if not parts:
            return None
        return "-d=" + "".join(parts)

    def _parse_candidate_line(self, line: str) -> Optional[DiscardCandidate]:
        line = line.strip()
        if not line:
            return None

        rank = None
        score = None
        m = re.match(r"^\s*(\d+)(?:\s*\[(\d+(?:\.\d+)?)\])?", line)
        if m:
            rank = int(m.group(1))
            if m.group(2):
                score = float(m.group(2))

        tile_token = None
        m = re.search(r"(?:切|ド)\s*([^\s=>,，]+)", line)
        if m:
            tile_token = m.group(1)
        else:
            m = re.search(
                r"(?:cut|discard)\s*([0-9][mps]|0[mps]|[1-7]z|[ESWNPFC])",
                line,
                re.IGNORECASE,
            )
            if m:
                tile_token = m.group(1)

        if not tile_token:
            return None

        tile = self._parse_tile_token(tile_token)
        if tile is None:
            return None

        call_type = self._detect_call_type(line)
        yaku_tags = self._extract_yaku_tags(line)
        no_yaku = "无役" in line or "无役" in yaku_tags
        return DiscardCandidate(
            tile=tile,
            score=score,
            rank=rank,
            raw_line=line,
            call_type=call_type,
            shanten=None,
            no_yaku=no_yaku,
            yaku_tags=yaku_tags,
        )

    def _parse_tile_token(self, token: str) -> Optional[str]:
        token = token.strip().translate(FULLWIDTH_DIGITS)
        token = token.strip(" ,，。. \t")
        if not token:
            return None

        m = re.match(r"^([0-9])([mpsz])$", token, re.IGNORECASE)
        if m:
            return self.helper_tile_to_mjai(m.group(1) + m.group(2).lower())

        m = re.match(r"^([1-7])z$", token, re.IGNORECASE)
        if m:
            return self.helper_tile_to_mjai(m.group(1) + "z")

        token_upper = token.upper()
        if token_upper in HONOR_TO_HELPER:
            return token_upper

        helper = self._tile_from_chinese(token)
        if helper:
            return self.helper_tile_to_mjai(helper)
        return None

    def _tile_from_chinese(self, token: str) -> Optional[str]:
        for ch in token:
            if ch in CHINESE_HONOR_TO_HELPER:
                return CHINESE_HONOR_TO_HELPER[ch]

        suit = None
        for ch, suit_char in CHINESE_SUIT_TO_HELPER.items():
            if ch in token:
                suit = suit_char
                break

        if suit is None:
            return None

        m = re.search(r"[0-9]", token)
        if not m:
            return None
        num = m.group(0)
        if ("赤" in token or "红" in token or "紅" in token) and num == "5":
            num = "0"
        return num + suit

    def _detect_call_type(self, line: str) -> Optional[str]:
        line_lower = line.lower()
        if "吃" in line or "chi" in line_lower:
            return "chi"
        if "碰" in line or "pon" in line_lower:
            return "pon"
        if "杠" in line or "槓" in line or "kan" in line_lower:
            return "daiminkan"
        return None

    def _header_shanten(self, line: str) -> Optional[int]:
        stripped = line.strip()
        if not stripped:
            return None
        if not (stripped.endswith("：") or stripped.endswith(":")):
            return None
        for word, value in SHANTEN_WORDS.items():
            if word in stripped:
                return value
        return None

    def _extract_yaku_tags(self, line: str) -> list[str]:
        tags: set[str] = set()
        for group in BRACKET_RE.findall(line):
            group = group.strip()
            if not group:
                continue
            if "无役" in group:
                tags.add("无役")
                continue
            if "宝牌" in group:
                tags.add("宝牌")
            for token in YAKU_TOKENS:
                if token in group:
                    tags.add(token)
        return sorted(tags)

    def _best_shanten_from_candidates(
        self,
        candidates: list[DiscardCandidate],
        lines: list[str],
    ) -> Optional[int]:
        shanten_values = [c.shanten for c in candidates if c.shanten is not None]
        if shanten_values:
            return min(shanten_values)
        for line in lines:
            header = self._header_shanten(line)
            if header is not None:
                return header
        return None

    def _select_best_tile(self, candidates: list[DiscardCandidate]) -> Optional[str]:
        best = self._select_best_candidate(candidates)
        return best.tile if best else None

    def _select_best_candidate(
        self,
        candidates: list[DiscardCandidate],
    ) -> Optional[DiscardCandidate]:
        if candidates:
            return candidates[0]
        return None

    def _format_analysis_text(
        self,
        lines: list[str],
        candidates: list[DiscardCandidate],
        max_count: int = 3,
        best_shanten: Optional[int] = None,
    ) -> Optional[str]:
        header = None
        for line in lines:
            if "向听" in line or "听牌" in line:
                if "：" in line or ":" in line:
                    header = line.strip()
                    break

        if not candidates:
            return header

        ranked = candidates[:]
        if best_shanten is not None:
            filtered = [c for c in ranked if c.shanten == best_shanten]
            if filtered:
                ranked = filtered
        scored = [c for c in ranked if c.score is not None]
        if scored:
            ranked = sorted(scored, key=lambda c: c.score, reverse=True)

        parts = []
        if header:
            parts.append(header)
        summary = []
        for cand in ranked[:max_count]:
            if cand.score is not None:
                summary.append(f"{cand.tile} ({cand.score:.2f})")
            else:
                summary.append(cand.tile)
        if summary:
            parts.append("Top discards: " + ", ".join(summary))
        return " | ".join(parts) if parts else None

    @staticmethod
    def mjai_tile_to_helper(tile: str) -> Optional[str]:
        if not tile or tile == "?":
            return None
        if len(tile) >= 2 and tile[1] in ("m", "p", "s"):
            if tile.endswith("r") and tile[0] == "5":
                return f"0{tile[1]}"
            if tile[0].isdigit():
                return tile[0] + tile[1]
        if tile in HONOR_TO_HELPER:
            return HONOR_TO_HELPER[tile]
        return None

    @staticmethod
    def helper_tile_to_mjai(tile: str) -> Optional[str]:
        if not tile:
            return None
        tile = tile.lower()
        if len(tile) != 2:
            return None
        num, suit = tile[0], tile[1]
        if suit in ("m", "p", "s"):
            if num == "0":
                return f"5{suit}r"
            return f"{num}{suit}"
        if suit == "z":
            return HELPER_TO_HONOR.get(tile)
        return None

    @staticmethod
    def meld_to_helper(tiles: Iterable[str], concealed: bool) -> Optional[str]:
        helper_tiles = []
        suit = None
        for tile in tiles:
            helper_tile = MahjongHelperEngine.mjai_tile_to_helper(tile)
            if helper_tile is None:
                continue
            if suit is None:
                suit = helper_tile[1]
            elif helper_tile[1] != suit:
                return None
            helper_tiles.append(int(helper_tile[0]))

        if not helper_tiles or suit is None:
            return None
        helper_tiles.sort(key=lambda d: 5 if d == 0 else d)
        suit_char = suit.upper() if concealed else suit
        return "".join(str(d) for d in helper_tiles) + suit_char

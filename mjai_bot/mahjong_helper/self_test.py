from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mjai_bot.mahjong_helper.engine import MahjongHelperEngine

SAMPLE_OUTPUT = """日本麻将助手 0.2.8 (by EndlessCheng)
34568m 5678p 23567s
===================
一向听：
27[27.62] 切5饼 =>  5.78听牌数 [27.13速度] [平和 断幺] [3678m 1234s]
27[27.62] 切8饼 =>  5.78听牌数 [27.13速度] [平和 断幺] [3678m 1234s]
20[25.75] 切8万 =>  7.20听牌数 [27.82速度] [平和 断幺] [36m 58p 14s]
...
"""


def test_parse_sample_output() -> None:
    engine = MahjongHelperEngine()
    analysis = engine.parse_discard_output(SAMPLE_OUTPUT)
    assert analysis.best_tile in ("5p", "8p")
    assert len(analysis.candidates) > 0


def test_smoke_if_available() -> None:
    engine = MahjongHelperEngine()
    try:
        engine.ensure_available()
    except RuntimeError:
        print("mahjong-helper not available; skipping smoke test")
        return

    analysis = engine.analyze_discard(
        hand_tiles=["3m", "4m", "5mr", "6m", "8m", "5p", "6p", "7p", "8p", "2s", "3s", "5s", "6s", "7s"],
        melds=None,
    )
    assert analysis.best_tile is not None


if __name__ == "__main__":
    test_parse_sample_output()
    test_smoke_if_available()
    print("mahjong_helper self-test OK")

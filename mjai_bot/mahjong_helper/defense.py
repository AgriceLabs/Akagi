from __future__ import annotations

from dataclasses import dataclass

HONOR_TILES = {"E", "S", "W", "N", "P", "F", "C"}


@dataclass
class DefenseContext:
    threat_levels: dict[int, float]
    discards_by_player: dict[int, list[str]]
    tiles_seen: dict[str, int]
    dora_tiles: set[str]


def normalize_tile(tile: str) -> str:
    return tile.replace("r", "")


def tile_num_suit(tile: str) -> tuple[int | None, str | None]:
    if tile in HONOR_TILES:
        return None, None
    if len(tile) >= 2 and tile[1] in ("m", "p", "s"):
        num = tile[0]
        if num == "0":
            num = "5"
        if num.isdigit():
            return int(num), tile[1]
    return None, None


def is_genbutsu(tile: str, discards: list[str]) -> bool:
    base = normalize_tile(tile)
    return base in [normalize_tile(t) for t in discards]


def is_suji_safe(tile: str, discards: list[str]) -> bool:
    num, suit = tile_num_suit(tile)
    if num is None or suit is None:
        return False
    for disc in discards:
        disc_num, disc_suit = tile_num_suit(disc)
        if disc_num is None or disc_suit != suit:
            continue
        if abs(disc_num - num) == 3:
            return True
    return False


def is_kabe_safe(tile: str, tiles_seen: dict[str, int]) -> bool:
    num, suit = tile_num_suit(tile)
    if num is None or suit is None:
        return False
    if num == 1:
        return tiles_seen.get(f"2{suit}", 0) >= 4
    if num == 9:
        return tiles_seen.get(f"8{suit}", 0) >= 4
    left_wall = tiles_seen.get(f"{num - 1}{suit}", 0) >= 4
    right_wall = tiles_seen.get(f"{num + 1}{suit}", 0) >= 4
    return left_wall or right_wall


def tile_danger(tile: str, ctx: DefenseContext) -> float:
    if not ctx.threat_levels:
        return 0.0

    base_tile = normalize_tile(tile)
    danger = 0.0
    for player, threat in ctx.threat_levels.items():
        discards = ctx.discards_by_player.get(player, [])
        if is_genbutsu(base_tile, discards):
            player_danger = 0.0
        else:
            player_danger = 1.0
            if base_tile in HONOR_TILES:
                player_danger *= 0.9
            if is_suji_safe(base_tile, discards):
                player_danger *= 0.75
            if is_kabe_safe(base_tile, ctx.tiles_seen):
                player_danger *= 0.8
            if base_tile in ctx.dora_tiles:
                player_danger *= 1.25
        player_danger *= threat
        danger = max(danger, player_danger)

    return min(danger, 3.0)

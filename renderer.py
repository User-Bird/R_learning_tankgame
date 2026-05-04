"""
renderer.py  ─  Phase 2B: All pygame drawing
──────────────────────────────────────────────
Zero game logic.  Reads a TankGame instance and a pygame.Surface,
translates tile coords → pixels, draws everything.

Public entry point:
    draw_game(surf, game, tile=32)
"""

import math
import pygame

from game import (
    COLS, ROWS, SPAWN1, SPAWN2,
    EMPTY, WALL, CHARGE_TILE,
    UP, RIGHT, DOWN, LEFT,
    MAX_HEALTH, MAX_AMMO, MAX_MINES, CHARGE_TICKS,
)

# ── Palette (identical to test_game.py) ─────────────────────────────────────────
C_ARENA_BG    = (18,  20,  26)
C_WALL        = (44,  52,  70)
C_WALL_BORDER = (60,  72,  98)
C_GRID        = (22,  26,  34)
C_SPAWN_1     = (40,  80,  50)
C_SPAWN_2     = (80,  40,  40)
C_CHARGE_TILE = (60,  50,  10)
C_CHARGE_GLOW = (220, 190,  40)

C_P1          = (80,  220, 120)
C_P1_DARK     = (40,  140,  70)
C_P2          = (230,  80,  80)
C_P2_DARK     = (160,  40,  40)

C_BULLET_P1   = (160, 255, 180)
C_BULLET_P2   = (255, 160, 140)

C_MINE_P1     = (180, 255,  50)
C_MINE_P2     = (255, 100, 200)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _dir_angle(direction):
    return {UP: 90, RIGHT: 0, DOWN: 270, LEFT: 180}[direction]


def _draw_arrow(surf, color, cx, cy, direction, size=10):
    rad  = math.radians(_dir_angle(direction))
    pts  = []
    for da, dist in [(0, size), (140, size * 0.6), (-140, size * 0.6)]:
        a = rad + math.radians(da)
        pts.append((cx + math.cos(a) * dist, cy - math.sin(a) * dist))
    pygame.draw.polygon(surf, color, pts)


# ── Per-object draw functions ──────────────────────────────────────────────────

def draw_tank(surf, tank, tile):
    cx = tank.x * tile + tile // 2
    cy = tank.y * tile + tile // 2
    body_c = C_P1 if tank.player_id == 1 else C_P2
    dark_c = C_P1_DARK if tank.player_id == 1 else C_P2_DARK

    # Shadow
    sh = pygame.Surface((tile - 4, tile - 4), pygame.SRCALPHA)
    sh.fill((0, 0, 0, 80))
    surf.blit(sh, (cx - tile // 2 + 3, cy - tile // 2 + 3))

    # Body
    body_rect = pygame.Rect(cx - tile // 2 + 1, cy - tile // 2 + 1, tile - 2, tile - 2)
    pygame.draw.rect(surf, dark_c, body_rect, border_radius=4)
    pygame.draw.rect(surf, body_c, body_rect.inflate(-4, -4), border_radius=3)

    # Facing arrow
    _draw_arrow(surf, (255, 255, 255), cx, cy, tank.direction, size=tile * 0.28)

    # Health pips
    pip_w, pip_gap = 5, 2
    total = MAX_HEALTH * (pip_w + pip_gap) - pip_gap
    px = cx - total // 2
    py = cy + tile // 2 + 3
    for i in range(MAX_HEALTH):
        color = body_c if i < tank.health else (40, 40, 50)
        pygame.draw.rect(surf, color, (px + i * (pip_w + pip_gap), py, pip_w, 4))

    # Ammo pips
    ammo_total = MAX_AMMO * (pip_w + pip_gap) - pip_gap
    apx = cx - ammo_total // 2
    apy = py + 7
    for i in range(MAX_AMMO):
        color = C_CHARGE_GLOW if i < tank.ammo else (40, 40, 50)
        pygame.draw.rect(surf, color, (apx + i * (pip_w + pip_gap), apy, pip_w, 3))

    # Charge progress arc
    if tank.charge_progress > 0:
        frac = tank.charge_progress / CHARGE_TICKS
        pygame.draw.arc(
            surf, C_CHARGE_GLOW,
            pygame.Rect(cx - tile // 2, cy - tile // 2, tile, tile),
            0, frac * 2 * math.pi, 3,
        )


def draw_bullet(surf, bullet, tile):
    cx = int(bullet.x * tile + tile // 2)
    cy = int(bullet.y * tile + tile // 2)
    color = C_BULLET_P1 if bullet.owner_id == 1 else C_BULLET_P2
    pygame.draw.circle(surf, color, (cx, cy), 5)
    pygame.draw.circle(surf, color, (cx, cy), 7, 1)


def draw_mine(surf, mine, tile, arena_w, arena_h):
    cx = int(mine.x * tile + tile // 2)
    cy = int(mine.y * tile + tile // 2)
    color = C_MINE_P1 if mine.owner_id == 1 else C_MINE_P2

    # Faint 5×5 warning ring
    r4x = max(0, (mine.x - 2) * tile)
    r4y = max(0, (mine.y - 2) * tile)
    r4w = min(arena_w, (mine.x + 3) * tile) - r4x
    r4h = min(arena_h, (mine.y + 3) * tile) - r4y
    warn = pygame.Surface((r4w, r4h), pygame.SRCALPHA)
    warn.fill((100, 100, 100, 12))
    surf.blit(warn, (r4x, r4y))
    pygame.draw.rect(surf, (100, 100, 100, 35),
                     pygame.Rect(r4x, r4y, r4w, r4h), 1)

    # 3×3 damage zone
    zx = max(0, (mine.x - 1) * tile)
    zy = max(0, (mine.y - 1) * tile)
    zw = min(arena_w, (mine.x + 2) * tile) - zx
    zh = min(arena_h, (mine.y + 2) * tile) - zy
    zone = pygame.Surface((zw, zh), pygame.SRCALPHA)
    zone.fill((*color, 22))
    surf.blit(zone, (zx, zy))
    pygame.draw.rect(surf, (*color, 55), pygame.Rect(zx, zy, zw, zh), 1)

    # Center icon
    pygame.draw.circle(surf, color, (cx, cy), 6)
    pygame.draw.circle(surf, (255, 255, 255), (cx, cy), 2)


def draw_charge_tile(surf, c, r, tile):
    """Draw the lightning-bolt icon on a charge tile."""
    mx = c * tile + tile // 2
    my = r * tile + tile // 2
    pts = [
        (mx,     my - 8),
        (mx - 4, my),
        (mx + 1, my),
        (mx,     my + 8),
        (mx + 4, my),
        (mx - 1, my),
    ]
    pygame.draw.lines(surf, C_CHARGE_GLOW, False, pts, 2)


# ── Main draw entry point ──────────────────────────────────────────────────────

def draw_game(surf: pygame.Surface, game, tile: int = 32):
    """
    Draw a complete TankGame frame onto surf.

    Parameters
    ----------
    surf : pygame.Surface   target surface (sized COLS*tile × ROWS*tile)
    game : TankGame         game instance to read state from
    tile : int              pixel size of one tile (default 32)
    """
    arena_w = COLS * tile
    arena_h = ROWS * tile

    surf.fill(C_ARENA_BG)

    # ── Tiles ──────────────────────────────────────────────────────────────────
    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(c * tile, r * tile, tile, tile)
            t    = game.grid[r][c]

            if t == WALL:
                pygame.draw.rect(surf, C_WALL,        rect)
                pygame.draw.rect(surf, C_WALL_BORDER, rect, 1)
            elif t == CHARGE_TILE:
                pygame.draw.rect(surf, C_CHARGE_TILE, rect)
                pygame.draw.rect(surf, C_CHARGE_GLOW, rect, 1)
                draw_charge_tile(surf, c, r, tile)
            else:
                pygame.draw.rect(surf, C_GRID, rect, 1)

    # ── Spawn zone tints ───────────────────────────────────────────────────────
    for (sx, sy), col in [(SPAWN1, C_SPAWN_1), (SPAWN2, C_SPAWN_2)]:
        tint = pygame.Surface((tile, tile), pygame.SRCALPHA)
        tint.fill((*col, 80))
        surf.blit(tint, (sx * tile, sy * tile))

    # ── Mines (drawn before bullets and tanks so they appear beneath) ──────────
    for m in game.active_mines:
        draw_mine(surf, m, tile, arena_w, arena_h)

    # ── Bullets ────────────────────────────────────────────────────────────────
    for b in game.bullets:
        draw_bullet(surf, b, tile)

    # ── Tanks ──────────────────────────────────────────────────────────────────
    if game.tank1.alive:
        draw_tank(surf, game.tank1, tile)
    if game.tank2.alive:
        draw_tank(surf, game.tank2, tile)
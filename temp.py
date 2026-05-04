"""
temp.py ─ Template Viewer for Report Screenshots
"""

import pygame
import sys
import math

# ── Arena Config ──────────────────────────────────────────────────────────────
TILE = 32
COLS = 25
ROWS = 19
ARENA_W = COLS * TILE
ARENA_H = ROWS * TILE
PANEL_W = 240  # Made narrower since we removed the text
WIN_W = ARENA_W + PANEL_W
WIN_H = ARENA_H

# ── Palette (Matching your game) ──────────────────────────────────────────────
C_ARENA_BG    = (18,  20,  26)
C_WALL        = (44,  52,  70)
C_WALL_BORDER = (60,  72,  98)
C_GRID        = (22,  26,  34)
C_PROTECTED   = (28,  34,  44) # Faint safety margin

C_PANEL_BG    = (14,  16,  22)
C_PANEL_LINE  = (36,  38,  54)
C_TEXT_PRI    = (210, 208, 200)
C_TEXT_DIM    = (60,  58,  54)

C_P1          = (80,  220, 120)
C_P1_DARK     = (40,  140,  70)
C_P2          = (230,  80,  80)
C_P2_DARK     = (160,  40,  40)
C_BULLET_P1   = (160, 255, 180)
C_BULLET_P2   = (255, 160, 140)
C_MINE_P1     = (180, 255,  50)
C_MINE_P2     = (255, 100, 200)
C_CHARGE_GLOW = (220, 190,  40)

C_BTN_IDLE    = (40, 100, 180)
C_BTN_HOVER   = (60, 120, 200)

UP = 0; RIGHT = 1; DOWN = 2; LEFT = 3
MAX_HEALTH = 5; MAX_AMMO = 5; CHARGE_TICKS = 2

# ── Templates ─────────────────────────────────────────────────────────────────
SPAWN_TEMPLATES = [
    [[2, 2, 2], [2, 1, 2], [2, 2, 2]],
    [[0, 2, 2], [0, 1, 2], [0, 1, 2], [0, 2, 2]],
    [[0, 2, 2], [0, 1, 2], [0, 1, 0], [0, 0, 0]],
    [[0, 0, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 0, 2]],
]

NON_SPAWN_TEMPLATES = [
    [[2, 2, 2, 2, 2], [0, 1, 1, 1, 2], [0, 2, 2, 1, 2], [0, 1, 2, 1, 2], [2, 0, 0, 0, 2]],
    [[2, 2, 2, 2, 2], [0, 1, 1, 1, 0], [0, 2, 2, 2, 0], [0, 1, 1, 1, 0], [2, 2, 2, 2, 2]],
    [[0, 0, 0], [2, 1, 2], [2, 1, 2], [0, 0, 0]],
]

ALL_TEMPLATES = SPAWN_TEMPLATES + NON_SPAWN_TEMPLATES
OFFSETS = [(2, 2), (8, 2), (14, 2), (20, 2), (2, 9), (10, 9), (18, 9)]

# ── Dummy Classes for Drawing ─────────────────────────────────────────────────
class DummyTank:
    def __init__(self, x, y, direction, pid):
        self.x = x; self.y = y; self.direction = direction; self.player_id = pid
        self.health = MAX_HEALTH; self.ammo = MAX_AMMO; self.charge_progress = 0

class DummyBullet:
    def __init__(self, x, y, pid):
        self.x = x; self.y = y; self.owner_id = pid

class DummyMine:
    def __init__(self, x, y, pid):
        self.x = x; self.y = y; self.owner_id = pid

# ── Drawing Functions ─────────────────────────────────────────────────────────
def _draw_arrow(surf, color, cx, cy, direction, size):
    rad = math.radians({UP: 90, RIGHT: 0, DOWN: 270, LEFT: 180}[direction])
    pts = []
    for da, d in [(0, size), (140, size * 0.6), (-140, size * 0.6)]:
        a = rad + math.radians(da)
        pts.append((cx + math.cos(a) * d, cy - math.sin(a) * d))
    pygame.draw.polygon(surf, color, pts)

def _draw_tank(surf, tank, tile, is_player):
    cx = tank.x * tile + tile // 2
    cy = tank.y * tile + tile // 2
    body_c = C_P1 if is_player else C_P2
    dark_c = C_P1_DARK if is_player else C_P2_DARK

    sh = pygame.Surface((tile - 4, tile - 4), pygame.SRCALPHA)
    sh.fill((0, 0, 0, 80))
    surf.blit(sh, (cx - tile // 2 + 3, cy - tile // 2 + 3))

    br = pygame.Rect(cx - tile // 2 + 1, cy - tile // 2 + 1, tile - 2, tile - 2)
    pygame.draw.rect(surf, dark_c, br, border_radius=4)
    pygame.draw.rect(surf, body_c, br.inflate(-4, -4), border_radius=3)

    # Original Main Game Arrow (White)
    _draw_arrow(surf, (255, 255, 255), cx, cy, tank.direction, tile * 0.28)

    pw, pg_ = 5, 2
    px = cx - (MAX_HEALTH * (pw + pg_) - pg_) // 2
    py = cy + tile // 2 + 3
    for i in range(MAX_HEALTH):
        c = body_c if i < tank.health else (40, 40, 50)
        pygame.draw.rect(surf, c, (px + i * (pw + pg_), py, pw, 4))

    apx = cx - (MAX_AMMO * (pw + pg_) - pg_) // 2
    apy = py + 7
    for i in range(MAX_AMMO):
        c = C_CHARGE_GLOW if i < tank.ammo else (40, 40, 50)
        pygame.draw.rect(surf, c, (apx + i * (pw + pg_), apy, pw, 3))

def draw_bullet(surf, bullet, tile):
    cx = int(bullet.x * tile + tile // 2)
    cy = int(bullet.y * tile + tile // 2)
    col = C_BULLET_P1 if bullet.owner_id == 1 else C_BULLET_P2
    pygame.draw.circle(surf, col, (cx, cy), 5)
    pygame.draw.circle(surf, col, (cx, cy), 7, 1)

def draw_mine(surf, mine, tile):
    cx_ = int(mine.x * tile + tile // 2)
    cy_ = int(mine.y * tile + tile // 2)
    color = C_MINE_P1 if mine.owner_id == 1 else C_MINE_P2

    zx = max(0, (mine.x - 1) * tile)
    zy = max(0, (mine.y - 1) * tile)
    zw = min(ARENA_W, (mine.x + 2) * tile) - zx
    zh = min(ARENA_H, (mine.y + 2) * tile) - zy
    zs = pygame.Surface((zw, zh), pygame.SRCALPHA)
    zs.fill((*color, 22))
    surf.blit(zs, (zx, zy))
    pygame.draw.rect(surf, (*color, 55), pygame.Rect(zx, zy, zw, zh), 1)
    pygame.draw.circle(surf, color, (cx_, cy_), 6)
    pygame.draw.circle(surf, (255, 255, 255), (cx_, cy_), 2)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Map Templates Viewer (Screenshot Mode)")

    font_md = pygame.font.SysFont("consolas", 16, bold=True)
    font_sm = pygame.font.SysFont("consolas", 14, bold=True)

    grid = [[-1 for _ in range(COLS)] for _ in range(ROWS)]
    for tmpl, (ox, oy) in zip(ALL_TEMPLATES, OFFSETS):
        for r, row in enumerate(tmpl):
            for c, val in enumerate(row):
                grid[oy + r][ox + c] = val

    # Setup dummy objects for the screenshot
    t1 = DummyTank(3, 16, UP, 1)
    t2 = DummyTank(7, 16, DOWN, 2)
    b1 = DummyBullet(11, 16, 1)
    b2 = DummyBullet(13, 16, 2)
    m1 = DummyMine(17, 16, 1)
    m2 = DummyMine(21, 16, 2)

    button_rect = pygame.Rect(ARENA_W + 20, ARENA_H // 2 - 25, PANEL_W - 40, 50)
    show_numbers = False

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if button_rect.collidepoint(mouse_pos):
                    show_numbers = not show_numbers

        screen.fill(C_ARENA_BG)

        # ── Draw Grid and Templates ──────────────────────────────────────────
        for r in range(ROWS):
            for c in range(COLS):
                rect = pygame.Rect(c * TILE, r * TILE, TILE, TILE)
                val = grid[r][c]

                if val == 1:
                    pygame.draw.rect(screen, C_WALL, rect)
                    pygame.draw.rect(screen, C_WALL_BORDER, rect, 1)
                elif val == 2:
                    pygame.draw.rect(screen, C_PROTECTED, rect)
                    pygame.draw.rect(screen, C_GRID, rect, 1)
                else:
                    pygame.draw.rect(screen, C_GRID, rect, 1)

                if show_numbers and val != -1:
                    col = C_P1 if val == 1 else ((180, 180, 240) if val == 2 else C_TEXT_DIM)
                    t_surf = font_sm.render(str(val), True, col)
                    screen.blit(t_surf, (rect.centerx - t_surf.get_width()//2,
                                         rect.centery - t_surf.get_height()//2))

        # ── Draw Entities ────────────────────────────────────────────────────
        _draw_tank(screen, t1, TILE, True)
        _draw_tank(screen, t2, TILE, False)
        draw_bullet(screen, b1, TILE)
        draw_bullet(screen, b2, TILE)
        draw_mine(screen, m1, TILE)
        draw_mine(screen, m2, TILE)

        # ── Draw Labels above Entities ───────────────────────────────────────
        labels = [
            ("PLAYER 1", 3, C_P1),
            ("PLAYER 2", 7, C_P2),
            ("BULLETS", 12, (255, 255, 255)),
            ("MINES", 19, (255, 255, 255))
        ]
        for text, grid_x, color in labels:
            surf = font_sm.render(text, True, color)
            px = grid_x * TILE + TILE // 2 - surf.get_width() // 2
            py = 14 * TILE + 10 # Draw slightly above row 16 (where objects are)
            screen.blit(surf, (px, py))

        # ── Draw Side Panel ──────────────────────────────────────────────────
        panel_rect = pygame.Rect(ARENA_W, 0, PANEL_W, ARENA_H)
        pygame.draw.rect(screen, C_PANEL_BG, panel_rect)
        pygame.draw.line(screen, C_PANEL_LINE, (ARENA_W, 0), (ARENA_W, ARENA_H), 2)

        # Button
        hover = button_rect.collidepoint(mouse_pos)
        btn_color = C_BTN_HOVER if hover else C_BTN_IDLE
        pygame.draw.rect(screen, btn_color, button_rect, border_radius=6)
        pygame.draw.rect(screen, C_TEXT_PRI, button_rect, 1, border_radius=6)

        btn_txt = "HIDE MATRIX NUMBERS" if show_numbers else "SHOW MATRIX NUMBERS"
        btn_surf = font_md.render(btn_txt, True, (255, 255, 255))
        screen.blit(btn_surf, (button_rect.centerx - btn_surf.get_width()//2,
                               button_rect.centery - btn_surf.get_height()//2))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
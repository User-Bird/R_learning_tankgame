"""
test_game.py  ─  Human vs Agent (or Random Bot) debug tool
────────────────────────────────────────────────────────────
Pick any saved .pt agent OR the built-in random bot to play
against.

Controls (Twin-Stick Style)
  W A S D    move  (multi-key, all held keys are checked each frame)
  MOUSE      aim
  R-CLICK    shoot          (Hold to auto-fire when ready)
  L-CLICK    drop mine      (instant keypress)
  R          full reset
  ESC        quit

Run:  python test_game.py
"""

import os, sys, glob, math, random
import numpy as np
import pygame
import torch

# ── Engine + RL ──────────────────────────────────────────────────────────────
import game as game_module

# OVERRIDES for human playability
game_module.MAX_TICKS = 999999
game_module.MINE_SPAWN_LOCKOUT = 0  # Let humans drop mines immediately

from game import (
    TankGame,
    COLS, ROWS, SPAWN1, SPAWN2,
    EMPTY, WALL, CHARGE_TILE,
    UP, RIGHT, DOWN, LEFT, DX, DY,
    MAX_HEALTH, MAX_AMMO, MAX_MINES, CHARGE_TICKS,
    SHOOT_COOLDOWN, MINE_MIN_SPACING,
)
from rl.state_encoder import encode_state
from rl.trainer import Trainer

MAX_TICKS = game_module.MAX_TICKS

# ─────────────────────────────────────────────────────────────────────────────
#  Layout constants
# ─────────────────────────────────────────────────────────────────────────────
AGENTS_DIR  = "saved_agents"
TILE        = 32
ARENA_W     = COLS * TILE          # 800
ARENA_H     = ROWS * TILE          # 608
HUD_W       = 340
INFO_H      = 30
WIN_W       = ARENA_W + HUD_W      # 1140
WIN_H       = ARENA_H + INFO_H     # 638
RESET_DELAY = 90                   # frames to display result before new ep

# ─────────────────────────────────────────────────────────────────────────────
#  Human-control cooldowns  (frames @ 60 fps)
# ─────────────────────────────────────────────────────────────────────────────
MOVE_CD   = 6    # ~100 ms between consecutive forward steps
ROTATE_CD = 6    # ~100 ms between bot rotations

# ─────────────────────────────────────────────────────────────────────────────
#  Palette
# ─────────────────────────────────────────────────────────────────────────────
C_BG          = (10,  12,  16)
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
C_HUD_BG      = (14,  16,  22)
C_HUD_LINE    = (36,  38,  54)
C_TEXT_PRI    = (210, 208, 200)
C_TEXT_SEC    = (130, 128, 118)
C_TEXT_DIM    = (60,  58,  54)
C_WARNING_BG  = (30,  20,  10)
C_WARNING_BD  = (220, 160,  40)
C_WARNING_TXT = (220, 180,  60)

DIR_NAMES    = {UP: "UP", RIGHT: "RIGHT", DOWN: "DOWN", LEFT: "LEFT"}
ACTION_NAMES = ["Rot L", "Rot R", "Move", "Shoot", "Stay", "Mine"]
ACTION_COLS  = [
    (100, 180, 240),   # Rot Left  – blue
    (100, 240, 180),   # Rot Right – teal
    (240, 220,  80),   # Move Fwd  – yellow
    (240,  80,  80),   # Shoot     – red
    (120, 120, 140),   # Stay      – grey
    (220, 100, 240),   # Mine      – purple
]


# ═════════════════════════════════════════════════════════════════════════════
#  Agent picker screen
# ═════════════════════════════════════════════════════════════════════════════

def _list_agents():
    os.makedirs(AGENTS_DIR, exist_ok=True)
    return sorted(os.path.basename(f)
                  for f in glob.glob(os.path.join(AGENTS_DIR, "*.pt")))


def pick_opponent(screen, fonts):
    """
    Shows a modal list: [Random Bot] + every .pt in saved_agents/.
    Returns None  → use random bot
    Returns path  → path to chosen .pt checkpoint
    """
    font_md, font_sm, font_xs = fonts
    W, H = screen.get_size()
    clock = pygame.time.Clock()

    agents  = _list_agents()
    items   = ["[Random Bot]"] + agents
    sel, scroll = 0, 0

    DW, DH  = 600, 580
    dx, dy  = (W - DW) // 2, (H - DH) // 2
    dlg     = pygame.Rect(dx, dy, DW, DH)

    # ── Movement info box (top area) ─────────────────────────────────────────
    INFO_H_BOX = 175
    list_top = dy + 64 + INFO_H_BOX + 10

    list_r  = pygame.Rect(dx + 14, list_top, DW - 28, DH - (list_top - dy) - 56)
    btn_ok  = pygame.Rect(dx + 14,             dy + DH - 46, (DW - 42) // 2, 34)
    btn_quit= pygame.Rect(btn_ok.right + 14,  dy + DH - 46, (DW - 42) // 2, 34)

    ROW_H = font_xs.get_linesize() + 6

    while True:
        mouse = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if ev.key == pygame.K_UP   and sel > 0:
                    sel -= 1
                if ev.key == pygame.K_DOWN and sel < len(items) - 1:
                    sel += 1
                if ev.key == pygame.K_RETURN:
                    if sel == 0:
                        return None
                    return os.path.join(AGENTS_DIR, items[sel])
            if ev.type == pygame.MOUSEBUTTONDOWN:
                if btn_ok.collidepoint(mouse):
                    if sel == 0:
                        return None
                    return os.path.join(AGENTS_DIR, items[sel])
                if btn_quit.collidepoint(mouse):
                    pygame.quit(); sys.exit()
                if list_r.collidepoint(mouse):
                    clicked = scroll + (mouse[1] - list_r.top) // ROW_H
                    if 0 <= clicked < len(items):
                        sel = clicked
            if ev.type == pygame.MOUSEWHEEL:
                visible = list_r.height // ROW_H
                scroll  = max(0, min(scroll - ev.y, max(0, len(items) - visible)))

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(C_BG)
        pygame.draw.rect(screen, (18, 18, 26), dlg, border_radius=10)
        pygame.draw.rect(screen, (40, 40, 60), dlg, 1, border_radius=10)

        ttl = font_md.render("SELECT OPPONENT", True, C_TEXT_PRI)
        screen.blit(ttl, (dlg.centerx - ttl.get_width() // 2, dy + 18))

        # Hint text
        hint_col = C_P1 if sel == 0 else C_TEXT_SEC
        hint     = "Random Bot" if sel == 0 else items[sel]
        ht = font_xs.render(f"Selected: {hint}", True, hint_col)
        screen.blit(ht, (dlg.centerx - ht.get_width() // 2, dy + 44))

        # ── Movement info box ─────────────────────────────────────────────────
        info_box = pygame.Rect(dx + 14, dy + 64, DW - 28, INFO_H_BOX)
        pygame.draw.rect(screen, C_WARNING_BG, info_box, border_radius=6)
        pygame.draw.rect(screen, C_WARNING_BD, info_box, 1, border_radius=6)

        bx = info_box.left + 10
        iy = info_box.top + 8

        hdr = font_sm.render("HOW TO PLAY / CONTROLS", True, C_WARNING_TXT)
        screen.blit(hdr, (bx, iy))
        iy += hdr.get_height() + 8

        lines = [
            ("IA (Agent) :   ", C_P2, "Se déplace comme un tank classique. Il doit pivoter"),
            ("               ", C_TEXT_DIM, "avant d'avancer. Impossible de se déplacer latéralement."),
            ("VOUS (Joueur) :", C_P1, "Style 'Twin-stick'. W/A/S/D vous déplace instantanément"),
            ("               ", C_TEXT_DIM, "dans cette direction. La souris dirige le canon de"),
            ("               ", C_TEXT_DIM, "façon autonome. Clic-D: Tirer, Clic-G: Poser une mine."),
            ("", (0, 0, 0), ""),  # Spacer
            ("⚠ ATTENTION :  ", (255, 100, 100), "Le moteur de base étant conçu pour des contrôles de tank,"),
            ("               ", (255, 100, 100), "ces mouvements 'twin-stick' peuvent sembler un peu saccadés !"),
        ]
        for label, lcol, desc in lines:
            if not label: # Handle spacer
                iy += 6
                continue

            lt = font_xs.render(label, True, lcol)
            dt = font_xs.render(desc,  True, C_TEXT_SEC)
            screen.blit(lt, (bx, iy))
            screen.blit(dt, (bx + lt.get_width(), iy))
            iy += lt.get_height() + 2

        # List box
        pygame.draw.rect(screen, (12, 12, 18), list_r, border_radius=4)
        pygame.draw.rect(screen, (36, 38, 54), list_r, 1, border_radius=4)
        visible = list_r.height // ROW_H
        for i, name in enumerate(items[scroll: scroll + visible]):
            ri  = i + scroll
            rr  = pygame.Rect(list_r.left + 2,
                              list_r.top + i * ROW_H,
                              list_r.width - 4, ROW_H)
            if ri == sel:
                pygame.draw.rect(screen, (60, 120, 200), rr, border_radius=3)
            col = C_P1 if ri == 0 else (C_TEXT_PRI if ri == sel else C_TEXT_SEC)
            t   = font_xs.render(name, True, col)
            screen.blit(t, (rr.left + 6, rr.top + 3))

        # Buttons
        for btn, label, col in [
            (btn_ok,   "Play",  (40, 160, 80)),
            (btn_quit, "Quit",  (160, 40, 40)),
        ]:
            hover = btn.collidepoint(mouse)
            c     = tuple(min(255, v + 25) for v in col) if hover else col
            pygame.draw.rect(screen, c,          btn, border_radius=6)
            pygame.draw.rect(screen, (40, 40, 60), btn, 1, border_radius=6)
            lbl = font_sm.render(label, True, C_TEXT_PRI)
            screen.blit(lbl, (btn.centerx - lbl.get_width() // 2,
                               btn.centery - lbl.get_height() // 2))

        pygame.display.flip()
        clock.tick(30)


# ═════════════════════════════════════════════════════════════════════════════
#  Session wrapper
# ═════════════════════════════════════════════════════════════════════════════

class TestSession:
    """
    Wraps TankGame with:
      - cross-episode score tracking
      - DQN agent for P2 (or random bot fallback)
      - last_q:  (6,) numpy Q-value array, None for random bot
      - state2:  raw state dict for P2 (used for AI Brain display)
    """

    def __init__(self, agent_path: str | None):
        self.game    = TankGame()
        self.is_dqn  = agent_path is not None
        self.trainer = None
        self.last_q  = None      # Q-values of the last AI decision
        self.agent_name = "Random Bot"

        # Bot constraints
        self.bot_move_cd = 0
        self.bot_rotate_cd = 0

        if self.is_dqn:
            self.trainer = Trainer()
            self.trainer.load_checkpoint(agent_path)
            self.trainer.epsilon = 0.0   # pure exploitation — no random moves
            self.agent_name = os.path.basename(agent_path)

        # Cross-episode score
        self.wins_p1  = self.wins_p2  = 0
        self.kills_p1 = self.kills_p2 = 0
        self.deaths_p1= self.deaths_p2= 0

        self.result_timer = 0
        self.result_text  = ""
        s1, s2 = self.game.reset()
        self.state2 = s2           # last known AI state dict

    # ── AI decision ──────────────────────────────────────────────────────────

    def _ai_action(self) -> int:
        if self.is_dqn:
            enc = encode_state(self.state2)
            t   = torch.tensor(enc, dtype=torch.float32,
                               device=self.trainer.device).unsqueeze(0)
            with torch.no_grad():
                q = self.trainer.online_net(t).squeeze(0).cpu().numpy()
            self.last_q = q
            return int(q.argmax())
        else:
            self.last_q = None
            pool = [0, 1, 2, 4]
            tank2 = self.game.tank2
            if tank2.can_shoot():
                pool += [3, 3]
            if tank2.mines > 0:
                pool += [5]
            return random.choice(pool)

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, player_action: int) -> bool:
        """Advance one tick. Returns True if episode just ended."""

        # 1. Ask the AI what it wants to do
        a2 = self._ai_action()

        # 2. Nerf the AI by applying cooldowns
        if a2 in [0, 1]:  # Rotate Left or Right
            if self.bot_rotate_cd > 0:
                a2 = 4    # Force "Stay" if on cooldown
            else:
                self.bot_rotate_cd = ROTATE_CD
        elif a2 == 2:     # Move
            if self.bot_move_cd > 0:
                a2 = 4    # Force "Stay" if on cooldown
            else:
                self.bot_move_cd = MOVE_CD

        # 3. Tick AI cooldowns down
        if self.bot_move_cd > 0: self.bot_move_cd -= 1
        if self.bot_rotate_cd > 0: self.bot_rotate_cd -= 1

        # 4. Advance the game
        (s1, s2), _rewards, done = self.game.step([player_action, a2])
        self.state2 = s2

        if done:
            txt = self.game.result_text
            self.result_text = txt
            if "1 WINS" in txt:
                self.wins_p1  += 1
                self.kills_p1 += 1
                self.deaths_p2+= 1
            elif "2 WINS" in txt:
                self.wins_p2  += 1
                self.kills_p2 += 1
                self.deaths_p1+= 1
            self.result_timer = RESET_DELAY

        return done

    def new_episode(self):
        s1, s2 = self.game.reset()
        self.state2 = s2
        self.last_q = None
        self.result_text = ""

    def full_reset(self):
        self.wins_p1 = self.wins_p2 = 0
        self.kills_p1= self.kills_p2= 0
        self.deaths_p1=self.deaths_p2=0
        self.result_timer = 0
        self.new_episode()


# ═════════════════════════════════════════════════════════════════════════════
#  Arena renderer
# ═════════════════════════════════════════════════════════════════════════════

def _draw_arrow(surf, color, cx, cy, direction, size):
    rad = math.radians({UP: 90, RIGHT: 0, DOWN: 270, LEFT: 180}[direction])
    pts = []
    for da, d in [(0, size), (140, size * 0.6), (-140, size * 0.6)]:
        a = rad + math.radians(da)
        pts.append((cx + math.cos(a) * d, cy - math.sin(a) * d))
    pygame.draw.polygon(surf, color, pts)


def _draw_tank(surf, tank, tile, is_player):
    cx     = tank.x * tile + tile // 2
    cy     = tank.y * tile + tile // 2
    body_c = C_P1 if is_player else C_P2
    dark_c = C_P1_DARK if is_player else C_P2_DARK

    sh = pygame.Surface((tile - 4, tile - 4), pygame.SRCALPHA)
    sh.fill((0, 0, 0, 80))
    surf.blit(sh, (cx - tile // 2 + 3, cy - tile // 2 + 3))

    br = pygame.Rect(cx - tile // 2 + 1, cy - tile // 2 + 1, tile - 2, tile - 2)
    pygame.draw.rect(surf, dark_c, br, border_radius=4)
    pygame.draw.rect(surf, body_c, br.inflate(-4, -4), border_radius=3)

    # Normal size black arrow with white outline for visibility
    _draw_arrow(surf, (255, 255, 255), cx, cy, tank.direction, tile * 0.28 + 2)
    _draw_arrow(surf, (0, 0, 0), cx, cy, tank.direction, tile * 0.28)

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

    if tank.charge_progress > 0:
        frac = tank.charge_progress / CHARGE_TICKS
        pygame.draw.arc(
            surf, C_CHARGE_GLOW,
            pygame.Rect(cx - tile // 2, cy - tile // 2, tile, tile),
            0, frac * 2 * math.pi, 3)


def draw_arena(surf, game):
    surf.fill(C_ARENA_BG)

    # Grid / walls / charge tiles
    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(c * TILE, r * TILE, TILE, TILE)
            t    = game.grid[r][c]
            if t == WALL:
                pygame.draw.rect(surf, C_WALL, rect)
                pygame.draw.rect(surf, C_WALL_BORDER, rect, 1)
            elif t == CHARGE_TILE:
                pygame.draw.rect(surf, C_CHARGE_TILE, rect)
                pygame.draw.rect(surf, C_CHARGE_GLOW, rect, 1)
                mx = c * TILE + TILE // 2
                my = r * TILE + TILE // 2
                pts = [(mx, my-8),(mx-4,my),(mx+1,my),(mx,my+8),(mx+4,my),(mx-1,my)]
                pygame.draw.lines(surf, C_CHARGE_GLOW, False, pts, 2)
            else:
                pygame.draw.rect(surf, C_GRID, rect, 1)

    # Spawn tints
    for (sx, sy), col in [(SPAWN1, C_SPAWN_1), (SPAWN2, C_SPAWN_2)]:
        tint = pygame.Surface((TILE, TILE), pygame.SRCALPHA)
        tint.fill((*col, 80))
        surf.blit(tint, (sx * TILE, sy * TILE))

    # Mines
    for m in game.active_mines:
        cx_   = int(m.x * TILE + TILE // 2)
        cy_   = int(m.y * TILE + TILE // 2)
        color = C_MINE_P1 if m.owner_id == 1 else C_MINE_P2
        # Danger zone overlay
        zx = max(0, (m.x - 1) * TILE)
        zy = max(0, (m.y - 1) * TILE)
        zw = min(ARENA_W, (m.x + 2) * TILE) - zx
        zh = min(ARENA_H, (m.y + 2) * TILE) - zy
        zs = pygame.Surface((zw, zh), pygame.SRCALPHA)
        zs.fill((*color, 22))
        surf.blit(zs, (zx, zy))
        pygame.draw.rect(surf, (*color, 55), pygame.Rect(zx, zy, zw, zh), 1)
        pygame.draw.circle(surf, color, (cx_, cy_), 6)
        pygame.draw.circle(surf, (255, 255, 255), (cx_, cy_), 2)

    # Bullets
    for b in game.bullets:
        cx_ = int(b.x * TILE + TILE // 2)
        cy_ = int(b.y * TILE + TILE // 2)
        col = C_BULLET_P1 if b.owner_id == 1 else C_BULLET_P2
        pygame.draw.circle(surf, col, (cx_, cy_), 5)
        pygame.draw.circle(surf, col, (cx_, cy_), 7, 1)

    # Tanks
    if game.tank1.alive: _draw_tank(surf, game.tank1, TILE, True)
    if game.tank2.alive: _draw_tank(surf, game.tank2, TILE, False)


def draw_result_overlay(surf, text, alpha_frac, font_lg, font_sm):
    ov = pygame.Surface((ARENA_W, ARENA_H), pygame.SRCALPHA)
    ov.fill((0, 0, 0, int(180 * alpha_frac)))
    surf.blit(ov, (0, 0))
    t1 = font_lg.render(text, True, (255, 255, 220))
    t2 = font_sm.render("Generating new map…", True, (160, 158, 150))
    cx, cy = ARENA_W // 2, ARENA_H // 2
    surf.blit(t1, (cx - t1.get_width() // 2, cy - t1.get_height()))
    surf.blit(t2, (cx - t2.get_width() // 2, cy + 8))


# ═════════════════════════════════════════════════════════════════════════════
#  HUD renderer
# ═════════════════════════════════════════════════════════════════════════════

def draw_hud(screen, session, hud_rect, font_md, font_sm, font_xs, move_cd: int):
    game = session.game
    pygame.draw.rect(screen, C_HUD_BG, hud_rect)
    pygame.draw.line(screen, C_HUD_LINE,
                     (hud_rect.left, hud_rect.top),
                     (hud_rect.left, hud_rect.bottom), 2)

    lx  = hud_rect.left + 12        # left text margin
    rx  = hud_rect.right - 12       # right margin (for bars)
    bar_max_w = rx - lx             # full bar width budget
    y   = 10

    def line(text, color=C_TEXT_SEC, gap=3, font=None):
        nonlocal y
        f  = font or font_xs
        t  = f.render(text, True, color)
        screen.blit(t, (lx, y))
        y += t.get_height() + gap

    def sep(gap=7):
        nonlocal y
        pygame.draw.line(screen, C_HUD_LINE, (lx, y), (rx, y), 1)
        y += gap

    def pip_row(label, filled, total, full_col, empty_col=(40, 40, 55)):
        """Draw a label + coloured pip squares."""
        nonlocal y
        lt = font_xs.render(label, True, C_TEXT_SEC)
        screen.blit(lt, (lx, y))
        px = lx + 78
        for i in range(total):
            c = full_col if i < filled else empty_col
            pygame.draw.rect(screen, c, (px + i * 13, y + 1, 10, 9), border_radius=2)
        y += lt.get_height() + 4

    # ── Title ─────────────────────────────────────────────────────────────────
    line("TEST GAME", C_TEXT_PRI, gap=1, font=font_md)
    opp_name = session.agent_name
    # Truncate long file names
    if len(opp_name) > 28:
        opp_name = opp_name[:25] + "…"
    line(f"vs {opp_name}", C_TEXT_DIM, gap=8)
    sep()

    # ── Player 1 ──────────────────────────────────────────────────────────────
    line("── YOU (P1) ──", C_P1, gap=5, font=font_sm)
    pip_row("  Health", game.tank1.health, MAX_HEALTH, C_P1)
    pip_row("  Ammo  ", game.tank1.ammo,   MAX_AMMO,   C_CHARGE_GLOW)
    pip_row("  Mines ", game.tank1.mines,  MAX_MINES,  C_MINE_P1)
    line(f"  Facing   {DIR_NAMES[game.tank1.direction]}", C_TEXT_SEC)

    scd = game.tank1.cooldown
    line(f"  Shoot CD {scd:>3}", (240, 80, 60) if scd > 0 else C_TEXT_DIM)
    mcd_col = C_CHARGE_GLOW if move_cd > 0 else C_TEXT_DIM
    line(f"  Move  CD {move_cd:>3}", mcd_col, gap=8)
    sep()

    # ── Bot ───────────────────────────────────────────────────────────────────
    bot_hdr = "── DQN BOT (P2) ──" if session.is_dqn else "── RANDOM BOT (P2) ──"
    line(bot_hdr, C_P2, gap=5, font=font_sm)
    pip_row("  Health", game.tank2.health, MAX_HEALTH, C_P2)
    pip_row("  Ammo  ", game.tank2.ammo,   MAX_AMMO,   C_CHARGE_GLOW)
    pip_row("  Mines ", game.tank2.mines,  MAX_MINES,  C_MINE_P2)
    line(f"  Facing   {DIR_NAMES[game.tank2.direction]}", C_TEXT_SEC, gap=8)
    sep()

    # ── Score ─────────────────────────────────────────────────────────────────
    line("SCORE", C_TEXT_PRI, gap=4, font=font_sm)
    line(f"  YOU  W:{session.wins_p1:<3}  K:{session.kills_p1:<3}  D:{session.deaths_p1}", C_P1)
    line(f"  BOT  W:{session.wins_p2:<3}  K:{session.kills_p2:<3}  D:{session.deaths_p2}", C_P2, gap=8)
    sep()

    # ── Game info ─────────────────────────────────────────────────────────────
    line("GAME INFO", C_TEXT_PRI, gap=4, font=font_sm)
    line(f"  Episode  {game.episode:>4}", C_TEXT_SEC)
    line(f"  Ticks    {game.ticks:>4}", C_TEXT_DIM)
    line(f"  Timeout  None (Free Play)", C_TEXT_DIM)
    line(f"  Bullets  {len(game.bullets):>4}", C_TEXT_DIM, gap=8)
    sep()

    # ── Controls ──────────────────────────────────────────────────────────────
    line("CONTROLS", C_TEXT_PRI, gap=4, font=font_sm)
    line("  W A S D   move", C_TEXT_DIM)
    line("  MOUSE     aim cannon", C_TEXT_DIM)
    line("  R-CLICK   shoot", C_TEXT_DIM)
    line("  L-CLICK   drop mine", C_MINE_P1)
    line("  R         full reset", C_TEXT_DIM)
    line("  ESC       quit", C_TEXT_DIM, gap=8)
    sep()

    # ── AI Brain ─────────────────────────────────────────────────────────────
    if not session.is_dqn:
        return

    line("AI BRAIN", C_TEXT_PRI, gap=5, font=font_sm)

    s = session.state2
    if s:
        # State variables
        dist  = s.get("distance_to_enemy", 0)
        los   = s.get("enemy_in_line_of_sight", False)
        angle = s.get("angle_to_enemy", 0.0)
        can_s = s.get("can_shoot", False)
        can_m = s.get("can_mine", False)
        on_ch = s.get("on_charge_tile", False)
        walls = s.get("walls_nearby", {})

        wf = "█" if walls.get("forward") else "░"
        wb = "█" if walls.get("back")    else "░"
        wl = "█" if walls.get("left")    else "░"
        wr_= "█" if walls.get("right")   else "░"

        angle_deg = int(angle * 180)

        line(f"  Dist to you  {dist:>3}", C_TEXT_SEC)
        line(f"  Angle off    {angle_deg:>3}°", C_TEXT_SEC)
        los_col = C_P2 if los else C_TEXT_DIM
        line(f"  Has LOS      {'YES' if los else 'no'}", los_col)
        cs_col  = C_P2 if can_s else C_TEXT_DIM
        cm_col  = C_MINE_P2 if can_m else C_TEXT_DIM
        line(f"  Can shoot    {'YES' if can_s else 'no'}", cs_col)
        line(f"  Can mine     {'YES' if can_m else 'no'}", cm_col)
        ch_col  = C_CHARGE_GLOW if on_ch else C_TEXT_DIM
        line(f"  On charger   {'YES' if on_ch else 'no'}", ch_col)
        line(f"  Walls  F:{wf} B:{wb} L:{wl} R:{wr_}", C_TEXT_DIM, gap=8)

    # Q-value bars
    line("Q-VALUES  (action weights)", C_TEXT_PRI, gap=6, font=font_sm)

    q = session.last_q
    if q is not None:
        best    = int(np.argmax(q))
        q_min   = q.min()
        q_max   = q.max()
        q_range = max(float(q_max - q_min), 1e-6)

        label_w = 46
        val_w   = 48
        bar_w   = bar_max_w - label_w - val_w - 4
        bar_h   = 13
        gap_    = 4

        for i, (name, color) in enumerate(zip(ACTION_NAMES, ACTION_COLS)):
            is_best  = (i == best)
            fill_w   = int((q[i] - q_min) / q_range * bar_w)
            fill_w   = max(0, min(fill_w, bar_w))
            bar_x    = lx + label_w
            dim_col  = tuple(max(0, c - 110) for c in color)

            # Label
            lbl_col = C_TEXT_PRI if is_best else C_TEXT_SEC
            lbl = font_xs.render(name, True, lbl_col)
            screen.blit(lbl, (lx, y + (bar_h - lbl.get_height()) // 2))

            # Bar background
            pygame.draw.rect(screen, (28, 28, 38),
                             (bar_x, y, bar_w, bar_h), border_radius=3)

            # Fill
            if fill_w > 0:
                fill_col = color if is_best else dim_col
                pygame.draw.rect(screen, fill_col,
                                 (bar_x, y, fill_w, bar_h), border_radius=3)

            # Border
            bdr = color if is_best else (50, 52, 68)
            pygame.draw.rect(screen, bdr,
                             (bar_x, y, bar_w, bar_h), 1, border_radius=3)

            # Star marker
            if is_best:
                star = font_xs.render("★", True, color)
                screen.blit(star, (bar_x + bar_w + val_w + 2,
                                   y + (bar_h - star.get_height()) // 2))

            # Numeric value
            val_str = f"{q[i]:+6.2f}"
            val_col = C_TEXT_PRI if is_best else C_TEXT_DIM
            vt = font_xs.render(val_str, True, val_col)
            screen.blit(vt, (bar_x + bar_w + 4,
                             y + (bar_h - vt.get_height()) // 2))

            y += bar_h + gap_
    else:
        line("  (waiting for first decision…)", C_TEXT_DIM)


def draw_info_bar(screen, info_rect, font_xs):
    pygame.draw.rect(screen, C_HUD_BG, info_rect)
    pygame.draw.line(screen, C_HUD_LINE,
                     info_rect.topleft, info_rect.topright, 1)
    tip = ("Test Game  │  WASD=Move (hold multiple)  Mouse=Aim  R-Click=Shoot  L-Click=Mine  R=Reset")
    t = font_xs.render(tip, True, C_TEXT_DIM)
    screen.blit(t, (12, info_rect.top + (INFO_H - t.get_height()) // 2))


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    pygame.init()
    pygame.display.set_caption("Combat Tank RL — Test Game")
    screen = pygame.display.set_mode((WIN_W, WIN_H))

    font_lg = pygame.font.SysFont("consolas", 34, bold=True)
    font_md = pygame.font.SysFont("consolas", 15, bold=True)
    font_sm = pygame.font.SysFont("consolas", 13, bold=True)
    font_xs = pygame.font.SysFont("consolas", 11)
    fonts   = (font_md, font_sm, font_xs)

    # ── Opponent picker ───────────────────────────────────────────────────────
    agent_path = pick_opponent(screen, fonts)
    session    = TestSession(agent_path)

    # ── Surfaces / rects ──────────────────────────────────────────────────────
    arena_surf = pygame.Surface((ARENA_W, ARENA_H))
    hud_rect   = pygame.Rect(ARENA_W, 0, HUD_W, ARENA_H)
    info_rect  = pygame.Rect(0, ARENA_H, WIN_W, INFO_H)

    # ── Per-frame human-control state ─────────────────────────────────────────
    move_cd   = 0

    warning_text  = ""
    warning_timer = 0

    clock   = pygame.time.Clock()
    running = True

    while running:
        # ── Determine mouse aim direction ─────────────────────────────────────
        mx, my = pygame.mouse.get_pos()
        tx = session.game.tank1.x * TILE + TILE // 2
        ty = session.game.tank1.y * TILE + TILE // 2

        dx = mx - tx
        dy = my - ty

        if abs(dx) > abs(dy):
            mouse_dir = RIGHT if dx > 0 else LEFT
        else:
            mouse_dir = DOWN if dy > 0 else UP

        player_action = 4   # default: stay

        # ── Event Loop (click-only events) ───────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    session.full_reset()
                    move_cd = 0
            if ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1 and not session.game.done:   # Left click -> Mine
                    t = session.game.tank1
                    active = sum(1 for m in session.game.active_mines if m.owner_id == 1)

                    if t.mines <= 0:
                        warning_text, warning_timer = "NO MINES LEFT IN INVENTORY!", 90
                    elif active >= MAX_MINES:
                        warning_text, warning_timer = f"MAX MINES ({MAX_MINES}) ALREADY ON FIELD!", 90
                    elif any(m.x == t.x and m.y == t.y for m in session.game.active_mines):
                        warning_text, warning_timer = "THERE IS ALREADY A MINE HERE!", 90
                    elif t.mine_cooldown > 0:
                        warning_text, warning_timer = "MINE PLANTING IS ON COOLDOWN!", 90
                    elif any(abs(t.x - m.x) + abs(t.y - m.y) < MINE_MIN_SPACING for m in session.game.active_mines):
                        warning_text, warning_timer = f"TOO CLOSE TO ANOTHER MINE! (Needs {MINE_MIN_SPACING} tiles clear)", 90
                    else:
                        player_action = 5

        if not running:
            break

        # ── Held actions: Shoot & Move ────────────────────────────────────────
        # Read ALL currently held keys/buttons at once — no elif blocking.
        mouse_btns = pygame.mouse.get_pressed()
        keys       = pygame.key.get_pressed()

        if player_action == 4 and not session.game.done:
            # Right-click shoot takes priority over movement
            if mouse_btns[2]:
                player_action = 3

            # Movement: check all four directions simultaneously.
            # Priority order when multiple keys are held: W > S > A > D
            # (feels natural; change order here if you prefer different priority)
            elif move_cd == 0:
                move_dir = None
                if   keys[pygame.K_w]: move_dir = UP
                elif keys[pygame.K_s]: move_dir = DOWN
                elif keys[pygame.K_a]: move_dir = LEFT
                elif keys[pygame.K_d]: move_dir = RIGHT

                if move_dir is not None:
                    # Set tank facing direction BEFORE the step so the engine
                    # moves in the right direction.
                    session.game.tank1.direction = move_dir
                    player_action = 2   # Move forward
                    move_cd = MOVE_CD

        # When not moving, snap the cannon to the mouse direction for aiming.
        if player_action != 2:
            session.game.tank1.direction = mouse_dir

        if move_cd > 0:
            move_cd -= 1

        # ── Game logic ────────────────────────────────────────────────────────
        if session.game.done:
            session.result_timer -= 1
            if session.result_timer <= 0:
                session.new_episode()
                move_cd = 0
        else:
            session.step(player_action)
            # After step, snap direction back to mouse for correct visual rendering
            session.game.tank1.direction = mouse_dir

        # ── Render ────────────────────────────────────────────────────────────
        screen.fill(C_BG)

        draw_arena(arena_surf, session.game)
        screen.blit(arena_surf, (0, 0))

        if session.game.done and session.result_text:
            alpha = min(1.0, (RESET_DELAY - session.result_timer) / 20.0)
            draw_result_overlay(screen, session.result_text,
                                alpha, font_lg, font_sm)

        draw_hud(screen, session, hud_rect, font_md, font_sm, font_xs, move_cd)
        draw_info_bar(screen, info_rect, font_xs)

        if warning_timer > 0:
            wt_surf = font_md.render(warning_text, True, (255, 100, 100))
            bx = ARENA_W // 2 - wt_surf.get_width() // 2
            by = 24
            pad = 8
            pygame.draw.rect(screen, (30, 10, 10),
                             (bx - pad, by - pad,
                              wt_surf.get_width() + pad*2,
                              wt_surf.get_height() + pad*2), border_radius=5)
            pygame.draw.rect(screen, (255, 100, 100),
                             (bx - pad, by - pad,
                              wt_surf.get_width() + pad*2,
                              wt_surf.get_height() + pad*2), 1, border_radius=5)
            screen.blit(wt_surf, (bx, by))
            warning_timer -= 1

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
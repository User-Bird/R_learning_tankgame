"""
agent_selector.py  —  Agent file picker + save-on-close logic
──────────────────────────────────────────────────────────────
Provides:
  show_session_mode_picker(screen, fonts)  →  ("NEW_VS_NEW" | "NEW_VS_AGENT" | "AGENT_VS_AGENT",
                                               agent_path_1_or_None, agent_path_2_or_None)

  save_agents(sessions, session_mode)  →  list of saved filenames

  show_save_and_close_dialog(screen, fonts, sessions, session_mode)
      →  list[str]  (saved filenames, 0-2 items)
      Shows a named-save dialog for each agent that should be saved, one at a
      time (P1 then P2 where applicable). Displays current ε, wins, and
      episodes next to the name input field.

What gets saved per mode
────────────────────────
  NEW_VS_NEW     → best P1 + best P2 (2 files — one dialog per player)
  NEW_VS_AGENT   → best P1 (new challenger) + best P2 (battle-hardened)
  AGENT_VS_AGENT → best P1 (agentAAA lineage) + best P2 (agentBB lineage)

Each .pt file is a full checkpoint dict:
  version, state_dict, epsilon, tick, episodes, win_rate, player, session_mode, saved_at

Call show_session_mode_picker() before starting the main loop.
Call save_agents() in your shutdown sequence after pygame.quit().
Call show_save_and_close_dialog() when the "Save & Close" button is pressed.
"""

import os
import glob
import datetime

import pygame
import torch
from game import generate_random_map, WALL, CHARGE_TILE, COLS, ROWS, SPAWN1, SPAWN2

# ── Palette (matches the rest of the project) ─────────────────────────────────
C_BG         = (10,  10,  14)
C_PANEL_BG   = (18,  18,  26)
C_BORDER     = (40,  40,  60)
C_TEXT_PRI   = (210, 208, 200)
C_TEXT_SEC   = (130, 128, 118)
C_TEXT_DIM   = (60,  58,  54)
C_HOVER      = (50,  50,  70)
C_SELECT     = (60,  120, 200)
C_BTN_NEW    = (40,  160,  80)
C_BTN_NVA    = (200, 130,  40)
C_BTN_AVA    = (160,  60, 200)
C_BTN_OK     = (40,  160,  80)
C_BTN_CANCEL = (160,  40,  40)
C_BTN_SAVE   = (40,  160, 200)   # distinct teal for the "Save & Close" button
C_STAT_GOOD  = (100, 220, 140)
C_STAT_WARN  = (220, 180,  60)

AGENTS_DIR   = "saved_agents"    # folder where .pt files live


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_agents_dir():
    os.makedirs(AGENTS_DIR, exist_ok=True)


def _list_agents() -> list[str]:
    """Return list of .pt filenames (basename only) in saved_agents/."""
    _ensure_agents_dir()
    files = glob.glob(os.path.join(AGENTS_DIR, "*.pt"))
    return sorted([os.path.basename(f) for f in files])


def _draw_btn(surf, font, text, rect, color, hover=False, disabled=False):
    if disabled:
        col = tuple(c // 3 for c in color)
    else:
        col = tuple(min(255, c + 30) for c in color) if hover else color
    pygame.draw.rect(surf, col, rect, border_radius=6)
    pygame.draw.rect(surf, C_BORDER, rect, 1, border_radius=6)
    t_col = C_TEXT_DIM if disabled else C_TEXT_PRI
    t = font.render(text, True, t_col)
    surf.blit(t, (rect.centerx - t.get_width()//2,
                  rect.centery - t.get_height()//2))


def _draw_list(surf, font, items, selected_idx, rect, scroll_offset):
    """Draw a scrollable file list inside rect. Returns the rects for each visible item."""
    pygame.draw.rect(surf, C_PANEL_BG, rect, border_radius=4)
    pygame.draw.rect(surf, C_BORDER,   rect, 1, border_radius=4)

    row_h   = font.get_linesize() + 6
    visible = rect.height // row_h
    item_rects = []

    for i, name in enumerate(items[scroll_offset: scroll_offset + visible]):
        real_idx = i + scroll_offset
        row_rect = pygame.Rect(rect.left + 2, rect.top + i * row_h,
                               rect.width - 4, row_h)
        if real_idx == selected_idx:
            pygame.draw.rect(surf, C_SELECT, row_rect, border_radius=3)
        t = font.render(name, True, C_TEXT_PRI if real_idx == selected_idx else C_TEXT_SEC)
        surf.blit(t, (row_rect.left + 6, row_rect.top + 3))
        item_rects.append((row_rect, real_idx))

    return item_rects


# ── Agent file picker ─────────────────────────────────────────────────────────

def _pick_agent(screen, fonts, title: str) -> str | None:
    """
    Shows a modal file-list dialog.
    Returns the full path to the chosen .pt file, or None if cancelled.
    """
    font_md, font_sm, font_xs = fonts
    W, H = screen.get_size()

    DW, DH = 420, 400
    dx = (W - DW) // 2
    dy = (H - DH) // 2
    dlg = pygame.Rect(dx, dy, DW, DH)

    agents      = _list_agents()
    selected    = 0 if agents else -1
    scroll      = 0
    list_rect   = pygame.Rect(dx + 14, dy + 60, DW - 28, DH - 120)
    btn_ok      = pygame.Rect(dx + 14,       dy + DH - 46, (DW - 42)//2, 34)
    btn_cancel  = pygame.Rect(btn_ok.right + 14, dy + DH - 46, (DW - 42)//2, 34)

    clock = pygame.time.Clock()
    while True:
        mouse = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_UP and selected > 0:
                    selected -= 1
                if event.key == pygame.K_DOWN and selected < len(agents) - 1:
                    selected += 1
                if event.key == pygame.K_RETURN and selected >= 0:
                    return os.path.join(AGENTS_DIR, agents[selected])
            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn_cancel.collidepoint(mouse):
                    return None
                if btn_ok.collidepoint(mouse) and selected >= 0:
                    return os.path.join(AGENTS_DIR, agents[selected])
                # click on list
                row_h    = font_xs.get_linesize() + 6
                visible  = list_rect.height // row_h
                if list_rect.collidepoint(mouse):
                    rel_y   = mouse[1] - list_rect.top
                    clicked = scroll + rel_y // row_h
                    if 0 <= clicked < len(agents):
                        selected = clicked
            if event.type == pygame.MOUSEWHEEL:
                scroll = max(0, min(scroll - event.y, max(0, len(agents) - 10)))

        # Draw
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))

        pygame.draw.rect(screen, C_BG, dlg, border_radius=10)
        pygame.draw.rect(screen, C_BORDER, dlg, 1, border_radius=10)

        ttl = font_md.render(title, True, C_TEXT_PRI)
        screen.blit(ttl, (dlg.centerx - ttl.get_width()//2, dy + 16))

        if not agents:
            nm = font_sm.render("No saved agents found in saved_agents/", True, C_TEXT_DIM)
            screen.blit(nm, (dlg.centerx - nm.get_width()//2, list_rect.centery))
        else:
            _draw_list(screen, font_xs, agents, selected, list_rect, scroll)

        _draw_btn(screen, font_sm, "Select",  btn_ok,     C_BTN_OK,
                  btn_ok.collidepoint(mouse))
        _draw_btn(screen, font_sm, "Cancel",  btn_cancel, C_BTN_CANCEL,
                  btn_cancel.collidepoint(mouse))

        pygame.display.flip()
        clock.tick(30)

# ── Fixed-map preview + picker ─────────────────────────────────────────────────

def _draw_map_preview(surf, grid, charge_tiles, rect):
    """
    Draw a miniature preview of a map inside rect.
    Walls are grey, charge tiles yellow, spawn zones tinted.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if cols == 0:
        return

    tile_sz = min(rect.width // cols, rect.height // rows)
    ox = rect.left + (rect.width  - tile_sz * cols) // 2
    oy = rect.top  + (rect.height - tile_sz * rows) // 2

    charge_set = set(map(tuple, charge_tiles))

    for r in range(rows):
        for c in range(cols):
            x = ox + c * tile_sz
            y = oy + r * tile_sz
            t = grid[r][c]
            if t == WALL:
                color = (55, 65, 88)
            elif (c, r) in charge_set:
                color = (80, 70, 15)
            else:
                color = (18, 20, 26)
            pygame.draw.rect(surf, color, (x, y, tile_sz, tile_sz))

    # Spawn tints
    for (sx, sy), col in [(SPAWN1, (40, 80, 50)), (SPAWN2, (80, 40, 40))]:
        pygame.draw.rect(surf, col,
                         (ox + sx * tile_sz, oy + sy * tile_sz, tile_sz, tile_sz))


def show_map_choice(screen, fonts):
    """
    Show a map-type selection screen.

    Returns
    -------
    (grid, charge_tiles)  →  train on this fixed map every episode
    None                  →  generate a fresh random map each episode
    """
    font_md, font_sm, font_xs = fonts
    W, H  = screen.get_size()
    clock = pygame.time.Clock()

    current_map = generate_random_map()

    DW, DH = 520, 440
    dx, dy = (W - DW) // 2, (H - DH) // 2
    dlg    = pygame.Rect(dx, dy, DW, DH)

    preview_rect = pygame.Rect(dx + 14, dy + 72, DW - 28, 230)

    btn_h  = 36
    btn_w  = (DW - 42) // 3
    btn_y  = dy + DH - 52
    btn_regen  = pygame.Rect(dx + 14,                   btn_y, btn_w, btn_h)
    btn_use    = pygame.Rect(dx + 14 + btn_w + 7,       btn_y, btn_w, btn_h)
    btn_random = pygame.Rect(dx + 14 + (btn_w + 7) * 2, btn_y, btn_w, btn_h)

    while True:
        mouse = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_r:
                    current_map = generate_random_map()
                if event.key == pygame.K_RETURN:
                    return current_map
            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn_regen.collidepoint(mouse):
                    current_map = generate_random_map()
                elif btn_use.collidepoint(mouse):
                    return current_map
                elif btn_random.collidepoint(mouse):
                    return None

        # ── Draw ─────────────────────────────────────────────────────────────
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))

        pygame.draw.rect(screen, C_BG,     dlg, border_radius=10)
        pygame.draw.rect(screen, C_BORDER, dlg, 1, border_radius=10)

        ttl = font_md.render("CHOOSE MAP TYPE", True, C_TEXT_PRI)
        screen.blit(ttl, (dlg.centerx - ttl.get_width() // 2, dy + 14))

        sub = font_xs.render(
            "Fixed: agents train on the same map every episode   "
            "  R = regenerate",
            True, C_TEXT_SEC)
        screen.blit(sub, (dlg.centerx - sub.get_width() // 2, dy + 48))

        # Map preview panel
        pygame.draw.rect(screen, C_PANEL_BG, preview_rect, border_radius=4)
        pygame.draw.rect(screen, C_BORDER,   preview_rect, 1, border_radius=4)
        grid, charge_tiles = current_map
        _draw_map_preview(screen, grid, charge_tiles, preview_rect.inflate(-6, -6))

        # Buttons
        _draw_btn(screen, font_sm, "Regenerate",   btn_regen,  C_BTN_AVA,
                  btn_regen.collidepoint(mouse))
        _draw_btn(screen, font_sm, "Use This Map", btn_use,    C_BTN_OK,
                  btn_use.collidepoint(mouse))
        _draw_btn(screen, font_sm, "Random Maps",  btn_random, C_BTN_CANCEL,
                  btn_random.collidepoint(mouse))

        pygame.display.flip()
        clock.tick(30)

# ── Session mode picker (main entry point) ────────────────────────────────────

def show_session_mode_picker(screen, fonts):
    """
    Shows the session-mode selection screen.

    Returns
    -------
    (session_mode, agent1_path, agent2_path)
      session_mode  :  "NEW_VS_NEW" | "NEW_VS_AGENT" | "AGENT_VS_AGENT"
      agent1_path   :  str path to .pt file, or None
      agent2_path   :  str path to .pt file, or None
    """
    font_md, font_sm, font_xs = fonts
    W, H = screen.get_size()
    clock = pygame.time.Clock()

    BW, BH = 340, 64
    bx     = W // 2 - BW // 2

    btn_nvn = pygame.Rect(bx, H//2 - 120, BW, BH)
    btn_nva = pygame.Rect(bx, H//2 -  40, BW, BH)
    btn_ava = pygame.Rect(bx, H//2 +  40, BW, BH)

    MODES = [
        (btn_nvn, "NEW_VS_NEW",       "  New vs New  (6 fresh agents)",         C_BTN_NEW),
        (btn_nva, "NEW_VS_AGENT",     "  New vs Agent  (pick opponent)",         C_BTN_NVA),
        (btn_ava, "AGENT_VS_AGENT",   "  Agent vs Agent  (pick both sides)",     C_BTN_AVA),
    ]

    desc = {
        "NEW_VS_NEW":     "6 games, all agents train from scratch.",
        "NEW_VS_AGENT":   "6 games: fresh P1 trains against a loaded P2.",
        "AGENT_VS_AGENT": "6 games: both sides loaded from saved files.",
    }

    while True:
        mouse = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.MOUSEBUTTONDOWN:
                for btn, mode, _, _ in MODES:
                    if btn.collidepoint(mouse):
                        if mode == "NEW_VS_NEW":
                            fixed_map = show_map_choice(screen, fonts)
                            return mode, None, None, fixed_map

                        elif mode == "NEW_VS_AGENT":
                            path = _pick_agent(screen, fonts,
                                               "Select P2 agent (.pt file)")
                            if path is None:
                                break
                            fixed_map = show_map_choice(screen, fonts)
                            return mode, None, path, fixed_map

                        elif mode == "AGENT_VS_AGENT":
                            p1 = _pick_agent(screen, fonts,
                                             "Select P1 agent (.pt file)")
                            if p1 is None:
                                break
                            p2 = _pick_agent(screen, fonts,
                                             "Select P2 agent (.pt file)")
                            if p2 is None:
                                break
                            # No map choice for agent-vs-agent: both are
                            # already trained, map type is irrelevant.
                            return mode, p1, p2, None

        # Draw picker screen
        screen.fill(C_BG)

        title = font_md.render("SELECT SESSION MODE", True, C_TEXT_PRI)
        screen.blit(title, (W//2 - title.get_width()//2, H//2 - 200))

        hovered_mode = None
        for btn, mode, label, color in MODES:
            hover = btn.collidepoint(mouse)
            if hover:
                hovered_mode = mode
            _draw_btn(screen, font_sm, label, btn, color, hover)

        # Description line
        if hovered_mode:
            d = font_xs.render(desc[hovered_mode], True, C_TEXT_DIM)
            screen.blit(d, (W//2 - d.get_width()//2, H//2 + 120))

        sub = font_xs.render("ESC or close window to quit", True, C_TEXT_DIM)
        screen.blit(sub, (W//2 - sub.get_width()//2, H - 40))

        pygame.display.flip()
        clock.tick(30)


# ── Internal: find best trainer per side ──────────────────────────────────────

def _find_best(sessions, player_id: int):
    """
    Among all sessions, find the trainer and stats for the best agent on
    player_id's side (1 or 2).

    'Best' = highest win-rate; ties broken by lower epsilon (more trained).

    Returns (session, trainer, total_ep, win_count, win_rate)  or  None.
    """
    best = None
    best_wr = -1.0

    for s in sessions:
        ep = s.episodes
        if ep == 0:
            continue

        if player_id == 1:
            wins = s.wins_p1
            trainer = s.trainer1
        else:
            wins = s.wins_p2
            trainer = s.trainer2

        wr = wins / ep

        # Prefer higher win-rate; break ties with lower epsilon
        if best is None or wr > best_wr or (
                wr == best_wr and trainer.epsilon < best[2].epsilon):
            best = (s, wins, trainer, ep, wr)
            best_wr = wr

    if best is None:
        return None
    s, wins, trainer, ep, wr = best
    return s, trainer, ep, wins, wr


def _save_one(trainer, ep: int, wins: int, wr: float,
              player_id: int, session_mode: str, tag: str) -> str:
    """
    Save a full checkpoint for one trainer.

    Filename:  saved_agents/agent_{tag}_p{player_id}_{date}_ep{N}.pt
    """
    _ensure_agents_dir()
    date  = datetime.datetime.now().strftime("%Y-%m-%d")
    fname = f"agent_{tag}_p{player_id}_{date}_ep{ep}.pt"
    fpath = os.path.join(AGENTS_DIR, fname)

    trainer.save_checkpoint(fpath, extra_info={
        "episodes":     ep,
        "wins":         wins,
        "win_rate":     wr,
        "player":       player_id,
        "session_mode": session_mode,
        "saved_at":     datetime.datetime.now().isoformat(),
    })
    print(f"[save] P{player_id} → {fpath}  "
          f"(win-rate {wr:.2%}, ε={trainer.epsilon:.4f}, ep={ep})")
    return fname


def _save_with_name(trainer, ep: int, wins: int, wr: float,
                    player_id: int, session_mode: str, name: str) -> str:
    """
    Save a full checkpoint with a user-supplied name.

    Filename:  saved_agents/{name}_p{player_id}_ep{N}_e{eps:.3f}_w{wins}.pt
    The stats are baked into the filename so you can see them in the picker
    without loading the file.
    """
    _ensure_agents_dir()
    eps   = trainer.epsilon
    # Sanitise: replace spaces/slashes with underscores, strip unsafe chars
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    safe_name = safe_name[:32]   # cap length
    fname = f"{safe_name}_p{player_id}_ep{ep}_e{eps:.3f}_w{wins}.pt"
    fpath = os.path.join(AGENTS_DIR, fname)

    trainer.save_checkpoint(fpath, extra_info={
        "episodes":     ep,
        "wins":         wins,
        "win_rate":     wr,
        "player":       player_id,
        "session_mode": session_mode,
        "saved_at":     datetime.datetime.now().isoformat(),
        "agent_name":   safe_name,
    })
    print(f"[save] '{safe_name}' P{player_id} → {fpath}  "
          f"(win-rate {wr:.2%}, ε={eps:.4f}, ep={ep})")
    return fname


# ── Internal: single-player named-save dialog ─────────────────────────────────

def _show_save_dialog_for_player(screen, fonts, sessions, session_mode: str,
                                  player_id: int) -> str | None:
    """
    Shows the naming dialog for one player's best agent.

    Returns the saved filename (basename) on success, or None if the user
    cancelled or there were no episodes played for this player.
    """
    font_md, font_sm, font_xs = fonts
    W, H = screen.get_size()

    # ── Find best for this player ─────────────────────────────────────────────
    result = _find_best(sessions, player_id)
    if result is None:
        _show_notice(screen, fonts, f"No episodes for P{player_id} — skipping.")
        return None

    _, trainer, ep, wins, wr = result
    eps = trainer.epsilon

    # ── Layout ────────────────────────────────────────────────────────────────
    DW, DH = 480, 290
    dx = (W - DW) // 2
    dy = (H - DH) // 2
    dlg = pygame.Rect(dx, dy, DW, DH)

    stats_top  = dy + 50
    label_y    = stats_top + 100
    input_rect = pygame.Rect(dx + 14, label_y + 22, DW - 28, 36)

    btn_save   = pygame.Rect(dx + 14,             dy + DH - 50,
                             (DW - 42) // 2, 34)
    btn_cancel = pygame.Rect(btn_save.right + 14, dy + DH - 50,
                             (DW - 42) // 2, 34)

    # ── State ─────────────────────────────────────────────────────────────────
    user_text = ""
    MAX_LEN   = 28
    clock     = pygame.time.Clock()

    while True:
        mouse    = pygame.mouse.get_pos()
        can_save = bool(user_text.strip())

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None

                elif event.key == pygame.K_RETURN and can_save:
                    return _save_with_name(
                        trainer, ep, wins, wr, player_id, session_mode,
                        user_text.strip()
                    )

                elif event.key == pygame.K_BACKSPACE:
                    user_text = user_text[:-1]

                elif len(user_text) < MAX_LEN:
                    ch = event.unicode
                    if ch and (ch.isalnum() or ch in " -_"):
                        user_text += ch

            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn_cancel.collidepoint(mouse):
                    return None
                if btn_save.collidepoint(mouse) and can_save:
                    return _save_with_name(
                        trainer, ep, wins, wr, player_id, session_mode,
                        user_text.strip()
                    )

        # ── Draw ──────────────────────────────────────────────────────────────
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        screen.blit(overlay, (0, 0))

        pygame.draw.rect(screen, C_BG,     dlg, border_radius=10)
        pygame.draw.rect(screen, C_BORDER, dlg, 1, border_radius=10)

        # Title
        ttl = font_md.render(f"SAVE BEST P{player_id} AGENT", True, C_TEXT_PRI)
        screen.blit(ttl, (dlg.centerx - ttl.get_width() // 2, dy + 14))

        # Divider
        pygame.draw.line(screen, C_BORDER,
                         (dx + 14, dy + 38), (dx + DW - 14, dy + 38), 1)

        # ── Stats block ───────────────────────────────────────────────────────
        col_label = dx + 24
        col_value = dx + 160
        sy = stats_top

        def _stat_row(label, value, value_color=C_TEXT_PRI):
            nonlocal sy
            tl = font_xs.render(label, True, C_TEXT_DIM)
            tv = font_sm.render(value, True, value_color)
            screen.blit(tl, (col_label, sy))
            screen.blit(tv, (col_value, sy - 1))
            sy += tl.get_height() + 8

        eps_col = C_STAT_GOOD if eps > 0.15 else C_STAT_WARN
        wr_col  = C_STAT_GOOD if wr >= 0.5  else C_STAT_WARN

        _stat_row("Episodes played:",    f"{ep:,}")
        _stat_row(f"Wins (P{player_id}):", f"{wins:,}  ({wr:.1%})", wr_col)
        _stat_row("Current  ε:",         f"{eps:.4f}",              eps_col)
        _stat_row("Tick count:",         f"{trainer._tick:,}")

        # ── Name input ────────────────────────────────────────────────────────
        lbl = font_xs.render("Name for this agent:", True, C_TEXT_SEC)
        screen.blit(lbl, (dx + 14, label_y))

        pygame.draw.rect(screen, C_PANEL_BG, input_rect, border_radius=5)
        pygame.draw.rect(screen, C_SELECT,   input_rect, 1, border_radius=5)

        cursor       = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else " "
        txt_surf     = font_sm.render(user_text + cursor, True, C_TEXT_PRI)
        screen.blit(txt_surf,
                    (input_rect.left + 8,
                     input_rect.centery - txt_surf.get_height() // 2))

        cc = font_xs.render(f"{len(user_text)}/{MAX_LEN}", True, C_TEXT_DIM)
        screen.blit(cc, (input_rect.right - cc.get_width() - 6,
                         input_rect.bottom + 3))

        # Filename preview
        if user_text.strip():
            safe = "".join(c if c.isalnum() or c in "-_" else "_"
                           for c in user_text.strip())[:32]
            preview = f"→ {safe}_p{player_id}_ep{ep}_e{eps:.3f}_w{wins}.pt"
        else:
            preview = "→ enter a name above"
        pv = font_xs.render(preview, True, C_TEXT_DIM)
        screen.blit(pv, (dx + 14, input_rect.bottom + 20))

        # ── Buttons ───────────────────────────────────────────────────────────
        _draw_btn(screen, font_sm, "Save & Close", btn_save,
                  C_BTN_SAVE, btn_save.collidepoint(mouse),
                  disabled=not can_save)
        _draw_btn(screen, font_sm, "Cancel",       btn_cancel,
                  C_BTN_CANCEL, btn_cancel.collidepoint(mouse))

        pygame.display.flip()
        clock.tick(30)


def _show_notice(screen, fonts, message: str, duration_ms: int = 2000):
    """Brief centred notice overlay — auto-dismisses after duration_ms."""
    font_md, font_sm, font_xs = fonts
    W, H = screen.get_size()
    start = pygame.time.get_ticks()
    clock = pygame.time.Clock()

    while pygame.time.get_ticks() - start < duration_ms:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                return

        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))

        t = font_sm.render(message, True, C_TEXT_PRI)
        bx = W // 2 - t.get_width() // 2 - 16
        by = H // 2 - t.get_height() // 2 - 10
        bw = t.get_width() + 32
        bh = t.get_height() + 20
        pygame.draw.rect(screen, C_BG,     (bx, by, bw, bh), border_radius=8)
        pygame.draw.rect(screen, C_BORDER, (bx, by, bw, bh), 1, border_radius=8)
        screen.blit(t, (W // 2 - t.get_width() // 2, H // 2 - t.get_height() // 2))
        pygame.display.flip()
        clock.tick(30)


# ── Public: Save & Close dialog ───────────────────────────────────────────────

def show_save_and_close_dialog(screen, fonts, sessions,
                                session_mode: str) -> list[str]:
    """
    Show a named-save dialog for each agent that should be saved, one at a
    time (P1 first, then P2).  All modes now save both players.

    Call this when the "Save & Close" button is pressed in main.py.
    After it returns, set running = False to exit the main loop.

    Returns
    -------
    list[str]  – saved filenames (0-2 items depending on how many had episodes
                 and how many the user confirmed rather than cancelling)
    """
    saved = []
    for player_id in (1, 2):
        fname = _show_save_dialog_for_player(
            screen, fonts, sessions, session_mode, player_id
        )
        if fname:
            saved.append(fname)
    return saved


# ── Public: save agents on shutdown ───────────────────────────────────────────

def save_agents(sessions, session_mode: str) -> list[str]:
    """
    Called by main.py after the main loop ends (fallback auto-save path).

    All modes save the best agent for each side independently (P1 + P2),
    keeping behaviour consistent with show_save_and_close_dialog.

    Each .pt file is a FULL checkpoint (weights + epsilon + tick + metadata).
    Returns a list of saved filenames (0-2 items).
    """
    label_map = {
        "NEW_VS_NEW":     "newnew",
        "NEW_VS_AGENT":   "nva",
        "AGENT_VS_AGENT": "ava",
    }
    tag   = label_map.get(session_mode, "unknown")
    saved = []

    for pid in (1, 2):
        result = _find_best(sessions, pid)
        if result is None:
            continue
        _, trainer, ep, wins, wr = result
        fname = _save_one(trainer, ep, wins, wr, pid, session_mode, tag)
        saved.append(fname)

    if not saved:
        print("[save] No episodes played — nothing saved.")

    return saved


# ── Backward-compat alias ─────────────────────────────────────────────────────
# old code called save_best_agent; keep it working as a single-return wrapper.

def save_best_agent(sessions, session_mode: str) -> str | None:
    saved = save_agents(sessions, session_mode)
    return saved[0] if saved else None


# ── Checkpoint migration helper ───────────────────────────────────────────────

def migrate_v1_checkpoints():
    """
    One-time utility: converts any v1 (raw state_dict) .pt files in
    saved_agents/ to v2 format so they load with correct epsilon.

    Because v1 files don't store epsilon, we can't recover the original value —
    but we mark them as version 2 with epsilon=None so at least the load path
    doesn't silently override epsilon to 0.05.

    Run once from a Python shell:
        from agent_selector import migrate_v1_checkpoints
        migrate_v1_checkpoints()
    """
    import torch
    files = glob.glob(os.path.join(AGENTS_DIR, "*.pt"))
    migrated = 0
    for fpath in files:
        try:
            raw = torch.load(fpath, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[migrate] Could not load {fpath}: {e}")
            continue

        if isinstance(raw, dict) and "state_dict" in raw:
            continue  # already v2

        # v1 — raw OrderedDict of weights
        print(f"[migrate] upgrading v1 → v2: {os.path.basename(fpath)}")
        checkpoint = {
            "version":    2,
            "state_dict": raw,
            "epsilon":    None,   # unknown — was not saved
            "tick":       0,
            "migrated":   True,
        }
        torch.save(checkpoint, fpath)
        migrated += 1

    print(f"[migrate] done — {migrated} file(s) upgraded.")
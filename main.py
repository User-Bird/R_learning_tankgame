"""
main.py  ─  Phase 5: Main Entry Point (Updated with Reward Curves)
──────────────────────────────────────────────────────────────────
Implements the 3-button logic (WATCH, FAST, HEADLESS).
Decouples the stats rendering to a separate process via stats_io.py.
Includes Agent Selection UI, Auto-saving on Quit, and Live Reward Tracking.
"""

import pygame
import sys
import time
import subprocess

from game import TankGame
from agent import DQNAgent
from rl.trainer import Trainer
from renderer import draw_game

from stats_io import write_stats, write_shutdown, clear_stats
from agent_selector import (
    show_session_mode_picker,
    save_agents,
    show_save_and_close_dialog,
)

# ── Session Wrapper ────────────────────────────────────────────────────────────

class Session:
    def __init__(self, idx: int, session_mode: str, agent1_path: str, agent2_path: str):
        self.idx = idx
        self.game = TankGame()
        self.trainer1 = Trainer()
        self.trainer2 = Trainer()
        self.episodes = 0
        self.wins_p1 = 0
        self.wins_p2 = 0
        self.ticks = 0
        self.ep_reward_p1 = 0.0
        self.ep_reward_p2 = 0.0
        self.last_ep_reward_p1 = 0.0
        self.last_ep_reward_p2 = 0.0
        if session_mode == "NEW_VS_AGENT" and agent2_path:
            ckpt = self.trainer2.load_checkpoint(agent2_path)
            if ckpt:
                self.episodes = ckpt.get("episodes", 0)
                self.wins_p2  = ckpt.get("wins",     0)
        elif session_mode == "AGENT_VS_AGENT":
            if agent1_path:
                ckpt1 = self.trainer1.load_checkpoint(agent1_path)
                if ckpt1:
                    self.episodes = max(self.episodes, ckpt1.get("episodes", 0))
                    self.wins_p1  = ckpt1.get("wins",     0)
            if agent2_path:
                ckpt2 = self.trainer2.load_checkpoint(agent2_path)
                if ckpt2:
                    self.episodes = max(self.episodes, ckpt2.get("episodes", 0))
                    self.wins_p2  = ckpt2.get("wins",     0)
        self.agent1 = DQNAgent(self.trainer1)
        self.agent2 = DQNAgent(self.trainer2)
        self.states = self.game.reset()
    def step(self):
        s1, s2 = self.states
        a1 = self.agent1.get_action(s1)
        a2 = self.agent2.get_action(s2)
        next_states, rewards, done = self.game.step([a1, a2])
        self.ticks += 1
        self.ep_reward_p1 += rewards[0]
        self.ep_reward_p2 += rewards[1]
        self.agent1.push(s1, a1, rewards[0], next_states[0], done)
        self.agent2.push(s2, a2, rewards[1], next_states[1], done)
        if done:
            self.episodes += 1
            if "1 WINS" in self.game.result_text:
                self.wins_p1 += 1
            elif "2 WINS" in self.game.result_text:
                self.wins_p2 += 1
            self.agent1.on_episode_end()
            self.agent2.on_episode_end()
            self.states = self.game.reset()
            self.last_ep_reward_p1 = self.ep_reward_p1
            self.last_ep_reward_p2 = self.ep_reward_p2
            self.ep_reward_p1 = 0.0
            self.ep_reward_p2 = 0.0
            return True
        else:
            self.states = next_states
            return False

# ── Render Helpers ─────────────────────────────────────────────────────────────

def draw_all_games(screen, sessions, tile=16):
    """Draws 6 games in a 3x2 grid."""
    pad = 10
    cols, rows = 25, 19
    w = cols * tile
    h = rows * tile

    for i, s in enumerate(sessions):
        col = i % 3
        row = i // 3
        x = pad + col * (w + pad)
        y = pad + row * (h + pad)

        surf = pygame.Surface((w, h))
        draw_game(surf, s.game, tile=tile)
        screen.blit(surf, (x, y))

        # Draw border & label
        pygame.draw.rect(screen, (60, 60, 80), (x, y, w, h), 2)
        font = pygame.font.SysFont("consolas", 14, bold=True)
        txt = font.render(f"GAME {i+1}", True, (255, 255, 255))
        screen.blit(txt, (x + 4, y + 4))

def draw_buttons(screen, current_mode, rects):
    font = pygame.font.SysFont("consolas", 18, bold=True)
    modes = ["WATCH", "FAST", "HEADLESS"]
    colors = {
        "WATCH": (60, 180, 80),
        "FAST": (220, 180, 40),
        "HEADLESS": (160, 80, 240)
    }

    for mode, rect in zip(modes, rects):
        bg_color = colors[mode] if current_mode == mode else (40, 40, 50)
        pygame.draw.rect(screen, bg_color, rect, border_radius=8)
        pygame.draw.rect(screen, (200, 200, 200), rect, 2, border_radius=8)
        txt = font.render(mode, True, (255, 255, 255))
        screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

def draw_save_close_btn(screen, btn_rect, font, mouse_pos):
    """Draws the new Save & Close button with hover effects."""
    C_BTN_SAVE = (40, 150, 140)  # A neat teal color
    hover = btn_rect.collidepoint(mouse_pos)
    col = tuple(min(255, c + 30) for c in C_BTN_SAVE) if hover else C_BTN_SAVE

    pygame.draw.rect(screen, col, btn_rect, border_radius=6)
    pygame.draw.rect(screen, (40, 40, 60), btn_rect, 1, border_radius=6)

    lbl = font.render("Save & Close", True, (210, 208, 200))
    screen.blit(lbl, (
        btn_rect.centerx - lbl.get_width()  // 2,
        btn_rect.centery - lbl.get_height() // 2,
    ))

# ── Main Loop ──────────────────────────────────────────────────────────────────

def main():
    # ── Wipe stale stats from last run before starting
    clear_stats()

    # ── AUTO-LAUNCH stats_window in a separate process ─────────────────────────
    stats_proc = subprocess.Popen([sys.executable, "stats_window.py"])

    pygame.init()

    # ── Window sizing (3x2 grid of games)
    TILE = 16
    W = 3 * (25 * TILE) + 4 * 10
    H = 2 * (19 * TILE) + 3 * 10
    UI_H = 80

    screen = pygame.display.set_mode((W, H + UI_H))
    pygame.display.set_caption("Combat Tank RL ─ Multi-Agent Training")
    clock = pygame.time.Clock()

    # ── SHOW SESSION PICKER UI ─────────────────────────────────────────────────
    font_md = pygame.font.SysFont("consolas", 18, bold=True)
    font_sm = pygame.font.SysFont("consolas", 14, bold=True)
    font_xs = pygame.font.SysFont("consolas", 12)
    fonts = (font_md, font_sm, font_xs)

    session_mode, agent1_path, agent2_path = show_session_mode_picker(screen, fonts)

    # Initialize the 6 games, passing down our selected modes
    sessions = [Session(i, session_mode, agent1_path, agent2_path) for i in range(6)]

    # ── Button geometry
    bw, bh = 140, 40
    bx_start = W // 2 - (3 * bw + 2 * 20) // 2
    by = H + 20
    rect_watch    = pygame.Rect(bx_start, by, bw, bh)
    rect_fast     = pygame.Rect(bx_start + bw + 20, by, bw, bh)
    rect_headless = pygame.Rect(bx_start + 2 * (bw + 20), by, bw, bh)
    btn_rects = [rect_watch, rect_fast, rect_headless]

    # ── Save & Close button geometry
    BTN_SC_W, BTN_SC_H = 160, 40
    btn_save_close = pygame.Rect(
        W - BTN_SC_W - 20,
        by,
        BTN_SC_W,
        BTN_SC_H,
    )

    current_mode = "WATCH"

    # ── Tracking state
    last_stats_write = 0.0
    STATS_WRITE_INTERVAL = 0.1
    tick_count_window = 0
    tps_timer = time.perf_counter()
    measured_tps = 0.0
    measured_fps = 0.0
    frame_count = 0

    # ── Reward History for Live Charts
    reward_history = {"p1": [], "p2": []}
    total_episodes_last_recorded = 0
    RECORD_EVERY_N_EPISODES = 6
    HISTORY_MAX_LEN = 1500

    C_BG = (10, 12, 16)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        # 1. Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if rect_watch.collidepoint(event.pos): current_mode = "WATCH"
                elif rect_fast.collidepoint(event.pos): current_mode = "FAST"
                elif rect_headless.collidepoint(event.pos): current_mode = "HEADLESS"
                elif btn_save_close.collidepoint(event.pos):
                    saved = show_save_and_close_dialog(screen, fonts, sessions, session_mode)
                    if saved:
                        print(f"[main] saved: {saved}")
                        running = False

        # 2. Step Games
        n_ticks = {"WATCH": 1, "FAST": 50, "HEADLESS": 200}[current_mode]
        for _ in range(n_ticks):
            for s in sessions:
                episode_ended = s.step()

                # If an episode just ended, record stats for the charts
                if episode_ended:
                    total_eps = sum(sess.episodes for sess in sessions)

                    # Only record a point every N total episodes to avoid spam
                    if total_eps - total_episodes_last_recorded >= RECORD_EVERY_N_EPISODES:
                        total_episodes_last_recorded = total_eps

                        for pid, key in ((1, "p1"), (2, "p2")):
                            active = [sess for sess in sessions if sess.episodes > 0]
                            if not active:
                                continue

                            # Average reward and win-rate across all active sessions
                            avg_r  = sum(
                                (sess.last_ep_reward_p1 if pid == 1 else sess.last_ep_reward_p2)
                                for sess in active
                            ) / len(active)

                            avg_wr = sum(
                                (sess.wins_p1 if pid == 1 else sess.wins_p2) / sess.episodes
                                for sess in active
                            ) / len(active)

                            reward_history[key].append([total_eps, avg_r, avg_wr])
                            if len(reward_history[key]) > HISTORY_MAX_LEN:
                                reward_history[key].pop(0)

        tick_count_window += n_ticks

        # 3. Render
        if current_mode == "WATCH":
            screen.fill(C_BG)
            draw_all_games(screen, sessions, TILE)
            draw_buttons(screen, current_mode, btn_rects)
            draw_save_close_btn(screen, btn_save_close, font_sm, mouse_pos)
            pygame.display.flip()

        elif current_mode == "FAST":
            if frame_count % 10 == 0:
                screen.fill(C_BG)
                draw_all_games(screen, sessions, TILE)
                draw_buttons(screen, current_mode, btn_rects)
                draw_save_close_btn(screen, btn_save_close, font_sm, mouse_pos)
                pygame.display.flip()

        else:  # HEADLESS
            if frame_count % 30 == 0:
                screen.fill(C_BG)
                draw_buttons(screen, current_mode, btn_rects)
                draw_save_close_btn(screen, btn_save_close, font_sm, mouse_pos)
                font = pygame.font.SysFont("consolas", 24, bold=True)
                txt = font.render("HEADLESS MODE ACTIVE - RENDERING PAUSED", True, (160, 80, 240))
                screen.blit(txt, (W // 2 - txt.get_width() // 2, H // 2 - txt.get_height() // 2))
                pygame.display.flip()

        # 4. Write stats to file (always on timer)
        now = time.perf_counter()
        if now - last_stats_write >= STATS_WRITE_INTERVAL:
            elapsed = now - tps_timer
            if elapsed >= 1.0:
                measured_tps = tick_count_window / elapsed
                measured_fps = clock.get_fps()
                tick_count_window = 0
                tps_timer = now

            write_stats(
                sessions,
                mode=current_mode,
                tps=measured_tps,
                fps=measured_fps,
                session_mode=session_mode,
                reward_history=reward_history
            )
            last_stats_write = now

        clock.tick(60)
        frame_count += 1

    # ── SHUTDOWN LOGIC ─────────────────────────────────────────────────────────
    write_shutdown()
    saved = save_agents(sessions, session_mode)
    for f in saved:
        print(f"Saved fallback: {f}")

    pygame.quit()
    stats_proc.wait(timeout=3)
    sys.exit(0)

if __name__ == "__main__":
    main()
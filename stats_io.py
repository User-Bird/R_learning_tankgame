"""
stats_io.py  —  shared stats bridge between main.py and stats_window.py
────────────────────────────────────────────────────────────────────────
main.py         calls  write_stats(sessions, ...)   every ~0.1 s
stats_window.py calls  read_stats()                 every ~0.1 s

Uses a local JSON file (stats_data.json) in the project folder.
Atomic write (write to tmp, rename) so the reader never sees half-written data.

Staleness: stats_window checks the 'ts' timestamp — if it's older than
STALE_AFTER seconds, main.py is considered dead and the window shows "Offline".

Shutdown: main.py writes {"shutdown": true} before exiting so stats_window
can close itself cleanly instead of waiting for the file to go stale.

Reward curves
─────────────
write_stats() now accepts an optional reward_history dict:
  {
    "p1": [(episode, avg_reward, win_rate), ...],   # history for best P1
    "p2": [(episode, avg_reward, win_rate), ...],   # history for best P2
  }
This is written into the JSON under "curves" and read back by stats_window
to draw the live reward / win-rate charts.  The history is accumulated by
main.py (not here) — this module just serialises and deserialises it.
"""

import json
import os
import sys
import time

STATS_FILE  = "stats_data.json"
STATS_TMP   = "stats_data.tmp.json"
STALE_AFTER = 3.0   # seconds — if data is older than this, main.py is dead

_IS_WINDOWS = sys.platform == "win32"


def write_stats(sessions, mode: str, tps: float, fps: float,
                session_mode: str = "NEW_VS_NEW",
                reward_history: dict | None = None):
    """
    Called by main.py every ~0.1 s.

    session_mode   : "NEW_VS_NEW" | "NEW_VS_AGENT" | "AGENT_VS_AGENT"
    reward_history : optional dict with keys "p1" and "p2", each a list of
                     (episode, avg_reward, win_rate) tuples for the best
                     agent on that side across all sessions.
    """
    data = {
        "ts":           time.time(),
        "shutdown":     False,
        "mode":         mode,           # WATCH / FAST / HEADLESS
        "session_mode": session_mode,
        "tps":          tps,
        "fps":          fps,
        "games": [
            {
                "idx":      s.idx,
                "episodes": s.episodes,
                "wins_p1":  s.wins_p1,
                "wins_p2":  s.wins_p2,
                "ticks":    s.ticks,
                "eps1":     s.trainer1.epsilon,
                "eps2":     s.trainer2.epsilon,
                "buf1":     len(s.trainer1.buffer),
                "buf2":     len(s.trainer2.buffer),
                "loss1":    s.trainer1.last_loss,
                "loss2":    s.trainer2.last_loss,
            }
            for s in sessions
        ],
        # Reward/win-rate curves — empty dicts if not provided
        "curves": reward_history or {"p1": [], "p2": []},
    }
    _atomic_write(data)


def write_shutdown():
    """
    Called by main.py just before pygame.quit().
    Tells stats_window to close itself.
    """
    _atomic_write({"shutdown": True, "ts": time.time()})


def _atomic_write(data: dict):
    """
    Write data to STATS_FILE atomically.

    On Linux/macOS, os.replace() is a true atomic rename.
    On Windows, os.replace() can raise PermissionError if the destination
    file is open by another process (e.g. stats_window.py reading it).
    We work around this with a retry loop and, as a last resort, a direct
    overwrite so the writer never crashes.
    """
    # Write to the temp file first (always safe)
    with open(STATS_TMP, "w") as f:
        json.dump(data, f)

    if not _IS_WINDOWS:
        # POSIX: rename is atomic and never fails due to the target being open
        os.replace(STATS_TMP, STATS_FILE)
        return

    # Windows: retry the replace a few times; readers hold the file open only
    # for the brief json.load() call, so collisions are rare and short-lived.
    for attempt in range(5):
        try:
            os.replace(STATS_TMP, STATS_FILE)
            return
        except PermissionError:
            time.sleep(0.01)   # 10 ms — wait for the reader to close the file

    # Last resort: direct overwrite (non-atomic but better than crashing)
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(data, f)
    except PermissionError:
        pass   # silently skip this write cycle; the next one will succeed


def read_stats() -> dict | None:
    """
    Called by stats_window.py.
    Returns parsed dict, or None if file missing / corrupt / stale.
    A stale file means main.py has died without writing a shutdown signal.
    """
    try:
        with open(STATS_FILE, "r") as f:
            data = json.load(f)
        # Treat stale data the same as no data
        age = time.time() - data.get("ts", 0)
        if age > STALE_AFTER:
            return None
        return data
    except (FileNotFoundError, json.JSONDecodeError, PermissionError,
            OSError):
        # PermissionError: writer is mid-replace on Windows — skip this poll.
        # OSError: catches other rare I/O failures gracefully.
        return None


def clear_stats():
    """Delete the stats file (call on main.py startup so old data never shows)."""
    for path in (STATS_FILE, STATS_TMP):
        try:
            os.remove(path)
        except (FileNotFoundError, PermissionError):
            pass
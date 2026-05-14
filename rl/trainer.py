"""
rl/trainer.py  ─  Phase 5: Optimised DQN Trainer
──────────────────────────────────────────────────
Key change: batch_act(encoded_states, epsilons) → one GPU forward pass for N agents.

Phase 5 checkpoint update:
  save_checkpoint(path, extra)  →  saves weights + epsilon + tick + metadata
  load_checkpoint(path)         →  restores all of the above (v1 weights-only files
                                   are still supported for backward compat)
"""

import copy
import numpy as np
import torch
import torch.nn as nn

from rl.model         import DQNModel
from rl.replay_buffer import ReplayBuffer
from rl.state_encoder import encode_state

BATCH_SIZE        = 64
GAMMA             = 0.99
LR                = 1e-3
TRAIN_EVERY       = 8
TARGET_SYNC_EVERY = 500
BUFFER_MIN        = 1_000
BUFFER_CAPACITY   = 50_000

EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.999


class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = DQNModel().to(self.device)
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=LR)

        self.loss_fn = nn.SmoothL1Loss()

        self.buffer = ReplayBuffer(BUFFER_CAPACITY)

        self.epsilon   = EPSILON_START
        self._tick     = 0
        self.last_loss = 0.0

        # ── Epsilon floor tracking ────────────────────────────────────────────
        # Logged once when epsilon first hits the floor so you can see how many
        # episodes it took.  Reset to False on load_checkpoint so resuming a
        # session that already hit the floor doesn't log a spurious message.
        self._epsilon_floor_logged = False
        self.episodes = 0   # incremented by on_episode_end(); used for the log
        self.fixed_map = None

    # ── Checkpoint save / load ────────────────────────────────────────────────

    def save_checkpoint(self, path: str, extra_info: dict = None):
        """
        Save full training state so a resumed session has the same epsilon,
        training progress, and weights.

        extra_info : optional dict of anything you want stored alongside
                     (e.g. episodes, win_rate, session_mode, saved_at)
        """
        checkpoint = {
            "version": 2,
            "state_dict": self.online_net.state_dict(),
            "epsilon": self.epsilon,
            "tick": self._tick,
            "episodes": self.episodes,
        }
        if self.fixed_map is not None:
            checkpoint["fixed_map"] = self.fixed_map  # saved alongside weights
        if extra_info:
            checkpoint.update(extra_info)
        torch.save(checkpoint, path)
        print(f"[trainer] checkpoint saved → {path}  (ε={self.epsilon:.4f}, tick={self._tick:,})")

    def load_checkpoint(self, path: str) -> dict:
        """
        Load a checkpoint from path.  Handles two formats:
          v1 (legacy) : raw state_dict only   → weights loaded, epsilon stays at 0.05
          v2          : full dict with 'state_dict' key → weights + epsilon + tick restored

        Returns the raw checkpoint dict so callers can read metadata (episodes, win_rate …).
        """
        raw = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(raw, dict) and "state_dict" in raw:
            # ── v2: full checkpoint ───────────────────────────────────────────
            self.online_net.load_state_dict(raw["state_dict"])
            self.epsilon  = raw.get("epsilon", EPSILON_END)
            self._tick    = raw.get("tick",    0)
            self.episodes = raw.get("episodes", 0)
            self.fixed_map = raw.get("fixed_map", None)
            # Re-arm the floor log: if the loaded epsilon is already at the
            # floor we don't want to log again, so mark it as already done.
            self._epsilon_floor_logged = (self.epsilon <= EPSILON_END)
            print(f"[trainer] loaded v2 checkpoint from {path}  "
                  f"(ε={self.epsilon:.4f}, tick={self._tick:,}, "
                  f"ep={raw.get('episodes', '?')})")
            return raw
        else:
            # ── v1: just weights (backward compat) ───────────────────────────
            self.online_net.load_state_dict(raw)
            self.epsilon = EPSILON_END    # safe exploitation default
            self._epsilon_floor_logged = True   # already at floor
            print(f"[trainer] loaded v1 weights from {path}  (ε set to {EPSILON_END})")
            return {}

    # ── Single-agent fallback ─────────────────────────────────────────────────
    def get_action(self, state: dict) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 6)
        enc = encode_state(state)
        t   = torch.tensor(enc, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.online_net(t)
        return int(q.argmax(dim=1).item())

    # ── BATCHED inference — one GPU call for all N states ────────────────────
    @torch.no_grad()
    def batch_act(self, encoded_states: np.ndarray, epsilons: np.ndarray) -> np.ndarray:
        """
        encoded_states : (N, 77) float32 — pre-encoded states for N agents
        epsilons       : (N,)    float32 — epsilon per agent
        returns        : (N,)    int64   — chosen actions
        """
        t        = torch.tensor(encoded_states, dtype=torch.float32, device=self.device)
        q_values = self.online_net(t)                       # (N, 6) — one pass
        greedy   = q_values.argmax(dim=1).cpu().numpy()    # (N,)
        explore  = np.random.random(len(epsilons)) < epsilons
        random_a = np.random.randint(0, 6, size=len(epsilons))
        return np.where(explore, random_a, greedy)

    # ── Experience storage + training ─────────────────────────────────────────
    def push(self, state: dict, action: int, reward: float,
             next_state: dict, done: bool):
        self.buffer.push(encode_state(state), action, reward,
                         encode_state(next_state), done)
        self._tick += 1
        if len(self.buffer) >= BUFFER_MIN and self._tick % TRAIN_EVERY == 0:
            self._train_step()
        if self._tick % TARGET_SYNC_EVERY == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def push_encoded(self, s: np.ndarray, action: int, reward: float,
                     ns: np.ndarray, done: bool):
        """Push already-encoded states — avoids double encoding in hot path."""
        self.buffer.push(s, action, reward, ns, done)
        self._tick += 1
        if len(self.buffer) >= BUFFER_MIN and self._tick % TRAIN_EVERY == 0:
            self._train_step()
        if self._tick % TARGET_SYNC_EVERY == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def on_episode_end(self):
        self.episodes += 1
        prev_epsilon = self.epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

        # Log once when epsilon first hits the floor
        if (prev_epsilon > EPSILON_END
                and self.epsilon <= EPSILON_END
                and not self._epsilon_floor_logged):
            self._epsilon_floor_logged = True
            print(
                f"[trainer] ε hit floor ({EPSILON_END}) after "
                f"{self.episodes:,} episodes  (tick={self._tick:,})"
            )

    def _train_step(self):
        states, actions, rewards, next_states, dones = \
            self.buffer.sample(BATCH_SIZE, self.device)

        q_values = self.online_net(states)
        q_taken  = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next   = self.target_net(next_states).max(dim=1).values
            q_target = rewards + GAMMA * q_next * (1.0 - dones)

        loss = self.loss_fn(q_taken, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.last_loss = loss.item()
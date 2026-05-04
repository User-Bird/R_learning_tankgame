"""
game.py  ─  Phase 2B: TankGame clean engine
────────────────────────────────────────────
"""

import random
import math
import numpy as np

# ── Arena constants ────────────────────────────────────────────────────────────
COLS = 25
ROWS = 19

SPAWN1 = (3, 1)
SPAWN2 = (21, 17)

# ── Tile types ─────────────────────────────────────────────────────────────────
EMPTY       = 0
WALL        = 1
CHARGE_TILE = 2

# ── Directions ─────────────────────────────────────────────────────────────────
UP    = 0
RIGHT = 1
DOWN  = 2
LEFT  = 3

DX = {UP: 0, RIGHT: 1, DOWN: 0,  LEFT: -1}
DY = {UP: -1, RIGHT: 0, DOWN: 1, LEFT:  0}

# ── Tunable constants ──────────────────────────────────────────────────────────
MAX_HEALTH     = 5
MAX_AMMO       = 5
MAX_MINES      = 3
SHOOT_COOLDOWN = 25
BULLET_MOVE_EVERY = 2
BULLET_LIFETIME   = 60
CHARGE_TICKS      = 2
MAX_TICKS         = 800      # ~13 seconds at 60Hz. Forces episodes to end.

# ── Mine placement restrictions ───────────────────────────────────────────────
MINE_COOLDOWN_TICKS  = 60    # Option 1 — ticks between mine plants (1 second at 60Hz)
MINE_MIN_SPACING     = 6     # Option 2 — min Manhattan distance between ANY two mines
MINE_SPAWN_LOCKOUT   = 120   # Option 3 — ticks after episode start before ANY mine allowed

# ── Reward constants ───────────────────────────────────────────────────────────
R_HIT_ENEMY      =  50.0
R_MINE_TRIGGER   =  80.0
R_KILL           = 200.0
R_TOOK_HIT       = -30.0
R_DIED           = -200.0
R_CHARGE_PICKUP  =  10.0
R_TIME_PENALTY   =  -1.0

MINE_PENALTY_RANGE = 7
R_STUPID_MINE      = -40.0   # Penalty to stop early mine dumping
R_CLOSER_ENEMY     =  0.5    # Reward for moving towards enemy
R_FACING_ENEMY     =  0.1    # Lowered to prevent sit-and-stare farming

# ── Templates + Helpers ────────────────────────────────────────────────────────

_SPAWN_TEMPLATES = [
    [[2, 2, 2], [2, 1, 2], [2, 2, 2]],
    [[0, 2, 2], [0, 1, 2], [0, 1, 2], [0, 2, 2]],
    [[0, 2, 2], [0, 1, 2], [0, 1, 0], [0, 0, 0]],
    [[0, 0, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 0, 2]],
]
_SPAWN_WEIGHTS = [1, 3, 3, 3]

_NON_SPAWN_TEMPLATES = [
    [[2, 2, 2, 2, 2], [0, 1, 1, 1, 2], [0, 2, 2, 1, 2], [0, 1, 2, 1, 2], [2, 0, 0, 0, 2]],
    [[2, 2, 2, 2, 2], [0, 1, 1, 1, 0], [0, 2, 2, 2, 0], [0, 1, 1, 1, 0], [2, 2, 2, 2, 2]],
]
_NON_SPAWN_WEIGHTS = [1, 1]


def _rotate_template_90(t):
    rows, cols = len(t), len(t[0])
    return [[t[rows - 1 - r][c] for r in range(rows)] for c in range(cols)]

def _rotate_template(t, times: int):
    for _ in range(times % 4):
        t = _rotate_template_90(t)
    return t

def _place_template(grid, template, origin_col: int, origin_row: int, protected: set):
    for r, row in enumerate(template):
        for c, val in enumerate(row):
            gc, gr = origin_col + c, origin_row + r
            if not (0 < gc < COLS - 1 and 0 < gr < ROWS - 1): continue
            if val == 1: grid[gr][gc] = WALL
            elif val == 2:
                grid[gr][gc] = EMPTY
                protected.add((gc, gr))

def _can_place(grid, template, origin_col: int, origin_row: int, protected: set) -> bool:
    for r, row in enumerate(template):
        for c, val in enumerate(row):
            if val == 0: continue
            gc, gr = origin_col + c, origin_row + r
            if not (0 < gc < COLS - 1 and 0 < gr < ROWS - 1): return False
            if grid[gr][gc] == WALL: return False
            if (gc, gr) in protected: return False
    return True

def _place_objects_in_quadrant(grid, templates, weights, count: int, qc: int, qr: int, qw: int, qh: int, protected: set):
    placed, attempts = 0, 0
    while placed < count and attempts < 300:
        attempts += 1
        tmpl = random.choices(templates, weights=weights, k=1)[0]
        tmpl = _rotate_template([row[:] for row in tmpl], random.randint(0, 3))
        t_h = len(tmpl)
        t_w = len(tmpl[0]) if t_h > 0 else 0

        max_dc, max_dr = qw - t_w - 1, qh - t_h - 1
        if max_dc < 0 or max_dr < 0: continue

        abs_c, abs_r = qc + random.randint(0, max_dc), qr + random.randint(0, max_dr)
        if _can_place(grid, tmpl, abs_c, abs_r, protected):
            _place_template(grid, tmpl, abs_c, abs_r, protected)
            placed += 1

def _clear_safe_zone(grid, cx: int, cy: int, radius: int = 1):
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r, c = cy + dr, cx + dc
            if 0 < r < ROWS - 1 and 0 < c < COLS - 1:
                grid[r][c] = EMPTY

def generate_random_map():
    grid = [[EMPTY] * COLS for _ in range(ROWS)]
    for c in range(COLS): grid[0][c] = grid[ROWS - 1][c] = WALL
    for r in range(ROWS): grid[r][0] = grid[r][COLS - 1] = WALL

    protected: set = set()
    _place_objects_in_quadrant(grid, _SPAWN_TEMPLATES, _SPAWN_WEIGHTS, random.randint(4, 7), 1, 1, 11, 8, protected)
    _place_objects_in_quadrant(grid, _NON_SPAWN_TEMPLATES, _NON_SPAWN_WEIGHTS, random.randint(2, 4), 13, 1, 11, 8, protected)
    _place_objects_in_quadrant(grid, _NON_SPAWN_TEMPLATES, _NON_SPAWN_WEIGHTS, random.randint(2, 4), 1, 10, 11, 8, protected)
    _place_objects_in_quadrant(grid, _SPAWN_TEMPLATES, _SPAWN_WEIGHTS, random.randint(4, 7), 13, 10, 11, 8, protected)

    for sx, sy in [SPAWN1, SPAWN2]:
        _clear_safe_zone(grid, sx, sy, radius=1)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                protected.discard((sx + dc, sy + dr))

    charge_tiles, attempts = [], 0
    while len(charge_tiles) < 6 and attempts < 2000:
        attempts += 1
        r, c = random.randint(1, ROWS - 2), random.randint(1, COLS - 2)
        if grid[r][c] != EMPTY or (c, r) in protected or (c, r) in (SPAWN1, SPAWN2): continue
        if any(abs(c - ec) <= 2 and abs(r - er) <= 2 for (ec, er) in charge_tiles): continue
        grid[r][c] = CHARGE_TILE
        charge_tiles.append((c, r))

    return grid, charge_tiles

# ── Data classes ───────────────────────────────────────────────────────────────

class Tank:
    # CHANGED: added mine_cooldown to __slots__
    __slots__ = ("x", "y", "direction", "health", "ammo", "mines",
                 "cooldown", "mine_cooldown", "charge_progress", "player_id")

    def __init__(self, x, y, direction, player_id):
        self.x, self.y, self.direction = x, y, direction
        self.health, self.ammo, self.mines = MAX_HEALTH, MAX_AMMO, MAX_MINES
        self.cooldown       = 0
        self.mine_cooldown  = 0   # NEW: Option 1 — ticks until next mine plant allowed
        self.charge_progress = 0
        self.player_id      = player_id

    @property
    def alive(self): return self.health > 0
    def can_shoot(self): return self.cooldown <= 0 and self.ammo > 0
    def can_plant_mine(self): return self.mine_cooldown <= 0  # NEW helper

class Bullet:
    __slots__ = ("x", "y", "direction", "owner_id", "lifetime", "move_timer")
    def __init__(self, x, y, direction, owner_id):
        self.x, self.y, self.direction = float(x), float(y), direction
        self.owner_id, self.lifetime, self.move_timer = owner_id, BULLET_LIFETIME, 0

class Mine:
    __slots__ = ("x", "y", "owner_id", "health")
    def __init__(self, x, y, owner_id):
        self.x, self.y, self.owner_id, self.health = x, y, owner_id, 2

# ── TankGame ───────────────────────────────────────────────────────────────────

class TankGame:
    def __init__(self):
        self.episode = 0
        self._new_episode_state()

    def reset(self):
        self._new_episode_state()
        return self._state_pair()

    def step(self, actions):
        if self.done:
            return self._state_pair(), [0.0, 0.0], True

        self.ticks += 1
        rewards = [R_TIME_PENALTY, R_TIME_PENALTY]

        # ── Pre-action state for per-agent distance tracking ───────────────────
        pos1_before = (self.tank1.x, self.tank1.y)
        pos2_before = (self.tank2.x, self.tank2.y)

        # ── actions ────────────────────────────────────────────────────────────
        mine1 = self._apply_action(self.tank1, actions[0])
        mine2 = self._apply_action(self.tank2, actions[1])

        # ── Reward Shaping: Movement ───────────────────────────────────────────
        dist_before = abs(pos1_before[0] - pos2_before[0]) + abs(pos1_before[1] - pos2_before[1])
        dist_after  = abs(self.tank1.x   - self.tank2.x)   + abs(self.tank1.y   - self.tank2.y)

        if actions[0] == 2 and dist_after < dist_before: rewards[0] += R_CLOSER_ENEMY
        if actions[1] == 2 and dist_after < dist_before: rewards[1] += R_CLOSER_ENEMY

        # ── Reward Shaping: Facing ─────────────────────────────────────────────
        if actions[0] != 4 and self._is_facing(self.tank1, self.tank2): rewards[0] += R_FACING_ENEMY
        if actions[1] != 4 and self._is_facing(self.tank2, self.tank1): rewards[1] += R_FACING_ENEMY

        # ── mine stupidity penalty ────────────────────────────────────────────
        if mine1 and dist_after > MINE_PENALTY_RANGE: rewards[0] += R_STUPID_MINE
        if mine2 and dist_after > MINE_PENALTY_RANGE: rewards[1] += R_STUPID_MINE

        # ── cooldowns & charge ────────────────────────────────────────────────
        if self.tank1.cooldown > 0: self.tank1.cooldown -= 1
        if self.tank2.cooldown > 0: self.tank2.cooldown -= 1

        # NEW: Option 1 — tick down mine cooldowns
        if self.tank1.mine_cooldown > 0: self.tank1.mine_cooldown -= 1
        if self.tank2.mine_cooldown > 0: self.tank2.mine_cooldown -= 1

        if self._update_charge(self.tank1): rewards[0] += R_CHARGE_PICKUP
        if self._update_charge(self.tank2): rewards[1] += R_CHARGE_PICKUP

        # ── combat updates ─────────────────────────────────────────────────────
        for pid, delta in self._update_bullets().items(): rewards[pid - 1] += delta
        for pid, delta in self._check_mines().items(): rewards[pid - 1] += delta
        for pid, delta in self._check_done().items(): rewards[pid - 1] += delta

        return self._state_pair(), rewards, self.done

    # ── Internal logic ─────────────────────────────────────────────────────────

    def _new_episode_state(self):
        self.grid, self.charge_tiles = generate_random_map()
        self.tank1 = Tank(SPAWN1[0], SPAWN1[1], UP,   1)
        self.tank2 = Tank(SPAWN2[0], SPAWN2[1], DOWN, 2)
        self.bullets, self.active_mines = [], []
        self.ticks, self.done, self.result_text = 0, False, ""
        self.episode += 1

    def is_walkable(self, x, y):
        if x < 0 or x >= COLS or y < 0 or y >= ROWS: return False
        return self.grid[y][x] != WALL

    def _is_facing(self, tank, enemy):
        if tank.direction == UP and enemy.y < tank.y: return True
        if tank.direction == DOWN and enemy.y > tank.y: return True
        if tank.direction == LEFT and enemy.x < tank.x: return True
        if tank.direction == RIGHT and enemy.x > tank.x: return True
        return False

    def _has_los(self, t1, t2):
        if t1.x != t2.x and t1.y != t2.y: return False
        if t1.x == t2.x:
            y1, y2 = min(t1.y, t2.y), max(t1.y, t2.y)
            for y in range(y1 + 1, y2):
                if self.grid[y][t1.x] == WALL: return False
        else:
            x1, x2 = min(t1.x, t2.x), max(t1.x, t2.x)
            for x in range(x1 + 1, x2):
                if self.grid[t1.y][x] == WALL: return False
        return True

    def _apply_action(self, tank, action) -> bool:
        if not tank.alive: return False
        if action == 0: tank.direction = (tank.direction - 1) % 4
        elif action == 1: tank.direction = (tank.direction + 1) % 4
        elif action == 2: self._try_move(tank)
        elif action == 3: self._shoot(tank)
        elif action == 5: return self._plant_mine(tank)
        return False

    def _try_move(self, tank):
        nx, ny = tank.x + DX[tank.direction], tank.y + DY[tank.direction]
        if not self.is_walkable(nx, ny): return
        other = self.tank2 if tank.player_id == 1 else self.tank1
        if other.alive and other.x == nx and other.y == ny: return
        tank.x, tank.y = nx, ny

    def _shoot(self, tank):
        if not tank.can_shoot(): return
        bx, by = tank.x + DX[tank.direction], tank.y + DY[tank.direction]
        if not (0 <= bx < COLS and 0 <= by < ROWS) or self.grid[by][bx] == WALL: return
        self.bullets.append(Bullet(bx, by, tank.direction, tank.player_id))
        tank.ammo -= 1
        tank.cooldown = SHOOT_COOLDOWN

    def _plant_mine(self, tank) -> bool:
        """
        Plant a mine at the tank's current position.

        Blocked if any of these are true:
          - Tank has no mines left in inventory
          - Tank already has MAX_MINES active on the field
          - There is already a mine on this exact tile
          - [Option 3] Episode is younger than MINE_SPAWN_LOCKOUT ticks
                       (prevents instant mine-dump at game start)
          - [Option 1] Tank's mine_cooldown has not expired yet
                       (enforces minimum time between consecutive plants)
          - [Option 2] Any mine anywhere on the field is within
                       MINE_MIN_SPACING Manhattan tiles of this position
                       (prevents mine clusters / suicide packs)
        """
        if tank.mines <= 0:
            return False
        if sum(1 for m in self.active_mines if m.owner_id == tank.player_id) >= MAX_MINES:
            return False
        if any(m.x == tank.x and m.y == tank.y for m in self.active_mines):
            return False

        # ── Option 3: spawn lockout — no mines in the opening phase ───────────
        if self.ticks < MINE_SPAWN_LOCKOUT:
            return False

        # ── Option 1: mine cooldown — must wait between plants ────────────────
        if tank.mine_cooldown > 0:
            return False

        # ── Option 2: spacing — no mine within MINE_MIN_SPACING tiles ─────────
        if any(
            abs(tank.x - m.x) + abs(tank.y - m.y) < MINE_MIN_SPACING
            for m in self.active_mines
        ):
            return False

        # All checks passed — plant the mine
        self.active_mines.append(Mine(tank.x, tank.y, tank.player_id))
        tank.mines -= 1
        tank.mine_cooldown = MINE_COOLDOWN_TICKS   # start per-tank cooldown
        return True

    def _update_charge(self, tank):
        tx, ty = tank.x, tank.y
        if self.grid[ty][tx] == CHARGE_TILE:
            tank.charge_progress += 1
            if tank.charge_progress >= CHARGE_TICKS:
                tank.ammo = MAX_AMMO
                tank.mines = min(MAX_MINES, tank.mines + 1)
                tank.charge_progress = 0
                self.grid[ty][tx] = EMPTY
                if (tx, ty) in self.charge_tiles: self.charge_tiles.remove((tx, ty))
                return True
        else:
            tank.charge_progress = 0
        return False

    def _update_bullets(self):
        rewards, alive = {}, []
        for b in self.bullets:
            b.lifetime -= 1
            b.move_timer += 1
            if b.lifetime <= 0: continue
            if b.move_timer < BULLET_MOVE_EVERY:
                alive.append(b); continue

            b.move_timer = 0
            nx, ny = int(b.x) + DX[b.direction], int(b.y) + DY[b.direction]
            if not (0 <= nx < COLS and 0 <= ny < ROWS) or self.grid[ny][nx] == WALL: continue
            b.x, b.y = float(nx), float(ny)

            mine_hit = False
            for m in self.active_mines:
                if int(b.x) == m.x and int(b.y) == m.y:
                    m.health -= 1
                    mine_hit = True
                    if m.health <= 0:
                        owner = self.tank1 if m.owner_id == 1 else self.tank2
                        owner.mines = min(MAX_MINES, owner.mines + 1)
                        self.active_mines.remove(m)
                    break
            if mine_hit: continue

            tank_hit = False
            for tank in (self.tank1, self.tank2):
                if not tank.alive or b.owner_id == tank.player_id: continue
                if int(b.x) == tank.x and int(b.y) == tank.y:
                    tank.health -= 1
                    tank_hit = True
                    rewards[b.owner_id] = rewards.get(b.owner_id, 0) + R_HIT_ENEMY
                    rewards[tank.player_id] = rewards.get(tank.player_id, 0) + R_TOOK_HIT
                    break
            if not tank_hit: alive.append(b)
        self.bullets = alive
        return rewards

    def _check_mines(self):
        rewards, surviving = {}, []
        for m in self.active_mines:
            triggered = False
            for tank in (self.tank1, self.tank2):
                if not tank.alive or tank.player_id == m.owner_id: continue
                if abs(tank.x - m.x) <= 1 and abs(tank.y - m.y) <= 1:
                    tank.health -= 2
                    triggered = True
                    owner = self.tank1 if m.owner_id == 1 else self.tank2
                    owner.mines = min(MAX_MINES, owner.mines + 1)
                    rewards[m.owner_id] = rewards.get(m.owner_id, 0) + R_MINE_TRIGGER
                    rewards[tank.player_id] = rewards.get(tank.player_id, 0) + R_TOOK_HIT
                    break
            if not triggered: surviving.append(m)
        self.active_mines = surviving
        return rewards

    def _check_done(self):
        t1_dead, t2_dead = not self.tank1.alive, not self.tank2.alive
        rewards = {}

        if self.ticks >= MAX_TICKS and not (t1_dead or t2_dead):
            self.result_text = "TIMEOUT — DRAW!"
            self.done = True
            return {1: R_DIED * 0.5, 2: R_DIED * 0.5}

        if not (t1_dead or t2_dead): return {}

        if t1_dead and t2_dead:
            self.result_text, rewards[1], rewards[2] = "DRAW!", R_DIED, R_DIED
        elif t2_dead:
            self.result_text, rewards[1], rewards[2] = "TANK 1 WINS!", R_KILL, R_DIED
        else:
            self.result_text, rewards[1], rewards[2] = "TANK 2 WINS!", R_DIED, R_KILL

        self.done = True
        return rewards

    # ── State builder ──────────────────────────────────────────────────────────

    def _state_for(self, my_tank, enemy_tank):
        dx_fwd, dy_fwd = DX[my_tank.direction], DY[my_tank.direction]
        dx_bk, dy_bk   = -dx_fwd, -dy_fwd
        dx_l, dy_l     = DX[(my_tank.direction - 1) % 4], DY[(my_tank.direction - 1) % 4]
        dx_r, dy_r     = DX[(my_tank.direction + 1) % 4], DY[(my_tank.direction + 1) % 4]

        def blocked(tx, ty):
            nx, ny = my_tank.x + tx, my_tank.y + ty
            if not (0 <= nx < COLS and 0 <= ny < ROWS): return True
            return self.grid[ny][nx] == WALL

        dist = abs(my_tank.x - enemy_tank.x) + abs(my_tank.y - enemy_tank.y)

        ex, ey = enemy_tank.x - my_tank.x, enemy_tank.y - my_tank.y
        target_angle = math.atan2(ey, ex)
        dir_angles = {UP: -math.pi/2, RIGHT: 0, DOWN: math.pi/2, LEFT: math.pi}
        my_angle = dir_angles[my_tank.direction]

        angle_diff = abs(target_angle - my_angle)
        if angle_diff > math.pi: angle_diff = 2 * math.pi - angle_diff
        angle_norm = angle_diff / math.pi

        return {
            "my_pos":          (my_tank.x, my_tank.y),
            "my_dir":          my_tank.direction,
            "my_health":       my_tank.health,
            "my_ammo":         my_tank.ammo,
            "my_mines":        my_tank.mines,
            "my_charge_prog":  my_tank.charge_progress,
            "enemy_pos":       (enemy_tank.x, enemy_tank.y),
            "enemy_dir":       enemy_tank.direction,
            "enemy_health":    enemy_tank.health,
            "bullets": [{"pos": (int(b.x), int(b.y)), "dir": b.direction, "owner": b.owner_id} for b in self.bullets],
            "mines": [{"pos": (m.x, m.y), "owner": m.owner_id} for m in self.active_mines],
            "walls_nearby": {
                "forward": blocked(dx_fwd, dy_fwd),
                "back":    blocked(dx_bk,  dy_bk),
                "left":    blocked(dx_l,   dy_l),
                "right":   blocked(dx_r,   dy_r),
            },
            # can_mine mirrors every check in _plant_mine so the agent sees the truth
            "can_shoot":              my_tank.can_shoot(),
            "can_mine":               (
                my_tank.mines > 0
                and self.ticks >= MINE_SPAWN_LOCKOUT
                and my_tank.mine_cooldown <= 0
                and not any(m.x == my_tank.x and m.y == my_tank.y for m in self.active_mines)
                and not any(
                    abs(my_tank.x - m.x) + abs(my_tank.y - m.y) < MINE_MIN_SPACING
                    for m in self.active_mines
                )
            ),
            "on_charge_tile":         self.grid[my_tank.y][my_tank.x] == CHARGE_TILE,
            "distance_to_enemy":      dist,
            "angle_to_enemy":         angle_norm,
            "enemy_in_line_of_sight": self._has_los(my_tank, enemy_tank),
            "arena_grid":             [row[:] for row in self.grid],
        }

    def _state_pair(self):
        return [self._state_for(self.tank1, self.tank2), self._state_for(self.tank2, self.tank1)]
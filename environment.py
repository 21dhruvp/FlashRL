from __future__ import annotations
"""
environment.py – World's Hardest Game RL wrapper
(fully self‑contained, RGB detector compatible)
"""

import os
from typing import Dict, List, Tuple, Optional

import ctypes
import ctypes.wintypes as wintypes

import cv2
import numpy as np

from game_capture import capture_screen, get_ruffle_window_region
from object_detection import detect_objects

# ─── optional faster keyboard backend ──────────────────────────────
try:
    import pydirectinput          # noqa: F401
    _FAST_INPUT = True
except ImportError:               # pragma: no cover
    _FAST_INPUT = False

VK = {"up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27}
PUL = ctypes.POINTER(ctypes.c_ulong)


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", PUL),
    ]


class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT)]

    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUT)]


SendInput = ctypes.windll.user32.SendInput  # type: ignore[attr-defined]

# ─── debug settings ───────────────────────────────────────────────
SHOW_DEBUG = True
DEBUG_INTERVAL = 25
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ─── reward constants ─────────────────────────────────────────────
STEP_PENALTY = -0.002
TIME_PENALTY_FACTOR = -0.001
LEAVE_SPAWN_BONUS = 0.0
APPROACH_COIN_REWARD = 5.0
COIN_COLLECT_REWARD = 500.0
CHECKPOINT_BONUS = 250.0
LEVEL_COMPLETE_REWARD = 8000.0
ENEMY_PROX_PENALTY = -3.0
ENEMY_PROX_THRESHOLD = 80
ENEMY_COLLISION_PENALTY = -120.0
RESPAWN_PENALTY = -50.0
NO_MOVE_BASE_PENALTY = -3.0
OUTSIDE_SPAWN_REWARD = 1.0
INSIDE_SPAWN_PENALTY = -1.5


# ─── geometry (px) ────────────────────────────────────────────────
ENEMY_HIT_RADIUS = 20
GOAL_HIT_RADIUS = 20
COIN_HIT_RADIUS = 20
WALL_HIT_RADIUS = 15
SPAWN_RADIUS = 25


class Environment:
    """Gym‑like wrapper around the World's Hardest Game running in Ruffle."""

    KEY_PRESS_DURATION = 0.001    # seconds key held
    REGION_REFRESH_EVERY = 10     # frames

    # ───────────────────────── initialisation ─────────────────────
    def __init__(self, *, epsilon: float = 0.0):
        self.action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        self.epsilon = epsilon
        self.region = get_ruffle_window_region()

        self._held_key: Optional[str] = None

        # episode‑state placeholders
        self.spawn_pos: Tuple[int, int] = (0, 0)
        self.start_pos: Tuple[int, int] = (0, 0)
        self.goal_pos: Tuple[int, int] = (0, 0)
        self.total_coins = 0
        self.collected = 0
        self.prev_coin_dist: Optional[float] = None
        self.left_spawn = False        # becomes True once player exits spawn

        # motion & event tracking
        self.prev_player_pos: Optional[np.ndarray] = None
        self.last_player: Tuple[int, int] = (0, 0)
        self.no_move_counter = 0
        self.enemy_hits = 0

        # misc
        self._refresh_cnt = self._dbg_cnt = 0
        self.episode_reward = 0.0

    # ───────────────────────── reset ───────────────────────────────
    def reset(self) -> np.ndarray:
        """Reset environment state after game reload or episode end."""
        self.region = get_ruffle_window_region()
        frame = capture_screen(self.region)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        objs = detect_objects(frame)        # detector converts BGR→RGB itself

        # robust spawn retrieval
        player_list = objs.get("player", [])
        self.spawn_pos = player_list[0] if player_list else (
            self.region["width"] // 2,
            self.region["height"] // 2,
        )
        self.last_player = self.spawn_pos

        # define start & main goal
        self.start_pos, self.goal_pos = self._find_start_goal(objs)

        # episode‑level counters / trackers
        self.total_coins = len(objs.get("coin", []))
        self.collected = 0
        self.prev_coin_dist = self._nearest_coin_dist(objs)
        self.left_spawn = False
        self.prev_player_pos = np.array(self.spawn_pos, dtype=np.float32)
        self.no_move_counter = 0
        self.enemy_hits = 0
        self._refresh_cnt = self._dbg_cnt = 0
        self.episode_reward = 0.0

        # ensure no keys are stuck
        if self._held_key is not None:
            self._send_input(VK[self._held_key], key_down=False)
            self._held_key = None

        return self._state_from_objects(objs)

    # ───────────────────────── step ────────────────────────────────
    def step(self, action: int):
        """Advance the environment by one action step."""
        self._press_key(action)

        # periodic window‑region refresh (handles window moves)
        self._refresh_cnt += 1
        if self._refresh_cnt >= self.REGION_REFRESH_EVERY:
            self.region = get_ruffle_window_region()
            self._refresh_cnt = 0

        # capture & detect (raw BGR frame)
        frame = capture_screen(self.region)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        objs  = detect_objects(frame)
        state = self._state_from_objects(objs)
        reward, done = self._compute_reward(state, objs)
        self.episode_reward += reward

        # debug overlay / screenshots
        if SHOW_DEBUG:
            self._dbg_cnt += 1
            if self._dbg_cnt % DEBUG_INTERVAL == 0:
                self._show_debug(frame.copy(), objs)

        return state, reward, done, {
            "step_penalty": STEP_PENALTY,
            "time_penalty": TIME_PENALTY_FACTOR,
            "coins": self.collected,
            "enemy_hits": self.enemy_hits,
            "episode_reward": self.episode_reward,
        }

    # ─────────────────── key‑press helper ──────────────────────────
    def _press_key(self, action: int):
        key = self.action_map[action]
        vk = VK[key]

        if key == self._held_key:
            return  # already pressed

        # release old key
        if self._held_key is not None:
            self._send_input(VK[self._held_key], key_down=False)

        # press new key
        self._send_input(vk, key_down=True)
        self._held_key = key

    @staticmethod
    def _send_input(vk: int, *, key_down: bool):
        KEYEVENTF_KEYUP = 0x0002
        flags = 0 if key_down else KEYEVENTF_KEYUP
        ctypes.windll.user32.keybd_event(vk, 0, flags, 0)  # type: ignore[attr-defined]

    # ─────────────────── state encoding ────────────────────────────
    def _state_from_objects(self, objs: Dict[str, List[Tuple[int, int]]]) -> np.ndarray:
        """Return [player_x, player_y, goal_x, goal_y] normalised to [0,1]."""
        w, h = self.region["width"], self.region["height"]

        player = objs.get("player", [self.last_player])[0]
        self.last_player = player

        # choose the furthest goal from the start to encourage progress
        goals = objs.get("goal", [])
        if goals:
            target = max(goals, key=lambda g: np.linalg.norm(np.array(g) - np.array(self.start_pos)))
        else:
            target = player

        return np.array([player[0] / w, player[1] / h,
                         target[0] / w, target[1] / h],
                        dtype=np.float32)

    # ─────────────────── reward function ───────────────────────────
    def _compute_reward(self, s: np.ndarray, objs: Dict[str, List[Tuple[int, int]]]):
        w, h = self.region["width"], self.region["height"]
        pxy = np.array([s[0] * w, s[1] * h])

        reward = STEP_PENALTY + TIME_PENALTY_FACTOR
        done = False

        # --- spawn handling ------------------------------------------------
        dist_from_spawn = np.linalg.norm(pxy - self.spawn_pos)

        if dist_from_spawn > SPAWN_RADIUS:
            # outside spawn
            if not self.left_spawn:
                self.left_spawn = True
                reward += LEAVE_SPAWN_BONUS          # one-time bonus
            reward += OUTSIDE_SPAWN_REWARD           # every step while outside
        else:
            # still in spawn
            reward += INSIDE_SPAWN_PENALTY           # discourage dithering


        # --- proximity penalty (help avoid enemies) -------------------
        if objs.get("enemy"):
            d_enemy = min(np.linalg.norm(pxy - e) for e in objs["enemy"])
            if d_enemy < ENEMY_PROX_THRESHOLD:
                # linear ramp: -3 at 0 px ➜ 0 at 80 px
                reward += ENEMY_PROX_PENALTY * (1 - d_enemy / ENEMY_PROX_THRESHOLD)

        # --- collisions ----------------------------------------------------
        collided = any(np.linalg.norm(pxy - e) < ENEMY_HIT_RADIUS
                       for e in objs.get("enemy", []))
        if collided:
            reward += ENEMY_COLLISION_PENALTY
            self.enemy_hits += 1

        # treat collision outside spawn as death / respawn
        respawn = any(
            np.linalg.norm(pxy - self.spawn_pos) > SPAWN_RADIUS and
            np.linalg.norm(pxy - e) < ENEMY_HIT_RADIUS
            for e in objs.get("enemy", [])
        )
        if respawn:
            reward += RESPAWN_PENALTY
            self.collected = 0
            self.left_spawn = False   # freeze idle penalty until player exits spawn
            self.no_move_counter = 0
            self.prev_coin_dist = None

        # --- spawn‑area bonus & per‑step reward ---------------------------
        outside_spawn = np.linalg.norm(pxy - self.spawn_pos) > SPAWN_RADIUS

        if outside_spawn:
            if not self.left_spawn:          # first time leaving
                self.left_spawn = True
                reward += LEAVE_SPAWN_BONUS
            reward += OUTSIDE_SPAWN_REWARD   # every step while outside


        # --- coin shaping --------------------------------------------------
        coin_dist = self._nearest_coin_dist(objs)
        if coin_dist is not None:
            if self.prev_coin_dist is not None and coin_dist < self.prev_coin_dist - 1:
                reward += APPROACH_COIN_REWARD
            self.prev_coin_dist = coin_dist

        # --- coin collected -----------------------------------------------
        coins_left = len(objs.get("coin", []))
        if coins_left < self.total_coins - self.collected:
            self.collected = self.total_coins - coins_left
            reward += COIN_COLLECT_REWARD

        # --- checkpoint bonus ---------------------------------------------
        for g in objs.get("goal", []):
            if np.linalg.norm(pxy - g) < GOAL_HIT_RADIUS and g != self.goal_pos:
                reward += CHECKPOINT_BONUS

        # --- level complete -----------------------------------------------
        if self.collected == self.total_coins and any(
            np.linalg.norm(pxy - g) < GOAL_HIT_RADIUS for g in objs.get("goal", [])
        ):
            reward += LEVEL_COMPLETE_REWARD
            done = True

        # --- wall / idle linear penalty -----------------------------------
        move = np.linalg.norm(pxy - self.prev_player_pos) if self.prev_player_pos is not None else 1
        wall_hit = any(np.linalg.norm(pxy - w_) < WALL_HIT_RADIUS for w_ in objs.get("wall", []))

        if self.left_spawn and (move < 1.0 or wall_hit):
            self.no_move_counter += 1                 # counts consecutive idle / wall frames
            reward += NO_MOVE_BASE_PENALTY * self.no_move_counter
        else:
            self.no_move_counter = 0

        # ⬇️ add these two lines ⬇️
        self.prev_player_pos = pxy
        return float(reward), done


    # ─────────────────── helper utilities ──────────────────────────
    def _find_start_goal(self, objs):
        checkpoints = objs.get("goal", [])
        if checkpoints:
            goal = min(checkpoints, key=lambda g: np.linalg.norm(np.array(g) - np.array(self.spawn_pos)))
            return self.spawn_pos, goal
        w, h = self.region["width"], self.region["height"]
        return self.spawn_pos, (w // 2, h // 2)

    def _nearest_coin_dist(self, objs) -> Optional[float]:
        player = self.last_player
        coins = objs.get("coin", [])
        if not coins:
            return None
        return min(np.linalg.norm(np.array(player) - np.array(c)) for c in coins)

    # ─────────────────── debug overlay ─────────────────────────────
    def _show_debug(self, frame: np.ndarray, objs):
        """
        Draws the *actual* detected shapes (full contours) instead of fixed-size
        markers, so you can see exactly how large the agent thinks each object is.
        Colours are the same as before.
        """
        dbg = frame.copy()                   # work on a copy

        # import the colour ranges from object_detection
        from object_detection import (
            R_RED_LO,  R_RED_HI,
            B_ENEMY_LO, B_ENEMY_HI,
            Y_COIN_LO, Y_COIN_HI,
            G_GOAL_LO, G_GOAL_HI,
        )

        # each entry: ((lower, upper), BGR colour to draw)
        mask_specs = [
            ((R_RED_LO,  R_RED_HI),  (255,   0, 255)),   # player  → magenta
            ((B_ENEMY_LO, B_ENEMY_HI), (255, 255, 255)), # enemy   → white
            ((Y_COIN_LO,  Y_COIN_HI),  (  0,   0,   0)), # coin    → black
            ((G_GOAL_LO,  G_GOAL_HI),  (  0, 255,   0)), # goal    → green
        ]

        for (lo, hi), colour in mask_specs:
            mask = cv2.inRange(dbg,
                               np.array(lo, dtype=np.uint8),
                               np.array(hi, dtype=np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dbg, contours, -1, colour, 1)

        # ─── HUD (unchanged) ──────────────────────────────────────
        hud_lines = [
            f"R_tot {self.episode_reward:+7.1f}",
            f"Coins {self.collected}/{self.total_coins}",
            f"Enemies {len(objs.get('enemy', []))}",
            f"Hits {self.enemy_hits}",
            f"Step {self._dbg_cnt}",
        ]
        for i, text in enumerate(hud_lines):
            cv2.putText(
                dbg, text,
                (10, 125 + 20 * i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA
            )

        cv2.imshow("Agent View", dbg)
        if cv2.waitKey(1) & 0xFF == 27:      # Esc closes debug window
            cv2.destroyAllWindows()


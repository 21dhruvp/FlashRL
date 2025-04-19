"""
environment.py  –  World’s Hardest Game RL wrapper
--------------------------------------------------
• 4‑value state     : [player_x, player_y, goal_x, goal_y] (normalised)
• Rich reward table : coins, goals, collisions, idle‑at‑wall penalty
• Debug overlay     : live preview + PNG dump every DEBUG_INTERVAL steps
"""

import os
import time
from typing import Tuple

import cv2
import numpy as np
import pyautogui

from game_capture import capture_screen, get_ruffle_window_region
from object_detection import detect_objects

# Try faster DirectInput key injection (Windows only)
try:
    import pydirectinput
    _FAST_INPUT = True
except ImportError:
    _FAST_INPUT = False

# ───────────────────────────── Debug settings ──────────────────────────────
SHOW_DEBUG      = True           # turn overlays / screenshots on or off
DEBUG_INTERVAL  = 20             # env steps between debug frames
SCREENSHOT_DIR  = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ───────────────────────────── Environment class ───────────────────────────
class Environment:
    """World's Hardest Game wrapper with richer penalties and debug preview."""

    # Timing / capture
    KEY_PRESS_DURATION   = 0.001   # seconds key is held
    REGION_REFRESH_EVERY = 100     # steps between window‑size refresh

    # Geometry thresholds (pixels)
    ENEMY_HIT_RADIUS = 20
    GOAL_HIT_RADIUS  = 20
    COIN_HIT_RADIUS  = 20
    WALL_HIT_RADIUS  = 15
    SPAWN_RADIUS     = 25

    # Rewards / penalties
    STEP_PENALTY            = -0.05
    LEAVE_SPAWN_BONUS       =  20.00
    APPROACH_COIN_REWARD    =  0.50
    COIN_COLLECT_REWARD     = 20.0
    LEVEL_COMPLETE_REWARD   = 60.0
    ENEMY_COLLISION_PENALTY = -10.0
    RESPAWN_PENALTY         = -5.0
    NO_MOVE_BASE_PENALTY    = -0.02   # doubled each consecutive frame

    # --------------------------------------------------------------------- #
    def __init__(self):
        self.action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        self.region = get_ruffle_window_region()

        # Episode‑specific state
        self.spawn_pos: Tuple[int, int] | None = None
        self.start_goal: Tuple[int, int] | None = None
        self.total_coins = 0
        self.collected_coins = 0
        self.prev_coin_dist: float | None = None
        self.left_spawn = False

        # Movement tracking
        self.prev_player_pos: np.ndarray | None = None
        self.no_move_counter = 0

        # Book‑keeping
        self._steps_since_refresh = 0
        self._dbg_cnt = 0
        self.episode_reward = 0.0

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def reset(self) -> np.ndarray:
        """Start a new episode and return initial state vector."""
        self.region = get_ruffle_window_region()
        frame   = capture_screen(region=self.region)
        objects = detect_objects(frame)

        # Spawn & goal
        self.spawn_pos = objects["player"][0] if objects["player"] else (0, 0)
        self.start_goal = (
            min(objects["goal"], key=lambda g: np.linalg.norm(np.array(g) - np.array(self.spawn_pos)))
            if objects["goal"] else None
        )

        # Coin bookkeeping
        self.total_coins     = len(objects["coin"])
        self.collected_coins = 0
        self.prev_coin_dist  = self._nearest_coin_dist(objects)

        # Flags / counters
        self.left_spawn          = False
        self.prev_player_pos     = np.array(self.spawn_pos)
        self.no_move_counter     = 0
        self._steps_since_refresh = 0
        self._dbg_cnt             = 0
        self.episode_reward       = 0.0

        if SHOW_DEBUG:
            cv2.namedWindow("Agent view", cv2.WINDOW_NORMAL)

        return self._state_from_objects(objects)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute action, update environment, return (state, reward, done)."""
        self._press_key(action)

        # Refresh capture region occasionally
        self._steps_since_refresh += 1
        if self._steps_since_refresh >= self.REGION_REFRESH_EVERY:
            self.region = get_ruffle_window_region()
            self._steps_since_refresh = 0
            print(f"[env] reward so far: {self.episode_reward:.2f}")

        # Capture & detect
        frame   = capture_screen(region=self.region)
        objects = detect_objects(frame)

        # Debug preview / screenshot
        if SHOW_DEBUG and self._dbg_cnt % DEBUG_INTERVAL == 0:
            self._show_debug(frame, objects)
        self._dbg_cnt += 1

        state  = self._state_from_objects(objects)
        reward, done = self._compute_reward(state, objects)

        self.episode_reward += reward
        return state, reward, done

    # --------------------------------------------------------------------- #
    # Key press helper
    # --------------------------------------------------------------------- #
    def _press_key(self, action: int) -> None:
        key = self.action_map[action]
        injector = pydirectinput if _FAST_INPUT else pyautogui
        injector.keyDown(key)
        time.sleep(self.KEY_PRESS_DURATION)
        injector.keyUp(key)

    # --------------------------------------------------------------------- #
    # State vector
    # --------------------------------------------------------------------- #
    def _state_from_objects(self, objects: dict) -> np.ndarray:
        w, h = self.region["width"], self.region["height"]
        player = objects["player"][0] if objects["player"] else (0, 0)

        goals = objects["goal"]
        if self.start_goal and len(goals) > 1:
            target_goal = max(goals, key=lambda g: np.linalg.norm(np.array(g) - np.array(self.start_goal)))
        elif goals:
            target_goal = goals[0]
        else:
            target_goal = player

        return np.array(
            [player[0] / w, player[1] / h, target_goal[0] / w, target_goal[1] / h],
            dtype=np.float32,
        )

    # --------------------------------------------------------------------- #
    # Reward logic (unchanged)
    # --------------------------------------------------------------------- #
    def _compute_reward(self, state: np.ndarray, objects: dict) -> Tuple[float, bool]:
        w, h = self.region["width"], self.region["height"]
        player_px = np.array([state[0] * w, state[1] * h])

        reward = self.STEP_PENALTY
        done   = False

        # Enemy collision
        if any(np.linalg.norm(player_px - e) < self.ENEMY_HIT_RADIUS for e in objects["enemy"]):
            reward += self.ENEMY_COLLISION_PENALTY

        # Respawn detection
        if self.left_spawn and np.linalg.norm(player_px - np.array(self.spawn_pos)) < self.SPAWN_RADIUS:
            reward += self.RESPAWN_PENALTY
            self.collected_coins = 0
            self.total_coins     = len(objects["coin"])
            self.prev_coin_dist  = self._nearest_coin_dist(objects)
            self.left_spawn      = False
            self.no_move_counter = 0

        # Leave spawn first time
        if not self.left_spawn and np.linalg.norm(player_px - np.array(self.spawn_pos)) > self.SPAWN_RADIUS:
            self.left_spawn = True
            reward += self.LEAVE_SPAWN_BONUS

        # Coin shaping
        coin_dist = self._nearest_coin_dist(objects)
        if self.total_coins > 0:
            if self.prev_coin_dist is not None and coin_dist < self.prev_coin_dist - 1.0:
                reward += self.APPROACH_COIN_REWARD
            self.prev_coin_dist = coin_dist

            current_coins = len(objects["coin"])
            if current_coins < self.total_coins - self.collected_coins:
                self.collected_coins += 1
                reward += self.COIN_COLLECT_REWARD

        # Goal completion
        if self.collected_coins == self.total_coins:
            goals = objects["goal"]
            if self.start_goal and len(goals) > 1:
                goals = [g for g in goals if np.linalg.norm(np.array(g) - np.array(self.start_goal)) > 5]
            for goal in goals:
                if np.linalg.norm(player_px - goal) < self.GOAL_HIT_RADIUS:
                    reward += self.LEVEL_COMPLETE_REWARD
                    done = True
                    break

        # Wall / no‑move exponential penalty
        move_dist = np.linalg.norm(player_px - self.prev_player_pos) if self.prev_player_pos is not None else 1.0
        wall_near = any(np.linalg.norm(player_px - w_) < self.WALL_HIT_RADIUS for w_ in objects["wall"])
        if move_dist < 1.0 or wall_near:
            self.no_move_counter += 1
            reward += self.NO_MOVE_BASE_PENALTY * (2 ** (self.no_move_counter - 1))
        else:
            self.no_move_counter = 0

        self.prev_player_pos = player_px
        return reward, done

    # --------------------------------------------------------------------- #
    # Utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def _nearest_coin_dist(objects: dict) -> float:
        player = objects["player"][0] if objects["player"] else (0, 0)
        if objects["coin"]:
            return min(np.linalg.norm(np.array(player) - np.array(c)) for c in objects["coin"])
        return 1e6

    def _show_debug(self, frame: np.ndarray, objects: dict) -> None:
        """Draw overlays and save PNG to screenshots/."""
        dbg = frame.copy()
        for x, y in objects["player"]:
            cv2.drawMarker(dbg, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=2)
        for x, y in objects["enemy"]:
            cv2.circle(dbg, (x, y), 8, (255, 0, 0), 2)
        for x, y in objects["coin"]:
            cv2.circle(dbg, (x, y), 6, (0, 255, 255), 2)
        for x, y in objects["goal"]:
            cv2.rectangle(dbg, (x - 4, y - 4), (x + 4, y + 4), (0, 255, 0), 1)

        cv2.imshow("Agent view", dbg)
        cv2.waitKey(1)

        ts = int(time.time() * 1000)
        cv2.imwrite(os.path.join(SCREENSHOT_DIR, f"dbg_{ts}.png"), dbg)

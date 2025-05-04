"""
environment.py – World's Hardest Game RL wrapper
------------------------------------------------
Crash‑proof: if the red player sprite is not detected in a frame, the
environment re‑uses the last known position instead of raising IndexError.
Reward shaping:
  • step/time penalties
  • distance‑to‑nearest‑coin shaping
  • wall/idle exponential penalties
  • coin, checkpoint, and level‑complete bonuses
"""

from __future__ import annotations
import os, time
from typing import Tuple, Dict, List, Optional

import cv2
import numpy as np
import pyautogui

from game_capture import capture_screen, get_ruffle_window_region
from object_detection import detect_objects

try:
    import pydirectinput
    _FAST_INPUT = True
except ImportError:
    _FAST_INPUT = False

# ─── debug settings ──────────────────────────────────────────────────
SHOW_DEBUG      = True
DEBUG_INTERVAL  = 20
SCREENSHOT_DIR  = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ─── reward constants ───────────────────────────────────────────────
STEP_PENALTY              = -0.05
TIME_PENALTY_FACTOR       = -0.01
LEAVE_SPAWN_BONUS         =  20.0
APPROACH_COIN_REWARD      =   0.5
COIN_COLLECT_REWARD       =  20.0
CHECKPOINT_BONUS          =  15.0
LEVEL_COMPLETE_REWARD     =  60.0
ENEMY_COLLISION_PENALTY   = -10.0
RESPAWN_PENALTY           =  -5.0
NO_MOVE_BASE_PENALTY      =  -0.02

# ─── geometry (px) ──────────────────────────────────────────────────
ENEMY_HIT_RADIUS = 20
GOAL_HIT_RADIUS  = 20
COIN_HIT_RADIUS  = 20
WALL_HIT_RADIUS  = 15
SPAWN_RADIUS     = 25


class Environment:
    KEY_PRESS_DURATION   = 0.001       # sec key held
    REGION_REFRESH_EVERY = 100         # frames

    def __init__(self, *, epsilon: float = 0.0):
        self.action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        self.epsilon = epsilon
        self.region = get_ruffle_window_region()

        # episode‑state placeholders
        self.spawn_pos : Tuple[int,int] = (0, 0)
        self.start_pos : Tuple[int,int] = (0, 0)
        self.goal_pos  : Tuple[int,int] = (0, 0)
        self.total_coins = 0
        self.collected   = 0
        self.prev_coin_dist: Optional[float] = None
        self.left_spawn  = False

        # motion tracking
        self.prev_player_pos: Optional[np.ndarray] = None
        self.last_player: Tuple[int,int] = (0, 0)
        self.no_move_counter = 0

        # misc
        self._refresh_cnt = self._dbg_cnt = 0
        self.episode_reward = 0.0

    # ───────────────────────── reset ────────────────────────────────
    def reset(self) -> np.ndarray:
        self.region  = get_ruffle_window_region()
        frame        = capture_screen(self.region)
        objs         = detect_objects(frame)

        # robust spawn retrieval
        player_list  = objs.get("player", [])
        self.spawn_pos = player_list[0] if player_list else (self.region["width"]//2,
                                                             self.region["height"]//2)
        self.last_player = self.spawn_pos

        # find first checkpoint/finish as goal
        self.start_pos, self.goal_pos = self._find_start_goal(objs)

        self.total_coins      = len(objs.get("coin", []))
        self.collected        = 0
        self.prev_coin_dist   = self._nearest_coin_dist(objs)
        self.left_spawn       = False
        self.prev_player_pos  = np.array(self.spawn_pos, dtype=np.float32)
        self.no_move_counter  = 0
        self._refresh_cnt     = self._dbg_cnt = 0
        self.episode_reward   = 0.0
        return self._state_from_objects(objs)

    # ───────────────────────── step ────────────────────────────────
    def step(self, action: int):
        self._press_key(action)

        # periodic window‑region refresh
        self._refresh_cnt += 1
        if self._refresh_cnt >= self.REGION_REFRESH_EVERY:
            self.region = get_ruffle_window_region()
            self._refresh_cnt = 0

        frame = capture_screen(self.region)
        objs  = detect_objects(frame)
        state = self._state_from_objects(objs)
        reward, done = self._compute_reward(state, objs)
        self.episode_reward += reward

        # debug overlay / screenshots
        if SHOW_DEBUG:
            self._dbg_cnt += 1
            if self._dbg_cnt % DEBUG_INTERVAL == 0:
                self._show_debug(frame, objs)

        return state, reward, done, {
            "step_penalty": STEP_PENALTY,
            "time_penalty": TIME_PENALTY_FACTOR,
            "coins": self.collected,
            "episode_reward": self.episode_reward,
        }

    # ─────────────────── key‑press helper ───────────────────────────
    def _press_key(self, action: int):
        key = self.action_map[action]
        injector = pydirectinput if _FAST_INPUT else pyautogui
        injector.keyDown(key)
        time.sleep(self.KEY_PRESS_DURATION)
        injector.keyUp(key)

    # ─────────────────── state encoding ─────────────────────────────
    def _state_from_objects(self, objs: Dict[str, List[Tuple[int, int]]]) -> np.ndarray:
        w, h = self.region["width"], self.region["height"]

        # robust player retrieval
        player_list = objs.get("player", [])
        if player_list:
            player = player_list[0]
            self.last_player = player
        else:
            player = self.last_player

        # simple target: farthest goal from spawn (encourages progress)
        goals = objs.get("goal", [])
        if goals:
            target = max(goals, key=lambda g: np.linalg.norm(np.array(g) - np.array(self.start_pos)))
        else:
            target = player

        return np.array([player[0] / w, player[1] / h,
                         target[0] / w, target[1] / h],
                        dtype=np.float32)

    # ─────────────────── reward function ────────────────────────────
    def _compute_reward(self, s: np.ndarray, objs):
        w, h = self.region["width"], self.region["height"]
        pxy  = np.array([s[0] * w, s[1] * h])

        reward = STEP_PENALTY + TIME_PENALTY_FACTOR
        done   = False

        # --- collisions ----------------------------------------------------
        if any(np.linalg.norm(pxy - e) < ENEMY_HIT_RADIUS for e in objs.get("enemy", [])):
            reward += ENEMY_COLLISION_PENALTY
        if any(np.linalg.norm(pxy - self.spawn_pos) > SPAWN_RADIUS and
               np.linalg.norm(pxy - e) < ENEMY_HIT_RADIUS for e in objs.get("enemy", [])):
            reward += RESPAWN_PENALTY
            self.collected = 0
            self.left_spawn = False
            self.no_move_counter = 0

        # --- left spawn check ----------------------------------------------
        if (not self.left_spawn and
            np.linalg.norm(pxy - np.array(self.spawn_pos)) > SPAWN_RADIUS):
            self.left_spawn = True
            reward += LEAVE_SPAWN_BONUS

        # --- coin shaping --------------------------------------------------
        coin_dist = self._nearest_coin_dist(objs)
        if coin_dist is not None:
            if self.prev_coin_dist is not None and coin_dist < self.prev_coin_dist - 1:
                reward += APPROACH_COIN_REWARD
            self.prev_coin_dist = coin_dist

        # --- coin collected ------------------------------------------------
        coins_left = len(objs.get("coin", []))
        if coins_left < self.total_coins - self.collected:
            self.collected = self.total_coins - coins_left
            reward += COIN_COLLECT_REWARD

        # --- checkpoint bonus ---------------------------------------------
        for g in objs.get("goal", []):
            if np.linalg.norm(pxy - g) < GOAL_HIT_RADIUS and g != self.goal_pos:
                reward += CHECKPOINT_BONUS

        # --- level complete ------------------------------------------------
        if (self.collected == self.total_coins and
            any(np.linalg.norm(pxy - g) < GOAL_HIT_RADIUS for g in objs.get("goal", []))):
            reward += LEVEL_COMPLETE_REWARD
            done = True

        # --- wall / idle exponential penalty ------------------------------
        move = (np.linalg.norm(pxy - self.prev_player_pos)
                if self.prev_player_pos is not None else 1)
        wall = any(np.linalg.norm(pxy - w_) < WALL_HIT_RADIUS for w_ in objs.get("wall", []))
        if move < 1.0 or wall:
            self.no_move_counter += 1
            reward += NO_MOVE_BASE_PENALTY * (2 ** (self.no_move_counter - 1))
        else:
            self.no_move_counter = 0
        self.prev_player_pos = pxy

        return float(reward), done

    # ─────────────────── helper utilities ───────────────────────────
    def _find_start_goal(self, objs):
        """Return (start_pos, goal_pos) or sensible fallbacks."""
        checkpoints = objs.get("goal", [])
        if checkpoints:
            goal = min(checkpoints,
                       key=lambda g: np.linalg.norm(np.array(g) - np.array(self.spawn_pos)))
            return self.spawn_pos, goal
        # fallback: treat centre as goal
        w, h = self.region["width"], self.region["height"]
        return self.spawn_pos, (w // 2, h // 2)

    def _nearest_coin_dist(self, objs) -> Optional[float]:
        player = self.last_player
        coins  = objs.get("coin", [])
        if not coins:
            return None
        return min(np.linalg.norm(np.array(player) - np.array(c)) for c in coins)

    # ─────────────────── debug overlay ──────────────────────────────
    def _show_debug(self, frame, objs):
        dbg = frame.copy()
        for x, y in objs.get("player", []):
            cv2.drawMarker(dbg, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 2)
        for x, y in objs.get("enemy", []):
            cv2.circle(dbg, (x, y), 6, (255, 0, 0), 1)
        for x, y in objs.get("coin", []):
            cv2.circle(dbg, (x, y), 4, (0, 255, 255), 1)
        for x, y in objs.get("goal", []):
            cv2.rectangle(dbg, (x - 3, y - 3), (x + 3, y + 3), (0, 255, 0), 1)

        hud = [
            f"R_tot {self.episode_reward:+7.1f}",
            f"Coins {self.collected}/{self.total_coins}",
            f"ε {self.epsilon:.2f}",
            f"Step {self._dbg_cnt}",
        ]
        for i, t in enumerate(hud):
            cv2.putText(dbg, t, (10, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Agent View", dbg)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit debug display
            cv2.destroyAllWindows()

        # periodic screenshot
        if self._dbg_cnt % (DEBUG_INTERVAL * 10) == 0:
            cv2.imwrite(os.path.join(SCREENSHOT_DIR, f"dbg_{self._dbg_cnt:06d}.png"), dbg)

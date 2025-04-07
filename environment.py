import time
from typing import Tuple

import numpy as np
import pyautogui

from game_capture import capture_screen, get_ruffle_window_region
from object_detection import detect_objects

# Faster key injection on Windows if available
try:
    import pydirectinput
    _FAST_INPUT = True
except ImportError:
    _FAST_INPUT = False


class Environment:
    """World's Hardest Game wrapper with richer penalties.

    • Detects *respawn* events (player teleported back to spawn) and applies a
      heavy negative reward without ending the episode.
    • Adds an exponentially growing penalty whenever the avatar fails to move
      (e.g. walks into a wall).
    • Keeps earlier reward‑shaping for coins, speed, and completion.
    """

    # Timing / capture
    KEY_PRESS_DURATION   = 0.004     # seconds key is held
    REGION_REFRESH_EVERY = 100       # env steps between window‑size refreshes

    # Geometry thresholds (pixels)
    ENEMY_HIT_RADIUS = 20
    GOAL_HIT_RADIUS  = 20
    COIN_HIT_RADIUS  = 20
    WALL_HIT_RADIUS  = 15
    SPAWN_RADIUS     = 25

    # Rewards / penalties
    STEP_PENALTY             = -0.05   # constant per step (speed incentive)
    LEAVE_SPAWN_BONUS        =  0.10
    APPROACH_COIN_REWARD     =  0.05
    COIN_COLLECT_REWARD      = 10.0
    LEVEL_COMPLETE_REWARD    = 50.0
    ENEMY_COLLISION_PENALTY  = -10.0
    RESPAWN_PENALTY          = -15.0   # applied when player returns to spawn
    NO_MOVE_BASE_PENALTY     = -0.02   # doubled each consecutive frame

    def __init__(self):
        self.action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        self.region = get_ruffle_window_region()

        # Episode‑specific state
        self.spawn_pos: Tuple[int, int] | None = None
        self.start_goal: Tuple[int, int] | None = None
        self.total_coins: int = 0
        self.collected_coins: int = 0
        self.prev_coin_dist: float | None = None
        self.left_spawn: bool = False

        # Movement tracking
        self.prev_player_pos: np.ndarray | None = None
        self.no_move_counter: int = 0

        # Bookkeeping
        self._steps_since_refresh = 0
        self.episode_reward = 0.0

    # ── Public API ──────────────────────────────────────────
    def reset(self) -> np.ndarray:
        """Initialise a new episode and return the first state."""
        self.region = get_ruffle_window_region()
        frame   = capture_screen(region=self.region)
        objects = detect_objects(frame)

        # Spawn & goal setup
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
        self.left_spawn       = False
        self.prev_player_pos  = np.array(self.spawn_pos)
        self.no_move_counter  = 0
        self._steps_since_refresh = 0
        self.episode_reward   = 0.0

        return self._state_from_objects(objects)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute *action* and return *(next_state, reward, done)*."""
        self._take_action(action)

        # Periodically refresh capture region (window may move)
        self._steps_since_refresh += 1
        if self._steps_since_refresh >= self.REGION_REFRESH_EVERY:
            self.region = get_ruffle_window_region()
            self._steps_since_refresh = 0
            print(f"[env] reward so far: {self.episode_reward:.2f}")

        frame   = capture_screen(region=self.region)
        objects = detect_objects(frame)
        state   = self._state_from_objects(objects)
        reward, done = self._compute_reward(state, objects)

        self.episode_reward += reward
        return state, reward, done

    # ── Helpers ────────────────────────────────────────────
    def _take_action(self, action: int) -> None:
        key = self.action_map[action]
        if _FAST_INPUT:
            pydirectinput.keyDown(key)
            time.sleep(self.KEY_PRESS_DURATION)
            pydirectinput.keyUp(key)
        else:
            pyautogui.keyDown(key)
            time.sleep(self.KEY_PRESS_DURATION)
            pyautogui.keyUp(key)

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

        return np.array([player[0]/w, player[1]/h, target_goal[0]/w, target_goal[1]/h], dtype=np.float32)

    # ── Reward function ───────────────────────────────────
    def _compute_reward(self, state: np.ndarray, objects: dict) -> Tuple[float, bool]:
        w, h = self.region["width"], self.region["height"]
        player_px = np.array([state[0]*w, state[1]*h])

        reward = self.STEP_PENALTY
        done   = False

        # 1) Enemy collision – big penalty but keep episode alive
        if any(np.linalg.norm(player_px - e) < self.ENEMY_HIT_RADIUS for e in objects["enemy"]):
            reward += self.ENEMY_COLLISION_PENALTY

        # 2) Detect respawn (player returns to spawn after having left)
        if self.left_spawn and np.linalg.norm(player_px - np.array(self.spawn_pos)) < self.SPAWN_RADIUS:
            reward += self.RESPAWN_PENALTY
            # reset coin tracking and flags
            self.collected_coins = 0
            self.total_coins     = len(objects["coin"])
            self.prev_coin_dist  = self._nearest_coin_dist(objects)
            self.left_spawn      = False
            self.no_move_counter = 0

        # 3) Leaving spawn for the first time
        if not self.left_spawn and np.linalg.norm(player_px - np.array(self.spawn_pos)) > self.SPAWN_RADIUS:
            self.left_spawn = True
            reward += self.LEAVE_SPAWN_BONUS

        # 4) Coin‑centric shaping
        coin_dist = self._nearest_coin_dist(objects)
        if self.total_coins > 0:
            if self.prev_coin_dist is not None and coin_dist < self.prev_coin_dist - 1.0:
                reward += self.APPROACH_COIN_REWARD
            self.prev_coin_dist = coin_dist

            # Coin collected (coin count decreased)
            current_coins = len(objects["coin"])
            if current_coins < self.total_coins - self.collected_coins:
                self.collected_coins += 1
                reward += self.COIN_COLLECT_REWARD

        # 5) Goal reached only if all coins collected
        if self.collected_coins == self.total_coins and self.total_coins >= 0:
            goals = objects["goal"]
            if self.start_goal and len(goals) > 1:
                goals = [g for g in goals if np.linalg.norm(np.array(g) - np.array(self.start_goal)) > 5]
            for goal in goals:
                if np.linalg.norm(player_px - goal) < self.GOAL_HIT_RADIUS:
                    reward += self.LEVEL_COMPLETE_REWARD
                    done = True
                    break

        # 6) Wall / no‑move penalty (exponential)
        move_dist = np.linalg.norm(player_px - self.prev_player_pos) if self.prev_player_pos is not None else 1.0
        wall_near = any(np.linalg.norm(player_px - w_) < self.WALL_HIT_RADIUS for w_ in objects["wall"])
        if move_dist < 1.0 or wall_near:
            self.no_move_counter += 1
            reward += self.NO_MOVE_BASE_PENALTY * (2 ** (self.no_move_counter - 1))
        else:
            self.no_move_counter = 0

        # Update prev position
        self.prev_player_pos = player_px
        return reward, done

    # ── Utility ────────────────────────────────────────────
    @staticmethod
    def _nearest_coin_dist(objects: dict) -> float:
        player = objects["player"][0] if objects["player"] else (0, 0)
        if objects["coin"]:
            return min(np.linalg.norm(np.array(player) - np.array(c)) for c in objects["coin"])
        return 1e6

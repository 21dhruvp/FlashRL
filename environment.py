import time
from typing import Tuple

import numpy as np
import pyautogui

from game_capture import capture_screen, get_ruffle_window_region
from object_detection import detect_objects

# Use pydirectinput if available (faster key injection on Windows)
try:
    import pydirectinput
    _FAST_INPUT = True
except ImportError:
    _FAST_INPUT = False


class Environment:
    KEY_PRESS_DURATION      = 0.006   # seconds
    REGION_REFRESH_EVERY    = 50     # steps
    ENEMY_HIT_RADIUS        = 20
    GOAL_HIT_RADIUS         = 20
    COIN_HIT_RADIUS         = 20

    def __init__(self):
        self.action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        self.start_goal = None
        self.region = get_ruffle_window_region()
        self._steps_since_refresh = 0
        self.episode_reward = 0.0      # <── NEW

    # ─────────────────────────────── reset / step ──
    def reset(self) -> np.ndarray:
        """Start a new episode and return the initial state."""
        self.region = get_ruffle_window_region()
        frame   = capture_screen(region=self.region)
        objects = detect_objects(frame)

        player_pos = objects["player"][0] if objects["player"] else (0, 0)
        self.start_goal = (
            min(objects["goal"], key=lambda g: np.linalg.norm(np.array(g) - np.array(player_pos)))
            if objects["goal"] else None
        )

        self._steps_since_refresh = 0
        self.episode_reward = 0.0      # reset counter
        return self._state_from_objects(objects)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Perform *action* and return (next_state, reward, done)."""
        self._take_action(action)

        # Refresh capture region periodically
        self._steps_since_refresh += 1
        if self._steps_since_refresh >= self.REGION_REFRESH_EVERY:
            self.region = get_ruffle_window_region()
            self._steps_since_refresh = 0
            # Print running reward counter  ───────────────
            print(f"[env] reward so far this episode: {self.episode_reward:.2f}")

        frame   = capture_screen(region=self.region)
        objects = detect_objects(frame)
        state   = self._state_from_objects(objects)
        reward, done = self._compute_reward(state, objects)

        self.episode_reward += reward      # update counter
        return state, reward, done

    # ─────────────────────────────── helpers ───────
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
        width, height = self.region["width"], self.region["height"]

        player_pos = objects["player"][0] if objects["player"] else (0, 0)
        goals      = objects["goal"]

        if self.start_goal and len(goals) > 1:
            target_goal = max(goals, key=lambda g: np.linalg.norm(np.array(g) - np.array(self.start_goal)))
        elif goals:
            target_goal = goals[0]
        else:
            target_goal = player_pos

        return np.array(
            [
                player_pos[0]  / width,
                player_pos[1]  / height,
                target_goal[0] / width,
                target_goal[1] / height,
            ],
            dtype=np.float32,
        )

    def _compute_reward(self, state: np.ndarray, objects: dict) -> Tuple[float, bool]:
        width, height = self.region["width"], self.region["height"]
        player_px = np.array([state[0] * width, state[1] * height])

        reward = -1   # time penalty
        done   = False

        # Enemy collision
        for enemy in objects["enemy"]:
            if np.linalg.norm(player_px - enemy) < self.ENEMY_HIT_RADIUS:
                return reward - 10, True

        # Goal reached
        goals = objects["goal"]
        if self.start_goal and len(goals) > 1:
            goals = [g for g in goals if np.linalg.norm(np.array(g) - np.array(self.start_goal)) > 5]
        for goal in goals:
            if np.linalg.norm(player_px - goal) < self.GOAL_HIT_RADIUS:
                return reward + 20, True

        # Coin collected
        for coin in objects["coin"]:
            if np.linalg.norm(player_px - coin) < self.COIN_HIT_RADIUS:
                reward += 5
                break

        return reward, done

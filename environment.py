from __future__ import annotations
import os, time, csv
from pathlib import Path
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

# ─── debug & logging settings ─────────────────────────────────────────────
SHOW_DEBUG          = True
DEBUG_INTERVAL      = 20          # frames between HUD refreshes
LOG_REWARDS         = True        # export reward components each step
REWARD_LOG_DIR      = Path("reward_logs")
REWARD_LOG_DIR.mkdir(exist_ok=True, parents=True)

# ─── reward constants ─────────────────────────────────────────────────────
STEP_PENALTY              = -0.01     # small negative for every frame
TIME_PENALTY_FACTOR       = -0.002    # scaled by elapsed steps
LEAVE_SPAWN_BONUS         =  20.0
DISTANCE_COIN_SCALE       =  0.05    # per‑pixel improvement towards nearest coin
COIN_COLLECT_REWARD       =  30.0
SECTION_BONUS             =  25.0     # hitting a green checkpoint tile
LEVEL_COMPLETE_REWARD     =  60.0
ENEMY_COLLISION_PENALTY   = -15.0
RESPAWN_PENALTY           = -10.0
WALL_CONTACT_PENALTY      =  -2.0
NO_MOVE_BASE_PENALTY      =  -0.02    # grows exponentially with idle frames

# ─── geometry (px) ────────────────────────────────────────────────────────
ENEMY_HIT_RADIUS = 20
GOAL_HIT_RADIUS  = 20
COIN_HIT_RADIUS  = 20
WALL_HIT_RADIUS  = 15
SPAWN_RADIUS     = 25

class Environment:
    KEY_PRESS_DURATION   = 0.001
    REGION_REFRESH_EVERY = 120

    def __init__(self, *, epsilon: float = 0.0, log_prefix: str | None = None):
        self.action_map = {0: "up", 1: "down", 2: "left", 3: "right"}
        self.region = get_ruffle_window_region()

        # episode state -----------------------------------------------------
        self.spawn_pos : Tuple[int,int] = (0,0)
        self.start_goal: Tuple[int,int] = (0,0)
        self.total_coins = 0
        self.collected   = 0
        self.prev_coin_dist: Optional[float] = None
        self.left_spawn  = False
        self._visited_checkpoints: set[Tuple[int,int]] = set()

        # motion tracking ---------------------------------------------------
        self.prev_player_pos: Optional[np.ndarray] = None
        self.last_player: Tuple[int,int] = (0,0)
        self.no_move_counter = 0

        # misc --------------------------------------------------------------
        self._refresh_cnt = 0
        self._dbg_cnt     = 0
        self.episode_reward = 0.0
        self.epsilon = epsilon
        self.reward_history: list[dict] = []  # per‑step reward breakdowns
        self._last_breakdown: dict[str,float] = {}
        self._log_file: Optional[csv.DictWriter] = None
        if LOG_REWARDS and log_prefix:
            fp = open(REWARD_LOG_DIR / f"{log_prefix}_rewards.csv", "w", newline="")
            self._log_file = csv.DictWriter(fp, fieldnames=[
                "step", "r_step", "r_time", "r_coin", "r_collect", "r_section",
                "r_idle", "r_wall", "r_enemy", "r_spawn", "r_done", "total"])
            self._log_file.writeheader()

        if SHOW_DEBUG:
            cv2.namedWindow("Agent\xa0view", cv2.WINDOW_NORMAL)

    # ───────────────────────── reset ────────────────────────────────
    def reset(self):
        self.region = get_ruffle_window_region()
        frame   = capture_screen(self.region)
        objs    = detect_objects(frame)

        self.spawn_pos   = objs.get("player", [(0,0)])[0]
        self.last_player = self.spawn_pos
        self.start_goal  = self._find_start_goal(objs)

        self.total_coins = len(objs.get("coin", []))
        self.collected   = 0
        self.prev_coin_dist = self._nearest_coin_dist(objs)
        self.left_spawn  = False
        self._visited_checkpoints.clear()

        self.prev_player_pos = np.array(self.spawn_pos, dtype=np.float32)
        self.no_move_counter = 0
        self._refresh_cnt = self._dbg_cnt = 0
        self.episode_reward = 0.0
        self.reward_history.clear()

        return self._state_from_objects(objs)

    # ───────────────────────── step ─────────────────────────────────
    def step(self, action:int):
        self._press_key(action)

        self._refresh_cnt += 1
        if self._refresh_cnt >= self.REGION_REFRESH_EVERY:
            self.region = get_ruffle_window_region()
            self._refresh_cnt = 0

        frame   = capture_screen(self.region)
        objs    = detect_objects(frame)

        if SHOW_DEBUG and self._dbg_cnt % DEBUG_INTERVAL == 0:
            self._show_debug(frame, objs)
        self._dbg_cnt += 1

        state  = self._state_from_objects(objs)
        reward, done, breakdown = self._compute_reward(state, objs)
        self.episode_reward += reward
        self._last_breakdown = breakdown
        if LOG_REWARDS:
            self.reward_history.append(breakdown)
            if self._log_file:
                self._log_file.writerow(breakdown)

        return state, reward, done, breakdown  # info dict useful for monitors

    # ─────────────────── key press helper ───────────────────────────
    def _press_key(self, action:int):
        key = self.action_map[action]
        injector = pydirectinput if _FAST_INPUT else pyautogui
        injector.keyDown(key); time.sleep(self.KEY_PRESS_DURATION); injector.keyUp(key)

    # ─────────────────── state encoding ─────────────────────────────
    def _state_from_objects(self, objs:Dict[str,List[Tuple[int,int]]]) -> np.ndarray:
        w,h = self.region["width"], self.region["height"]

        # robust player retrieval ------------------------------------------
        player_list = objs.get("player", [])
        if player_list:
            player = player_list[0]
            self.last_player = player            # update cache
        else:
            player = self.last_player            # reuse last valid position

        goals = objs.get("goal", [])
        if goals:
            target = max(goals, key=lambda g: np.linalg.norm(np.array(g)-np.array(self.start_goal)))
        else:
            target = player
        return np.array([player[0]/w, player[1]/h, target[0]/w, target[1]/h], dtype=np.float32)

    # ─────────────────── reward function ────────────────────────────
    def _compute_reward(self, s:np.ndarray, objs):
        """Return (total, done, breakdown_dict)."""
        w,h = self.region["width"], self.region["height"]
        pxy = np.array([s[0]*w, s[1]*h])

        breakdown = {
            "step": self._dbg_cnt,
            "r_step": STEP_PENALTY,
            "r_time": TIME_PENALTY_FACTOR * self._dbg_cnt,
            "r_coin": 0.0,
            "r_collect": 0.0,
            "r_section": 0.0,
            "r_idle": 0.0,
            "r_wall": 0.0,
            "r_enemy": 0.0,
            "r_spawn": 0.0,
            "r_done": 0.0,
            "total": 0.0,
        }
        reward = breakdown["r_step"] + breakdown["r_time"]
        done   = False

        # enemy collision ---------------------------------------------------
        if any(np.linalg.norm(pxy-e)<ENEMY_HIT_RADIUS for e in objs.get("enemy",[])):
            breakdown["r_enemy"] = ENEMY_COLLISION_PENALTY
            reward += breakdown["r_enemy"]

        # respawn detection -------------------------------------------------
        if self.left_spawn and np.linalg.norm(pxy-np.array(self.spawn_pos)) < SPAWN_RADIUS:
            breakdown["r_spawn"] = RESPAWN_PENALTY
            reward += breakdown["r_spawn"]
            self.collected = 0
            self.left_spawn = False
            self.no_move_counter = 0

        if (not self.left_spawn and np.linalg.norm(pxy-np.array(self.spawn_pos)) > SPAWN_RADIUS):
            self.left_spawn = True
            reward += LEAVE_SPAWN_BONUS

        # coin distance shaping --------------------------------------------
        coin_dist = self._nearest_coin_dist(objs)
        if coin_dist is not None and self.prev_coin_dist is not None:
            delta = self.prev_coin_dist - coin_dist  # positive when moving closer
            breakdown["r_coin"] = DISTANCE_COIN_SCALE * delta
            reward += breakdown["r_coin"]
        self.prev_coin_dist = coin_dist

        # coin collection reward -------------------------------------------
        coins_left = len(objs.get("coin", []))
        if coins_left < self.total_coins - self.collected:
            self.collected = self.total_coins - coins_left
            breakdown["r_collect"] = COIN_COLLECT_REWARD
            reward += breakdown["r_collect"]

        # checkpoint / section bonus ---------------------------------------
        for g in objs.get("goal", []):
            if np.linalg.norm(pxy-g) < GOAL_HIT_RADIUS and g not in self._visited_checkpoints:
                self._visited_checkpoints.add(g)
                breakdown["r_section"] += SECTION_BONUS
                reward += SECTION_BONUS

        # level completion --------------------------------------------------
        if (self.collected == self.total_coins and any(np.linalg.norm(pxy-g)<GOAL_HIT_RADIUS for g in objs.get("goal",[]))):
            breakdown["r_done"] = LEVEL_COMPLETE_REWARD
            reward += LEVEL_COMPLETE_REWARD
            done = True

        # idle / wall penalties --------------------------------------------
        move = np.linalg.norm(pxy - self.prev_player_pos) if self.prev_player_pos is not None else 1
        wall_contact = any(np.linalg.norm(pxy-w_)<WALL_HIT_RADIUS for w_ in objs.get("wall",[]))
        if wall_contact:
            breakdown["r_wall"] = WALL_CONTACT_PENALTY
            reward += breakdown["r_wall"]
        if move < 1.0:
            self.no_move_counter += 1
            idle_pen = NO_MOVE_BASE_PENALTY * (2 ** (self.no_move_counter-1))
            breakdown["r_idle"] = idle_pen
            reward += idle_pen
        else:
            self.no_move_counter = 0
        self.prev_player_pos = pxy

        breakdown["total"] = reward
        return float(reward), done, breakdown

    # ─────────────────── helper utilities ───────────────────────────
    def _find_start_goal(self, objs):
        goals = objs.get("goal", [])
        if not goals:
            return (0,0)
        return min(goals, key=lambda g: np.linalg.norm(np.array(g)-np.array(self.spawn_pos)))

    def _nearest_coin_dist(self, objs)->Optional[float]:
        player = self.last_player
        coins  = objs.get("coin", [])
        if not coins:
            return None
        return min(np.linalg.norm(np.array(player)-np.array(c)) for c in coins)

    # ─────────────────── debug overlay ──────────────────────────────
    def _show_debug(self, frame, objs):
        dbg = frame.copy()
        for x,y in objs.get("player",[]): cv2.drawMarker(dbg,(x,y),(0,0,255),cv2.MARKER_CROSS,2)
        for x,y in objs.get("enemy", []): cv2.circle(dbg,(x,y),6,(255,0,0),1)
        for x,y in objs.get("coin" , []): cv2.circle(dbg,(x,y),4,(0,255,255),1)
        for x,y in objs.get("goal" , []): cv2.rectangle(dbg,(x-3,y-3),(x+3,y+3),(0,255,0),1)

        hud = [
            f"R_step {self._last_breakdown.get('total',0):+6.1f}",
            f"R_tot  {self.episode_reward:+7.1f}",
            f"Coins  {self.collected}/{self.total_coins}",
            f"ε {self.epsilon:.2f}",
            f"Step {self._dbg_cnt}"
        ]
        for i,t in enumerate(hud):
            cv2.putText(dbg,t,(10,20+20*i),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)

        cv2.imshow("Agent\xa0view", dbg); cv2.waitKey(1)
        cv2.imwrite(os.path.join("screenshots",f"dbg_{int(time.time()*1000)}.png"),dbg)

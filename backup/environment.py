import numpy as np
import cv2
import pyautogui
import time
from game_capture import capture_screen
from object_detection import detect_objects

class Environment:
    def __init__(self):
        self.action_map = {
            0: 'up',
            1: 'down',
            2: 'left',
            3: 'right'
        }
        self.last_positions = []
        
    def reset(self):
        """Reset the game and return initial state"""
        # You'll need to implement actual game reset logic
        # For now just return current state
        return self._get_state()
        
    def step(self, action):
        """Execute one action and return (state, reward, done)"""
        self._take_action(action)
        state = self._get_state()
        reward, done = self._compute_reward(state)
        return state, reward, done
        
    def _take_action(self, action):
        """Press the corresponding key"""
        key = self.action_map[action]
        pyautogui.keyDown(key)
        time.sleep(0.05)  # Short key press
        pyautogui.keyUp(key)
        
    def _get_state(self):
        """Extract meaningful state information"""
        frame = capture_screen()
        objects = detect_objects(frame)
        
        # Get player position (use first detected or default)
        player_pos = objects['player'][0] if len(objects['player']) > 0 else (0, 0)
        
        # Get nearest coin position or goal if no coins
        if len(objects['coin']) > 0:
            target_pos = min(objects['coin'], 
                           key=lambda c: np.linalg.norm(np.array(c) - np.array(player_pos)))
        elif len(objects['goal']) > 0:
            target_pos = objects['goal'][0]
        else:
            target_pos = player_pos  # Fallback
            
        # Normalize positions to 0-1 range (assuming 800x600 game area)
        player_x = player_pos[0] / 800
        player_y = player_pos[1] / 600
        target_x = target_pos[0] / 800
        target_y = target_pos[1] / 600
        
        return np.array([player_x, player_y, target_x, target_y], dtype=np.float32)
        
    def _compute_reward(self, state):
        """Calculate reward and done flag"""
        reward = -0.1  # Small negative reward for each step
        done = False
        
        frame = capture_screen()
        objects = detect_objects(frame)
        
        # Check for collisions with enemies or walls
        player_pos = (int(state[0]*800), int(state[1]*600))
        
        # Check if player is touching any enemies
        for enemy in objects['enemy']:
            if np.linalg.norm(np.array(player_pos) - np.array(enemy)) < 20:
                reward -= 10
                done = True
                break
                
        # Check if player reached goal
        if len(objects['goal']) > 0:
            if np.linalg.norm(np.array(player_pos) - np.array(objects['goal'][0])) < 20:
                reward += 20
                done = True
                
        # Check if player collected coin
        for i, coin in enumerate(objects['coin']):
            if np.linalg.norm(np.array(player_pos) - np.array(coin)) < 20:
                reward += 5
                break
                
        return reward, done
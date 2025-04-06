import time
import numpy as np
from agent import Agent
from environment import Environment

def main():
    env = Environment()
    state_dim = 4  # [player_x, player_y, nearest_coin_x, nearest_coin_y]
    action_dim = 4  # Up, Down, Left, Right
    
    agent = Agent(state_dim, action_dim)
    
    for episode in range(1000):
        state = env.reset()
        done = False
        total_reward = 0
        episode_steps = 0
        
        while not done and episode_steps < 1000:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay(32)  # Continuous training
            
            state = next_state
            total_reward += reward
            episode_steps += 1
            
        print(f"Episode {episode}, Total Reward: {total_reward}, Steps: {episode_steps}")
        
        if episode % 10 == 0:
            agent.save(f"models/agent_ep{episode}.h5")

if __name__ == "__main__":
    main()
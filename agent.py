import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        logits = self.fc(state)
        probs = torch.clamp(logits, min=1e-8, max=1.0)  # Clamp probabilities
        return probs

class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        return self.fc(state)


class Agent:
    """Deep‑Q agent with CUDA acceleration and prioritized experience replay."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        memory_size: int = 10000,
        gamma: float = 0.97,
        lr_actor: float = 0.001,
        lr_critic: float = 0.001
    ) -> None:

        # CUDA setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        
        # Actor and Critic networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        # replay memory
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=memory_size
        )
        # optimisers
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.scaler = torch.cuda.amp.GradScaler('cuda')

        # Warmup CUDA (first call is slower)
        if 'cuda' in str(self.device):
            self.actor(torch.zeros(1, state_dim).to(self.device))

    # ── public API ─────────────────────────────────────────
    def act(self, state: np.ndarray) -> int:
        """Sample an action from the policy distribution"""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state_t).to(self.device)
        action_probs = action_probs.cpu().numpy().flatten()
        action_probs /= action_probs.sum()
        action = np.random.choice(self.action_dim, p=action_probs)
        return action

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int) -> None:
        """Train on a batch of experiences with CUDA acceleration."""
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        
        # Prepare batch tensors on GPU
        states = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in batch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in batch])).to(self.device)

        # Compute value targets
        with torch.no_grad():
            target_values = rewards + self.gamma * self.critic(next_states).squeeze(1) * (1 - dones)

        # Update Critic
        values = self.critic(states).squeeze(1)
        critic_loss = nn.MSELoss()(values, target_values)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Compute advantages
        advantages = (target_values - values).detach()

        # Update Actor
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = -(log_probs.squeeze(1) * advantages).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

    def save(self, path: str) -> None:
        """Save model ensuring it's on CPU for compatibility."""
        # Move model to CPU before saving
        torch.save(self.actor.to('cpu').state_dict(), path+"Actor.h5")
        torch.save(self.critic.to('cpu').state_dict(), path+"Critic.h5")
        # Move back to original device
        self.actor.to(self.device)
        self.critic.to(self.device)

    def load(self, path: str) -> None:
        """Load model and ensure it's on the right device."""
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        self.actor.to(self.device)
        self.actor.eval()

import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )
        self.net.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class Agent:
    """Deep‑Q agent with CUDA acceleration and prioritized experience replay."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        memory_size: int = 10_000,
        gamma: float = 0.97,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
        lr: float = 0.001,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim

        # ε‑greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        # replay memory
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=memory_size
        )

        # CUDA setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # model & optimiser
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss().to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == 'cuda')

        # Warmup CUDA (first call is slower)
        if 'cuda' in str(self.device):
            self.model(torch.zeros(1, state_dim).to(self.device))

    # ── public API ─────────────────────────────────────────
    def act(self, state: np.ndarray) -> int:
        """Return an action chosen ε‑greedily from the Q‑network with CUDA acceleration."""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():  # No gradient needed for inference
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_t)
            return int(torch.argmax(q_values).item())

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

        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            # Current Q values
            current_q = self.model(states).gather(1, actions)
            
            # Target Q values
            with torch.no_grad():
                next_q = self.model(next_states).max(1)[0]
                target_q = rewards + (1 - dones) * self.gamma * next_q
                target_q = target_q.unsqueeze(1)
            
            # Compute loss
            loss = self.criterion(current_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # ε decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path: str) -> None:
        """Save model ensuring it's on CPU for compatibility."""
        # Move model to CPU before saving
        torch.save(self.model.to('cpu').state_dict(), path)
        # Move back to original device
        self.model.to(self.device)

    def load(self, path: str) -> None:
        """Load model and ensure it's on the right device."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

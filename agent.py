import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    """A minimal 3‑layer fully‑connected network."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    """Deep‑Q agent **with built‑in macro‑action holding**.

    The environment still expects an *integer* action each step.  The agent now
    decides only every *hold_frames* steps; in between it keeps returning the
    same action, effectively *holding* the key so the avatar moves multiple
    tiles per high‑level decision.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hold_frames: int = 5,
        memory_size: int = 10_000,
        gamma: float = 0.97,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
        lr: float = 0.001,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hold_frames = hold_frames  # how many steps to keep an action

        # ε‑greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        # replay memory
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=memory_size
        )

        # model & optimiser
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # macro‑action bookkeeping
        self._hold_counter = 0
        self._current_action = 0

    # ───────────────────────────────────────── public API ──
    def act(self, state: np.ndarray) -> int:
        """Return an action; repeat the previous one while *holding*."""
        # keep holding?
        if self._hold_counter > 0:
            self._hold_counter -= 1
            return self._current_action

        # choose new action
        if np.random.rand() < self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action = int(torch.argmax(self.model(state_t)))

        # start holding
        self._current_action = action
        self._hold_counter = self.hold_frames - 1  # minus this step
        return action

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int) -> None:
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            state_t      = torch.FloatTensor(state).unsqueeze(0)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0)

            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state_t)).item()

            target_f = self.model(state_t)
            target_vec = target_f.clone().detach()
            target_vec[0][action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(target_f, target_vec)
            loss.backward()
            self.optimizer.step()

        # ε decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

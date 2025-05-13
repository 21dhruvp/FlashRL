import os
import glob
from pathlib import Path
from typing import Optional

from agent import Agent
from environment import Environment


CHECKPOINT_DIR = Path("models")
CHECKPOINT_DIR.mkdir(exist_ok=True)
CHECKPOINT_EVERY = 10  # episodes


def _latest_checkpoint() -> Optional[Path]:
    """Return the most recent .pth/.h5 file in the models directory."""
    files = sorted(glob.glob(str(CHECKPOINT_DIR / "agent_ep*.h5")))
    return Path(files[-1]) if files else None


def main() -> None:
    env = Environment()
    state_dim = 4   # [player_x, player_y, goal_x, goal_y]
    action_dim = 4  # up, down, left, right

    agent = Agent(state_dim, action_dim)

    # ── Resume from latest checkpoint if available ─────────
    start_ep = 0
    ckpt = _latest_checkpoint()
    if ckpt:
        print(f"[train] Resuming from {ckpt}")
        agent.load(str(ckpt))
        # Extract episode number from filename
        try:
            start_ep = int(ckpt.stem.split("ep")[-1]) + 1
        except ValueError:
            pass

    # ── Training loop ──────────────────────────────────────
    for ep in range(start_ep, 1000):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < 1000:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)   # 'info' holds the reward components
            agent.remember(state, action, reward, next_state, done)
            agent.replay(32)

            state = next_state
            total_reward += reward
            steps += 1

        print(f"Episode {ep:04d} | Reward {total_reward:7.2f} | Steps {steps}")

        if ep % CHECKPOINT_EVERY == 0:
            ckpt_path = CHECKPOINT_DIR / f"agent_ep{ep}"
            agent.save(str(ckpt_path))
            print(f"[train] Saved checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()

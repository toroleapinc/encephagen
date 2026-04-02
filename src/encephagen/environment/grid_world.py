"""GridWorld: simple 2D environment for the brain to interact with.

An agent (dot) moves on a 2D grid trying to reach a target (food).
The brain controls the agent through motor output and receives
sensory information about the relative direction to the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GridWorldParams:
    """Parameters for the grid world."""

    size: float = 10.0          # World size (square, -size/2 to +size/2)
    step_size: float = 0.3      # How far agent moves per action
    target_radius: float = 0.5  # Distance to "reach" target
    max_steps: int = 1000       # Maximum steps per episode


# Actions: 0=up, 1=down, 2=left, 3=right
ACTION_VECTORS = np.array([
    [0.0, 1.0],    # up
    [0.0, -1.0],   # down
    [-1.0, 0.0],   # left
    [1.0, 0.0],    # right
], dtype=np.float64)


class GridWorld:
    """Simple 2D navigation environment.

    The agent starts at a random position and must reach a target.
    Sensory input: 4 values encoding relative direction to target
    (up/down/left/right components, each 0-1).
    """

    def __init__(
        self,
        params: GridWorldParams | None = None,
        seed: int | None = None,
    ):
        self.p = params or GridWorldParams()
        self.rng = np.random.default_rng(seed)

        self.agent_pos = np.zeros(2, dtype=np.float64)
        self.target_pos = np.zeros(2, dtype=np.float64)
        self.steps = 0
        self.done = False
        self.total_reward = 0.0
        self._prev_distance = 0.0

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to new random positions.

        Returns:
            Sensory observation [4] (direction encoding).
        """
        half = self.p.size / 2
        self.agent_pos = self.rng.uniform(-half * 0.8, half * 0.8, size=2)
        self.target_pos = self.rng.uniform(-half * 0.8, half * 0.8, size=2)

        # Ensure minimum distance
        while np.linalg.norm(self.agent_pos - self.target_pos) < self.p.target_radius * 3:
            self.target_pos = self.rng.uniform(-half * 0.8, half * 0.8, size=2)

        self.steps = 0
        self.done = False
        self.total_reward = 0.0
        self._prev_distance = float(np.linalg.norm(self.agent_pos - self.target_pos))

        return self.observe()

    def observe(self) -> np.ndarray:
        """Get sensory observation: relative direction to target.

        Returns:
            [4] array: [up, down, left, right] components in [0, 1].
            The direction toward the target has the highest value.
        """
        diff = self.target_pos - self.agent_pos
        dist = np.linalg.norm(diff)

        if dist < 1e-6:
            return np.full(4, 0.5)

        # Normalize direction
        direction = diff / dist

        # Encode as 4 channels: positive-y (up), negative-y (down),
        # negative-x (left), positive-x (right)
        obs = np.zeros(4, dtype=np.float64)
        obs[0] = max(0, direction[1])     # up component
        obs[1] = max(0, -direction[1])    # down component
        obs[2] = max(0, -direction[0])    # left component
        obs[3] = max(0, direction[0])     # right component

        # Scale by proximity (closer = stronger signal)
        proximity = 1.0 / (1.0 + dist * 0.2)
        obs *= proximity

        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Take an action and return (observation, reward, done).

        Args:
            action: 0=up, 1=down, 2=left, 3=right.

        Returns:
            observation: [4] sensory observation.
            reward: Scalar reward signal.
            done: Whether episode is finished.
        """
        if self.done:
            return self.observe(), 0.0, True

        # Move agent
        if 0 <= action < 4:
            movement = ACTION_VECTORS[action] * self.p.step_size
            self.agent_pos += movement

        # Clip to world bounds
        half = self.p.size / 2
        self.agent_pos = np.clip(self.agent_pos, -half, half)

        self.steps += 1

        # Compute reward
        distance = float(np.linalg.norm(self.agent_pos - self.target_pos))
        reward = self._prev_distance - distance  # Positive if getting closer
        self._prev_distance = distance

        # Check if target reached
        if distance < self.p.target_radius:
            reward += 10.0  # Bonus for reaching target
            self.done = True

        # Check max steps
        if self.steps >= self.p.max_steps:
            self.done = True

        self.total_reward += reward

        return self.observe(), reward, self.done

    @property
    def distance_to_target(self) -> float:
        return float(np.linalg.norm(self.agent_pos - self.target_pos))

    @property
    def target_reached(self) -> bool:
        return self.distance_to_target < self.p.target_radius

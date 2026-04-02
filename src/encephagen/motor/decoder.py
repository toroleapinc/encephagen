"""Motor decoder: read firing rates from motor regions to produce actions.

Maps population firing rates in motor region neurons to discrete
or continuous actions. Uses population vector decoding — each
neuron group "votes" for a direction, the population average
determines the action.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MotorParams:
    """Parameters for motor decoding."""

    n_actions: int = 4           # Number of discrete actions (up/down/left/right)
    window_ms: float = 50.0      # Averaging window for rate estimation (ms)
    noise_sigma: float = 0.1     # Action noise (exploration)


class MotorDecoder:
    """Decode motor region firing into actions.

    Divides motor region neurons into n_actions groups.
    The group with the highest firing rate determines the action.
    """

    def __init__(
        self,
        n_neurons: int = 200,
        params: MotorParams | None = None,
        seed: int | None = None,
    ):
        self.n_neurons = n_neurons
        self.p = params or MotorParams()
        self.rng = np.random.default_rng(seed)

        # Divide neurons into action groups
        neurons_per_action = n_neurons // self.p.n_actions
        self.action_groups: list[list[int]] = []
        for a in range(self.p.n_actions):
            start = a * neurons_per_action
            end = start + neurons_per_action if a < self.p.n_actions - 1 else n_neurons
            self.action_groups.append(list(range(start, end)))

        # Spike count accumulator for rate estimation
        self._spike_buffer: list[np.ndarray] = []
        self._buffer_max = 1  # Updated based on window_ms and dt

    def update(self, spikes: np.ndarray, dt: float) -> None:
        """Add a timestep of spikes to the buffer.

        Args:
            spikes: Boolean spike array [n_neurons].
            dt: Timestep in ms.
        """
        self._buffer_max = max(1, int(self.p.window_ms / dt))
        self._spike_buffer.append(spikes.copy())
        if len(self._spike_buffer) > self._buffer_max:
            self._spike_buffer.pop(0)

    def get_action_rates(self) -> np.ndarray:
        """Get firing rate per action group from buffered spikes.

        Returns:
            Firing rates [n_actions] in Hz.
        """
        if not self._spike_buffer:
            return np.zeros(self.p.n_actions)

        # Sum spikes across buffer
        total = np.zeros(self.n_neurons, dtype=np.float64)
        for s in self._spike_buffer:
            total += s.astype(np.float64)

        # Average rate per action group
        n_steps = len(self._spike_buffer)
        rates = np.zeros(self.p.n_actions)
        for a in range(self.p.n_actions):
            group = self.action_groups[a]
            if group:
                rates[a] = total[group].sum() / (len(group) * n_steps)

        return rates

    def decode_action(self) -> int:
        """Get the discrete action (argmax of rates with noise).

        Returns:
            Action index (0 to n_actions-1).
        """
        rates = self.get_action_rates()
        # Add exploration noise
        noisy_rates = rates + self.rng.normal(0, self.p.noise_sigma, size=len(rates))
        return int(np.argmax(noisy_rates))

    def decode_continuous(self) -> np.ndarray:
        """Get continuous action vector (normalized rates).

        Returns:
            Action vector [n_actions] in [0, 1].
        """
        rates = self.get_action_rates()
        max_rate = rates.max()
        if max_rate > 0:
            return rates / max_rate
        return rates

    def reset(self) -> None:
        """Clear the spike buffer."""
        self._spike_buffer.clear()

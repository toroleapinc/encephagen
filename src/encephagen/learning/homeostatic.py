"""Homeostatic Plasticity.

Keeps firing rates within a target range by globally scaling
synaptic weights per neuron. Prevents runaway excitation or
complete silencing during STDP learning.

References:
    Turrigiano (2008). The Self-Tuning Neuron: Synaptic Scaling
    of Excitatory Synapses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class HomeostaticParams:
    """Parameters for homeostatic plasticity."""

    target_rate: float = 10.0     # Target firing rate (Hz)
    tau_homeo: float = 1000.0     # Homeostatic time constant (ms) — slow
    eta: float = 0.01             # Scaling rate per adjustment


class HomeostaticPlasticity:
    """Homeostatic synaptic scaling.

    Each neuron tracks its running average firing rate.
    If firing too fast: scale down all incoming excitatory weights.
    If firing too slow: scale up all incoming excitatory weights.

    Operates on a slow timescale (seconds) relative to STDP (ms).
    """

    def __init__(
        self,
        n_neurons: int,
        params: HomeostaticParams | None = None,
    ):
        self.n = n_neurons
        self.p = params or HomeostaticParams()

        # Running average firing rate (exponential moving average)
        self.running_rate = np.full(n_neurons, self.p.target_rate, dtype=np.float64)

    def update_rates(self, dt: float, spikes: np.ndarray) -> None:
        """Update running average firing rates.

        Args:
            dt: Timestep in ms.
            spikes: Boolean spike array [n_neurons].
        """
        # Instantaneous rate: spike/dt converted to Hz
        inst_rate = spikes.astype(np.float64) * (1000.0 / dt)

        # Exponential moving average
        alpha = dt / self.p.tau_homeo
        self.running_rate = (1 - alpha) * self.running_rate + alpha * inst_rate

    def compute_scaling_factors(self) -> np.ndarray:
        """Compute multiplicative scaling factor per neuron.

        Returns:
            Scaling factors [n_neurons]. >1 means scale up, <1 means scale down.
        """
        # Ratio of target to actual rate
        ratio = self.p.target_rate / (self.running_rate + 1e-6)

        # Gentle scaling: move toward ratio by eta
        # factor = 1 + eta * (ratio - 1)
        # Clipped to prevent extreme changes
        factors = 1.0 + self.p.eta * (ratio - 1.0)
        factors = np.clip(factors, 0.9, 1.1)  # Max 10% change per step

        return factors

    def apply_scaling(
        self,
        weights: sparse.csr_matrix,
        scaling_factors: np.ndarray,
    ) -> sparse.csr_matrix:
        """Scale incoming weights for each post-synaptic neuron.

        Args:
            weights: Excitatory weight matrix [n_pre, n_post].
            scaling_factors: Per-post-neuron scaling [n_post].

        Returns:
            Scaled weight matrix.
        """
        # Multiply each column by its scaling factor using diagonal matrix
        # weights @ diag(scaling_factors) scales each column
        diag = sparse.diags(scaling_factors)
        scaled = weights @ diag
        scaled.data[:] = np.clip(scaled.data, 0, 20.0)
        return scaled.tocsr()

    def step(
        self,
        dt: float,
        spikes: np.ndarray,
        weights: sparse.csr_matrix,
        apply_every_ms: float = 100.0,
        current_time: float = 0.0,
    ) -> sparse.csr_matrix:
        """Update rates and optionally apply scaling.

        Scaling is applied infrequently (every apply_every_ms) to
        reduce computational cost.

        Args:
            dt: Timestep in ms.
            spikes: Boolean [n_neurons].
            weights: Current exc weight matrix.
            apply_every_ms: How often to actually scale weights.
            current_time: Current simulation time in ms.

        Returns:
            Weight matrix (possibly scaled).
        """
        self.update_rates(dt, spikes)

        # Apply scaling periodically
        if current_time > 0 and abs(current_time % apply_every_ms) < dt:
            factors = self.compute_scaling_factors()
            weights = self.apply_scaling(weights, factors)

        return weights

    def reset(self) -> None:
        """Reset running rates to target."""
        self.running_rate[:] = self.p.target_rate

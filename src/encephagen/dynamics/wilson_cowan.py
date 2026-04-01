"""Wilson-Cowan excitatory-inhibitory oscillator model.

Each brain region is modeled as a coupled excitatory-inhibitory population.
When placed on a connectivity graph, these oscillators interact to produce
emergent dynamics shaped by the network topology.

References:
    Wilson & Cowan (1972). Excitatory and Inhibitory Interactions in
    Localized Populations of Model Neurons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WilsonCowanParams:
    """Parameters for the Wilson-Cowan model."""

    tau_e: float = 1.0       # Excitatory time constant (ms)
    tau_i: float = 2.0       # Inhibitory time constant (ms)
    w_ee: float = 16.0       # E→E coupling
    w_ei: float = 12.0       # I→E coupling (inhibition of excitatory)
    w_ie: float = 15.0       # E→I coupling
    w_ii: float = 3.0        # I→I coupling
    theta_e: float = 2.0     # Excitatory sigmoid threshold
    theta_i: float = 3.7     # Inhibitory sigmoid threshold
    a_e: float = 1.5         # Excitatory sigmoid gain
    a_i: float = 1.0         # Inhibitory sigmoid gain
    noise_sigma: float = 0.01  # Additive noise std


def _sigmoid(x: np.ndarray, a: float, theta: float) -> np.ndarray:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-a * (x - theta)))


class WilsonCowanModel:
    """Wilson-Cowan dynamics on a connectivity graph.

    Each region i has excitatory (E_i) and inhibitory (I_i) populations.
    Inter-region coupling is through excitatory populations only, weighted
    by the connectivity matrix.
    """

    def __init__(
        self,
        connectivity: np.ndarray,
        global_coupling: float = 1.0,
        params: WilsonCowanParams | None = None,
    ):
        if connectivity.ndim != 2 or connectivity.shape[0] != connectivity.shape[1]:
            raise ValueError(f"connectivity must be square 2D, got {connectivity.shape}")
        self.C = connectivity.astype(np.float64)
        self.n = connectivity.shape[0]
        self.G = global_coupling
        self.p = params or WilsonCowanParams()

    def simulate(
        self,
        duration: float = 10000.0,
        dt: float = 0.1,
        transient: float = 2000.0,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the simulation.

        Args:
            duration: Total simulation time (ms), including transient.
            dt: Integration timestep (ms).
            transient: Initial transient to discard (ms).
            seed: Random seed for reproducibility.

        Returns:
            Dict with keys:
                'E': Excitatory activity [timesteps, regions]
                'I': Inhibitory activity [timesteps, regions]
                'time': Time vector [timesteps]
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        total_steps = int(duration / dt)
        transient_steps = int(transient / dt)
        record_steps = total_steps - transient_steps

        if record_steps <= 0:
            raise ValueError(
                f"duration ({duration}) must be greater than transient ({transient})"
            )

        p = self.p
        n = self.n
        C = self.C * self.G

        # Initialize near resting state with small perturbation
        E = rng.uniform(0.0, 0.1, size=n)
        I = rng.uniform(0.0, 0.1, size=n)

        # Pre-allocate recording arrays
        E_record = np.zeros((record_steps, n), dtype=np.float64)
        I_record = np.zeros((record_steps, n), dtype=np.float64)

        record_idx = 0

        for step in range(total_steps):
            # Inter-region coupling (excitatory only)
            coupling = C @ E

            # Excitatory dynamics
            input_e = p.w_ee * E - p.w_ei * I + coupling
            dE = (-E + _sigmoid(input_e, p.a_e, p.theta_e)) / p.tau_e

            # Inhibitory dynamics
            input_i = p.w_ie * E - p.w_ii * I
            dI = (-I + _sigmoid(input_i, p.a_i, p.theta_i)) / p.tau_i

            # Euler integration with noise
            noise_e = p.noise_sigma * rng.standard_normal(n) * np.sqrt(dt)
            noise_i = p.noise_sigma * rng.standard_normal(n) * np.sqrt(dt)

            E = np.clip(E + dt * dE + noise_e, 0.0, 1.0)
            I = np.clip(I + dt * dI + noise_i, 0.0, 1.0)

            # Record after transient
            if step >= transient_steps:
                E_record[record_idx] = E
                I_record[record_idx] = I
                record_idx += 1

        time = np.arange(record_steps) * dt

        return {
            "E": E_record,
            "I": I_record,
            "time": time,
        }

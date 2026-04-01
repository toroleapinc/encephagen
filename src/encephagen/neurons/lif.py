"""Leaky Integrate-and-Fire (LIF) neuron model.

Vectorized implementation for simulating large populations efficiently.
All neurons share the same parameters (identical dynamics) — any
differentiation emerges from connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LIFParams:
    """Parameters for LIF neurons."""

    tau_m_exc: float = 20.0      # Excitatory membrane time constant (ms)
    tau_m_inh: float = 10.0      # Inhibitory membrane time constant (ms)
    v_rest: float = -65.0        # Resting potential (mV)
    v_threshold: float = -50.0   # Spike threshold (mV)
    v_reset: float = -70.0       # Reset potential after spike (mV)
    t_ref: float = 2.0           # Refractory period (ms)
    r_m: float = 100.0           # Membrane resistance (MΩ)
    tau_syn_exc: float = 5.0     # Excitatory synaptic time constant (ms)
    tau_syn_inh: float = 10.0    # Inhibitory synaptic time constant (ms)
    w_exc: float = 0.5           # Excitatory synaptic weight (nA)
    w_inh: float = -2.0          # Inhibitory synaptic weight (nA)


class LIFNeurons:
    """Vectorized LIF neuron population.

    Manages membrane potentials, refractory states, and synaptic currents
    for N neurons simultaneously using NumPy arrays.
    """

    def __init__(
        self,
        n: int,
        n_exc: int,
        params: LIFParams | None = None,
        seed: int | None = None,
    ):
        if n_exc > n:
            raise ValueError(f"n_exc ({n_exc}) > n ({n})")

        self.n = n
        self.n_exc = n_exc
        self.n_inh = n - n_exc
        self.p = params or LIFParams()
        self.rng = np.random.default_rng(seed)

        # State arrays
        self.v = np.full(n, self.p.v_rest, dtype=np.float64)
        self.v += self.rng.uniform(-5, 5, size=n)  # Small initial perturbation

        # Refractory timer (ms remaining)
        self.refrac_timer = np.zeros(n, dtype=np.float64)

        # Synaptic current accumulators
        self.i_syn = np.zeros(n, dtype=np.float64)

        # Spike output (binary)
        self.spikes = np.zeros(n, dtype=bool)

        # Per-neuron membrane time constant
        self.tau_m = np.full(n, self.p.tau_m_exc, dtype=np.float64)
        self.tau_m[n_exc:] = self.p.tau_m_inh

        # Is excitatory mask
        self.is_exc = np.zeros(n, dtype=bool)
        self.is_exc[:n_exc] = True

    def step(self, dt: float, i_ext: np.ndarray | None = None) -> np.ndarray:
        """Advance one timestep.

        Args:
            dt: Timestep in ms.
            i_ext: External current per neuron [n], in nA. Optional.

        Returns:
            Boolean spike array [n].
        """
        # Total input current
        i_total = self.i_syn.copy()
        if i_ext is not None:
            i_total += i_ext

        # Neurons not in refractory period
        active = self.refrac_timer <= 0

        # Membrane dynamics: dV/dt = (-(V - V_rest) + R * I) / tau_m
        dv = (-(self.v - self.p.v_rest) + self.p.r_m * i_total) / self.tau_m
        self.v[active] += dt * dv[active]

        # Spike detection
        self.spikes[:] = (self.v >= self.p.v_threshold) & active

        # Reset spiking neurons
        self.v[self.spikes] = self.p.v_reset
        self.refrac_timer[self.spikes] = self.p.t_ref

        # Decrement refractory timer
        self.refrac_timer = np.maximum(0, self.refrac_timer - dt)

        # Decay synaptic currents
        exc_decay = np.exp(-dt / self.p.tau_syn_exc)
        inh_decay = np.exp(-dt / self.p.tau_syn_inh)
        # Apply appropriate decay based on whether current is from exc or inh
        # Simplified: just use average decay
        self.i_syn *= 0.5 * (exc_decay + inh_decay)

        return self.spikes.copy()

    def receive_spikes(self, pre_spikes: np.ndarray, weights: np.ndarray) -> None:
        """Receive synaptic input from presynaptic spikes.

        Args:
            pre_spikes: Boolean array of presynaptic spikes [n_pre].
            weights: Synaptic weight matrix [n_pre, n_post].
                     Rows = presynaptic, columns = postsynaptic.
                     Only rows where pre_spikes=True are used.
        """
        if not pre_spikes.any():
            return

        # Sum weighted input from all spiking presynaptic neurons
        spiking_indices = np.where(pre_spikes)[0]
        self.i_syn += weights[spiking_indices].sum(axis=0)

    @property
    def firing_rates(self) -> np.ndarray:
        """Current spikes as float (for compatibility)."""
        return self.spikes.astype(np.float64)

    def reset(self) -> None:
        """Reset all state to initial conditions."""
        self.v[:] = self.p.v_rest
        self.v += self.rng.uniform(-5, 5, size=self.n)
        self.refrac_timer[:] = 0
        self.i_syn[:] = 0
        self.spikes[:] = False

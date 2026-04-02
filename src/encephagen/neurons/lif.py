"""Leaky Integrate-and-Fire (LIF) neuron model.

Vectorized implementation with separate excitatory and inhibitory
synaptic currents. Parameters calibrated following Brunel (2000)
balanced network theory for the fluctuation-driven regime.

References:
    Brunel (2000). Dynamics of Sparsely Connected Networks of Excitatory
    and Inhibitory Spiking Neurons. J Comput Neurosci.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LIFParams:
    """Parameters for LIF neurons.

    Defaults calibrated for a balanced network in the asynchronous
    irregular (AI) regime (~5-15 Hz firing rates). Based on Brunel (2000)
    and Potjans & Diesmann (2014).
    """

    # Membrane
    tau_m: float = 20.0          # Membrane time constant (ms)
    v_rest: float = 0.0          # Resting potential (mV) — normalized
    v_threshold: float = 20.0    # Spike threshold (mV) — normalized
    v_reset: float = 0.0         # Reset potential after spike (mV)
    t_ref: float = 2.0           # Refractory period (ms)

    # Synaptic
    tau_syn_exc: float = 5.0     # Excitatory synaptic time constant (ms)
    tau_syn_inh: float = 5.0     # Inhibitory synaptic time constant (ms)

    # Weights (in mV — PSP amplitude)
    j_exc: float = 2.0           # Excitatory PSP amplitude (mV) — scaled for C_E~16
    g_inh: float = 5.0           # Relative inhibitory strength (|J_inh| = g * J_exc)
    # So J_inh = -g_inh * j_exc = -0.5 mV


class LIFNeurons:
    """Vectorized LIF neuron population with separate E/I synaptic currents.

    Uses the Brunel (2000) formulation where synaptic input is in units
    of PSP amplitude (mV), not current (nA). This simplifies E/I balance
    calibration.

    The membrane equation is:
        tau_m * dV/dt = -V + I_syn_exc + I_syn_inh + I_ext
    where V is measured from V_rest (so V_rest = 0).
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

        # Membrane potential (relative to V_rest, so range [0, V_threshold])
        self.v = self.rng.uniform(0, self.p.v_threshold, size=n)

        # Refractory timer (ms remaining)
        self.refrac_timer = np.zeros(n, dtype=np.float64)

        # SEPARATE excitatory and inhibitory synaptic currents
        self.i_exc = np.zeros(n, dtype=np.float64)
        self.i_inh = np.zeros(n, dtype=np.float64)

        # Spike output
        self.spikes = np.zeros(n, dtype=bool)

        # Is excitatory mask
        self.is_exc = np.zeros(n, dtype=bool)
        self.is_exc[:n_exc] = True

    def step(self, dt: float, i_ext: np.ndarray | None = None) -> np.ndarray:
        """Advance one timestep.

        Args:
            dt: Timestep in ms.
            i_ext: External input per neuron [n], in mV. Optional.

        Returns:
            Boolean spike array [n].
        """
        # Total synaptic input
        i_total = self.i_exc + self.i_inh
        if i_ext is not None:
            i_total = i_total + i_ext

        # Neurons not in refractory period
        active = self.refrac_timer <= 0

        # Membrane dynamics: tau_m * dV/dt = -V + I_total
        dv = (-self.v + i_total) / self.p.tau_m
        self.v[active] += dt * dv[active]

        # Spike detection
        self.spikes[:] = (self.v >= self.p.v_threshold) & active

        # Reset spiking neurons
        self.v[self.spikes] = self.p.v_reset
        self.refrac_timer[self.spikes] = self.p.t_ref

        # Decrement refractory timer
        self.refrac_timer = np.maximum(0, self.refrac_timer - dt)

        # Decay synaptic currents SEPARATELY
        self.i_exc *= np.exp(-dt / self.p.tau_syn_exc)
        self.i_inh *= np.exp(-dt / self.p.tau_syn_inh)

        return self.spikes.copy()

    def receive_exc_spikes(self, n_spikes_per_neuron: np.ndarray) -> None:
        """Receive excitatory synaptic input.

        Args:
            n_spikes_per_neuron: Number of excitatory presynaptic spikes
                                 arriving at each neuron [n].
        """
        self.i_exc += n_spikes_per_neuron * self.p.j_exc

    def receive_inh_spikes(self, n_spikes_per_neuron: np.ndarray) -> None:
        """Receive inhibitory synaptic input.

        Args:
            n_spikes_per_neuron: Number of inhibitory presynaptic spikes
                                 arriving at each neuron [n].
        """
        self.i_inh -= n_spikes_per_neuron * self.p.j_exc * self.p.g_inh

    @property
    def firing_rates(self) -> np.ndarray:
        """Current spikes as float."""
        return self.spikes.astype(np.float64)

    def reset(self) -> None:
        """Reset all state to initial conditions."""
        self.v = self.rng.uniform(0, self.p.v_threshold, size=self.n)
        self.refrac_timer[:] = 0
        self.i_exc[:] = 0
        self.i_inh[:] = 0
        self.spikes[:] = False

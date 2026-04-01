"""BrainSimulator: continuous simulation of human brain connectome dynamics.

Runs Wilson-Cowan E/I dynamics on the connectome with IDENTICAL parameters
at every region. Records full activity traces for region-by-region analysis
of emergent functional roles.

The key question: does the network topology alone cause different regions
to develop distinct functional behaviors (gating, sustained activity,
sensory responsiveness, etc.)?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from encephagen.connectome.loader import Connectome
from encephagen.dynamics.wilson_cowan import WilsonCowanParams


def _sigmoid(x: np.ndarray, a: float, theta: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-a * (x - theta)))


@dataclass
class StimulusEvent:
    """A stimulus injected into specific regions at a specific time."""

    region_indices: list[int]
    onset: float       # ms
    duration: float     # ms
    amplitude: float    # strength of input current


@dataclass
class SimulationResult:
    """Full recording of a brain simulation."""

    E: np.ndarray           # Excitatory activity [timesteps, regions]
    I: np.ndarray           # Inhibitory activity [timesteps, regions]
    time: np.ndarray        # Time vector [timesteps] in ms
    labels: list[str]       # Region labels
    stimuli: list[StimulusEvent]  # Stimuli that were applied
    dt: float               # Timestep in ms
    params: WilsonCowanParams

    @property
    def num_regions(self) -> int:
        return self.E.shape[1]

    @property
    def num_timesteps(self) -> int:
        return self.E.shape[0]

    @property
    def duration_ms(self) -> float:
        return float(self.time[-1] - self.time[0])

    def region_activity(self, label: str) -> np.ndarray:
        """Get excitatory activity for a named region."""
        idx = self.labels.index(label)
        return self.E[:, idx]

    def region_index(self, label: str) -> int:
        return self.labels.index(label)


class BrainSimulator:
    """Simulate Wilson-Cowan dynamics on a human brain connectome.

    All regions use IDENTICAL parameters — any functional differentiation
    that emerges is purely from the network topology.
    """

    def __init__(
        self,
        connectome: Connectome,
        global_coupling: float = 0.01,
        params: WilsonCowanParams | None = None,
    ):
        self.connectome = connectome
        self.C = connectome.weights.astype(np.float64)
        self.n = connectome.num_regions
        self.G = global_coupling
        self.p = params or WilsonCowanParams()

    def simulate(
        self,
        duration: float = 10000.0,
        dt: float = 0.1,
        transient: float = 2000.0,
        stimuli: list[StimulusEvent] | None = None,
        seed: int | None = None,
    ) -> SimulationResult:
        """Run the brain simulation.

        Args:
            duration: Total simulation time (ms), including transient.
            dt: Integration timestep (ms).
            transient: Initial transient to discard (ms).
            stimuli: List of stimulus events to inject.
            seed: Random seed for reproducibility.

        Returns:
            SimulationResult with full activity traces.
        """
        rng = np.random.default_rng(seed)
        stimuli = stimuli or []

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

        # Initialize near resting state
        E = rng.uniform(0.0, 0.1, size=n)
        I = rng.uniform(0.0, 0.1, size=n)

        # Pre-allocate recording
        E_record = np.zeros((record_steps, n), dtype=np.float64)
        I_record = np.zeros((record_steps, n), dtype=np.float64)

        record_idx = 0

        for step in range(total_steps):
            t = step * dt  # Current time in ms

            # External input from stimuli
            P_ext = np.zeros(n, dtype=np.float64)
            for stim in stimuli:
                if stim.onset <= t < stim.onset + stim.duration:
                    for ri in stim.region_indices:
                        P_ext[ri] += stim.amplitude

            # Inter-region coupling (excitatory only)
            coupling = C @ E

            # Wilson-Cowan dynamics
            input_e = p.w_ee * E - p.w_ei * I + coupling + P_ext
            dE = (-E + _sigmoid(input_e, p.a_e, p.theta_e)) / p.tau_e

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

        time = np.arange(record_steps) * dt + transient

        return SimulationResult(
            E=E_record,
            I=I_record,
            time=time,
            labels=list(self.connectome.labels),
            stimuli=stimuli,
            dt=dt,
            params=p,
        )

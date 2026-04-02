"""SpikingBrain: full brain simulation with balanced E/I spiking populations.

Connects RegionPopulations via the macro-connectome. Between-region
connections are excitatory (from exc neurons in source to random neurons
in target). Background Poisson drive calibrated using Brunel (2000)
balanced network theory.

References:
    Brunel (2000). Dynamics of Sparsely Connected Networks of Excitatory
    and Inhibitory Spiking Neurons.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy import sparse

from encephagen.connectome.loader import Connectome
from encephagen.neurons.lif import LIFParams
from encephagen.neurons.population import RegionPopulation


@dataclass
class SpikingBrainResult:
    """Recording from a spiking brain simulation."""

    firing_rates: np.ndarray    # Mean firing rate per region per bin [steps, n_regions]
    spike_counts: np.ndarray    # Total spikes per region per bin [steps, n_regions]
    time: np.ndarray            # Time vector [steps] in ms
    labels: list[str]
    dt: float
    n_neurons_per_region: int

    @property
    def num_regions(self) -> int:
        return len(self.labels)

    def region_rate(self, label: str) -> np.ndarray:
        idx = self.labels.index(label)
        return self.firing_rates[:, idx]


class SpikingBrain:
    """Spiking neural network brain with connectome topology.

    Uses Brunel's balanced network recipe:
    - Each neuron receives external Poisson excitatory input at rate nu_ext
    - Internal E/I recurrence provides balanced feedback
    - Background drive calibrated so mean input ≈ threshold
    - Firing is driven by fluctuations → irregular, ~5-15 Hz
    """

    def __init__(
        self,
        connectome: Connectome,
        neurons_per_region: int = 1000,
        exc_ratio: float = 0.8,
        internal_conn_prob: float = 0.1,
        between_conn_prob: float = 0.02,
        global_coupling: float = 1.0,
        ext_rate: float = 2.0,
        params: LIFParams | None = None,
        seed: int | None = None,
    ):
        """
        Args:
            ext_rate: External Poisson rate as a multiple of threshold rate.
                      Brunel uses nu_ext/nu_thr. Values 1.0-2.5 give different
                      regimes. 2.0 = slightly above threshold → AI regime.
        """
        self.connectome = connectome
        self.n_regions = connectome.num_regions
        self.neurons_per_region = neurons_per_region
        self.global_coupling = global_coupling
        self.params = params or LIFParams()

        rng = np.random.default_rng(seed)
        self._rng = rng

        # Brunel's balanced network calibration
        # Each neuron receives C_E excitatory inputs from within the region.
        # External input modeled as C_ext independent Poisson sources.
        # At nu_ext = nu_thr, mean membrane input = V_threshold.
        # nu_thr = V_threshold / (J_exc * C_ext * tau_m)
        # We set C_ext = C_E (standard Brunel assumption).
        c_e = int(neurons_per_region * exc_ratio * internal_conn_prob)
        c_e = max(c_e, 1)
        self._c_e = c_e
        c_ext = c_e

        # Scale J_exc with C_E to maintain constant mean input.
        # Reference: j_exc=2.0 was calibrated for C_E=16.
        # At larger C_E, each PSP must be smaller to keep
        # mean membrane input near threshold.
        j_eff = self.params.j_exc * (16.0 / c_e)
        self._j_eff = j_eff

        # Override params with scaled j_exc so populations use it
        from dataclasses import replace
        self.params = replace(self.params, j_exc=j_eff)

        nu_thr = self.params.v_threshold / (
            j_eff * c_ext * self.params.tau_m
        )
        self._nu_ext_per_ms = c_ext * nu_thr * ext_rate
        print(f"  Brunel calibration: C_E={c_e}, j_eff={j_eff:.3f}mV, "
              f"nu_ext={self._nu_ext_per_ms:.2f}/ms "
              f"({self._nu_ext_per_ms*1000:.0f} Hz)")

        # Create region populations
        print(f"  Creating {self.n_regions} populations "
              f"({neurons_per_region} neurons each)...", end=" ", flush=True)
        t0 = time.time()
        self.regions: list[RegionPopulation] = []
        for i in range(self.n_regions):
            pop = RegionPopulation(
                name=connectome.labels[i],
                n_neurons=neurons_per_region,
                exc_ratio=exc_ratio,
                conn_prob=internal_conn_prob,
                params=self.params,
                seed=rng.integers(0, 2**31) if seed is not None else None,
            )
            self.regions.append(pop)
        print(f"{time.time() - t0:.1f}s")

        # Build between-region connectivity
        print(f"  Building between-region connectivity...", end=" ", flush=True)
        t0 = time.time()
        self._build_between_connectivity(between_conn_prob, rng)
        print(f"{time.time() - t0:.1f}s")

        total_neurons = self.n_regions * neurons_per_region
        print(f"  Total: {total_neurons:,} neurons, {self.n_regions} regions")

    def _build_between_connectivity(
        self, conn_prob: float, rng: np.random.Generator,
    ) -> None:
        """Build sparse between-region excitatory projections."""
        self.between_connections: list[tuple[int, int, sparse.csr_matrix]] = []
        n = self.neurons_per_region

        for src, dst, w in self.connectome.edges():
            if src == dst:
                continue

            src_pop = self.regions[src]
            n_exc_src = src_pop.n_exc

            effective_prob = min(1.0, conn_prob * w * self.global_coupling)
            if effective_prob < 1e-6:
                continue

            # Binary connectivity — weights are handled via j_exc
            mask = rng.random((n_exc_src, n)) < effective_prob
            if mask.sum() == 0:
                continue

            self.between_connections.append(
                (src, dst, sparse.csr_matrix(mask.astype(np.float64)))
            )

        total_between = sum(m.nnz for _, _, m in self.between_connections)
        print(f"({len(self.between_connections)} projections, "
              f"{total_between:,} synapses)", end=" ")

    def step(self, dt: float, external_currents: dict[int, np.ndarray] | None = None) -> None:
        """Advance one timestep."""
        # 1. Propagate between-region excitatory spikes
        for src_idx, dst_idx, conn_matrix in self.between_connections:
            src_pop = self.regions[src_idx]
            exc_spikes = src_pop.exc_spikes
            if exc_spikes.any():
                # Count spikes arriving at each target neuron
                n_arriving = conn_matrix[exc_spikes].toarray().sum(axis=0)
                self.regions[dst_idx].neurons.receive_exc_spikes(n_arriving)

        # 2. Background Poisson drive + step each region
        for i, pop in enumerate(self.regions):
            # External Poisson excitatory input (Brunel's nu_ext)
            n_bg = self._rng.poisson(self._nu_ext_per_ms * dt, size=pop.n_neurons)
            pop.neurons.receive_exc_spikes(n_bg.astype(np.float64))

            ext = None
            if external_currents and i in external_currents:
                ext = external_currents[i]
            pop.step(dt, external_input=ext)

    def simulate(
        self,
        duration: float = 5000.0,
        dt: float = 0.1,
        transient: float = 1000.0,
        record_interval: float = 1.0,
        external_currents: dict[int, np.ndarray] | None = None,
    ) -> SpikingBrainResult:
        """Run simulation and record firing rates."""
        total_steps = int(duration / dt)
        transient_steps = int(transient / dt)
        record_every = max(1, int(record_interval / dt))

        record_steps = (total_steps - transient_steps) // record_every
        if record_steps <= 0:
            raise ValueError("Duration must be greater than transient")

        rates = np.zeros((record_steps, self.n_regions), dtype=np.float64)
        counts = np.zeros((record_steps, self.n_regions), dtype=np.float64)
        acc_counts = np.zeros(self.n_regions, dtype=np.float64)

        record_idx = 0
        acc_steps = 0

        print(f"  Simulating {duration/1000:.1f}s ({total_steps:,} steps)...", end=" ", flush=True)
        t0 = time.time()

        for step in range(total_steps):
            self.step(dt, external_currents)

            if step >= transient_steps:
                for i, pop in enumerate(self.regions):
                    acc_counts[i] += pop.neurons.spikes.sum()
                acc_steps += 1

                if acc_steps >= record_every:
                    interval_sec = acc_steps * dt / 1000.0
                    rates[record_idx] = acc_counts / (self.neurons_per_region * interval_sec)
                    counts[record_idx] = acc_counts
                    acc_counts[:] = 0
                    acc_steps = 0
                    record_idx += 1

            if step > 0 and step % (total_steps // 5) == 0:
                elapsed = time.time() - t0
                pct = step / total_steps * 100
                print(f"{pct:.0f}%", end=" ", flush=True)

        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")

        time_vec = np.arange(record_idx) * record_interval + transient

        return SpikingBrainResult(
            firing_rates=rates[:record_idx],
            spike_counts=counts[:record_idx],
            time=time_vec,
            labels=list(self.connectome.labels),
            dt=dt,
            n_neurons_per_region=self.neurons_per_region,
        )

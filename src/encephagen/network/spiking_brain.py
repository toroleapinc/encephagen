"""SpikingBrain: full brain simulation with spiking neuron populations.

Connects RegionPopulations via the macro-connectome. Between-region
connections are from excitatory neurons in the source region to
random neurons in the target region, weighted by the connectome.
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

    firing_rates: np.ndarray    # Mean firing rate per region per timestep [steps, n_regions]
    spike_counts: np.ndarray    # Total spikes per region per timestep [steps, n_regions]
    time: np.ndarray            # Time vector [steps] in ms
    labels: list[str]           # Region labels
    dt: float
    n_neurons_per_region: int

    @property
    def num_regions(self) -> int:
        return len(self.labels)

    def region_rate(self, label: str) -> np.ndarray:
        """Get firing rate trace for a named region."""
        idx = self.labels.index(label)
        return self.firing_rates[:, idx]


class SpikingBrain:
    """Spiking neural network brain with connectome topology.

    Each brain region is a population of LIF neurons.
    Between-region connections follow the structural connectome.
    All regions have IDENTICAL neuron parameters.
    """

    def __init__(
        self,
        connectome: Connectome,
        neurons_per_region: int = 1000,
        exc_ratio: float = 0.8,
        internal_conn_prob: float = 0.1,
        between_conn_prob: float = 0.05,
        global_coupling: float = 1.0,
        background_rate: float = 1000.0,
        background_weight: float = 0.005,
        params: LIFParams | None = None,
        seed: int | None = None,
    ):
        self.connectome = connectome
        self.n_regions = connectome.num_regions
        self.neurons_per_region = neurons_per_region
        self.global_coupling = global_coupling
        self.background_rate = background_rate
        self.background_weight = background_weight
        self.params = params or LIFParams()

        rng = np.random.default_rng(seed)
        self._rng = rng

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
                internal_conn_prob=internal_conn_prob,
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
        """Build sparse between-region connection matrices.

        For each edge in the connectome, create sparse connections from
        excitatory neurons in the source to random neurons in the target.
        """
        # Store as list of (src_region, target_region, weight_matrix)
        self.between_connections: list[tuple[int, int, sparse.csr_matrix]] = []

        n = self.neurons_per_region

        for src, dst, w in self.connectome.edges():
            if src == dst:
                continue

            src_pop = self.regions[src]
            n_exc_src = src_pop.n_exc

            # Scale: connectome weight * global coupling * base prob
            effective_prob = min(1.0, conn_prob * w * self.global_coupling)
            if effective_prob < 1e-6:
                continue

            # Sparse random connections: exc neurons in src → all neurons in dst
            mask = rng.random((n_exc_src, n)) < effective_prob
            n_connections = mask.sum()
            if n_connections == 0:
                continue

            # All between-region connections are excitatory (from exc neurons)
            weight_val = self.params.w_exc * w * self.global_coupling
            weights = sparse.csr_matrix(mask * weight_val)

            self.between_connections.append((src, dst, weights))

        total_between = sum(w.nnz for _, _, w in self.between_connections)
        print(f"({len(self.between_connections)} projections, "
              f"{total_between:,} synapses)", end=" ")

    def step(self, dt: float, external_currents: dict[int, np.ndarray] | None = None) -> None:
        """Advance one timestep for the entire brain.

        Args:
            dt: Timestep in ms.
            external_currents: Optional dict of region_index → current array [n_neurons].
        """
        # 1. Propagate between-region spikes
        for src_idx, dst_idx, weights in self.between_connections:
            src_pop = self.regions[src_idx]
            exc_spikes = src_pop.exc_spikes
            if exc_spikes.any():
                syn_input = weights[exc_spikes].toarray().sum(axis=0)
                self.regions[dst_idx].neurons.i_syn += syn_input

        # 2. Step each region with background Poisson input
        for i, pop in enumerate(self.regions):
            # Background drive: Poisson spikes add to synaptic current
            # (accumulates with time constant, just like real synaptic input)
            bg_spikes = self._rng.random(pop.n_neurons) < (self.background_rate * dt / 1000.0)
            pop.neurons.i_syn += bg_spikes.astype(np.float64) * self.background_weight

            ext = None
            if external_currents and i in external_currents:
                ext = external_currents[i]
            pop.step(dt, external_current=ext)

    def simulate(
        self,
        duration: float = 5000.0,
        dt: float = 0.1,
        transient: float = 1000.0,
        record_interval: float = 1.0,
        external_currents: dict[int, np.ndarray] | None = None,
        seed: int | None = None,
    ) -> SpikingBrainResult:
        """Run simulation and record firing rates.

        Args:
            duration: Total time in ms (including transient).
            dt: Timestep in ms.
            transient: Discard this many ms from the start.
            record_interval: Record firing rates every N ms.
            external_currents: Static external currents per region.

        Returns:
            SpikingBrainResult with firing rate traces.
        """
        total_steps = int(duration / dt)
        transient_steps = int(transient / dt)
        record_every = max(1, int(record_interval / dt))

        record_steps = (total_steps - transient_steps) // record_every
        if record_steps <= 0:
            raise ValueError("Duration must be greater than transient")

        # Recording arrays
        rates = np.zeros((record_steps, self.n_regions), dtype=np.float64)
        counts = np.zeros((record_steps, self.n_regions), dtype=np.float64)

        # Spike count accumulators (reset every record_interval)
        acc_counts = np.zeros(self.n_regions, dtype=np.float64)

        record_idx = 0
        acc_steps = 0

        print(f"  Simulating {duration/1000:.1f}s ({total_steps:,} steps)...", end=" ", flush=True)
        t0 = time.time()

        for step in range(total_steps):
            self.step(dt, external_currents)

            if step >= transient_steps:
                # Accumulate spike counts
                for i, pop in enumerate(self.regions):
                    acc_counts[i] += pop.neurons.spikes.sum()
                acc_steps += 1

                # Record at interval
                if acc_steps >= record_every:
                    interval_sec = acc_steps * dt / 1000.0
                    rates[record_idx] = acc_counts / (self.neurons_per_region * interval_sec)
                    counts[record_idx] = acc_counts
                    acc_counts[:] = 0
                    acc_steps = 0
                    record_idx += 1

            # Progress
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

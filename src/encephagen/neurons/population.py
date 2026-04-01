"""RegionPopulation: a brain region as a population of spiking neurons.

Each region contains excitatory and inhibitory LIF neurons with
random internal connectivity. Between-region connections follow
the macro-connectome.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from encephagen.neurons.lif import LIFNeurons, LIFParams


class RegionPopulation:
    """A brain region modeled as a population of spiking neurons.

    Contains N neurons (80% excitatory, 20% inhibitory by default)
    with sparse random internal connectivity.
    """

    def __init__(
        self,
        name: str,
        n_neurons: int = 1000,
        exc_ratio: float = 0.8,
        internal_conn_prob: float = 0.1,
        params: LIFParams | None = None,
        seed: int | None = None,
    ):
        self.name = name
        self.n_neurons = n_neurons
        self.n_exc = int(n_neurons * exc_ratio)
        self.n_inh = n_neurons - self.n_exc
        self.params = params or LIFParams()

        rng = np.random.default_rng(seed)

        # Create neurons
        self.neurons = LIFNeurons(
            n=n_neurons, n_exc=self.n_exc, params=self.params, seed=seed,
        )

        # Internal connectivity (sparse)
        # Excitatory neurons → all neurons (weight > 0)
        # Inhibitory neurons → all neurons (weight < 0)
        self.internal_weights = self._build_internal_connectivity(
            n_neurons, self.n_exc, internal_conn_prob, rng,
        )

    def _build_internal_connectivity(
        self, n: int, n_exc: int, prob: float, rng: np.random.Generator,
    ) -> sparse.csr_matrix:
        """Build sparse random internal connectivity matrix."""
        # Generate random connections
        mask = rng.random((n, n)) < prob
        np.fill_diagonal(mask, False)  # No self-connections

        # Set weights based on pre-synaptic neuron type
        weights = np.zeros((n, n), dtype=np.float64)
        weights[:n_exc, :] = mask[:n_exc, :] * self.params.w_exc   # Exc → all
        weights[n_exc:, :] = mask[n_exc:, :] * self.params.w_inh   # Inh → all

        return sparse.csr_matrix(weights)

    def step(self, dt: float, external_current: np.ndarray | None = None) -> np.ndarray:
        """Advance one timestep.

        Args:
            dt: Timestep in ms.
            external_current: External current per neuron [n_neurons]. Optional.

        Returns:
            Boolean spike array [n_neurons].
        """
        # Internal synaptic input from previous spikes
        if self.neurons.spikes.any():
            spiking = self.neurons.spikes
            # Sparse matrix-vector multiply: sum weights from spiking neurons
            syn_input = self.internal_weights[spiking].toarray().sum(axis=0)
            self.neurons.i_syn += syn_input

        # Step neurons
        spikes = self.neurons.step(dt, i_ext=external_current)
        return spikes

    @property
    def mean_firing_rate_hz(self) -> float:
        """Instantaneous fraction of neurons spiking (proxy for rate)."""
        return float(self.neurons.spikes.sum()) / self.n_neurons

    @property
    def exc_spikes(self) -> np.ndarray:
        """Spikes from excitatory neurons only."""
        return self.neurons.spikes[:self.n_exc]

    @property
    def inh_spikes(self) -> np.ndarray:
        """Spikes from inhibitory neurons only."""
        return self.neurons.spikes[self.n_exc:]

    def get_exc_spike_indices(self) -> np.ndarray:
        """Indices of excitatory neurons that spiked."""
        return np.where(self.neurons.spikes[:self.n_exc])[0]

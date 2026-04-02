"""RegionPopulation: a brain region as a balanced E/I spiking network.

Each region contains excitatory and inhibitory LIF neurons with
sparse random internal connectivity. Uses Brunel (2000) balanced
network recipe: mean input is near threshold, firing is driven
by fluctuations.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from encephagen.neurons.lif import LIFNeurons, LIFParams


class RegionPopulation:
    """A brain region modeled as a balanced E/I spiking population.

    Internal connectivity follows Brunel's balanced network:
    - Each neuron receives C_E excitatory and C_I inhibitory connections
    - E/I balance: g * C_I / C_E ≈ 1 for balance
    - Background external input pushes mean voltage near threshold
    """

    def __init__(
        self,
        name: str,
        n_neurons: int = 1000,
        exc_ratio: float = 0.8,
        conn_prob: float = 0.1,
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

        # Internal connectivity as sparse binary matrices
        # exc_conn[i,j] = 1 means excitatory neuron i projects to neuron j
        # inh_conn[i,j] = 1 means inhibitory neuron i projects to neuron j
        exc_mask = rng.random((self.n_exc, n_neurons)) < conn_prob
        inh_mask = rng.random((self.n_inh, n_neurons)) < conn_prob
        # No self-connections for exc neurons (inh are indexed separately)
        for i in range(self.n_exc):
            exc_mask[i, i] = False

        self.exc_conn = sparse.csr_matrix(exc_mask.astype(np.float64))
        self.inh_conn = sparse.csr_matrix(inh_mask.astype(np.float64))

        # Count connections per neuron (for diagnostics)
        self.c_exc = int(self.exc_conn.sum() / n_neurons)  # avg exc inputs per neuron
        self.c_inh = int(self.inh_conn.sum() / n_neurons)  # avg inh inputs per neuron

    def step(self, dt: float, external_input: np.ndarray | None = None) -> np.ndarray:
        """Advance one timestep.

        Args:
            dt: Timestep in ms.
            external_input: External input per neuron [n_neurons] in mV.

        Returns:
            Boolean spike array [n_neurons].
        """
        # Internal synaptic input from previous timestep's spikes
        exc_spikes = self.neurons.spikes[:self.n_exc]
        inh_spikes = self.neurons.spikes[self.n_exc:]

        if exc_spikes.any():
            # Count exc spikes arriving at each neuron
            exc_input = self.exc_conn[exc_spikes].toarray().sum(axis=0)
            self.neurons.receive_exc_spikes(exc_input)

        if inh_spikes.any():
            # Count inh spikes arriving at each neuron
            inh_input = self.inh_conn[inh_spikes].toarray().sum(axis=0)
            self.neurons.receive_inh_spikes(inh_input)

        # Step neurons
        spikes = self.neurons.step(dt, i_ext=external_input)
        return spikes

    @property
    def mean_firing_rate_hz(self) -> float:
        """Instantaneous fraction of neurons spiking."""
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

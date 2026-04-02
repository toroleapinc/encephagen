"""Spike-Timing Dependent Plasticity (STDP).

If presynaptic neuron fires BEFORE postsynaptic → strengthen (LTP).
If presynaptic neuron fires AFTER postsynaptic → weaken (LTD).
Only modifies excitatory synapses (Dale's law).

References:
    Bi & Poo (1998). Synaptic Modifications in Cultured Hippocampal
    Neurons: Dependence on Spike Timing, Synaptic Strength, and
    Postsynaptic Cell Type.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class STDPParams:
    """Parameters for STDP learning rule."""

    a_plus: float = 0.005       # LTP amplitude
    a_minus: float = 0.005      # LTD amplitude (asymmetric: slightly stronger)
    tau_plus: float = 20.0      # LTP time window (ms)
    tau_minus: float = 20.0     # LTD time window (ms)
    w_max: float = 10.0         # Maximum synaptic weight (mV)
    w_min: float = 0.0          # Minimum synaptic weight (mV)


class STDPRule:
    """Manages STDP traces and weight updates for a synapse population.

    Each neuron maintains two eligibility traces:
    - pre_trace: incremented on pre spike, decays exponentially
    - post_trace: incremented on post spike, decays exponentially

    When a post neuron spikes: potentiate all pre→post synapses
    proportional to pre_trace (pre fired recently → LTP).

    When a pre neuron spikes: depress all pre→post synapses
    proportional to post_trace (post fired recently → LTD).
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        params: STDPParams | None = None,
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.p = params or STDPParams()

        # Eligibility traces
        self.pre_trace = np.zeros(n_pre, dtype=np.float64)
        self.post_trace = np.zeros(n_post, dtype=np.float64)

    def update_traces(
        self, dt: float, pre_spikes: np.ndarray, post_spikes: np.ndarray
    ) -> None:
        """Decay traces and increment on spikes.

        Args:
            dt: Timestep in ms.
            pre_spikes: Boolean array [n_pre].
            post_spikes: Boolean array [n_post].
        """
        # Decay
        self.pre_trace *= np.exp(-dt / self.p.tau_plus)
        self.post_trace *= np.exp(-dt / self.p.tau_minus)

        # Increment on spike
        self.pre_trace[pre_spikes] += 1.0
        self.post_trace[post_spikes] += 1.0

    def compute_weight_updates(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        weights: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """Compute weight changes based on current spikes and traces.

        Args:
            pre_spikes: Boolean array [n_pre].
            post_spikes: Boolean array [n_post].
            weights: Current weight matrix [n_pre, n_post] (sparse).

        Returns:
            Sparse matrix of weight changes [n_pre, n_post].
        """
        # LTP: post spikes → strengthen pre→post proportional to pre_trace
        # Only for connections that exist (nonzero in weights)
        dw = sparse.csr_matrix(weights.shape, dtype=np.float64)

        post_spiking = np.where(post_spikes)[0]
        pre_spiking = np.where(pre_spikes)[0]

        if len(post_spiking) > 0 and self.pre_trace.sum() > 0:
            # For each post neuron that spiked, potentiate incoming connections
            for j in post_spiking:
                # Get pre neurons connected to j
                col = weights[:, j].toarray().ravel()
                connected = col > 0
                if connected.any():
                    # LTP: proportional to pre_trace
                    ltp = self.p.a_plus * self.pre_trace * connected
                    # Apply additively, clip later
                    dw[:, j] += sparse.csr_matrix(ltp.reshape(-1, 1))

        if len(pre_spiking) > 0 and self.post_trace.sum() > 0:
            # For each pre neuron that spiked, depress outgoing connections
            for i in pre_spiking:
                # Get post neurons connected from i
                row = weights[i, :].toarray().ravel()
                connected = row > 0
                if connected.any():
                    # LTD: proportional to post_trace
                    ltd = -self.p.a_minus * self.post_trace * connected
                    dw[i, :] += sparse.csr_matrix(ltd.reshape(1, -1))

        return dw

    def apply_updates(
        self,
        weights: sparse.csr_matrix,
        dw: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """Apply weight changes with bounds.

        Args:
            weights: Current weights [n_pre, n_post].
            dw: Weight changes [n_pre, n_post].

        Returns:
            Updated weights (clipped to [w_min, w_max]).
        """
        new_weights = weights + dw
        # Clip to bounds
        new_weights.data = np.clip(new_weights.data, self.p.w_min, self.p.w_max)
        # Ensure no new connections are created (maintain sparsity pattern)
        new_weights.eliminate_zeros()
        return new_weights

    def step(
        self,
        dt: float,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        weights: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """Full STDP step: update traces, compute and apply weight changes.

        Args:
            dt: Timestep in ms.
            pre_spikes: Boolean [n_pre].
            post_spikes: Boolean [n_post].
            weights: Current weight matrix [n_pre, n_post].

        Returns:
            Updated weight matrix.
        """
        self.update_traces(dt, pre_spikes, post_spikes)

        if pre_spikes.any() or post_spikes.any():
            dw = self.compute_weight_updates(pre_spikes, post_spikes, weights)
            if dw.nnz > 0:
                weights = self.apply_updates(weights, dw)

        return weights

    def reset(self) -> None:
        """Reset traces."""
        self.pre_trace[:] = 0
        self.post_trace[:] = 0

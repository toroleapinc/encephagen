"""Spike-Timing Dependent Plasticity (STDP) — Vectorized.

Uses trace-based STDP: each pre/post neuron maintains an eligibility
trace. Weight updates are computed as outer products of traces and
spikes, masked by existing connectivity. No per-neuron loops.

References:
    Bi & Poo (1998). Synaptic Modifications in Cultured Hippocampal Neurons.
    Morrison et al. (2008). Phenomenological models of synaptic plasticity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class STDPParams:
    """Parameters for STDP learning rule."""

    a_plus: float = 0.005       # LTP amplitude
    a_minus: float = 0.005      # LTD amplitude
    tau_plus: float = 20.0      # LTP time window (ms)
    tau_minus: float = 20.0     # LTD time window (ms)
    w_max: float = 10.0         # Maximum synaptic weight (mV)
    w_min: float = 0.0          # Minimum synaptic weight (mV)


class STDPRule:
    """Vectorized trace-based STDP.

    Weight update rule (applied every timestep):
      When post neuron j spikes: dw_ij += a_plus * pre_trace_i  (LTP)
      When pre neuron i spikes:  dw_ij -= a_minus * post_trace_j (LTD)

    This is computed as:
      dW = a_plus * (pre_trace outer post_spikes) - a_minus * (pre_spikes outer post_trace)
    masked by the connectivity pattern.
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

        self.pre_trace = np.zeros(n_pre, dtype=np.float64)
        self.post_trace = np.zeros(n_post, dtype=np.float64)

    def update_traces(self, dt: float, pre_spikes: np.ndarray, post_spikes: np.ndarray) -> None:
        """Decay traces and increment on spikes."""
        self.pre_trace *= np.exp(-dt / self.p.tau_plus)
        self.post_trace *= np.exp(-dt / self.p.tau_minus)
        self.pre_trace[pre_spikes] += 1.0
        self.post_trace[post_spikes] += 1.0

    def step(
        self,
        dt: float,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        weights: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """Full STDP step: update traces and modify weights.

        Only modifies existing connections (preserves sparsity pattern).
        """
        self.update_traces(dt, pre_spikes, post_spikes)

        has_post = post_spikes.any()
        has_pre = pre_spikes.any()

        if not has_post and not has_pre:
            return weights

        # Get the connectivity mask (nonzero pattern)
        # We only modify weights where connections exist
        rows, cols = weights.nonzero()
        if len(rows) == 0:
            return weights

        # Current weight values
        w = np.array(weights[rows, cols]).ravel()

        # LTP: post spikes, strengthen by pre_trace
        if has_post:
            post_spike_mask = post_spikes[cols]  # Which synapses have spiking post?
            if post_spike_mask.any():
                ltp = self.p.a_plus * self.pre_trace[rows] * post_spike_mask
                w += ltp

        # LTD: pre spikes, weaken by post_trace
        if has_pre:
            pre_spike_mask = pre_spikes[rows]  # Which synapses have spiking pre?
            if pre_spike_mask.any():
                ltd = self.p.a_minus * self.post_trace[cols] * pre_spike_mask
                w -= ltd

        # Clip to bounds
        w = np.clip(w, self.p.w_min, self.p.w_max)

        # Rebuild sparse matrix with updated weights
        new_weights = sparse.csr_matrix(
            (w, (rows, cols)), shape=weights.shape
        )
        return new_weights

    def reset(self) -> None:
        self.pre_trace[:] = 0
        self.post_trace[:] = 0

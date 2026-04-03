"""GPU-accelerated spiking brain using PyTorch + SpikingJelly.

Everything runs on the GPU as torch tensors. No Python loops over
neurons or regions. Sparse connectivity stored as torch.sparse.

This replaces the NumPy-based SpikingBrain for training speed.
Target: 50-200x speedup over CPU NumPy implementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from encephagen.connectome.loader import Connectome


@dataclass
class GPUBrainResult:
    """Recording from GPU brain simulation."""

    firing_rates: np.ndarray    # [timesteps, n_neurons] or [timesteps, n_regions]
    time: np.ndarray
    region_rates: np.ndarray    # [timesteps, n_regions] mean rate per region
    labels: list[str]


class SpikingBrainGPU(nn.Module):
    """Fully GPU-accelerated spiking brain.

    All neurons are in a single flat tensor. Connectivity is a sparse
    matrix on GPU. One matrix multiply per timestep replaces all the
    Python loops.

    Architecture:
      N_total neurons = n_regions × neurons_per_region
      Sparse weight matrix W [N_total, N_total] on GPU
      Membrane potential V [batch, N_total] on GPU
      Background Poisson drive via torch.poisson
    """

    def __init__(
        self,
        connectome: Connectome | None = None,
        n_total: int = 1000,
        n_regions: int = 1,
        neurons_per_region: int = 1000,
        exc_ratio: float = 0.8,
        internal_conn_prob: float = 0.1,
        between_conn_prob: float = 0.02,
        global_coupling: float = 0.05,
        ext_rate_factor: float = 3.5,
        # LIF parameters (normalized units)
        tau_m: float = 20.0,
        v_threshold: float = 20.0,
        v_reset: float = 0.0,
        t_ref: float = 2.0,
        dt: float = 0.1,
        # Synaptic
        tau_syn: float = 5.0,
        j_exc: float = 2.0,
        g_inh: float = 5.0,
        device: str = "cuda",
    ):
        super().__init__()

        if connectome is not None:
            n_regions = connectome.num_regions
            n_total = n_regions * neurons_per_region

        self.n_total = n_total
        self.n_regions = n_regions
        self.neurons_per_region = neurons_per_region
        self.n_exc_per_region = int(neurons_per_region * exc_ratio)
        self.dt = dt
        self.tau_m = tau_m
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.t_ref = t_ref
        self.tau_syn = tau_syn
        self.device = torch.device(device)

        # Scale j_exc by C_E
        c_e = max(1, int(neurons_per_region * exc_ratio * internal_conn_prob))
        self.j_eff = j_exc * (16.0 / c_e)
        self.j_inh = self.j_eff * g_inh

        # Background rate (Brunel calibration)
        c_ext = c_e
        nu_thr = v_threshold / (self.j_eff * c_ext * tau_m)
        self.bg_rate_per_step = c_ext * nu_thr * ext_rate_factor * dt

        print(f"  GPU Brain: {n_total} neurons, {n_regions} regions")
        print(f"  j_eff={self.j_eff:.3f}mV, bg_rate={self.bg_rate_per_step:.4f}/step")

        # Build connectivity matrix
        t0 = time.time()
        W = self._build_connectivity(
            connectome, n_regions, neurons_per_region,
            exc_ratio, internal_conn_prob, between_conn_prob,
            global_coupling,
        )
        # Store as sparse tensor on GPU
        indices = torch.tensor(np.array(W.nonzero()), dtype=torch.long)
        values = torch.tensor(W[W.nonzero()].A1, dtype=torch.float32)
        self.register_buffer(
            "W", torch.sparse_coo_tensor(indices, values, (n_total, n_total)).coalesce()
        )
        elapsed = time.time() - t0
        nnz = len(values)
        print(f"  Connectivity: {nnz:,} synapses ({elapsed:.1f}s)")

        # Neuron type masks
        is_exc = torch.zeros(n_total, dtype=torch.bool)
        for r in range(n_regions):
            start = r * neurons_per_region
            is_exc[start:start + self.n_exc_per_region] = True
        self.register_buffer("is_exc", is_exc)

        # Region boundaries for rate computation
        self.region_starts = [r * neurons_per_region for r in range(n_regions)]
        self.region_ends = [(r + 1) * neurons_per_region for r in range(n_regions)]

        if connectome is not None:
            self.labels = list(connectome.labels)
        else:
            self.labels = [f"region_{i}" for i in range(n_regions)]

        # Move to device
        self.to(self.device)

    def _build_connectivity(self, connectome, n_regions, npr, exc_ratio,
                            int_prob, bet_prob, g_coupling):
        """Build full sparse connectivity matrix [N, N]."""
        from scipy import sparse
        import numpy as np

        N = n_regions * npr
        n_exc = int(npr * exc_ratio)
        rng = np.random.default_rng(42)

        rows, cols, vals = [], [], []

        for r in range(n_regions):
            offset = r * npr

            # Internal excitatory connections
            for i in range(n_exc):
                targets = rng.random(npr) < int_prob
                targets[i] = False  # No self-connection
                for j in np.where(targets)[0]:
                    rows.append(offset + i)
                    cols.append(offset + j)
                    vals.append(self.j_eff)

            # Internal inhibitory connections
            for i in range(n_exc, npr):
                targets = rng.random(npr) < int_prob
                for j in np.where(targets)[0]:
                    rows.append(offset + i)
                    cols.append(offset + j)
                    vals.append(-self.j_inh)

        # Between-region connections (excitatory only)
        if connectome is not None:
            for src, dst, w in connectome.edges():
                if src == dst:
                    continue
                src_off = src * npr
                dst_off = dst * npr
                eff_prob = min(1.0, bet_prob * w * g_coupling)
                if eff_prob < 1e-6:
                    continue
                n_exc_src = int(npr * exc_ratio)
                for i in range(n_exc_src):
                    targets = rng.random(npr) < eff_prob
                    for j in np.where(targets)[0]:
                        rows.append(src_off + i)
                        cols.append(dst_off + j)
                        vals.append(self.j_eff * w * g_coupling)

        W = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(N, N), dtype=np.float32,
        )
        return W

    def init_state(self, batch_size: int = 1) -> dict:
        """Initialize neuron state."""
        return {
            "v": torch.rand(batch_size, self.n_total, device=self.device) * self.v_threshold,
            "i_syn": torch.zeros(batch_size, self.n_total, device=self.device),
            "refrac": torch.zeros(batch_size, self.n_total, device=self.device),
        }

    def step(self, state: dict, external: torch.Tensor | None = None) -> tuple[dict, torch.Tensor]:
        """One simulation timestep — fully on GPU.

        Args:
            state: dict with 'v', 'i_syn', 'refrac' tensors.
            external: [batch, n_total] external input in mV. Optional.

        Returns:
            (new_state, spikes) where spikes is [batch, n_total] bool.
        """
        v = state["v"]
        i_syn = state["i_syn"]
        refrac = state["refrac"]

        # Background Poisson drive
        bg_spikes = torch.poisson(
            torch.full_like(v, self.bg_rate_per_step)
        )
        i_syn = i_syn + bg_spikes * self.j_eff

        # External input
        if external is not None:
            i_syn = i_syn + external

        # Total input
        i_total = i_syn

        # Membrane dynamics (only for non-refractory neurons)
        active = refrac <= 0
        dv = (-v + i_total) / self.tau_m
        v = v + self.dt * dv * active.float()

        # Spike detection
        spikes = (v >= self.v_threshold) & active

        # Reset spiking neurons
        v = torch.where(spikes, torch.tensor(self.v_reset, device=self.device), v)
        refrac = torch.where(spikes, torch.tensor(self.t_ref, device=self.device), refrac)

        # Decrement refractory
        refrac = torch.clamp(refrac - self.dt, min=0)

        # Synaptic current from spikes (sparse matmul)
        # W is [N, N], spikes is [batch, N] → need spikes as float for matmul
        spike_float = spikes.float()
        # Sparse matmul: each spike propagates through W
        # W[i,j] means neuron i projects to neuron j with weight W[i,j]
        # synaptic_input[j] = sum_i W[i,j] * spike[i] = (W^T @ spike^T)^T
        syn_input = torch.sparse.mm(self.W.t(), spike_float.t()).t()

        # Decay + new input
        i_syn = i_syn * np.exp(-self.dt / self.tau_syn) + syn_input

        new_state = {"v": v, "i_syn": i_syn, "refrac": refrac}
        return new_state, spikes

    def simulate(
        self,
        duration_ms: float = 1000.0,
        transient_ms: float = 200.0,
        record_every: int = 100,
        external: torch.Tensor | None = None,
    ) -> GPUBrainResult:
        """Run simulation on GPU.

        Args:
            duration_ms: Total simulation time.
            transient_ms: Discard initial transient.
            record_every: Record region rates every N steps.
            external: Static external input [1, n_total].

        Returns:
            GPUBrainResult with firing rates.
        """
        total_steps = int(duration_ms / self.dt)
        transient_steps = int(transient_ms / self.dt)
        record_steps = (total_steps - transient_steps) // record_every

        region_rates = []
        state = self.init_state(batch_size=1)

        t0 = time.time()
        spike_acc = torch.zeros(1, self.n_total, device=self.device)
        acc_count = 0

        with torch.no_grad():
            for step in range(total_steps):
                state, spikes = self.step(state, external)

                if step >= transient_steps:
                    spike_acc += spikes.float()
                    acc_count += 1

                    if acc_count >= record_every:
                        # Compute per-region rates
                        rates = []
                        for r in range(self.n_regions):
                            s = self.region_starts[r]
                            e = self.region_ends[r]
                            region_spikes = spike_acc[0, s:e].sum().item()
                            rate_hz = region_spikes / (self.neurons_per_region * acc_count * self.dt / 1000)
                            rates.append(rate_hz)
                        region_rates.append(rates)
                        spike_acc.zero_()
                        acc_count = 0

                if step > 0 and step % (total_steps // 5) == 0:
                    pct = step / total_steps * 100
                    print(f"  {pct:.0f}%", end=" ", flush=True)

        elapsed = time.time() - t0
        sim_time_sec = duration_ms / 1000
        speedup = sim_time_sec / elapsed
        print(f"done ({elapsed:.1f}s, {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than real-time)")

        region_rates_arr = np.array(region_rates)
        time_arr = np.arange(len(region_rates)) * record_every * self.dt + transient_ms

        return GPUBrainResult(
            firing_rates=region_rates_arr,
            time=time_arr,
            region_rates=region_rates_arr,
            labels=self.labels,
        )

    def get_region_rates(self, spikes: torch.Tensor) -> np.ndarray:
        """Compute mean firing rate per region from spike tensor."""
        rates = np.zeros(self.n_regions)
        for r in range(self.n_regions):
            s = self.region_starts[r]
            e = self.region_ends[r]
            rates[r] = spikes[0, s:e].float().mean().item()
        return rates

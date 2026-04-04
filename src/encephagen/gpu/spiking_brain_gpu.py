"""GPU-accelerated spiking brain using PyTorch.

Everything runs on the GPU as torch tensors. No Python loops over
neurons or regions. Sparse connectivity stored as torch.sparse.

Features (v2):
  - Conduction delays from tract lengths (Euclidean distance / velocity)
  - Region-specific neuron types (fast-spiking PV+, regular pyramidal, thalamic)
  - ALIF adaptation for longer temporal credit assignment
  - NMDA slow synapses in PFC for working memory
  - E-prop learning integration
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import squareform, pdist

from encephagen.connectome.loader import Connectome
from encephagen.learning.eprop import EpropLearner, EpropParams


@dataclass
class GPUBrainResult:
    """Recording from GPU brain simulation."""

    firing_rates: np.ndarray    # [timesteps, n_neurons] or [timesteps, n_regions]
    time: np.ndarray
    region_rates: np.ndarray    # [timesteps, n_regions] mean rate per region
    labels: list[str]


class SpikingBrainGPU(nn.Module):
    """Fully GPU-accelerated spiking brain with biophysical diversity.

    Architecture:
      N_total neurons = n_regions × neurons_per_region
      Sparse weight matrix W [N_total, N_total] on GPU
      Per-neuron membrane time constant (fast-spiking vs regular)
      ALIF adaptation for temporal credit assignment
      Conduction delays from inter-region distances
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
        tau_nmda: float = 150.0,
        nmda_ratio: float = 0.3,
        j_exc: float = 2.0,
        g_inh: float = 5.0,
        pfc_regions: list | None = None,
        # New features
        use_delays: bool = False,
        conduction_velocity: float = 3.5,  # mm/ms
        use_neuron_types: bool = False,
        use_adaptation: bool = False,
        tau_adapt: float = 200.0,     # Adaptation time constant (ms)
        beta_adapt: float = 1.6,      # Adaptation strength (mV)
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
        self.tau_m_default = tau_m
        self.v_threshold_base = v_threshold
        self.v_reset = v_reset
        self.t_ref = t_ref
        self.tau_syn = tau_syn
        self.tau_nmda = tau_nmda
        self.nmda_ratio = nmda_ratio
        self.use_delays = use_delays
        self.use_adaptation = use_adaptation
        self.tau_adapt = tau_adapt
        self.beta_adapt = beta_adapt
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

        # ---- Per-neuron membrane time constants (neuron type diversity) ----
        tau_m_arr = torch.full((n_total,), tau_m)
        if use_neuron_types and connectome is not None:
            tau_m_arr = self._assign_neuron_types(
                connectome, n_regions, neurons_per_region, exc_ratio, tau_m
            )
        self.register_buffer("tau_m", tau_m_arr)

        # ---- Build connectivity matrix ----
        t0 = time.time()
        W = self._build_connectivity(
            connectome, n_regions, neurons_per_region,
            exc_ratio, internal_conn_prob, between_conn_prob,
            global_coupling,
        )
        indices = torch.tensor(np.array(W.nonzero()), dtype=torch.long)
        values = torch.tensor(W[W.nonzero()].A1, dtype=torch.float32)
        self.register_buffer(
            "W", torch.sparse_coo_tensor(indices, values, (n_total, n_total)).coalesce()
        )
        elapsed = time.time() - t0
        nnz = len(values)
        print(f"  Connectivity: {nnz:,} synapses ({elapsed:.1f}s)")

        # ---- Conduction delays ----
        if use_delays and connectome is not None and connectome.positions is not None:
            self._build_delay_buffer(
                connectome, n_regions, neurons_per_region,
                conduction_velocity, indices
            )
        else:
            self.use_delays = False

        # ---- Neuron type masks ----
        is_exc = torch.zeros(n_total, dtype=torch.bool)
        for r in range(n_regions):
            start = r * neurons_per_region
            is_exc[start:start + self.n_exc_per_region] = True
        self.register_buffer("is_exc", is_exc)

        # Region boundaries
        self.region_starts = [r * neurons_per_region for r in range(n_regions)]
        self.region_ends = [(r + 1) * neurons_per_region for r in range(n_regions)]

        if connectome is not None:
            self.labels = list(connectome.labels)
        else:
            self.labels = [f"region_{i}" for i in range(n_regions)]

        # PFC mask for NMDA
        pfc_mask = torch.zeros(n_total, dtype=torch.bool)
        if pfc_regions is not None:
            for ri in pfc_regions:
                pfc_mask[ri * neurons_per_region:(ri + 1) * neurons_per_region] = True
        self.register_buffer("pfc_mask", pfc_mask)
        n_pfc = pfc_mask.sum().item()
        if n_pfc > 0:
            print(f"  NMDA slow synapses: {n_pfc} PFC neurons (tau={tau_nmda}ms)")

        # Adaptation
        if use_adaptation:
            print(f"  ALIF adaptation: tau={tau_adapt}ms, beta={beta_adapt}mV")

        # E-prop learner (initialized on demand)
        self.learner: EpropLearner | None = None

        # Move to device
        self.to(self.device)

    def _assign_neuron_types(self, connectome, n_regions, npr, exc_ratio, default_tau):
        """Assign per-neuron tau_m based on region type and E/I identity.

        Neuron types:
          - Cortical pyramidal (excitatory): tau_m = 20ms (default)
          - Cortical fast-spiking PV+ (inhibitory): tau_m = 10ms
          - Thalamic relay (excitatory): tau_m = 15ms
          - Thalamic reticular (inhibitory): tau_m = 10ms
          - Subcortical (basal ganglia, amygdala): tau_m = 25ms
        """
        tau_arr = torch.full((n_regions * npr,), default_tau)
        n_exc = int(npr * exc_ratio)

        # Classify regions
        thalamic = set()
        subcortical = set()
        for i, label in enumerate(connectome.labels):
            lu = label.upper()
            if any(p in lu for p in ['TM', 'THAL']):
                thalamic.add(i)
            elif any(p in lu for p in ['BG', 'AMYG', 'HC', 'PHC', 'NAC', 'PUT', 'CAUD']):
                subcortical.add(i)

        n_thal = 0
        n_sub = 0
        n_fs = 0

        for r in range(n_regions):
            offset = r * npr
            if r in thalamic:
                # Thalamic relay (exc): faster
                tau_arr[offset:offset + n_exc] = 15.0
                # Thalamic reticular (inh): fast
                tau_arr[offset + n_exc:offset + npr] = 10.0
                n_thal += npr
            elif r in subcortical:
                # Subcortical: slower
                tau_arr[offset:offset + npr] = 25.0
                n_sub += npr
            else:
                # Cortical pyramidal (exc): default
                tau_arr[offset:offset + n_exc] = default_tau
                # Cortical PV+ fast-spiking (inh): fast
                tau_arr[offset + n_exc:offset + npr] = 10.0
                n_fs += (npr - n_exc)

        print(f"  Neuron types: {n_thal} thalamic, {n_sub} subcortical, {n_fs} fast-spiking inh")
        return tau_arr

    def _build_delay_buffer(self, connectome, n_regions, npr, velocity, W_indices):
        """Build conduction delay buffer for between-region synapses.

        Delays are computed from Euclidean distances between region centroids.
        Within-region delay = 0 (local circuits).
        Between-region delay = distance / velocity, discretized to timesteps.
        """
        # Compute region-to-region distances
        positions = connectome.positions
        dists = squareform(pdist(positions))  # [n_regions, n_regions]

        # Convert to delays in timesteps
        delays_ms = dists / velocity
        delays_steps = np.round(delays_ms / self.dt).astype(int)
        max_delay = int(delays_steps.max())

        # For each synapse, compute its delay
        pre_neurons = W_indices[0].numpy()
        post_neurons = W_indices[1].numpy()
        pre_regions = pre_neurons // npr
        post_regions = post_neurons // npr

        # Per-synapse delays
        syn_delays = np.zeros(len(pre_neurons), dtype=int)
        for i in range(len(pre_neurons)):
            r_pre = pre_regions[i]
            r_post = post_regions[i]
            if r_pre != r_post:
                syn_delays[i] = delays_steps[r_pre, r_post]
            # Within-region: delay = 0

        self.register_buffer("syn_delays", torch.tensor(syn_delays, dtype=torch.long))
        self.max_delay = max_delay

        # Spike history buffer: [max_delay+1, batch=1, n_total]
        # Circular buffer indexed by step % (max_delay + 1)
        self.delay_buffer_size = max_delay + 1

        # Store W indices for delay computation
        self.register_buffer("W_pre_idx", torch.tensor(pre_neurons, dtype=torch.long))
        self.register_buffer("W_post_idx", torch.tensor(post_neurons, dtype=torch.long))
        self.register_buffer("W_values_raw", W_indices.clone())  # not needed, but keep indices

        mean_delay = delays_ms[delays_ms > 0].mean()
        n_delayed = int((syn_delays > 0).sum())
        print(f"  Conduction delays: max={max_delay * self.dt:.1f}ms, "
              f"mean={mean_delay:.1f}ms, {n_delayed:,} delayed synapses")

    def _build_connectivity(self, connectome, n_regions, npr, exc_ratio,
                            int_prob, bet_prob, g_coupling):
        """Build full sparse connectivity matrix [N, N]."""
        from scipy import sparse

        N = n_regions * npr
        n_exc = int(npr * exc_ratio)
        rng = np.random.default_rng(42)

        rows, cols, vals = [], [], []

        for r in range(n_regions):
            offset = r * npr

            # Internal excitatory connections
            for i in range(n_exc):
                targets = rng.random(npr) < int_prob
                targets[i] = False
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
        state = {
            "v": torch.rand(batch_size, self.n_total, device=self.device) * self.v_threshold_base,
            "i_syn": torch.zeros(batch_size, self.n_total, device=self.device),
            "i_nmda": torch.zeros(batch_size, self.n_total, device=self.device),
            "refrac": torch.zeros(batch_size, self.n_total, device=self.device),
            "step_count": 0,
        }

        # ALIF adaptation variable
        if self.use_adaptation:
            state["adaptation"] = torch.zeros(batch_size, self.n_total, device=self.device)

        # Delay buffer
        if self.use_delays:
            state["spike_history"] = torch.zeros(
                self.delay_buffer_size, self.n_total, device=self.device
            )

        return state

    def enable_learning(self, params: EpropParams | None = None) -> EpropLearner:
        """Enable e-prop learning on this brain's synapses."""
        self.learner = EpropLearner(
            n_neurons=self.n_total,
            W_sparse=self.W,
            dt=self.dt,
            v_threshold=self.v_threshold_base,
            tau_m=self.tau_m_default,
            params=params,
            device=str(self.device),
            use_adaptation=self.use_adaptation,
            tau_adapt=self.tau_adapt,
            beta_adapt=self.beta_adapt,
        )
        return self.learner

    def apply_reward(self, spikes: torch.Tensor, reward: float) -> None:
        """Apply reward-modulated e-prop weight update."""
        if self.learner is None:
            return
        W_vals = self.W.coalesce().values()
        new_vals = self.learner.apply_reward(spikes, reward, W_vals)
        indices = self.W.coalesce().indices()
        self.W = torch.sparse_coo_tensor(
            indices, new_vals, self.W.shape
        ).coalesce()

    def step(self, state: dict, external: torch.Tensor | None = None) -> tuple[dict, torch.Tensor]:
        """One simulation timestep — fully on GPU.

        Args:
            state: dict with neuron state tensors.
            external: [batch, n_total] external input in mV. Optional.

        Returns:
            (new_state, spikes) where spikes is [batch, n_total] bool.
        """
        v = state["v"]
        i_syn = state["i_syn"]
        i_nmda = state["i_nmda"]
        refrac = state["refrac"]
        step_count = state["step_count"]

        # Background Poisson drive
        bg_spikes = torch.poisson(
            torch.full_like(v, self.bg_rate_per_step)
        )
        i_syn = i_syn + bg_spikes * self.j_eff

        # External input
        if external is not None:
            i_syn = i_syn + external

        # Total input: fast AMPA + slow NMDA
        i_total = i_syn + i_nmda

        # Membrane dynamics with per-neuron tau_m
        active = refrac <= 0
        dv = (-v + i_total) / self.tau_m.unsqueeze(0)
        v = v + self.dt * dv * active.float()

        # Adaptive threshold (ALIF)
        if self.use_adaptation:
            adaptation = state["adaptation"]
            v_thr = self.v_threshold_base + self.beta_adapt * adaptation
        else:
            v_thr = self.v_threshold_base

        # Spike detection
        spikes = (v >= v_thr) & active

        # Update eligibility traces BEFORE reset (need pre-spike voltage)
        if self.learner is not None:
            self.learner.step(v, spikes, adaptation if self.use_adaptation else None)

        # Reset spiking neurons (soft reset: subtract threshold)
        if self.use_adaptation:
            v = torch.where(spikes, v - v_thr, v)
            # Update adaptation: increment on spike, decay otherwise
            rho = np.exp(-self.dt / self.tau_adapt)
            adaptation = rho * adaptation + spikes.float()
        else:
            v = torch.where(spikes, torch.tensor(self.v_reset, device=self.device), v)

        refrac = torch.where(spikes, torch.tensor(self.t_ref, device=self.device), refrac)
        refrac = torch.clamp(refrac - self.dt, min=0)

        # Synaptic current from spikes
        spike_float = spikes.float()

        if self.use_delays:
            # Store current spikes in history buffer
            buf_idx = step_count % self.delay_buffer_size
            state["spike_history"][buf_idx] = spike_float[0]

            # For each synapse, look up the spike from (step - delay) ago
            # Gather delayed spikes per synapse
            W_coalesced = self.W.coalesce()
            W_indices = W_coalesced.indices()
            W_vals = W_coalesced.values()
            pre_idx = W_indices[0]  # [nnz] source neurons
            post_idx = W_indices[1]  # [nnz] target neurons

            # Compute buffer index for each synapse's delay
            delayed_buf_idx = (step_count - self.syn_delays) % self.delay_buffer_size
            # Clamp to valid range (before enough history exists)
            delayed_buf_idx = torch.clamp(delayed_buf_idx, min=0)

            # Gather delayed presynaptic spikes
            delayed_spikes = state["spike_history"][delayed_buf_idx, pre_idx]

            # Compute synaptic input: sum of W * delayed_spike for each postsynaptic neuron
            syn_input = torch.zeros(1, self.n_total, device=self.device)
            weighted_spikes = W_vals * delayed_spikes
            syn_input[0].scatter_add_(0, post_idx, weighted_spikes)
        else:
            syn_input = torch.sparse.mm(self.W.t(), spike_float.t()).t()

        # Fast AMPA decay + new input
        i_syn = i_syn * np.exp(-self.dt / self.tau_syn) + syn_input

        # Slow NMDA: only for PFC neurons
        if self.pfc_mask.any():
            nmda_input = syn_input * self.pfc_mask.float().unsqueeze(0) * self.nmda_ratio
            i_nmda = i_nmda * np.exp(-self.dt / self.tau_nmda) + nmda_input

        new_state = {
            "v": v, "i_syn": i_syn, "i_nmda": i_nmda,
            "refrac": refrac, "step_count": step_count + 1,
        }
        if self.use_adaptation:
            new_state["adaptation"] = adaptation
        if self.use_delays:
            new_state["spike_history"] = state["spike_history"]

        return new_state, spikes

    def simulate(
        self,
        duration_ms: float = 1000.0,
        transient_ms: float = 200.0,
        record_every: int = 100,
        external: torch.Tensor | None = None,
    ) -> GPUBrainResult:
        """Run simulation on GPU."""
        total_steps = int(duration_ms / self.dt)
        transient_steps = int(transient_ms / self.dt)

        region_rates = []
        state = self.init_state(batch_size=1)

        t0 = time.time()
        spike_acc = torch.zeros(1, self.n_total, device=self.device)
        acc_count = 0

        with torch.no_grad():
            for step_i in range(total_steps):
                state, spikes = self.step(state, external)

                if step_i >= transient_steps:
                    spike_acc += spikes.float()
                    acc_count += 1

                    if acc_count >= record_every:
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

                if step_i > 0 and step_i % (total_steps // 5) == 0:
                    pct = step_i / total_steps * 100
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

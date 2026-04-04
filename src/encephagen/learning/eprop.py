"""E-prop (Eligibility Propagation) for GPU spiking networks.

Implements the three-factor learning rule from Bellec et al. (2020):
  dW_ji = -eta * sum_t L_j^t * e_ji^t

Supports both LIF and ALIF (Adaptive LIF) neurons. ALIF adds a slow
adaptation variable that creates a recursive eligibility trace,
enabling temporal credit assignment over seconds (not just ~50ms).

Two reward modes:
  - Continuous: reward modulates eligibility at every timestep (Gerstner's recommendation)
  - Snapshot: capture eligibility at CS offset, apply reward later (simpler but weaker)

References:
    Bellec et al. (2020). A solution to the learning dilemma for
    recurrent networks of spiking neurons. Nature Communications.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EpropParams:
    """Parameters for e-prop learning."""

    lr: float = 1e-3                # Learning rate
    tau_e: float = 20.0             # Eligibility trace filter time constant (ms)
    gamma: float = 0.3              # Surrogate gradient dampening
    w_max: float = 15.0             # Maximum weight magnitude
    w_min: float = -15.0            # Minimum weight (inhibitory)
    reward_decay: float = 0.95      # Reward baseline exponential decay
    regularization: float = 1e-4    # Firing rate regularization strength
    target_rate: float = 0.02       # Target firing probability per step


class EpropLearner:
    """GPU-native e-prop learning for SpikingBrainGPU.

    Supports both LIF and ALIF neurons. For ALIF, the eligibility trace
    has a recursive adaptation component that carries credit over the
    adaptation timescale (typically ~200ms), enabling much longer
    temporal credit assignment than LIF (~20ms).
    """

    def __init__(
        self,
        n_neurons: int,
        W_sparse: torch.Tensor,
        dt: float = 0.1,
        v_threshold: float = 20.0,
        tau_m: float = 20.0,
        params: EpropParams | None = None,
        device: str = "cuda",
        use_adaptation: bool = False,
        tau_adapt: float = 200.0,
        beta_adapt: float = 1.6,
    ):
        self.n = n_neurons
        self.dt = dt
        self.v_thr = v_threshold
        self.tau_m = tau_m
        self.p = params or EpropParams()
        self.device = torch.device(device)
        self.alpha = np.exp(-dt / tau_m)
        self.use_adaptation = use_adaptation

        if use_adaptation:
            self.rho = np.exp(-dt / tau_adapt)
            self.beta = beta_adapt

        # Extract sparse connectivity indices
        indices = W_sparse.coalesce().indices()
        self.pre_idx = indices[0]
        self.post_idx = indices[1]
        self.nnz = indices.shape[1]

        print(f"  E-prop: {self.nnz:,} plastic synapses, lr={self.p.lr}")

        # Per-neuron traces
        self.z_bar = torch.zeros(n_neurons, device=self.device)

        # Per-synapse eligibility traces
        self.e_filtered = torch.zeros(self.nnz, device=self.device)

        # ALIF recursive adaptation eligibility (per-synapse)
        if use_adaptation:
            self.e_adapt = torch.zeros(self.nnz, device=self.device)

        # Snapshot for delayed reward (kept for backward compatibility)
        self.e_snapshot = torch.zeros(self.nnz, device=self.device)

        # Reward baseline
        self.reward_baseline = torch.tensor(0.0, device=self.device)
        self.trial_count = 0

        # Accumulator for continuous reward mode
        self.dW_accumulator = torch.zeros(self.nnz, device=self.device)

    def surrogate_grad(self, v: torch.Tensor, v_thr: torch.Tensor | float | None = None) -> torch.Tensor:
        """Piecewise linear surrogate gradient.

        Args:
            v: membrane potential [n_neurons]
            v_thr: actual spike threshold (scalar or per-neuron). If None, uses base threshold.
                   For ALIF, this should be v_threshold_base + beta * adaptation.
        """
        if v_thr is None:
            v_thr = self.v_thr
        return self.p.gamma * torch.clamp(
            1.0 - torch.abs(v - v_thr) / self.v_thr, min=0.0
        )

    def step(
        self,
        v: torch.Tensor,
        spikes: torch.Tensor,
        adaptation: torch.Tensor | None = None,
    ) -> None:
        """Update eligibility traces for one timestep.

        For ALIF neurons, the eligibility trace has a recursive component:
          e_adapt_ji^{t+1} = psi_j * z_bar_i + (rho - psi_j * beta) * e_adapt_ji^t
          e_ji = psi_j * z_bar_i - beta * e_adapt_ji

        This recursion carries credit over the adaptation timescale (~200ms),
        enabling learning over much longer temporal windows.
        """
        v_flat = v[0]
        z_flat = spikes[0].float()

        # Update filtered presynaptic trace (normalized)
        alpha_norm = 1.0 - self.alpha
        self.z_bar = self.alpha * self.z_bar + alpha_norm * z_flat

        # Compute actual threshold (adaptive for ALIF)
        if self.use_adaptation and adaptation is not None:
            actual_thr = self.v_thr + self.beta * adaptation[0]
        else:
            actual_thr = None

        # Surrogate gradient at postsynaptic neurons (using actual threshold)
        psi = self.surrogate_grad(v_flat, actual_thr)

        # Basic eligibility: psi_j * z_bar_i
        psi_post = psi[self.post_idx]
        z_bar_pre = self.z_bar[self.pre_idx]
        e_basic = psi_post * z_bar_pre

        if self.use_adaptation and adaptation is not None:
            # ALIF recursive eligibility trace
            # e_adapt carries credit through the slow adaptation dynamics
            self.e_adapt = e_basic + (self.rho - psi_post * self.beta) * self.e_adapt
            e_new = e_basic - self.beta * self.e_adapt
        else:
            e_new = e_basic

        # Filter eligibility trace
        kappa = np.exp(-self.dt / self.p.tau_e)
        kappa_norm = 1.0 - kappa
        self.e_filtered = kappa * self.e_filtered + kappa_norm * e_new

    def snapshot_eligibility(self) -> None:
        """Save current eligibility for delayed reward (backward compat)."""
        self.e_snapshot = self.e_filtered.clone()

    def apply_continuous_reward(
        self,
        reward: float,
        W_values: torch.Tensor,
    ) -> torch.Tensor:
        """Continuous reward modulation — apply at every timestep during reward.

        The reward modulates CURRENT eligibility (which naturally decays from
        CS-phase activity). No snapshot needed — the exponential decay of
        eligibility IS the temporal credit assignment.

        This is the correct e-prop formulation per Gerstner.
        """
        # Reward prediction error
        reward_signal = reward - self.reward_baseline.item()

        # Weight update from current eligibility
        dW = self.p.lr * reward_signal * self.e_filtered

        new_values = W_values + dW
        new_values = torch.clamp(new_values, self.p.w_min, self.p.w_max)
        return new_values

    def apply_reward(
        self,
        spikes: torch.Tensor,
        reward: float,
        W_values: torch.Tensor,
    ) -> torch.Tensor:
        """Apply reward — uses continuous mode (modulates current eligibility).

        For backward compatibility, still accepts spikes argument.
        """
        self.trial_count += 1
        self.reward_baseline = (
            self.p.reward_decay * self.reward_baseline
            + (1 - self.p.reward_decay) * reward
        )
        return self.apply_continuous_reward(reward, W_values)

    def apply_supervised(
        self,
        target_spikes: torch.Tensor,
        actual_spikes: torch.Tensor,
        W_values: torch.Tensor,
        output_neurons: torch.Tensor,
    ) -> torch.Tensor:
        """Supervised e-prop with random feedback alignment."""
        z_flat = actual_spikes[0].float()
        actual_output = z_flat[output_neurons]
        error = actual_output - target_spikes.float()

        n_out = len(output_neurons)
        if not hasattr(self, '_B_feedback') or self._B_feedback.shape[1] != n_out:
            self._B_feedback = torch.randn(
                self.n, n_out, device=self.device
            ) * 0.01

        L = self._B_feedback @ error
        L_per_synapse = L[self.post_idx]
        dW = -self.p.lr * L_per_synapse * self.e_filtered

        new_values = W_values + dW
        new_values = torch.clamp(new_values, self.p.w_min, self.p.w_max)
        return new_values

    def reset(self) -> None:
        """Reset all traces (between episodes)."""
        self.z_bar.zero_()
        self.e_filtered.zero_()
        self.e_snapshot.zero_()
        if self.use_adaptation:
            self.e_adapt.zero_()
        self.dW_accumulator.zero_()

"""E-prop (Eligibility Propagation) for GPU spiking networks.

Implements the three-factor learning rule from Bellec et al. (2020):
  dW_ji = -eta * sum_t L_j^t * e_ji^t

where:
  e_ji^t = psi_j^t * z_bar_i^{t-1}   (eligibility trace)
  psi_j^t = surrogate gradient at postsynaptic neuron
  z_bar_i = low-pass filtered presynaptic spikes
  L_j^t = learning signal (reward-modulated or supervised)

This replaces the Hebbian outer product used in experiments 15-21.
Unlike STDP which only captures spike correlations within ~20ms,
e-prop eligibility traces carry temporal credit assignment through
the membrane dynamics.

References:
    Bellec et al. (2020). A solution to the learning dilemma for
    recurrent networks of spiking neurons. Nature Communications.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    Maintains per-synapse eligibility traces and per-neuron filtered
    spike traces. Weight updates are computed as sparse operations
    to match the brain's sparse connectivity.

    Uses the "random e-prop" variant (most biologically plausible):
    learning signals are broadcast through fixed random feedback
    weights, no weight transport required.
    """

    def __init__(
        self,
        n_neurons: int,
        W_sparse: torch.Tensor,  # sparse COO weight matrix
        dt: float = 0.1,
        v_threshold: float = 20.0,
        tau_m: float = 20.0,
        params: EpropParams | None = None,
        device: str = "cuda",
    ):
        self.n = n_neurons
        self.dt = dt
        self.v_thr = v_threshold
        self.tau_m = tau_m
        self.p = params or EpropParams()
        self.device = torch.device(device)
        self.alpha = torch.exp(torch.tensor(-dt / tau_m, device=self.device))

        # Extract sparse connectivity indices
        indices = W_sparse.coalesce().indices()  # [2, nnz]
        self.pre_idx = indices[0]   # source neuron indices
        self.post_idx = indices[1]  # target neuron indices
        self.nnz = indices.shape[1]

        print(f"  E-prop: {self.nnz:,} plastic synapses, lr={self.p.lr}")

        # Per-neuron traces (cheap — only n_neurons floats)
        self.z_bar = torch.zeros(n_neurons, device=self.device)  # filtered pre spikes

        # Per-synapse eligibility traces (the main memory cost)
        self.e_trace = torch.zeros(self.nnz, device=self.device)
        # Filtered eligibility (for delayed reward)
        self.e_filtered = torch.zeros(self.nnz, device=self.device)
        # Snapshot of eligibility at CS offset (for delayed reward)
        self.e_snapshot = torch.zeros(self.nnz, device=self.device)

        # Reward baseline (running average for variance reduction)
        self.reward_baseline = torch.tensor(0.0, device=self.device)
        self.trial_count = 0

    def surrogate_grad(self, v: torch.Tensor) -> torch.Tensor:
        """Piecewise linear surrogate gradient (pseudo-derivative).

        psi_j = gamma * max(0, 1 - |v_j - v_thr| / v_thr)

        Non-zero only when membrane potential is near threshold,
        approximating the derivative of the Heaviside spike function.
        """
        return self.p.gamma * torch.clamp(
            1.0 - torch.abs(v - self.v_thr) / self.v_thr, min=0.0
        )

    def step(
        self,
        v: torch.Tensor,
        spikes: torch.Tensor,
    ) -> None:
        """Update eligibility traces for one timestep.

        Must be called AFTER brain.step() with the resulting v and spikes.
        v: [batch, n_neurons] membrane potential (pre-spike for surrogate)
        spikes: [batch, n_neurons] binary spike tensor
        """
        # Use batch dim 0
        v_flat = v[0]
        z_flat = spikes[0].float()

        # Update filtered presynaptic trace: z_bar = alpha * z_bar + z
        # Scale by (1-alpha) to normalize: z_bar converges to firing_rate/(1-alpha)
        # becomes independent of dt
        self.z_bar = self.alpha * self.z_bar + (1.0 - self.alpha) * z_flat

        # Surrogate gradient at postsynaptic neurons
        psi = self.surrogate_grad(v_flat)

        # Eligibility trace: e_ji = psi_j * z_bar_i (for each synapse)
        e_new = psi[self.post_idx] * self.z_bar[self.pre_idx]

        # Filter eligibility trace (exponential moving average)
        kappa = torch.exp(torch.tensor(-self.dt / self.p.tau_e, device=self.device))
        self.e_filtered = kappa * self.e_filtered + (1.0 - kappa) * e_new

        # Store raw trace too
        self.e_trace = e_new

    def snapshot_eligibility(self) -> None:
        """Save current eligibility traces for delayed reward.

        Call this at CS offset — the snapshot captures which synapses
        causally contributed to the CS response. When reward arrives
        later (US phase), it modulates these saved traces.
        """
        self.e_snapshot = self.e_filtered.clone()

    def apply_reward(
        self,
        spikes: torch.Tensor,
        reward: float,
        W_values: torch.Tensor,
    ) -> torch.Tensor:
        """Apply reward-modulated weight update using snapshotted eligibility.

        Three-factor rule: dW_ji = lr * reward_signal * e_snapshot_ji

        Uses the eligibility snapshot (from CS phase) rather than current
        eligibility, implementing temporal credit assignment: the reward
        arrives after the CS, but modulates the traces that were active
        DURING the CS.

        Args:
            spikes: [batch, n_neurons] current spikes (unused but kept for API)
            reward: scalar reward signal
            W_values: current weight values [nnz] (sparse tensor values)

        Returns:
            Updated weight values [nnz]
        """
        self.trial_count += 1

        # Reward baseline: slow-moving average (over trials, not steps)
        self.reward_baseline = (
            self.p.reward_decay * self.reward_baseline
            + (1 - self.p.reward_decay) * reward
        )

        # Reward prediction error
        reward_signal = reward - self.reward_baseline.item()

        # Weight update using SNAPSHOT eligibility (from CS phase)
        dW = self.p.lr * reward_signal * self.e_snapshot

        # Apply update
        new_values = W_values + dW

        # Enforce weight bounds
        new_values = torch.clamp(new_values, self.p.w_min, self.p.w_max)

        return new_values

    def apply_supervised(
        self,
        target_spikes: torch.Tensor,
        actual_spikes: torch.Tensor,
        W_values: torch.Tensor,
        output_neurons: torch.Tensor,
    ) -> torch.Tensor:
        """Apply supervised e-prop weight update.

        For supervised tasks, the learning signal is the output error:
          L_j = sum_k B_jk * (y_k - y_k_target)

        Uses random feedback weights B (feedback alignment).

        Args:
            target_spikes: desired output pattern [n_output]
            actual_spikes: actual output [batch, n_neurons]
            W_values: current weight values [nnz]
            output_neurons: indices of output neurons

        Returns:
            Updated weight values [nnz]
        """
        z_flat = actual_spikes[0].float()

        # Output error
        actual_output = z_flat[output_neurons]
        error = actual_output - target_spikes.float()

        # Broadcast error to all neurons via random feedback
        # (feedback alignment — biologically plausible)
        n_out = len(output_neurons)
        if not hasattr(self, '_B_feedback') or self._B_feedback.shape[1] != n_out:
            # Initialize random feedback weights (fixed, never updated)
            self._B_feedback = torch.randn(
                self.n, n_out, device=self.device
            ) * 0.01

        # Learning signal per neuron
        L = self._B_feedback @ error

        # Weight update
        L_per_synapse = L[self.post_idx]
        dW = -self.p.lr * L_per_synapse * self.e_filtered

        new_values = W_values + dW
        new_values = torch.clamp(new_values, self.p.w_min, self.p.w_max)

        return new_values

    def reset(self) -> None:
        """Reset all traces (between episodes)."""
        self.z_bar.zero_()
        self.e_trace.zero_()
        self.e_filtered.zero_()
        self.e_snapshot.zero_()

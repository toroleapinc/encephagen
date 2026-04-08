"""Basal Ganglia: Action selection through direct/indirect pathways.

The brain's action selection mechanism. Without it, the brain can sense
and react (reflexes) but can't CHOOSE between competing actions.

Architecture (Albin et al. 1989, DeLong 1990, Gurney et al. 2001):

  DIRECT pathway (GO):
    Cortex → D1 MSNs (striatum) →|inhib|→ GPi →|inhib|→ Thalamus
    D1 inhibits GPi, which releases thalamus → ACTION SELECTED

  INDIRECT pathway (STOP):
    Cortex → D2 MSNs (striatum) →|inhib|→ GPe →|inhib|→ STN →|excit|→ GPi
    Net effect: strengthens GPi → more thalamic inhibition → ACTION SUPPRESSED

  HYPERDIRECT pathway (EMERGENCY STOP):
    Cortex → STN → GPi (fast, bypasses striatum)
    Global suppression — stops everything quickly

  DOPAMINE modulation:
    D1 receptors: dopamine excites D1 MSNs → promotes GO
    D2 receptors: dopamine inhibits D2 MSNs → reduces STOP
    Net: dopamine = "do more of this action"

Neurons (~200):
  D1 MSNs: 40 (direct pathway, 4 action channels × 10)
  D2 MSNs: 40 (indirect pathway, 4 action channels × 10)
  GPe: 20
  STN: 20
  GPi/SNr: 30 (output, tonically active)
  SNc/VTA: 10 (dopamine neurons)
  Thalamic motor: 40 (receives BG output, drives cortex motor)
"""

from __future__ import annotations

import numpy as np
import torch


class BasalGanglia:
    """Spiking basal ganglia with direct/indirect/hyperdirect pathways.

    Implements action selection: multiple competing actions enter through
    cortical input to striatum. The winning action disinhibits thalamus.
    Dopamine biases selection toward rewarded actions.
    """

    def __init__(self, n_actions=4, device="cuda", dt=0.1):
        self.device = torch.device(device)
        self.dt = dt
        self.n_actions = n_actions

        # Neuron counts
        self.neurons_per_action = 10
        self.n_d1 = n_actions * self.neurons_per_action   # 40
        self.n_d2 = n_actions * self.neurons_per_action   # 40
        self.n_gpe = 20
        self.n_stn = 20
        self.n_gpi = 30    # output nucleus
        self.n_snc = 10    # dopamine neurons
        self.n_thal_motor = 40  # motor thalamus (BG output target)
        self.n_total = (self.n_d1 + self.n_d2 + self.n_gpe +
                         self.n_stn + self.n_gpi + self.n_snc + self.n_thal_motor)

        # Index map
        offset = 0
        self.idx = {}
        for name, count in [
            ('d1', self.n_d1), ('d2', self.n_d2),
            ('gpe', self.n_gpe), ('stn', self.n_stn),
            ('gpi', self.n_gpi), ('snc', self.n_snc),
            ('thal_motor', self.n_thal_motor),
        ]:
            self.idx[name] = slice(offset, offset + count)
            offset += count

        # Action channel indices within D1/D2
        self.action_idx = {}
        for a in range(n_actions):
            s = a * self.neurons_per_action
            e = s + self.neurons_per_action
            self.action_idx[a] = {
                'd1': slice(self.idx['d1'].start + s, self.idx['d1'].start + e),
                'd2': slice(self.idx['d2'].start + s, self.idx['d2'].start + e),
            }

        # Neuron parameters
        tau_m = torch.full((self.n_total,), 15.0)
        v_threshold = torch.full((self.n_total,), 8.0)
        # MSNs are harder to activate (high threshold, low excitability)
        tau_m[self.idx['d1']] = 20.0
        tau_m[self.idx['d2']] = 20.0
        # GPi is TONICALLY active (low threshold, high drive)
        tau_m[self.idx['gpi']] = 12.0
        # STN is excitatory and fast
        tau_m[self.idx['stn']] = 10.0

        self.tau_m = tau_m.to(self.device)
        self.v_threshold = v_threshold.to(self.device)

        # Build circuit
        W = self._build_circuit()
        self.W = W.to(self.device)

        # Tonic drive
        self.tonic_drive = torch.zeros(self.n_total, device=self.device)
        # GPi is tonically active — it continuously INHIBITS thalamus
        # Only when D1 inhibits GPi does thalamus get released
        self.tonic_drive[self.idx['gpi']] = 10.0  # tonically active
        self.tonic_drive[self.idx['gpe']] = 9.0
        self.tonic_drive[self.idx['stn']] = 6.0
        self.tonic_drive[self.idx['thal_motor']] = 8.5  # ABOVE threshold — GPi inhibition suppresses it
        self.tonic_drive[self.idx['snc']] = 7.0

        # Dopamine level (0 = no dopamine, 1 = high dopamine)
        self.dopamine = 0.5  # baseline

        print(f"  Basal ganglia: {self.n_total} neurons, {n_actions} action channels")
        print(f"    D1={self.n_d1}, D2={self.n_d2}, GPe={self.n_gpe}, "
              f"STN={self.n_stn}, GPi={self.n_gpi}, SNc={self.n_snc}, "
              f"ThalMotor={self.n_thal_motor}")

    def _build_circuit(self):
        """Build the basal ganglia circuit."""
        W = torch.zeros(self.n_total, self.n_total)

        def connect(src, dst, weight, prob=0.3, normalize=True):
            n_src = src.stop - src.start
            w = weight / max(n_src, 1) if normalize else weight
            for i in range(src.start, src.stop):
                for j in range(dst.start, dst.stop):
                    if i != j and np.random.random() < prob:
                        W[i, j] = w

        # === DIRECT PATHWAY: D1 →|inhib|→ GPi (CHANNEL-SPECIFIC) ===
        # Each action's D1 MSNs inhibit a SPECIFIC portion of GPi
        gpi_per_action = self.n_gpi // self.n_actions
        for a in range(self.n_actions):
            d1_s = self.action_idx[a]['d1']
            gpi_s = slice(self.idx['gpi'].start + a * gpi_per_action,
                          self.idx['gpi'].start + (a + 1) * gpi_per_action)
            connect(d1_s, gpi_s, -8.0, prob=0.7, normalize=False)  # must overcome GPi tonic drive

        # === INDIRECT PATHWAY: D2 →|inhib|→ GPe →|inhib|→ STN →|excit|→ GPi ===
        connect(self.idx['d2'], self.idx['gpe'], -2.5, prob=0.4)
        connect(self.idx['gpe'], self.idx['stn'], -2.0, prob=0.3)
        connect(self.idx['stn'], self.idx['gpi'], 2.0, prob=0.3)  # excitatory!

        # === HYPERDIRECT: STN → GPi (already covered above) ===
        # STN directly excites GPi for fast global stopping

        # === GPi → Thalamus motor (CHANNEL-SPECIFIC inhibitory output) ===
        thal_per_action = self.n_thal_motor // self.n_actions
        for a in range(self.n_actions):
            gpi_s = slice(self.idx['gpi'].start + a * gpi_per_action,
                          self.idx['gpi'].start + (a + 1) * gpi_per_action)
            thal_s = slice(self.idx['thal_motor'].start + a * thal_per_action,
                           self.idx['thal_motor'].start + (a + 1) * thal_per_action)
            connect(gpi_s, thal_s, -3.0, prob=0.5)

        # === GPe → GPi (weak inhibition, helps modulate) ===
        connect(self.idx['gpe'], self.idx['gpi'], -1.0, prob=0.2)

        # === Thalamus motor recurrent (sustains selected action) ===
        connect(self.idx['thal_motor'], self.idx['thal_motor'], 0.3, prob=0.1)

        # === Lateral inhibition within striatum (competition between actions) ===
        connect(self.idx['d1'], self.idx['d1'], -0.5, prob=0.1)
        connect(self.idx['d2'], self.idx['d2'], -0.5, prob=0.1)

        return W

    def init_state(self):
        return {
            'v': torch.rand(self.n_total, device=self.device) * 4.0,
            'refrac': torch.zeros(self.n_total, device=self.device),
            'i_syn': torch.zeros(self.n_total, device=self.device),
        }

    def step(self, state, cortex_input=None, reward=0.0):
        """One BG timestep.

        Args:
            state: neuron state dict
            cortex_input: dict {action_idx: intensity} — cortical drive to striatum
            reward: scalar reward signal → modulates dopamine

        Returns:
            state, selected_actions dict {action_idx: thalamic_output_rate}
        """
        v = state['v']; refrac = state['refrac']; i_syn = state['i_syn']

        # Dopamine modulation
        # Reward → dopamine → biases D1 (GO) and D2 (STOP)
        self.dopamine = np.clip(self.dopamine + reward * 0.1, 0.0, 1.5)
        self.dopamine *= 0.99  # decay toward baseline

        # External input
        ext = torch.zeros(self.n_total, device=self.device)

        # Cortical input → striatum (D1 and D2)
        if cortex_input is not None:
            for action_id, intensity in cortex_input.items():
                if action_id < self.n_actions:
                    # D1 gets input modulated by dopamine (D1 receptor: dopamine = more excitable)
                    d1_idx = self.action_idx[action_id]['d1']
                    ext[d1_idx] = intensity * (1.0 + self.dopamine * 0.5) * 12.0

                    # D2 gets input modulated inversely by dopamine
                    d2_idx = self.action_idx[action_id]['d2']
                    ext[d2_idx] = intensity * (1.0 - self.dopamine * 0.3) * 10.0

        # Noise
        noise = torch.randn(self.n_total, device=self.device) * 0.5

        # Membrane dynamics
        i_total = i_syn + self.tonic_drive + ext + noise
        active = refrac <= 0
        dv = (-v + i_total) / self.tau_m
        v = v + self.dt * dv * active.float()

        spikes = (v >= self.v_threshold) & active
        v = torch.where(spikes, torch.zeros_like(v), v)
        refrac = torch.where(spikes, torch.full_like(refrac, 1.5), refrac)
        refrac = torch.clamp(refrac - self.dt, min=0)

        syn_input = spikes.float() @ self.W  # W[src, dst]: spikes @ W gives input to each dst
        i_syn = i_syn * np.exp(-self.dt / 5.0) + syn_input

        state = {'v': v, 'refrac': refrac, 'i_syn': i_syn}

        # Output: which actions are selected (thalamic motor output per action channel)
        thal_rate = spikes[self.idx['thal_motor']].float().mean().item()
        selected_actions = {}
        thal_per_action = self.n_thal_motor // self.n_actions
        for a in range(self.n_actions):
            s = self.idx['thal_motor'].start + a * thal_per_action
            e = s + thal_per_action
            selected_actions[a] = spikes[s:e].float().mean().item()

        return state, selected_actions

    def get_status(self):
        """Get current BG state for display."""
        return {
            'dopamine': self.dopamine,
            'n_actions': self.n_actions,
        }

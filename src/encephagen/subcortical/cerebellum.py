"""Cerebellum: Motor coordination and error correction.

The cerebellum contains 80% of the brain's neurons but has a remarkably
uniform circuit (Marr 1969, Albus 1971):

  Mossy fibers → Granule cells → Parallel fibers → Purkinje cells → DCN → output
  Climbing fibers (error) → Purkinje cells (teaches the circuit)

Simplified to ~500 neurons:
  Granule cells: 200 (massive expansion layer, excitatory)
  Purkinje cells: 100 (output, inhibitory — each integrates many granule inputs)
  Deep cerebellar nuclei (DCN): 50 (final output, excitatory)
  Golgi cells: 50 (inhibitory, regulate granule cell activity)
  Inferior olive: 50 (climbing fiber error signal)
  Basket/stellate: 50 (inhibit Purkinje cells)

Function: receives motor command copy + sensory feedback,
computes error, adjusts timing of motor output.
"""

import numpy as np
import torch


class Cerebellum:
    def __init__(self, device="cuda", dt=0.1):
        self.device = torch.device(device)
        self.dt = dt

        self.n_granule = 200
        self.n_purkinje = 100
        self.n_dcn = 50
        self.n_golgi = 50
        self.n_olive = 50
        self.n_basket = 50
        self.n_total = (self.n_granule + self.n_purkinje + self.n_dcn +
                         self.n_golgi + self.n_olive + self.n_basket)

        offset = 0
        self.idx = {}
        for name, count in [('granule', self.n_granule), ('purkinje', self.n_purkinje),
                             ('dcn', self.n_dcn), ('golgi', self.n_golgi),
                             ('olive', self.n_olive), ('basket', self.n_basket)]:
            self.idx[name] = slice(offset, offset + count)
            offset += count

        tau_m = torch.full((self.n_total,), 15.0)
        tau_m[self.idx['granule']] = 8.0    # very fast
        tau_m[self.idx['purkinje']] = 25.0  # slow, integrative
        tau_m[self.idx['basket']] = 8.0     # fast inhibitory
        v_thr = torch.full((self.n_total,), 8.0)

        self.tau_m = tau_m.to(self.device)
        self.v_threshold = v_thr.to(self.device)
        self.W = self._build_circuit().to(self.device)

        self.tonic_drive = torch.zeros(self.n_total, device=self.device)
        self.tonic_drive[self.idx['granule']] = 5.0
        self.tonic_drive[self.idx['dcn']] = 8.0   # DCN tonically active
        self.tonic_drive[self.idx['olive']] = 5.0

        print(f"  Cerebellum: {self.n_total} neurons (granule={self.n_granule}, "
              f"purkinje={self.n_purkinje}, DCN={self.n_dcn})")

    def _build_circuit(self):
        W = torch.zeros(self.n_total, self.n_total)
        def conn(s, d, w, p=0.3, norm=True):
            ns = s.stop - s.start
            wn = w / max(ns, 1) if norm else w
            for i in range(s.start, s.stop):
                for j in range(d.start, d.stop):
                    if i != j and np.random.random() < p:
                        W[i, j] = wn

        # Granule → Purkinje (parallel fibers, massive convergence)
        conn(self.idx['granule'], self.idx['purkinje'], 0.8, p=0.4)
        # Granule → Golgi (feedback)
        conn(self.idx['granule'], self.idx['golgi'], 0.5, p=0.2)
        # Golgi → Granule (inhibitory feedback — regulates activity)
        conn(self.idx['golgi'], self.idx['granule'], -1.5, p=0.3, norm=False)
        # Purkinje → DCN (inhibitory output — Purkinje SUPPRESSES DCN)
        conn(self.idx['purkinje'], self.idx['dcn'], -3.0, p=0.5, norm=False)
        # Olive → Purkinje (climbing fibers — error signal, very strong)
        conn(self.idx['olive'], self.idx['purkinje'], 3.0, p=0.3, norm=False)
        # Basket → Purkinje (lateral inhibition)
        conn(self.idx['basket'], self.idx['purkinje'], -2.0, p=0.3, norm=False)
        # Granule → Basket (drives basket cells)
        conn(self.idx['granule'], self.idx['basket'], 0.6, p=0.3)
        # DCN → Olive (inhibitory feedback — controls error signal)
        conn(self.idx['dcn'], self.idx['olive'], -1.5, p=0.2, norm=False)
        return W

    def init_state(self):
        return {
            'v': torch.rand(self.n_total, device=self.device) * 4.0,
            'refrac': torch.zeros(self.n_total, device=self.device),
            'i_syn': torch.zeros(self.n_total, device=self.device),
        }

    def step(self, state, motor_copy=0.0, sensory_feedback=0.0, error_signal=0.0):
        """One cerebellar timestep.

        Args:
            motor_copy: copy of motor command (mossy fiber input to granule)
            sensory_feedback: proprioceptive feedback (mossy fiber)
            error_signal: mismatch between predicted and actual (olive input)
        """
        v = state['v']; refrac = state['refrac']; i_syn = state['i_syn']
        ext = torch.zeros(self.n_total, device=self.device)
        ext[self.idx['granule']] = (motor_copy + sensory_feedback) * 5.0
        ext[self.idx['olive']] = error_signal * 10.0

        noise = torch.randn(self.n_total, device=self.device) * 0.5
        i_total = i_syn + self.tonic_drive + ext + noise
        active = refrac <= 0
        dv = (-v + i_total) / self.tau_m
        v = v + self.dt * dv * active.float()
        spikes = (v >= self.v_threshold) & active
        v = torch.where(spikes, torch.zeros_like(v), v)
        refrac = torch.where(spikes, torch.full_like(refrac, 1.5), refrac)
        refrac = torch.clamp(refrac - self.dt, min=0)
        syn_input = spikes.float() @ self.W
        i_syn = i_syn * np.exp(-self.dt / 5.0) + syn_input

        state = {'v': v, 'refrac': refrac, 'i_syn': i_syn}
        dcn_output = spikes[self.idx['dcn']].float().mean().item()
        return state, dcn_output

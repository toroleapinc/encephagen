"""Amygdala: Fear, emotion, and valence.

Circuit (LeDoux 2000, Pape & Pare 2010):
  Lateral nucleus (LA): receives sensory input (thalamic + cortical)
  Basolateral (BLA): integrates context, connects to cortex
  Central nucleus (CeA): output → brainstem fear responses
    CeA → PAG (freezing), hypothalamus (stress hormones), brainstem (startle)

  Intercalated cells (ITC): inhibitory gate between BLA and CeA
    (extinction learning happens here — not modeled for innate phase)

Simplified to ~100 neurons.
"""

import numpy as np
import torch


class Amygdala:
    def __init__(self, device="cuda", dt=0.1):
        self.device = torch.device(device)
        self.dt = dt

        self.n_la = 30    # lateral (sensory input)
        self.n_bla = 30   # basolateral (context)
        self.n_cea = 20   # central (output → fear responses)
        self.n_itc = 20   # intercalated (inhibitory gate)
        self.n_total = 100

        offset = 0
        self.idx = {}
        for name, count in [('la', self.n_la), ('bla', self.n_bla),
                             ('cea', self.n_cea), ('itc', self.n_itc)]:
            self.idx[name] = slice(offset, offset + count)
            offset += count

        tau_m = torch.full((self.n_total,), 15.0)
        v_thr = torch.full((self.n_total,), 8.0)
        self.tau_m = tau_m.to(self.device)
        self.v_threshold = v_thr.to(self.device)
        self.W = self._build_circuit().to(self.device)

        self.tonic_drive = torch.zeros(self.n_total, device=self.device)
        self.tonic_drive[self.idx['la']] = 5.0
        self.tonic_drive[self.idx['bla']] = 5.0
        self.tonic_drive[self.idx['cea']] = 4.0  # low — needs LA/BLA input to fire

        self.fear_level = 0.0

        print(f"  Amygdala: {self.n_total} neurons "
              f"(LA={self.n_la}, BLA={self.n_bla}, CeA={self.n_cea})")

    def _build_circuit(self):
        W = torch.zeros(self.n_total, self.n_total)
        def conn(s, d, w, p=0.3, norm=True):
            ns = s.stop - s.start
            wn = w / max(ns, 1) if norm else w
            for i in range(s.start, s.stop):
                for j in range(d.start, d.stop):
                    if i != j and np.random.random() < p:
                        W[i, j] = wn

        conn(self.idx['la'], self.idx['bla'], 1.5, p=0.3)
        conn(self.idx['bla'], self.idx['cea'], 2.0, p=0.3)
        conn(self.idx['la'], self.idx['cea'], 1.0, p=0.2)   # direct fast path
        conn(self.idx['bla'], self.idx['itc'], 0.8, p=0.2)
        conn(self.idx['itc'], self.idx['cea'], -2.0, p=0.3, norm=False)  # inhibitory gate
        conn(self.idx['la'], self.idx['la'], 0.3, p=0.1)    # recurrent
        conn(self.idx['bla'], self.idx['bla'], 0.3, p=0.1)
        return W

    def init_state(self):
        return {
            'v': torch.rand(self.n_total, device=self.device) * 4.0,
            'refrac': torch.zeros(self.n_total, device=self.device),
            'i_syn': torch.zeros(self.n_total, device=self.device),
        }

    def step(self, state, threat_input=0.0, context_input=0.0):
        v = state['v']; refrac = state['refrac']; i_syn = state['i_syn']
        ext = torch.zeros(self.n_total, device=self.device)
        ext[self.idx['la']] = threat_input * 12.0
        ext[self.idx['bla']] = context_input * 8.0

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
        cea_output = spikes[self.idx['cea']].float().mean().item()
        self.fear_level = self.fear_level * 0.995 + cea_output * 20.0
        return state, self.fear_level

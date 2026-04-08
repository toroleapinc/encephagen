"""Hippocampus: Memory formation and spatial navigation.

Circuit (Andersen et al. 2007):
  Entorhinal cortex → Dentate gyrus → CA3 → CA1 → Subiculum → back to EC
  (The "trisynaptic loop")

  DG: pattern separation (orthogonalizes similar inputs)
  CA3: autoassociative memory (pattern completion, recurrent excitation)
  CA1: output/comparison (compares EC input with CA3 recalled memory)

Simplified to ~200 neurons.
"""

import numpy as np
import torch


class Hippocampus:
    def __init__(self, device="cuda", dt=0.1):
        self.device = torch.device(device)
        self.dt = dt

        self.n_ec = 40     # entorhinal cortex (input)
        self.n_dg = 60     # dentate gyrus (pattern separation — sparse)
        self.n_ca3 = 40    # CA3 (autoassociative, recurrent)
        self.n_ca1 = 40    # CA1 (output)
        self.n_inh = 20    # inhibitory interneurons
        self.n_total = self.n_ec + self.n_dg + self.n_ca3 + self.n_ca1 + self.n_inh

        offset = 0
        self.idx = {}
        for name, count in [('ec', self.n_ec), ('dg', self.n_dg),
                             ('ca3', self.n_ca3), ('ca1', self.n_ca1),
                             ('inh', self.n_inh)]:
            self.idx[name] = slice(offset, offset + count)
            offset += count

        tau_m = torch.full((self.n_total,), 20.0)
        tau_m[self.idx['dg']] = 15.0   # DG is selective
        tau_m[self.idx['inh']] = 8.0   # fast inhibitory
        v_thr = torch.full((self.n_total,), 8.0)
        v_thr[self.idx['dg']] = 10.0   # high threshold → sparse coding

        self.tau_m = tau_m.to(self.device)
        self.v_threshold = v_thr.to(self.device)
        self.W = self._build_circuit().to(self.device)

        self.tonic_drive = torch.zeros(self.n_total, device=self.device)
        self.tonic_drive[self.idx['ec']] = 6.0
        self.tonic_drive[self.idx['ca3']] = 6.0
        self.tonic_drive[self.idx['ca1']] = 6.0

        print(f"  Hippocampus: {self.n_total} neurons "
              f"(EC={self.n_ec}, DG={self.n_dg}, CA3={self.n_ca3}, CA1={self.n_ca1})")

    def _build_circuit(self):
        W = torch.zeros(self.n_total, self.n_total)
        def conn(s, d, w, p=0.3, norm=True):
            ns = s.stop - s.start
            wn = w / max(ns, 1) if norm else w
            for i in range(s.start, s.stop):
                for j in range(d.start, d.stop):
                    if i != j and np.random.random() < p:
                        W[i, j] = wn

        # Trisynaptic loop: EC → DG → CA3 → CA1
        conn(self.idx['ec'], self.idx['dg'], 1.5, p=0.3)
        conn(self.idx['dg'], self.idx['ca3'], 2.0, p=0.2)  # sparse but strong (mossy fibers)
        conn(self.idx['ca3'], self.idx['ca1'], 1.0, p=0.3)  # Schaffer collaterals
        # CA3 recurrent (autoassociative memory)
        conn(self.idx['ca3'], self.idx['ca3'], 0.5, p=0.15)
        # Direct EC → CA3 and EC → CA1 (perforant path)
        conn(self.idx['ec'], self.idx['ca3'], 0.8, p=0.2)
        conn(self.idx['ec'], self.idx['ca1'], 0.6, p=0.2)
        # Inhibition (DG and CA3 drive inhibitory interneurons)
        conn(self.idx['dg'], self.idx['inh'], 0.8, p=0.3)
        conn(self.idx['ca3'], self.idx['inh'], 0.8, p=0.3)
        conn(self.idx['inh'], self.idx['dg'], -2.0, p=0.3, norm=False)
        conn(self.idx['inh'], self.idx['ca3'], -2.0, p=0.3, norm=False)
        return W

    def init_state(self):
        return {
            'v': torch.rand(self.n_total, device=self.device) * 4.0,
            'refrac': torch.zeros(self.n_total, device=self.device),
            'i_syn': torch.zeros(self.n_total, device=self.device),
        }

    def step(self, state, cortical_input=0.0):
        v = state['v']; refrac = state['refrac']; i_syn = state['i_syn']
        ext = torch.zeros(self.n_total, device=self.device)
        ext[self.idx['ec']] = cortical_input * 10.0

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
        ca1_output = spikes[self.idx['ca1']].float().mean().item()
        return state, ca1_output

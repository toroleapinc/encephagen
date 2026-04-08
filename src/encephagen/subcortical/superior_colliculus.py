"""Superior Colliculus: Visual orienting — "look at that!"

The fastest visual pathway. Retina → SC → brainstem motor → eye/head turn.
Faster than cortical visual processing. Present and functional at birth.

Simplified to ~100 neurons:
  Superficial layer: 40 (visual input, retinotopic map)
  Deep layer: 40 (motor output, saccade commands)
  Inhibitory: 20 (competition between locations — winner-take-all)

Function: detects visual salience (motion, sudden appearance),
commands an orienting movement toward it.
"""

import numpy as np
import torch


class SuperiorColliculus:
    def __init__(self, n_locations=4, device="cuda", dt=0.1):
        self.device = torch.device(device)
        self.dt = dt
        self.n_locations = n_locations
        self.neurons_per_loc = 10

        self.n_superficial = n_locations * self.neurons_per_loc  # 40
        self.n_deep = n_locations * self.neurons_per_loc          # 40
        self.n_inh = 20
        self.n_total = self.n_superficial + self.n_deep + self.n_inh

        offset = 0
        self.idx = {}
        for name, count in [('superficial', self.n_superficial),
                             ('deep', self.n_deep), ('inh', self.n_inh)]:
            self.idx[name] = slice(offset, offset + count)
            offset += count

        tau_m = torch.full((self.n_total,), 10.0)  # fast responses
        v_thr = torch.full((self.n_total,), 8.0)
        self.tau_m = tau_m.to(self.device)
        self.v_threshold = v_thr.to(self.device)
        self.W = self._build_circuit().to(self.device)
        self.tonic_drive = torch.zeros(self.n_total, device=self.device)
        self.tonic_drive[self.idx['deep']] = 5.0

        print(f"  Superior colliculus: {self.n_total} neurons, {n_locations} locations")

    def _build_circuit(self):
        W = torch.zeros(self.n_total, self.n_total)
        npl = self.neurons_per_loc
        # Superficial → Deep (same location: excitatory)
        for loc in range(self.n_locations):
            s_start = self.idx['superficial'].start + loc * npl
            d_start = self.idx['deep'].start + loc * npl
            for i in range(s_start, s_start + npl):
                for j in range(d_start, d_start + npl):
                    if np.random.random() < 0.4:
                        W[i, j] = 2.0

        # Deep → Inhibitory (drives competition)
        for i in range(self.idx['deep'].start, self.idx['deep'].stop):
            for j in range(self.idx['inh'].start, self.idx['inh'].stop):
                if np.random.random() < 0.3:
                    W[i, j] = 1.0 / self.n_deep

        # Inhibitory → Deep (suppresses competing locations — winner-take-all)
        for i in range(self.idx['inh'].start, self.idx['inh'].stop):
            for j in range(self.idx['deep'].start, self.idx['deep'].stop):
                if np.random.random() < 0.3:
                    W[i, j] = -2.0 / self.n_inh

        return W

    def init_state(self):
        return {
            'v': torch.rand(self.n_total, device=self.device) * 4.0,
            'refrac': torch.zeros(self.n_total, device=self.device),
            'i_syn': torch.zeros(self.n_total, device=self.device),
        }

    def step(self, state, visual_input=None):
        """One SC timestep.

        Args:
            visual_input: dict {location_idx: intensity} or None
        Returns:
            state, orienting_direction (which location won)
        """
        v = state['v']; refrac = state['refrac']; i_syn = state['i_syn']
        ext = torch.zeros(self.n_total, device=self.device)

        if visual_input is not None:
            for loc, intensity in visual_input.items():
                if loc < self.n_locations:
                    npl = self.neurons_per_loc
                    s = self.idx['superficial'].start + loc * npl
                    ext[s:s + npl] = intensity * 15.0

        noise = torch.randn(self.n_total, device=self.device) * 0.3
        i_total = i_syn + self.tonic_drive + ext + noise
        active = refrac <= 0
        dv = (-v + i_total) / self.tau_m
        v = v + self.dt * dv * active.float()
        spikes = (v >= self.v_threshold) & active
        v = torch.where(spikes, torch.zeros_like(v), v)
        refrac = torch.where(spikes, torch.full_like(refrac, 1.0), refrac)
        refrac = torch.clamp(refrac - self.dt, min=0)
        syn_input = spikes.float() @ self.W
        i_syn = i_syn * np.exp(-self.dt / 5.0) + syn_input

        state = {'v': v, 'refrac': refrac, 'i_syn': i_syn}

        # Output: which location has highest deep layer activity
        npl = self.neurons_per_loc
        location_rates = {}
        for loc in range(self.n_locations):
            s = self.idx['deep'].start + loc * npl
            location_rates[loc] = spikes[s:s + npl].float().mean().item()

        return state, location_rates

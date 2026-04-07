"""Thalamus: The gateway to cortex.

Every sensory modality (except smell) passes through the thalamus
before reaching cortex. The thalamus filters, relays, and gates
sensory information through the thalamocortical loop.

Architecture:
  RELAY NUCLEI (excitatory, glutamatergic):
    LGN (Lateral Geniculate) — vision → visual cortex L4
    MGN (Medial Geniculate) — hearing → auditory cortex L4
    VPL (Ventral Posterolateral) — touch/proprioception → somatosensory cortex L4
    MD (Mediodorsal) — higher-order → prefrontal cortex L4

  RETICULAR NUCLEUS (TRN) (inhibitory, GABAergic):
    Shell surrounding relay nuclei
    Receives collaterals from ALL thalamocortical and corticothalamic fibers
    Inhibits relay nuclei → GATES what reaches cortex
    Generates alpha rhythm (~10Hz) via post-inhibitory rebound

  THALAMOCORTICAL LOOP:
    Relay → Cortex L4 (feedforward, driving)
    Cortex L6 → Relay (feedback, modulatory)
    Cortex L6 → TRN (controls the gate)
    TRN → Relay (inhibitory gate)

References:
    Sherman & Guillery (2006) Exploring the Thalamus
    Jones (2007) The Thalamus
    Crunelli & Hughes (2010) The slow (<1 Hz) rhythm of thalamocortical oscillations
"""

from __future__ import annotations

import numpy as np
import torch


class Thalamus:
    """Spiking thalamus with relay nuclei and reticular nucleus.

    ~200 LIF neurons organized as:
      LGN relay: 30 neurons (vision)
      MGN relay: 20 neurons (hearing)
      VPL relay: 30 neurons (touch/proprioception)
      MD relay: 20 neurons (higher-order → PFC)
      TRN: 50 neurons (inhibitory gate)
      Higher-order relay: 50 neurons (pulvinar-like, cortical relay)
    """

    def __init__(self, device="cuda", dt=0.1):
        self.device = torch.device(device)
        self.dt = dt

        # Neuron counts
        self.n_lgn = 30      # visual relay
        self.n_mgn = 20      # auditory relay
        self.n_vpl = 30      # somatosensory relay
        self.n_md = 20       # mediodorsal (→PFC)
        self.n_trn = 50      # reticular nucleus (inhibitory)
        self.n_ho = 50       # higher-order relay (pulvinar)
        self.n_total = self.n_lgn + self.n_mgn + self.n_vpl + self.n_md + self.n_trn + self.n_ho

        # Index map
        offset = 0
        self.idx = {}
        for name, count in [('lgn', self.n_lgn), ('mgn', self.n_mgn),
                             ('vpl', self.n_vpl), ('md', self.n_md),
                             ('trn', self.n_trn), ('ho', self.n_ho)]:
            self.idx[name] = slice(offset, offset + count)
            offset += count

        # Neuron parameters
        # Relay neurons: two modes (tonic = awake relay, burst = sleep/alpha)
        tau_m = torch.full((self.n_total,), 15.0)
        v_threshold = torch.full((self.n_total,), 8.0)
        # TRN: faster (inhibitory interneurons)
        tau_m[self.idx['trn']] = 10.0

        self.tau_m = tau_m.to(self.device)
        self.v_threshold = v_threshold.to(self.device)

        # Build connectivity
        W = self._build_circuit()
        self.W = W.to(self.device)

        # Tonic drive (background excitation — represents brainstem arousal)
        self.tonic_drive = torch.zeros(self.n_total, device=self.device)
        for name in ['lgn', 'mgn', 'vpl', 'md', 'ho']:
            self.tonic_drive[self.idx[name]] = 6.0  # subthreshold — needs sensory input to fire
        self.tonic_drive[self.idx['trn']] = 5.0  # TRN also needs input

        print(f"  Thalamus: {self.n_total} neurons "
              f"(LGN={self.n_lgn}, MGN={self.n_mgn}, VPL={self.n_vpl}, "
              f"MD={self.n_md}, TRN={self.n_trn}, HO={self.n_ho})")

    def _build_circuit(self):
        """Build thalamic circuit."""
        W = torch.zeros(self.n_total, self.n_total)

        def connect(src, dst, weight, prob=0.3):
            n_src = src.stop - src.start
            for i in range(src.start, src.stop):
                for j in range(dst.start, dst.stop):
                    if i != j and np.random.random() < prob:
                        W[i, j] = weight / max(n_src, 1)

        # === RELAY → TRN (collateral copies — relay excites TRN) ===
        # Every relay neuron sends a copy to TRN
        for relay in ['lgn', 'mgn', 'vpl', 'md', 'ho']:
            connect(self.idx[relay], self.idx['trn'], 1.5, prob=0.4)

        # === TRN → RELAY (inhibitory gate — TRN suppresses relay) ===
        # TRN inhibits ALL relay nuclei (the gate)
        for relay in ['lgn', 'mgn', 'vpl', 'md', 'ho']:
            connect(self.idx['trn'], self.idx[relay], -2.5, prob=0.3)

        # === TRN → TRN (mutual inhibition — creates competitive dynamics) ===
        connect(self.idx['trn'], self.idx['trn'], -1.0, prob=0.2)

        # === Recurrent excitation within relay (sustains activity) ===
        for relay in ['lgn', 'mgn', 'vpl', 'md', 'ho']:
            connect(self.idx[relay], self.idx[relay], 0.3, prob=0.15)

        return W

    def init_state(self):
        return {
            'v': torch.rand(self.n_total, device=self.device) * 4.0,
            'refrac': torch.zeros(self.n_total, device=self.device),
            'i_syn': torch.zeros(self.n_total, device=self.device),
        }

    def step(self, state, sensory_input=None, cortical_feedback=None):
        """One thalamic timestep.

        Args:
            state: neuron state dict
            sensory_input: dict with keys 'visual', 'auditory', 'somatosensory'
                          each a scalar [0, 1] intensity
            cortical_feedback: tensor [n_total] from cortex L6 (modulates gate)

        Returns:
            state, relay_output dict with firing rates per modality
        """
        v = state['v']; refrac = state['refrac']; i_syn = state['i_syn']

        # Sensory input → relay nuclei
        ext = torch.zeros(self.n_total, device=self.device)
        if sensory_input is not None:
            if 'visual' in sensory_input:
                ext[self.idx['lgn']] = sensory_input['visual'] * 15.0
            if 'auditory' in sensory_input:
                ext[self.idx['mgn']] = sensory_input['auditory'] * 15.0
            if 'somatosensory' in sensory_input:
                ext[self.idx['vpl']] = sensory_input['somatosensory'] * 15.0

        # Cortical feedback → relay + TRN (L6 modulates the gate)
        if cortical_feedback is not None:
            ext += cortical_feedback

        # Noise
        noise = torch.randn(self.n_total, device=self.device) * 0.5

        # Membrane dynamics
        i_total = i_syn + self.tonic_drive + ext + noise
        active = refrac <= 0
        dv = (-v + i_total) / self.tau_m
        v = v + self.dt * dv * active.float()

        # Spike
        spikes = (v >= self.v_threshold) & active
        v = torch.where(spikes, torch.zeros_like(v), v)
        refrac = torch.where(spikes, torch.full_like(refrac, 1.5), refrac)
        refrac = torch.clamp(refrac - self.dt, min=0)

        # Synaptic
        syn_input = self.W @ spikes.float()
        i_syn = i_syn * np.exp(-self.dt / 5.0) + syn_input

        state = {'v': v, 'refrac': refrac, 'i_syn': i_syn}

        # Output: relay firing rates (what reaches cortex)
        relay_output = {}
        for name in ['lgn', 'mgn', 'vpl', 'md', 'ho']:
            relay_output[name] = spikes[self.idx[name]].float().mean().item()

        return state, relay_output, spikes

    def get_cortex_input(self, relay_output, cell_map, n_regions, npr, tau_labels):
        """Convert relay output to cortex L4 input current.

        Routes thalamic relay output to the appropriate cortical region's L4:
          LGN → visual cortex L4
          MGN → auditory cortex L4
          VPL → somatosensory cortex L4
          MD → prefrontal cortex L4
        """
        n_total = n_regions * npr
        cortex_input = torch.zeros(n_total, device=self.device)

        # Map relay → cortical regions
        routing = {
            'lgn': ['Calcarine', 'Cuneus', 'Lingual', 'Occipital'],
            'mgn': ['Heschl', 'Temporal_Sup'],
            'vpl': ['Postcentral', 'Paracentral'],
            'md': ['Frontal_Sup', 'Frontal_Mid'],
            'ho': ['Parietal_Sup', 'Precuneus', 'Angular'],
        }

        for relay_name, cortex_patterns in routing.items():
            rate = relay_output[relay_name]
            if rate < 0.001:
                continue
            # Find matching cortical regions and inject into their L4
            for r, label in enumerate(tau_labels):
                if any(p in label for p in cortex_patterns):
                    l4_start, l4_end = cell_map.get((r, 'L4'), (0, 0))
                    if l4_end > l4_start:
                        cortex_input[l4_start:l4_end] = rate * 500.0  # very strong thalamic drive

        return cortex_input

    def get_status(self):
        """Get current thalamic activity for display."""
        return {name: 0.0 for name in ['lgn', 'mgn', 'vpl', 'md', 'trn', 'ho']}

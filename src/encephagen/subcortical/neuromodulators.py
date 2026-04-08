"""Neuromodulatory Systems: The brain's state controllers.

Four systems that RECONFIGURE the entire brain based on state:
  Dopamine (VTA/SNc): reward, motivation, learning signal
  Serotonin (Raphe nuclei): mood, impulse control, satiety
  Norepinephrine (Locus coeruleus): arousal, attention, fight/flight
  Acetylcholine (Basal forebrain): attention, learning, memory

Each system: small nucleus (~20 neurons) that broadcasts GLOBALLY.
One dopamine neuron can influence millions of targets.

Simplified to ~100 neurons total.
"""

import numpy as np
import torch


class NeuromodulatorSystem:
    """All four neuromodulatory systems as spiking neurons."""

    def __init__(self, device="cuda", dt=0.1):
        self.device = torch.device(device)
        self.dt = dt

        self.n_dopamine = 25   # VTA/SNc
        self.n_serotonin = 25  # Raphe
        self.n_norepinephrine = 25  # Locus coeruleus
        self.n_acetylcholine = 25   # Basal forebrain
        self.n_total = 100

        offset = 0
        self.idx = {}
        for name, count in [('dopamine', self.n_dopamine),
                             ('serotonin', self.n_serotonin),
                             ('norepinephrine', self.n_norepinephrine),
                             ('acetylcholine', self.n_acetylcholine)]:
            self.idx[name] = slice(offset, offset + count)
            offset += count

        # Slow-firing neurons (2-5 Hz tonic in real brain)
        tau_m = torch.full((self.n_total,), 25.0)  # slow
        v_thr = torch.full((self.n_total,), 8.0)
        self.tau_m = tau_m.to(self.device)
        self.v_threshold = v_thr.to(self.device)

        # Tonic drive (baseline firing)
        self.tonic_drive = torch.zeros(self.n_total, device=self.device)
        self.tonic_drive[self.idx['dopamine']] = 7.0      # baseline DA
        self.tonic_drive[self.idx['serotonin']] = 7.5      # tonic 5-HT
        self.tonic_drive[self.idx['norepinephrine']] = 6.0 # low arousal baseline
        self.tonic_drive[self.idx['acetylcholine']] = 6.0  # low attention baseline

        # Global modulation levels (what the rest of the brain reads)
        self.levels = {
            'dopamine': 0.5,        # 0=no reward, 1=high reward
            'serotonin': 0.5,       # 0=impulsive, 1=calm
            'norepinephrine': 0.3,  # 0=drowsy, 1=hyperaroused
            'acetylcholine': 0.3,   # 0=unfocused, 1=highly attentive
        }

        print(f"  Neuromodulators: {self.n_total} neurons "
              f"(DA={self.n_dopamine}, 5HT={self.n_serotonin}, "
              f"NE={self.n_norepinephrine}, ACh={self.n_acetylcholine})")

    def init_state(self):
        return {
            'v': torch.rand(self.n_total, device=self.device) * 4.0,
            'refrac': torch.zeros(self.n_total, device=self.device),
            'i_syn': torch.zeros(self.n_total, device=self.device),
        }

    def step(self, state, reward=0.0, threat=0.0, novelty=0.0, effort=0.0):
        """One neuromodulator timestep.

        Args:
            reward: positive reward signal → boosts dopamine
            threat: danger signal → boosts norepinephrine (fight/flight)
            novelty: new stimulus → boosts norepinephrine + acetylcholine
            effort: sustained effort → reduces serotonin (impulse control)
        """
        v = state['v']; refrac = state['refrac']; i_syn = state['i_syn']

        ext = torch.zeros(self.n_total, device=self.device)
        # Reward → dopamine burst
        ext[self.idx['dopamine']] = reward * 10.0
        # Threat → norepinephrine surge
        ext[self.idx['norepinephrine']] = threat * 12.0
        # Novelty → NE + ACh
        ext[self.idx['norepinephrine']] += novelty * 5.0
        ext[self.idx['acetylcholine']] = novelty * 8.0
        # Effort → reduce serotonin
        ext[self.idx['serotonin']] -= effort * 3.0

        noise = torch.randn(self.n_total, device=self.device) * 0.5
        i_total = i_syn + self.tonic_drive + ext + noise
        active = refrac <= 0
        dv = (-v + i_total) / self.tau_m
        v = v + self.dt * dv * active.float()
        spikes = (v >= self.v_threshold) & active
        v = torch.where(spikes, torch.zeros_like(v), v)
        refrac = torch.where(spikes, torch.full_like(refrac, 2.0), refrac)
        refrac = torch.clamp(refrac - self.dt, min=0)
        i_syn = i_syn * np.exp(-self.dt / 10.0)  # slow synaptic (neuromodulators are slow)

        state = {'v': v, 'refrac': refrac, 'i_syn': i_syn}

        # Update global levels (slow exponential tracking of firing rates)
        decay = 0.999
        for name in self.levels:
            rate = spikes[self.idx[name]].float().mean().item()
            self.levels[name] = self.levels[name] * decay + rate * (1 - decay) * 50

        return state, dict(self.levels)

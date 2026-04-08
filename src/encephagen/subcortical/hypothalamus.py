"""Hypothalamus: Survival drives and homeostasis.

Controls: hunger, thirst, temperature, sleep-wake cycle, hormones.
The "basic needs" computer.

Simplified to ~50 neurons:
  SCN: 15 (suprachiasmatic nucleus — circadian clock, ~24h oscillator)
  LH: 15 (lateral hypothalamus — hunger/arousal, orexin neurons)
  PVN: 10 (paraventricular — stress response, CRH → cortisol)
  VLPO: 10 (ventrolateral preoptic — sleep switch)

The SCN produces a slow oscillation (~0.01 Hz in real time, accelerated here)
that drives the sleep-wake cycle through VLPO ↔ LH mutual inhibition (flip-flop).
"""

import numpy as np
import torch


class Hypothalamus:
    def __init__(self, device="cuda", dt=0.1):
        self.device = torch.device(device)
        self.dt = dt

        self.n_scn = 15   # circadian clock
        self.n_lh = 15    # hunger/arousal (orexin)
        self.n_pvn = 10   # stress
        self.n_vlpo = 10  # sleep
        self.n_total = 50

        offset = 0
        self.idx = {}
        for name, count in [('scn', self.n_scn), ('lh', self.n_lh),
                             ('pvn', self.n_pvn), ('vlpo', self.n_vlpo)]:
            self.idx[name] = slice(offset, offset + count)
            offset += count

        tau_m = torch.full((self.n_total,), 30.0)  # very slow neurons
        v_thr = torch.full((self.n_total,), 8.0)
        self.tau_m = tau_m.to(self.device)
        self.v_threshold = v_thr.to(self.device)
        self.W = self._build_circuit().to(self.device)

        self.tonic_drive = torch.zeros(self.n_total, device=self.device)
        self.tonic_drive[self.idx['scn']] = 7.0   # circadian pacemaker
        self.tonic_drive[self.idx['lh']] = 7.0    # arousal drive
        self.tonic_drive[self.idx['vlpo']] = 5.0  # sleep pressure (builds up)

        # Internal state
        self.arousal = 0.7      # 0=deep sleep, 1=fully awake
        self.hunger = 0.3       # 0=satiated, 1=starving
        self.stress = 0.2       # 0=calm, 1=maximum stress
        self.circadian_phase = 0.0  # 0-2π

        print(f"  Hypothalamus: {self.n_total} neurons "
              f"(SCN={self.n_scn}, LH={self.n_lh}, PVN={self.n_pvn}, VLPO={self.n_vlpo})")

    def _build_circuit(self):
        W = torch.zeros(self.n_total, self.n_total)
        def conn(s, d, w, p=0.3, norm=True):
            ns = s.stop - s.start
            wn = w / max(ns, 1) if norm else w
            for i in range(s.start, s.stop):
                for j in range(d.start, d.stop):
                    if i != j and np.random.random() < p:
                        W[i, j] = wn

        # Sleep-wake flip-flop (Saper 2005): LH ↔ VLPO mutual inhibition
        conn(self.idx['lh'], self.idx['vlpo'], -2.0, p=0.3, norm=False)
        conn(self.idx['vlpo'], self.idx['lh'], -2.0, p=0.3, norm=False)
        # SCN → VLPO (circadian input to sleep switch)
        conn(self.idx['scn'], self.idx['vlpo'], 1.0, p=0.2)
        # Stress: LH → PVN (arousal drives stress response)
        conn(self.idx['lh'], self.idx['pvn'], 0.8, p=0.2)
        # SCN recurrent (pacemaker)
        conn(self.idx['scn'], self.idx['scn'], 0.3, p=0.1)
        return W

    def init_state(self):
        return {
            'v': torch.rand(self.n_total, device=self.device) * 4.0,
            'refrac': torch.zeros(self.n_total, device=self.device),
            'i_syn': torch.zeros(self.n_total, device=self.device),
        }

    def step(self, state, food_input=0.0, threat_input=0.0):
        v = state['v']; refrac = state['refrac']; i_syn = state['i_syn']
        ext = torch.zeros(self.n_total, device=self.device)

        # Circadian oscillation (slow phase advance)
        self.circadian_phase += self.dt * 0.0001  # very slow
        circadian_signal = np.sin(self.circadian_phase) * 3.0
        ext[self.idx['scn']] = circadian_signal

        # Hunger drives arousal
        ext[self.idx['lh']] = self.hunger * 5.0 + food_input * (-5.0)
        # Threat drives stress
        ext[self.idx['pvn']] = threat_input * 10.0

        noise = torch.randn(self.n_total, device=self.device) * 0.3
        i_total = i_syn + self.tonic_drive + ext + noise
        active = refrac <= 0
        dv = (-v + i_total) / self.tau_m
        v = v + self.dt * dv * active.float()
        spikes = (v >= self.v_threshold) & active
        v = torch.where(spikes, torch.zeros_like(v), v)
        refrac = torch.where(spikes, torch.full_like(refrac, 2.0), refrac)
        refrac = torch.clamp(refrac - self.dt, min=0)
        syn_input = spikes.float() @ self.W
        i_syn = i_syn * np.exp(-self.dt / 10.0) + syn_input

        state = {'v': v, 'refrac': refrac, 'i_syn': i_syn}

        # Update internal states
        lh_rate = spikes[self.idx['lh']].float().mean().item()
        vlpo_rate = spikes[self.idx['vlpo']].float().mean().item()
        pvn_rate = spikes[self.idx['pvn']].float().mean().item()

        self.arousal = self.arousal * 0.999 + lh_rate * 5.0
        self.stress = self.stress * 0.999 + pvn_rate * 5.0
        self.hunger = min(self.hunger + 0.00001, 1.0)  # slowly increases

        return state, {
            'arousal': np.clip(self.arousal, 0, 1),
            'hunger': self.hunger,
            'stress': np.clip(self.stress, 0, 1),
            'circadian': (np.sin(self.circadian_phase) + 1) / 2,
        }

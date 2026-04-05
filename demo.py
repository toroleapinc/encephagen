"""Encephagen Demo: A miniature human brain — alive and responsive.

16,000 spiking neurons across 80 brain regions, each with a unique
timescale from the Human Connectome Project T1w/T2w myelination map.
Connected by real structural connectivity with conduction delays.
Feedforward inhibition creates balanced dynamics.

This brain:
  - Has spontaneous activity at rest (different patterns per region)
  - Responds to sensory stimulation (visual, auditory, touch)
  - Shows timescale hierarchy (sensory fast, frontal slow)
  - Modulates a CPG for walking rhythm
  - Can learn from experience (e-prop)

Commands:
  look         → flash visual cortex
  sound        → activate auditory cortex
  touch        → stimulate somatosensory
  walk         → engage motor cortex → CPG
  status       → show all regions
  rhythm       → show CPG walking pattern
  learn <cmd>  → repeated stimulus + reward
  rest <N>     → observe resting activity for N seconds
  quit         → exit
"""

import json
import sys
import time

import numpy as np
import torch

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU
from encephagen.spinal.cpg import SpinalCPG, CPGParams
from encephagen.learning.eprop import EpropParams


class BrainDemo:
    def __init__(self):
        print("=" * 60)
        print("  ENCEPHAGEN — Miniature Human Brain")
        print("  16,000 spiking neurons | 80 regions | HCP connectome")
        print("=" * 60)

        # Load connectome
        sc = np.load('src/encephagen/connectome/bundled/neurolib80_weights.npy')
        tl = np.load('src/encephagen/connectome/bundled/neurolib80_tract_lengths.npy')
        labels = json.load(open('src/encephagen/connectome/bundled/neurolib80_labels.json'))
        self.tau_labels = json.load(open(
            'src/encephagen/connectome/bundled/neurolib80_t1t2_labels.json'))
        self.tau_m = np.load('src/encephagen/connectome/bundled/neurolib80_tau_m.npy')

        c = Connectome(sc, labels)
        c.tract_lengths = tl

        self.npr = 200
        self.n_regions = 80
        self.n_total = self.n_regions * self.npr
        self.device = "cuda"

        # Build brain
        self.brain = SpikingBrainGPU(
            connectome=c, neurons_per_region=self.npr,
            internal_conn_prob=0.05, between_conn_prob=0.03,
            global_coupling=5.0, ext_rate_factor=3.5,
            pfc_regions=[], device=self.device,
            use_delays=True, conduction_velocity=3.5,
            use_t1t2_gradient=True,
        )

        # Region groups
        self.groups = {}
        for key, patterns in [
            ('visual', ['Calcarine', 'Cuneus', 'Lingual', 'Occipital']),
            ('auditory', ['Heschl', 'Temporal_Sup']),
            ('somatosensory', ['Postcentral', 'Paracentral']),
            ('motor', ['Precentral', 'Supp_Motor']),
            ('frontal', ['Frontal_Sup', 'Frontal_Mid', 'Frontal_Inf']),
            ('parietal', ['Parietal', 'Angular', 'SupraMarginal']),
            ('temporal', ['Temporal_Mid', 'Temporal_Inf', 'Fusiform']),
            ('cingulate', ['Cingulate']),
        ]:
            self.groups[key] = [i for i, l in enumerate(self.tau_labels)
                                if any(p in l for p in patterns)]

        # CPG for walking
        self.cpg = SpinalCPG(CPGParams(
            tau=50.0, tau_adapt=500.0, drive=1.0,
            w_mutual=2.5, w_crossed=1.5, beta=2.5,
        ))

        # Init state
        print("\n  Warming up brain (3 seconds of neural time)...", flush=True)
        self.state = self.brain.init_state(batch_size=1)
        with torch.no_grad():
            for _ in range(3000):
                self.state, _ = self.brain.step(self.state)
        print("  Brain is alive.\n")

    def get_rates(self, steps=500):
        """Get current firing rates per group."""
        group_rates = {}
        with torch.no_grad():
            spikes_total = torch.zeros(self.n_regions, device=self.device)
            for _ in range(steps):
                self.state, spikes = self.brain.step(self.state)
                for r in range(self.n_regions):
                    spikes_total[r] += spikes[0, r*self.npr:(r+1)*self.npr].sum()
            rates = (spikes_total / (self.npr * steps)).cpu().numpy()

        for gname, gidx in self.groups.items():
            if gidx:
                group_rates[gname] = float(np.mean(rates[gidx]))
        return group_rates, rates

    def stimulate(self, group_name, amplitude=50.0, duration_steps=1000):
        """Stimulate a brain region group."""
        gidx = self.groups.get(group_name, [])
        if not gidx:
            print(f"  Unknown region: {group_name}")
            return

        # Before
        rates_before, _ = self.get_rates(500)

        # Stimulate
        ext = torch.zeros(1, self.n_total, device=self.device)
        for ri in gidx:
            ext[0, ri*self.npr:(ri+1)*self.npr] = amplitude

        with torch.no_grad():
            for _ in range(duration_steps):
                self.state, _ = self.brain.step(self.state, ext)

        # After (while stimulus still echoing)
        rates_after, _ = self.get_rates(500)

        # Display
        tau_mean = np.mean([self.tau_m[i] for i in gidx])
        print(f"\n  Stimulated {group_name} ({len(gidx)} regions, tau={tau_mean:.0f}ms)")
        print(f"  {'Region':<15} {'Before':>8} {'After':>8} {'Change':>8}")
        print(f"  {'─'*42}")
        for gn in ['visual', 'auditory', 'somatosensory', 'motor',
                    'frontal', 'parietal', 'temporal', 'cingulate']:
            b = rates_before.get(gn, 0)
            a = rates_after.get(gn, 0)
            ch = (a - b) / (b + 1e-10) * 100
            bar_b = "█" * int(b * 200)
            bar_a = "█" * int(a * 200)
            marker = " ← stimulated" if gn == group_name else ""
            print(f"  {gn:<15} {b:>7.4f}  {a:>7.4f}  {ch:>+6.1f}%{marker}")

    def show_status(self):
        """Show all region activity."""
        rates, raw = self.get_rates(1000)
        print(f"\n  Brain Status (16,000 neurons, 80 regions)")
        print(f"  {'Region':<15} {'Rate':>8} {'tau_m':>6} {'Activity'}")
        print(f"  {'─'*55}")
        for gn in ['visual', 'auditory', 'somatosensory', 'motor',
                    'frontal', 'parietal', 'temporal', 'cingulate']:
            r = rates.get(gn, 0)
            gidx = self.groups.get(gn, [])
            tau = np.mean([self.tau_m[i] for i in gidx]) if gidx else 0
            bar = "█" * int(r * 200) + "░" * max(0, 10 - int(r * 200))
            print(f"  {gn:<15} {r:>7.4f}  {tau:>4.0f}ms  {bar}")

    def walk_demo(self, steps=2000):
        """Brain modulates CPG walking rhythm."""
        print(f"\n  Walking demo ({steps*0.1:.0f}ms)")
        self.cpg.reset()

        motor_idx = self.groups.get('motor', [])
        dt = 0.1

        print(f"  {'Step':>6} {'R_hip':>7} {'R_knee':>7} {'L_hip':>7} {'L_knee':>7} {'Brain_motor':>12}")
        for s in range(0, steps, 200):
            # Get motor cortex drive
            motor_rate = 0
            with torch.no_grad():
                for _ in range(200):
                    self.state, spikes = self.brain.step(self.state)
                    for ri in motor_idx:
                        motor_rate += spikes[0, ri*self.npr:(ri+1)*self.npr].sum().item()
            motor_rate /= (len(motor_idx) * self.npr * 200)

            # Brain modulates CPG drive
            brain_drive = (motor_rate - 0.04) * 10  # center around baseline
            torques = self.cpg.step(dt * 200, brain_drive=brain_drive)

            print(f"  {s:>6} {torques[0]:>+7.3f} {torques[1]:>+7.3f} "
                  f"{torques[2]:>+7.3f} {torques[3]:>+7.3f} {motor_rate:>12.5f}")

    def run(self):
        """Interactive loop."""
        print("  Type 'help' for commands.\n")
        while True:
            try:
                cmd = input("brain> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if not cmd:
                continue
            elif cmd == 'quit' or cmd == 'exit':
                break
            elif cmd == 'help':
                print("  look     — visual stimulus")
                print("  sound    — auditory stimulus")
                print("  touch    — somatosensory stimulus")
                print("  walk     — CPG walking demo")
                print("  status   — show all regions")
                print("  rest N   — observe N seconds of resting activity")
                print("  quit     — exit")
            elif cmd == 'look':
                self.stimulate('visual')
            elif cmd == 'sound':
                self.stimulate('auditory')
            elif cmd == 'touch':
                self.stimulate('somatosensory')
            elif cmd == 'status':
                self.show_status()
            elif cmd == 'walk':
                self.walk_demo()
            elif cmd.startswith('rest'):
                parts = cmd.split()
                secs = int(parts[1]) if len(parts) > 1 else 1
                steps = int(secs * 10000)
                print(f"\n  Resting for {secs}s ({steps} steps)...")
                with torch.no_grad():
                    for _ in range(steps):
                        self.state, _ = self.brain.step(self.state)
                self.show_status()
            else:
                print(f"  Unknown command: {cmd}. Type 'help'.")

        print("\n  Brain shutting down.")


if __name__ == "__main__":
    demo = BrainDemo()
    demo.run()

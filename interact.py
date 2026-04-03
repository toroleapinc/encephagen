"""Interactive brain session.

Run: python interact.py

Commands:
  look A      — show pattern A to visual cortex
  look B      — show pattern B to visual cortex
  sound       — play tone to auditory cortex
  touch       — touch somatosensory cortex
  reward      — deliver reward to amygdala
  teach A     — pair pattern A with reward (one trial)
  train A 10  — pair pattern A with reward (10 trials)
  test        — compare A vs B response
  memory      — show a pattern, remove it, check PFC persistence
  status      — show brain activity summary
  wait N      — let brain run N seconds with no input
  quit        — exit
"""

import sys
import time
import numpy as np
import torch

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU


def classify_regions(labels):
    groups = {}
    for key, patterns in [
        ('visual', ['V1', 'V2', 'VAC']),
        ('auditory', ['A1', 'A2']),
        ('somatosensory', ['S1', 'S2']),
        ('prefrontal', ['PFC', 'FEF']),
        ('hippocampus', ['HC', 'PHC']),
        ('amygdala', ['AMYG']),
        ('basal_ganglia', ['BG']),
        ('thalamus', ['TM']),
        ('temporal', ['TC']),
        ('motor', ['M1', 'PMC']),
        ('cingulate', ['CC']),
        ('parietal', ['PC']),
    ]:
        groups[key] = [i for i, l in enumerate(labels)
                       if any(p in l.upper() for p in patterns)]
    return groups


class InteractiveBrain:
    def __init__(self):
        print("=" * 60)
        print("  ENCEPHAGEN — Interactive Brain Session")
        print("  19,200 spiking neurons | Human connectome | GPU")
        print("=" * 60)

        self.connectome = Connectome.from_bundled("tvb96")
        self.groups = classify_regions(self.connectome.labels)
        self.npr = 200
        self.n_total = 96 * self.npr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        pfc_regions = self.groups['prefrontal']
        self.brain = SpikingBrainGPU(
            connectome=self.connectome, neurons_per_region=self.npr,
            global_coupling=0.15, ext_rate_factor=3.5,
            tau_nmda=150.0, nmda_ratio=0.4,
            pfc_regions=pfc_regions,
            device=self.device,
        )

        # Create stimulus patterns
        rng = np.random.default_rng(42)
        self.patterns = {}
        for name in ['A', 'B', 'C', 'D']:
            p = np.zeros(self.npr, dtype=np.float32)
            p[rng.choice(self.npr, 60, replace=False)] = 1.0
            self.patterns[name] = p

        # Learning weights (dense copy for modification)
        self.W_dense = self.brain.W.to_dense()

        # Warm up
        print("\n  Warming up brain...", end=" ", flush=True)
        self.state = self.brain.init_state(batch_size=1)
        with torch.no_grad():
            for _ in range(5000):
                self.state, _ = self.brain.step(self.state)
        print("ready.")

        # Baseline rates
        self._measure_baseline()
        print(f"  Baseline rates measured.")
        print()

    def _measure_baseline(self):
        """Measure baseline activity for each region group."""
        self.baselines = {}
        counts = {name: 0.0 for name in self.groups}
        steps = 2000
        with torch.no_grad():
            for _ in range(steps):
                self.state, spikes = self.brain.step(self.state)
                for name, indices in self.groups.items():
                    for ri in indices:
                        counts[name] += spikes[0, ri*self.npr:(ri+1)*self.npr].sum().item()
        for name, indices in self.groups.items():
            if indices:
                self.baselines[name] = counts[name] / (len(indices) * self.npr * steps)

    def _get_activity(self, steps=500, external=None):
        """Run brain and return per-group activity rates."""
        counts = {name: 0.0 for name in self.groups}
        with torch.no_grad():
            for _ in range(steps):
                self.state, spikes = self.brain.step(self.state, external)
                for name, indices in self.groups.items():
                    for ri in indices:
                        counts[name] += spikes[0, ri*self.npr:(ri+1)*self.npr].sum().item()
        rates = {}
        for name, indices in self.groups.items():
            if indices:
                rates[name] = counts[name] / (len(indices) * self.npr * steps)
        return rates

    def _activity_bar(self, rate, baseline, width=30):
        """Render an ASCII activity bar."""
        if baseline > 0:
            ratio = rate / baseline
        else:
            ratio = 1.0
        filled = int(min(ratio, 3.0) / 3.0 * width)
        bar = "█" * filled + "░" * (width - filled)
        change = rate - baseline
        marker = "▲" if change > baseline * 0.1 else "▼" if change < -baseline * 0.1 else " "
        return f"{bar} {marker} {rate:.4f} ({change:+.4f})"

    def show_activity(self, rates, label=""):
        """Display brain activity with bars."""
        if label:
            print(f"\n  {label}")
        print(f"  {'Region':<16} {'Activity':^45}")
        print(f"  {'─'*62}")
        for name in ['visual', 'auditory', 'somatosensory', 'prefrontal',
                      'temporal', 'parietal', 'hippocampus', 'amygdala',
                      'basal_ganglia', 'thalamus', 'motor', 'cingulate']:
            if name in rates:
                bar = self._activity_bar(rates[name], self.baselines.get(name, 0.001))
                print(f"  {name:<16} {bar}")

    def cmd_look(self, pattern_name):
        """Present a visual pattern."""
        p = self.patterns.get(pattern_name.upper())
        if p is None:
            print(f"  Unknown pattern '{pattern_name}'. Available: {list(self.patterns.keys())}")
            return

        external = torch.zeros(1, self.n_total, device=self.device)
        pat_t = torch.tensor(p, device=self.device)
        for ri in self.groups['visual']:
            external[0, ri*self.npr:(ri+1)*self.npr] = pat_t * 12.0

        print(f"\n  Showing pattern {pattern_name.upper()} to visual cortex...")
        rates = self._get_activity(steps=500, external=external)
        self.show_activity(rates, f"Brain response to pattern {pattern_name.upper()}")

    def cmd_sound(self):
        """Play a tone to auditory cortex."""
        external = torch.zeros(1, self.n_total, device=self.device)
        for ri in self.groups['auditory']:
            external[0, ri*self.npr:(ri+1)*self.npr] = 15.0

        print(f"\n  Playing tone to auditory cortex...")
        rates = self._get_activity(steps=500, external=external)
        self.show_activity(rates, "Brain response to sound")

    def cmd_touch(self):
        """Stimulate somatosensory cortex."""
        external = torch.zeros(1, self.n_total, device=self.device)
        for ri in self.groups['somatosensory']:
            external[0, ri*self.npr:(ri+1)*self.npr] = 15.0

        print(f"\n  Touch stimulus to somatosensory cortex...")
        rates = self._get_activity(steps=500, external=external)
        self.show_activity(rates, "Brain response to touch")

    def cmd_reward(self):
        """Deliver reward signal."""
        external = torch.zeros(1, self.n_total, device=self.device)
        for ri in self.groups['amygdala']:
            external[0, ri*self.npr:(ri+1)*self.npr] = 20.0

        print(f"\n  Delivering reward to amygdala...")
        rates = self._get_activity(steps=500, external=external)
        self.show_activity(rates, "Brain response to reward")

    def cmd_teach(self, pattern_name, n_trials=1):
        """Pair pattern with reward (one or more trials)."""
        p = self.patterns.get(pattern_name.upper())
        if p is None:
            print(f"  Unknown pattern '{pattern_name}'.")
            return

        print(f"\n  Teaching: pattern {pattern_name.upper()} → reward ({n_trials} trials)")
        learning_rate = 0.003

        for trial in range(n_trials):
            # Present pattern
            ext_cs = torch.zeros(1, self.n_total, device=self.device)
            pat_t = torch.tensor(p, device=self.device)
            for ri in self.groups['visual']:
                ext_cs[0, ri*self.npr:(ri+1)*self.npr] = pat_t * 12.0

            cs_activity = torch.zeros(self.n_total, device=self.device)
            with torch.no_grad():
                for _ in range(500):
                    self.state, spikes = self.brain.step(self.state, ext_cs)
                    cs_activity += spikes[0]

            # Deliver reward
            ext_us = torch.zeros(1, self.n_total, device=self.device)
            for ri in self.groups['amygdala']:
                ext_us[0, ri*self.npr:(ri+1)*self.npr] = 20.0

            us_activity = torch.zeros(self.n_total, device=self.device)
            with torch.no_grad():
                for _ in range(500):
                    self.state, spikes = self.brain.step(self.state, ext_us)
                    us_activity += spikes[0]

            # Three-factor learning
            cs_active = (cs_activity > 2).float()
            us_active = (us_activity > 2).float()
            dW = learning_rate * torch.outer(cs_active, us_active)
            mask = (self.W_dense != 0).float()
            self.W_dense = torch.clamp(self.W_dense + dW * mask, -20, 20)

            # Gap
            with torch.no_grad():
                for _ in range(300):
                    self.state, _ = self.brain.step(self.state)

            if (trial + 1) % max(1, n_trials // 5) == 0:
                print(f"    Trial {trial+1}/{n_trials}")

        # Update brain weights
        indices = self.brain.W.coalesce().indices()
        new_vals = self.W_dense[indices[0], indices[1]]
        self.brain.W = torch.sparse_coo_tensor(
            indices, new_vals, self.brain.W.shape
        ).coalesce()

        print(f"  Learning complete. Pattern {pattern_name.upper()} associated with reward.")

    def cmd_test(self):
        """Compare response to all patterns."""
        print(f"\n  Testing all patterns...")
        results = {}
        for name in sorted(self.patterns.keys()):
            external = torch.zeros(1, self.n_total, device=self.device)
            pat_t = torch.tensor(self.patterns[name], device=self.device)
            for ri in self.groups['visual']:
                external[0, ri*self.npr:(ri+1)*self.npr] = pat_t * 12.0

            rates = self._get_activity(steps=500, external=external)
            # Response = amygdala + PFC deviation from baseline
            response = (rates.get('amygdala', 0) - self.baselines.get('amygdala', 0) +
                        rates.get('prefrontal', 0) - self.baselines.get('prefrontal', 0))
            results[name] = response

            # Gap
            with torch.no_grad():
                for _ in range(500):
                    self.state, _ = self.brain.step(self.state)

        print(f"\n  Pattern responses (amygdala + PFC above baseline):")
        max_resp = max(results.values()) if results else 1
        for name, resp in sorted(results.items(), key=lambda x: -x[1]):
            bar_len = int(resp / max(max_resp, 0.0001) * 30) if resp > 0 else 0
            bar = "█" * bar_len
            print(f"    {name}: {bar} {resp:+.5f}")

    def cmd_memory(self, pattern_name='A'):
        """Test working memory: show pattern, remove, check PFC."""
        p = self.patterns.get(pattern_name.upper())
        if p is None:
            print(f"  Unknown pattern '{pattern_name}'.")
            return

        print(f"\n  Working memory test: pattern {pattern_name.upper()}")

        # Baseline
        bl = self._get_activity(steps=300)
        bl_pfc = bl.get('prefrontal', 0)

        # Stimulus
        external = torch.zeros(1, self.n_total, device=self.device)
        pat_t = torch.tensor(p, device=self.device)
        for ri in self.groups['visual']:
            external[0, ri*self.npr:(ri+1)*self.npr] = pat_t * 15.0

        stim = self._get_activity(steps=500, external=external)
        stim_pfc = stim.get('prefrontal', 0)

        # Delay (no stimulus)
        delay = self._get_activity(steps=500)
        delay_pfc = delay.get('prefrontal', 0)

        # Late delay
        late = self._get_activity(steps=500)
        late_pfc = late.get('prefrontal', 0)

        stim_change = stim_pfc - bl_pfc
        persistence = (delay_pfc - bl_pfc) / max(stim_change, 0.00001) * 100

        print(f"    Baseline PFC:     {bl_pfc:.5f}")
        print(f"    During stimulus:  {stim_pfc:.5f} ({stim_change:+.5f})")
        print(f"    After removal:    {delay_pfc:.5f} ({delay_pfc-bl_pfc:+.5f})")
        print(f"    Late delay:       {late_pfc:.5f} ({late_pfc-bl_pfc:+.5f})")
        print(f"    Persistence:      {persistence:.0f}%")

    def cmd_status(self):
        """Show current brain state."""
        rates = self._get_activity(steps=300)
        self.show_activity(rates, "Current brain activity")

    def cmd_wait(self, seconds=1.0):
        """Let brain run with no input."""
        steps = int(seconds * 10000)
        print(f"\n  Running brain for {seconds}s ({steps} steps)...", end=" ", flush=True)
        with torch.no_grad():
            for _ in range(steps):
                self.state, _ = self.brain.step(self.state)
        print("done.")

    def run(self):
        """Main interactive loop."""
        print("\n  Commands: look A | sound | touch | reward | teach A | train A 10")
        print("            test | memory A | status | wait 2 | quit")
        print()

        while True:
            try:
                cmd = input("  brain> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye.")
                break

            if not cmd:
                continue

            parts = cmd.split()
            action = parts[0]

            try:
                if action == 'quit' or action == 'exit':
                    print("  Goodbye.")
                    break
                elif action == 'look' and len(parts) > 1:
                    self.cmd_look(parts[1])
                elif action == 'sound':
                    self.cmd_sound()
                elif action == 'touch':
                    self.cmd_touch()
                elif action == 'reward':
                    self.cmd_reward()
                elif action == 'teach' and len(parts) > 1:
                    self.cmd_teach(parts[1])
                elif action == 'train' and len(parts) > 2:
                    self.cmd_train_alias(parts[1], int(parts[2]))
                elif action == 'train' and len(parts) > 1:
                    self.cmd_teach(parts[1], n_trials=10)
                elif action == 'test':
                    self.cmd_test()
                elif action == 'memory' and len(parts) > 1:
                    self.cmd_memory(parts[1])
                elif action == 'memory':
                    self.cmd_memory()
                elif action == 'status':
                    self.cmd_status()
                elif action == 'wait':
                    secs = float(parts[1]) if len(parts) > 1 else 1.0
                    self.cmd_wait(secs)
                elif action == 'help':
                    print(__doc__)
                else:
                    print(f"  Unknown command: {cmd}")
                    print("  Commands: look A | sound | touch | reward | teach A | train A 10 | test | memory | status | wait | quit")
            except Exception as e:
                print(f"  Error: {e}")

    def cmd_train_alias(self, pattern_name, n_trials):
        self.cmd_teach(pattern_name, n_trials=n_trials)


if __name__ == "__main__":
    brain = InteractiveBrain()
    brain.run()

"""Experiment 17: Working memory — the brain can REMEMBER.

Present a stimulus, remove it, and test whether the brain
maintains a trace of what it saw.

Working memory in real brains uses persistent activity in PFC
maintained by recurrent excitation. Our 20 PFC regions (4000 neurons)
should be able to sustain activity after stimulus removal.

Test:
  1. Present stimulus → activity rises in visual + PFC
  2. Remove stimulus → visual activity drops, PFC should PERSIST
  3. Measure: how long does PFC maintain the trace?
  4. Compare: PFC persistence vs visual persistence
"""

import time
import numpy as np
import torch

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU


def classify_regions(labels):
    groups = {}
    for key, patterns in [
        ('visual', ['V1', 'V2', 'VAC']),
        ('prefrontal', ['PFC', 'FEF']),
        ('hippocampus', ['HC', 'PHC']),
        ('temporal', ['TC']),
        ('parietal', ['PC']),
        ('cingulate', ['CC']),
        ('thalamus', ['TM']),
    ]:
        groups[key] = [i for i, l in enumerate(labels) if any(p in l.upper() for p in patterns)]
    return groups


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 17: Working Memory")
    print("Does the brain maintain a trace after stimulus removal?")
    print("=" * 70)

    connectome = Connectome.from_bundled("tvb96")
    groups = classify_regions(connectome.labels)
    npr = 200
    n_total = 96 * npr
    device = "cuda"

    brain = SpikingBrainGPU(
        connectome=connectome, neurons_per_region=npr,
        global_coupling=0.15, ext_rate_factor=3.5, device=device,
    )

    visual_idx = groups['visual']
    pfc_idx = groups['prefrontal']
    hippo_idx = groups['hippocampus']
    thal_idx = groups['thalamus']

    print(f"\n  Visual: {len(visual_idx)} regions ({len(visual_idx)*npr} neurons)")
    print(f"  PFC: {len(pfc_idx)} regions ({len(pfc_idx)*npr} neurons)")
    print(f"  Hippocampus: {len(hippo_idx)} regions ({len(hippo_idx)*npr} neurons)")

    # Warm up
    print(f"\nWarming up...")
    state = brain.init_state(batch_size=1)
    with torch.no_grad():
        for _ in range(5000):
            state, _ = brain.step(state)

    # --- Record activity over time ---
    print(f"\n{'='*60}")
    print("Protocol: baseline(1s) → stimulus(0.5s) → delay(2s)")
    print(f"{'='*60}")

    region_groups = {
        'visual': visual_idx,
        'PFC': pfc_idx,
        'hippocampus': hippo_idx,
        'thalamus': thal_idx,
    }

    # Recording bins (every 50ms = 500 steps)
    bin_size = 500
    record = {name: [] for name in region_groups}

    # Phase 1: Baseline (1 second = 10000 steps)
    print("  Recording baseline (1s)...")
    for bin_i in range(20):  # 20 bins × 50ms = 1s
        bin_spikes = {name: 0.0 for name in region_groups}
        with torch.no_grad():
            for _ in range(bin_size):
                state, spikes = brain.step(state)
                for name, indices in region_groups.items():
                    for ri in indices:
                        bin_spikes[name] += spikes[0, ri*npr:(ri+1)*npr].sum().item()
        for name in region_groups:
            n_neurons = len(region_groups[name]) * npr
            rate = bin_spikes[name] / (n_neurons * bin_size)
            record[name].append(rate)

    # Phase 2: Stimulus (0.5 second = 5000 steps)
    print("  Presenting stimulus (0.5s)...")
    external = torch.zeros(1, n_total, device=device)
    for ri in visual_idx:
        external[0, ri*npr:(ri+1)*npr] = 15.0

    for bin_i in range(10):  # 10 bins × 50ms = 0.5s
        bin_spikes = {name: 0.0 for name in region_groups}
        with torch.no_grad():
            for _ in range(bin_size):
                state, spikes = brain.step(state, external)
                for name, indices in region_groups.items():
                    for ri in indices:
                        bin_spikes[name] += spikes[0, ri*npr:(ri+1)*npr].sum().item()
        for name in region_groups:
            n_neurons = len(region_groups[name]) * npr
            rate = bin_spikes[name] / (n_neurons * bin_size)
            record[name].append(rate)

    # Phase 3: Delay (2 seconds = 20000 steps) — NO stimulus
    print("  Delay period (2s) — stimulus removed...")
    for bin_i in range(40):  # 40 bins × 50ms = 2s
        bin_spikes = {name: 0.0 for name in region_groups}
        with torch.no_grad():
            for _ in range(bin_size):
                state, spikes = brain.step(state)
                for name, indices in region_groups.items():
                    for ri in indices:
                        bin_spikes[name] += spikes[0, ri*npr:(ri+1)*npr].sum().item()
        for name in region_groups:
            n_neurons = len(region_groups[name]) * npr
            rate = bin_spikes[name] / (n_neurons * bin_size)
            record[name].append(rate)

    # --- Analysis ---
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    time_ms = np.arange(len(record['visual'])) * 50  # ms

    for name in region_groups:
        trace = np.array(record[name])
        baseline = trace[:20].mean()
        stim = trace[20:30].mean()
        early_delay = trace[30:40].mean()  # First 0.5s after stimulus
        late_delay = trace[50:].mean()     # Last 1s of delay

        stim_response = stim - baseline
        early_persistence = early_delay - baseline
        late_persistence = late_delay - baseline

        print(f"\n  {name}:")
        print(f"    Baseline rate:     {baseline:.5f}")
        print(f"    During stimulus:   {stim:.5f} ({stim_response:+.5f})")
        print(f"    Early delay (0-0.5s): {early_delay:.5f} ({early_persistence:+.5f})")
        print(f"    Late delay (1-2s):    {late_delay:.5f} ({late_persistence:+.5f})")

        if abs(stim_response) > 0.0001:
            persistence_ratio = early_persistence / stim_response * 100
            print(f"    Persistence: {persistence_ratio:.0f}% of stimulus response maintained")

    # Key comparison: PFC vs Visual persistence
    vis_trace = np.array(record['visual'])
    pfc_trace = np.array(record['PFC'])

    vis_baseline = vis_trace[:20].mean()
    pfc_baseline = pfc_trace[:20].mean()
    vis_stim = vis_trace[20:30].mean() - vis_baseline
    pfc_stim = pfc_trace[20:30].mean() - pfc_baseline
    vis_persist = vis_trace[30:40].mean() - vis_baseline
    pfc_persist = pfc_trace[30:40].mean() - pfc_baseline

    print(f"\n  {'='*50}")
    print(f"  KEY COMPARISON: PFC vs Visual after stimulus removal")
    print(f"  {'='*50}")
    if vis_stim > 0.00001:
        print(f"  Visual persistence: {vis_persist/vis_stim*100:.0f}% of stimulus response")
    if pfc_stim > 0.00001:
        print(f"  PFC persistence:    {pfc_persist/pfc_stim*100:.0f}% of stimulus response")

    if pfc_persist > vis_persist and pfc_persist > 0:
        print(f"\n  ✓ WORKING MEMORY: PFC maintains trace longer than visual cortex!")
    elif pfc_persist > 0.00001:
        print(f"\n  ~ PFC shows some persistence but not clearly better than visual")
    else:
        print(f"\n  ✗ No persistent activity detected in PFC")


if __name__ == "__main__":
    run_experiment()

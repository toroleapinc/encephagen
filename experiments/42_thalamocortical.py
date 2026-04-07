"""Experiment 42: Thalamocortical loop — does the thalamus make the cortex work?

The thalamus is the gateway. All sensory input passes through it.
Now that we have it, test: does sensory → thalamus → cortex L4 → cortex L2/3
produce proper hierarchical propagation?

Architecture:
  Sensory → Thalamus relay → (gated by TRN) → Cortex L4 → L2/3 → L5 → output
"""

import numpy as np
import torch
import json
import time
from scipy import stats

from encephagen.connectome import Connectome
from encephagen.subcortical.thalamus import Thalamus
from encephagen.cortex.microcircuit import (
    build_microcircuit_connectivity,
    build_between_region_connectivity,
    CellTypeConfig,
)


def main():
    print("=" * 60)
    print("  THALAMOCORTICAL LOOP TEST")
    print("  Sensory → Thalamus → Cortex: does it propagate?")
    print("=" * 60)

    # Load connectome
    sc = np.load('src/encephagen/connectome/bundled/neurolib80_weights.npy')
    tl = np.load('src/encephagen/connectome/bundled/neurolib80_tract_lengths.npy')
    labels = json.load(open('src/encephagen/connectome/bundled/neurolib80_labels.json'))
    tau_labels = json.load(open('src/encephagen/connectome/bundled/neurolib80_t1t2_labels.json'))
    c = Connectome(sc, labels); c.tract_lengths = tl

    device = "cuda"
    npr = 200; n_regions = 80; n_total = n_regions * npr

    # Build thalamus
    thalamus = Thalamus(device=device)
    thal_state = thalamus.init_state()

    # Build microcircuit cortex
    print("  Building microcircuit cortex...", flush=True)
    config = CellTypeConfig()
    rng = np.random.default_rng(42)
    (int_r, int_c, int_v), tau_m, cell_map = \
        build_microcircuit_connectivity(npr, n_regions, config, rng)
    bet_r, bet_c, bet_v = build_between_region_connectivity(
        c.weights, cell_map, n_regions, npr, 5.0, rng)

    from scipy import sparse
    all_r = np.concatenate([int_r, bet_r])
    all_c = np.concatenate([int_c, bet_c])
    all_v = np.concatenate([int_v, bet_v])
    W = sparse.csr_matrix((all_v, (all_r, all_c)), shape=(n_total, n_total))
    indices = torch.tensor(np.array(W.nonzero()), dtype=torch.long)
    values = torch.tensor(W[W.nonzero()].A1, dtype=torch.float32)
    W_t = torch.sparse_coo_tensor(indices, values, (n_total, n_total)).coalesce().to(device)
    print(f"  Cortex: {n_total} neurons, {len(values):,} synapses")

    tau_m_d = torch.tensor(tau_m, dtype=torch.float32, device=device)
    tonic = torch.full((n_total,), 10.0, device=device)
    for r in range(n_regions):
        s, e = cell_map.get((r, 'L4'), (0, 0))
        tonic[s:e] = 11.0

    # Cortex state
    v = torch.rand(n_total, device=device) * 5.0
    refrac = torch.zeros(n_total, device=device)
    i_syn = torch.zeros(n_total, device=device)
    dt = 0.1; v_thr = 8.0

    # Warmup (3000 steps)
    print("  Warming up...", flush=True)
    for _ in range(3000):
        thal_state, relay, _ = thalamus.step(thal_state)
        thal_input = thalamus.get_cortex_input(relay, cell_map, n_regions, npr, tau_labels)
        noise = torch.randn(n_total, device=device) * 0.5
        i_total = i_syn + tonic + noise + thal_input
        active = refrac <= 0; dv = (-v + i_total) / tau_m_d
        v = v + dt * dv * active.float()
        spikes = (v >= v_thr) & active
        v = torch.where(spikes, torch.zeros_like(v), v)
        refrac = torch.where(spikes, torch.full_like(refrac, 1.5), refrac)
        refrac = torch.clamp(refrac - dt, min=0)
        syn_input = torch.sparse.mm(W_t.t(), spikes.float().unsqueeze(1)).squeeze(1)
        i_syn = i_syn * np.exp(-dt / 5.0) + syn_input

    # Regions of interest
    vis_regions = [i for i, l in enumerate(tau_labels) if 'Calcarine' in l or 'Cuneus' in l]
    aud_regions = [i for i, l in enumerate(tau_labels) if 'Heschl' in l or 'Temporal_Sup' in l]
    soma_regions = [i for i, l in enumerate(tau_labels) if 'Postcentral' in l]
    front_regions = [i for i, l in enumerate(tau_labels) if 'Frontal_Sup' in l]
    parietal_regions = [i for i, l in enumerate(tau_labels) if 'Parietal_Sup' in l]

    def measure_rates(regions, steps=2000):
        """Measure firing rates for given regions."""
        nonlocal v, refrac, i_syn, thal_state
        total = 0
        for _ in range(steps):
            thal_state, relay, _ = thalamus.step(thal_state)
            thal_input = thalamus.get_cortex_input(relay, cell_map, n_regions, npr, tau_labels)
            noise = torch.randn(n_total, device=device) * 0.5
            i_total = i_syn + tonic + noise + thal_input
            active = refrac <= 0; dv = (-v + i_total) / tau_m_d
            v = v + dt * dv * active.float()
            spikes = (v >= v_thr) & active
            v = torch.where(spikes, torch.zeros_like(v), v)
            refrac = torch.where(spikes, torch.full_like(refrac, 1.5), refrac)
            refrac = torch.clamp(refrac - dt, min=0)
            syn_input = torch.sparse.mm(W_t.t(), spikes.float().unsqueeze(1)).squeeze(1)
            i_syn = i_syn * np.exp(-dt / 5.0) + syn_input
            for r in regions:
                total += spikes[r*npr:(r+1)*npr].sum().item()
        return total / (len(regions) * npr * steps)

    # === TEST 1: Baseline (no sensory input) ===
    print(f"\n  TEST 1: Baseline (thalamus relay, no external stimulus)")
    bl = {}
    for name, regions in [('visual', vis_regions), ('auditory', aud_regions),
                            ('somatosensory', soma_regions), ('frontal', front_regions),
                            ('parietal', parietal_regions)]:
        bl[name] = measure_rates(regions)
    print(f"  {'Region':<18} {'Rate':>10}")
    for name, rate in bl.items():
        print(f"  {name:<18} {rate:>10.5f}")

    # === TEST 2: Visual stimulus through thalamus ===
    print(f"\n  TEST 2: Visual stimulus → LGN → visual cortex L4")
    # Set thalamic visual input
    thalamus_sensory = {'visual': 0.8, 'auditory': 0.0, 'somatosensory': 0.0}

    stim = {}
    for name, regions in [('visual', vis_regions), ('auditory', aud_regions),
                            ('somatosensory', soma_regions), ('frontal', front_regions),
                            ('parietal', parietal_regions)]:
        # Measure with sensory input going through thalamus
        total = 0
        for _ in range(3000):
            thal_state, relay, _ = thalamus.step(thal_state, sensory_input=thalamus_sensory)
            thal_input = thalamus.get_cortex_input(relay, cell_map, n_regions, npr, tau_labels)
            noise = torch.randn(n_total, device=device) * 0.5
            i_total = i_syn + tonic + noise + thal_input
            active = refrac <= 0; dv = (-v + i_total) / tau_m_d
            v = v + dt * dv * active.float()
            spikes = (v >= v_thr) & active
            v = torch.where(spikes, torch.zeros_like(v), v)
            refrac = torch.where(spikes, torch.full_like(refrac, 1.5), refrac)
            refrac = torch.clamp(refrac - dt, min=0)
            syn_input = torch.sparse.mm(W_t.t(), spikes.float().unsqueeze(1)).squeeze(1)
            i_syn = i_syn * np.exp(-dt / 5.0) + syn_input
            for r in regions:
                total += spikes[r*npr:(r+1)*npr].sum().item()
        stim[name] = total / (len(regions) * npr * 3000)

    print(f"  {'Region':<18} {'Baseline':>10} {'Stimulus':>10} {'Change':>10}")
    print(f"  {'─'*50}")
    for name in bl:
        change = (stim[name] - bl[name]) / (bl[name] + 1e-10) * 100
        marker = " ← through thalamus" if name == 'visual' else ""
        print(f"  {name:<18} {bl[name]:>10.5f} {stim[name]:>10.5f} {change:>+9.1f}%{marker}")

    # Check propagation
    vis_change = (stim['visual'] - bl['visual']) / (bl['visual'] + 1e-10) * 100
    temp_change = (stim.get('auditory', bl.get('auditory', 0)) - bl.get('auditory', 0)) / (bl.get('auditory', 0) + 1e-10) * 100
    front_change = (stim['frontal'] - bl['frontal']) / (bl['frontal'] + 1e-10) * 100
    par_change = (stim['parietal'] - bl['parietal']) / (bl['parietal'] + 1e-10) * 100

    print(f"\n  Thalamic relay rates: LGN={relay.get('lgn', 0):.4f} "
          f"TRN(gate)=estimated")

    print(f"\n{'='*60}")
    print(f"  VERDICT")
    print(f"{'='*60}")
    if vis_change > 10:
        print(f"\n  ✓ Visual cortex responds through thalamocortical loop (+{vis_change:.0f}%)")
        if abs(front_change) > 1 or abs(par_change) > 1:
            print(f"  ✓ Signal propagates to frontal ({front_change:+.1f}%) and parietal ({par_change:+.1f}%)")
            print(f"  The thalamus ENABLES hierarchical cortical processing!")
        else:
            print(f"  ✗ But signal doesn't propagate beyond visual cortex")
    else:
        print(f"\n  ✗ Visual cortex doesn't respond to thalamic input")


if __name__ == "__main__":
    main()

"""Experiment 40: Canonical Microcircuit Brain — does structure finally matter?

Replace random internal wiring with the canonical cortical microcircuit:
  - 7 cell types per region (L4, L2/3, L5, L6, PV+, SST+, VIP+)
  - Layer-specific connectivity rules (from MICrONS/Blue Brain Project)
  - Feedforward (L2/3→L4) vs feedback (L5/6→L2/3) between regions
  - Feedforward inhibition via PV+ interneurons

Test:
  1. Does stimulus NOW propagate? (it was blocked before)
  2. Does the structured microcircuit produce different dynamics than random?
  3. Does the connectome now provide advantage over random wiring?
"""

import numpy as np
import torch
import json
import time
from scipy import stats, signal

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU
from encephagen.cortex.microcircuit import (
    build_microcircuit_connectivity,
    build_between_region_connectivity,
    CellTypeConfig,
)


def build_microcircuit_brain(connectome, device="cuda"):
    """Build a brain with canonical microcircuit internal wiring."""
    npr = 200; n_regions = connectome.num_regions
    n_total = n_regions * npr

    print(f"  Building microcircuit brain ({n_total} neurons, {n_regions} regions)")

    config = CellTypeConfig()
    rng = np.random.default_rng(42)

    # 1. Build internal connectivity (canonical microcircuit per region)
    print(f"  Building internal microcircuit wiring...", flush=True)
    (int_rows, int_cols, int_vals), tau_m, cell_map = \
        build_microcircuit_connectivity(npr, n_regions, config, rng)

    # 2. Build between-region connectivity (feedforward/feedback)
    print(f"  Building feedforward/feedback pathways...", flush=True)
    bet_rows, bet_cols, bet_vals = build_between_region_connectivity(
        connectome.weights, cell_map, n_regions, npr,
        global_coupling=5.0, rng=rng)

    # 3. Combine into sparse matrix
    from scipy import sparse
    all_rows = np.concatenate([int_rows, bet_rows])
    all_cols = np.concatenate([int_cols, bet_cols])
    all_vals = np.concatenate([int_vals, bet_vals])

    W_scipy = sparse.csr_matrix(
        (all_vals, (all_rows, all_cols)), shape=(n_total, n_total))

    # Convert to torch sparse
    indices = torch.tensor(np.array(W_scipy.nonzero()), dtype=torch.long)
    values = torch.tensor(W_scipy[W_scipy.nonzero()].A1, dtype=torch.float32)
    W_torch = torch.sparse_coo_tensor(indices, values, (n_total, n_total)).coalesce()

    nnz = len(values)
    n_internal = len(int_vals)
    n_between = len(bet_vals)
    print(f"  Connectivity: {nnz:,} synapses ({n_internal:,} internal + {n_between:,} between-region)")

    # 4. Build tonic drive (background input)
    # L4 gets more drive (receives thalamic input)
    tonic = torch.full((n_total,), 7.0)  # base drive
    for r in range(n_regions):
        s, e = cell_map.get((r, 'L4'), (0, 0))
        tonic[s:e] = 9.0  # L4 gets thalamic drive

    # 5. Build the brain object manually
    tau_m_tensor = torch.tensor(tau_m, dtype=torch.float32)

    return W_torch, tau_m_tensor, tonic, cell_map, n_total, n_regions, npr


def run_brain(W, tau_m, tonic, cell_map, n_total, n_regions, npr,
              device="cuda", warmup_steps=3000, sim_steps=5000):
    """Run the microcircuit brain and record activity."""
    dt = 0.1; v_thr = 8.0

    v = torch.rand(n_total, device=device) * 5.0
    refrac = torch.zeros(n_total, device=device)
    i_syn = torch.zeros(n_total, device=device)
    tau_m_d = tau_m.to(device)
    tonic_d = tonic.to(device)
    W_d = W.to(device)

    # Warmup
    for _ in range(warmup_steps):
        noise = torch.randn(n_total, device=device) * 0.5
        i_total = i_syn + tonic_d + noise
        active = refrac <= 0
        dv = (-v + i_total) / tau_m_d
        v = v + dt * dv * active.float()
        spikes = (v >= v_thr) & active
        v = torch.where(spikes, torch.zeros_like(v), v)
        refrac = torch.where(spikes, torch.full_like(refrac, 1.5), refrac)
        refrac = torch.clamp(refrac - dt, min=0)
        syn_input = torch.sparse.mm(W_d.t(), spikes.float().unsqueeze(1)).squeeze(1)
        i_syn = i_syn * np.exp(-dt / 5.0) + syn_input

    # Record region rates (100-step bins = 10ms)
    region_ts = []
    spike_acc = torch.zeros(n_regions, device=device)
    acc = 0

    for step in range(sim_steps):
        noise = torch.randn(n_total, device=device) * 0.5
        i_total = i_syn + tonic_d + noise
        active = refrac <= 0
        dv = (-v + i_total) / tau_m_d
        v = v + dt * dv * active.float()
        spikes = (v >= v_thr) & active
        v = torch.where(spikes, torch.zeros_like(v), v)
        refrac = torch.where(spikes, torch.full_like(refrac, 1.5), refrac)
        refrac = torch.clamp(refrac - dt, min=0)
        syn_input = torch.sparse.mm(W_d.t(), spikes.float().unsqueeze(1)).squeeze(1)
        i_syn = i_syn * np.exp(-dt / 5.0) + syn_input

        for r in range(n_regions):
            spike_acc[r] += spikes[r*npr:(r+1)*npr].sum()
        acc += 1
        if acc >= 100:
            region_ts.append((spike_acc / (npr * acc)).cpu().numpy())
            spike_acc.zero_(); acc = 0

    return np.array(region_ts), v, i_syn, spikes


def test_propagation(W, tau_m, tonic, cell_map, n_total, n_regions, npr,
                     device="cuda"):
    """Test: does a visual stimulus propagate through the hierarchy?"""
    dt = 0.1; v_thr = 8.0

    tau_labels = json.load(open('src/encephagen/connectome/bundled/neurolib80_t1t2_labels.json'))
    vis_regions = [i for i, l in enumerate(tau_labels) if 'Calcarine' in l or 'Cuneus' in l]
    temp_regions = [i for i, l in enumerate(tau_labels) if 'Temporal_Mid' in l]
    front_regions = [i for i, l in enumerate(tau_labels) if 'Frontal_Sup' in l]

    v = torch.rand(n_total, device=device) * 5.0
    refrac = torch.zeros(n_total, device=device)
    i_syn = torch.zeros(n_total, device=device)
    tau_m_d = tau_m.to(device); tonic_d = tonic.to(device); W_d = W.to(device)

    # Warmup
    for _ in range(3000):
        noise = torch.randn(n_total, device=device) * 0.5
        i_total = i_syn + tonic_d + noise
        active = refrac <= 0; dv = (-v + i_total) / tau_m_d
        v = v + dt * dv * active.float()
        spikes = (v >= v_thr) & active
        v = torch.where(spikes, torch.zeros_like(v), v)
        refrac = torch.where(spikes, torch.full_like(refrac, 1.5), refrac)
        refrac = torch.clamp(refrac - dt, min=0)
        syn_input = torch.sparse.mm(W_d.t(), spikes.float().unsqueeze(1)).squeeze(1)
        i_syn = i_syn * np.exp(-dt / 5.0) + syn_input

    # Baseline
    def measure_rate(regions, steps=1000):
        nonlocal v, refrac, i_syn
        total = 0
        for _ in range(steps):
            noise = torch.randn(n_total, device=device) * 0.5
            i_total = i_syn + tonic_d + noise
            active = refrac <= 0; dv = (-v + i_total) / tau_m_d
            v = v + dt * dv * active.float()
            spikes = (v >= v_thr) & active
            v = torch.where(spikes, torch.zeros_like(v), v)
            refrac = torch.where(spikes, torch.full_like(refrac, 1.5), refrac)
            refrac = torch.clamp(refrac - dt, min=0)
            syn_input = torch.sparse.mm(W_d.t(), spikes.float().unsqueeze(1)).squeeze(1)
            i_syn = i_syn * np.exp(-dt / 5.0) + syn_input
            for r in regions:
                total += spikes[r*npr:(r+1)*npr].sum().item()
        return total / (len(regions) * npr * steps)

    bl_vis = measure_rate(vis_regions)
    bl_temp = measure_rate(temp_regions)
    bl_front = measure_rate(front_regions)

    # Stimulus: inject current into visual L4 neurons
    ext = torch.zeros(n_total, device=device)
    for r in vis_regions:
        s, e = cell_map.get((r, 'L4'), (0, 0))
        ext[s:e] = 30.0  # strong input to visual L4

    # Measure with stimulus
    def measure_rate_stim(regions, steps=3000):
        nonlocal v, refrac, i_syn
        total = 0
        for _ in range(steps):
            noise = torch.randn(n_total, device=device) * 0.5
            i_total = i_syn + tonic_d + noise + ext
            active = refrac <= 0; dv = (-v + i_total) / tau_m_d
            v = v + dt * dv * active.float()
            spikes = (v >= v_thr) & active
            v = torch.where(spikes, torch.zeros_like(v), v)
            refrac = torch.where(spikes, torch.full_like(refrac, 1.5), refrac)
            refrac = torch.clamp(refrac - dt, min=0)
            syn_input = torch.sparse.mm(W_d.t(), spikes.float().unsqueeze(1)).squeeze(1)
            i_syn = i_syn * np.exp(-dt / 5.0) + syn_input
            for r in regions:
                total += spikes[r*npr:(r+1)*npr].sum().item()
        return total / (len(regions) * npr * steps)

    stim_vis = measure_rate_stim(vis_regions)
    stim_temp = measure_rate_stim(temp_regions)
    stim_front = measure_rate_stim(front_regions)

    return {
        'visual': {'baseline': bl_vis, 'stim': stim_vis,
                   'change': (stim_vis - bl_vis) / (bl_vis + 1e-10) * 100},
        'temporal': {'baseline': bl_temp, 'stim': stim_temp,
                     'change': (stim_temp - bl_temp) / (bl_temp + 1e-10) * 100},
        'frontal': {'baseline': bl_front, 'stim': stim_front,
                    'change': (stim_front - bl_front) / (bl_front + 1e-10) * 100},
    }


def main():
    print("=" * 60)
    print("  CANONICAL MICROCIRCUIT BRAIN")
    print("  7 cell types × 80 regions = principled human cortex")
    print("=" * 60)

    # Load connectome
    sc = np.load('src/encephagen/connectome/bundled/neurolib80_weights.npy')
    tl = np.load('src/encephagen/connectome/bundled/neurolib80_tract_lengths.npy')
    labels = json.load(open('src/encephagen/connectome/bundled/neurolib80_labels.json'))
    c = Connectome(sc, labels); c.tract_lengths = tl

    # Build microcircuit brain
    t0 = time.time()
    W, tau_m, tonic, cell_map, n_total, n_regions, npr = \
        build_microcircuit_brain(c)
    print(f"  Build time: {time.time()-t0:.1f}s")

    # Test 1: Does stimulus propagate?
    print(f"\n  TEST 1: Stimulus Propagation")
    print(f"  {'Region':<12} {'Baseline':>10} {'Stimulus':>10} {'Change':>10}")
    print(f"  {'─'*44}")

    prop = test_propagation(W, tau_m, tonic, cell_map, n_total, n_regions, npr)
    for region, data in prop.items():
        marker = " ← stimulated" if region == 'visual' else ""
        print(f"  {region:<12} {data['baseline']:>10.5f} {data['stim']:>10.5f} "
              f"{data['change']:>+9.1f}%{marker}")

    propagated = (abs(prop['temporal']['change']) > 1.0 or
                  abs(prop['frontal']['change']) > 1.0)

    if propagated:
        print(f"\n  ✓ STIMULUS PROPAGATES! The microcircuit structure enables signal cascade.")
    else:
        print(f"\n  ✗ Stimulus still doesn't propagate significantly.")

    # Test 2: SC-FC correlation
    print(f"\n  TEST 2: SC-FC Correlation")
    fc_emp = np.load('src/encephagen/connectome/bundled/neurolib80_fc_empirical.npy')
    region_ts, _, _, _ = run_brain(W, tau_m, tonic, cell_map, n_total, n_regions, npr,
                                   sim_steps=30000)
    fc_sim = np.corrcoef(region_ts.T)
    idx = np.triu_indices(n_regions, k=1)
    valid = ~np.isnan(fc_sim[idx]) & ~np.isnan(fc_emp[idx])
    if valid.sum() > 10:
        r_fcfc, _ = stats.pearsonr(fc_emp[idx][valid], fc_sim[idx][valid])
    else:
        r_fcfc = 0.0
    print(f"  FC-FC(empirical): r = {r_fcfc:.4f} (benchmark: >0.3)")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()

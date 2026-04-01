"""Experiment 3: Deep analysis of emergent functional hierarchy.

1. Fix stimulus response detection (baseline deviation method)
2. Map the full hierarchy (rank all regions by dynamics)
3. Test each prediction against null models
4. Characterize the phase transition across coupling strengths
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import signal, stats

from encephagen.connectome import Connectome
from encephagen.dynamics.brain_sim import BrainSimulator, StimulusEvent
from encephagen.dynamics.wilson_cowan import WilsonCowanParams
from encephagen.analysis.functional_roles import (
    compute_regional_profiles,
    _classify_tvb76_regions,
)


def _oscillatory_params():
    return WilsonCowanParams(
        w_ee=16.0, w_ei=12.0, w_ie=15.0, w_ii=3.0,
        theta_e=2.0, theta_i=3.7, a_e=1.5, a_i=1.0,
        noise_sigma=0.01,
    )


def _randomize_connectome(connectome, seed):
    rng = np.random.default_rng(seed)
    w = connectome.weights.copy()
    n = w.shape[0]
    rows, cols = np.where(w > 0)
    edges = list(zip(rows.tolist(), cols.tolist()))
    weights_list = [float(w[r, c]) for r, c in edges]
    n_edges = len(edges)
    for _ in range(10 * n_edges):
        i1, i2 = rng.choice(n_edges, size=2, replace=False)
        a, b = edges[i1]
        c, d = edges[i2]
        if a == d or c == b or w[a, d] > 0 or w[c, b] > 0:
            continue
        w[a, b], w[c, d] = 0, 0
        w[a, d], w[c, b] = weights_list[i1], weights_list[i2]
        edges[i1], edges[i2] = (a, d), (c, b)
    return Connectome.from_numpy(w, list(connectome.labels))


def part1_stimulus_response(connectome, groups, params):
    """Fix stimulus detection: use baseline deviation instead of absolute threshold."""
    print("=" * 70)
    print("PART 1: Fixed stimulus response detection")
    print("=" * 70)

    G = 0.01  # Use weaker coupling where regions aren't saturated
    sim = BrainSimulator(connectome, global_coupling=G, params=params)

    sensory_idx = groups.get("sensory", [])
    stim = StimulusEvent(
        region_indices=sensory_idx, onset=3000.0, duration=100.0, amplitude=5.0,
    )

    print(f"Simulating at G={G} with stimulus at t=3000ms...", end=" ", flush=True)
    t0 = time.time()
    result = sim.simulate(duration=6000, dt=0.1, transient=1000, stimuli=[stim], seed=42)
    print(f"{time.time() - t0:.1f}s")

    # Baseline: 500ms before stimulus
    stim_step = int((stim.onset - result.time[0]) / result.dt)
    baseline_start = max(0, stim_step - int(500 / result.dt))
    baseline = result.E[baseline_start:stim_step, :]
    bl_mean = np.mean(baseline, axis=0)
    bl_std = np.std(baseline, axis=0) + 1e-8

    # Post-stimulus: compute z-scored response for each region
    window = int(1000 / result.dt)  # 1 second after stimulus
    post = result.E[stim_step:stim_step + window, :]

    # Response magnitude: max z-score in post-stimulus window
    z_response = (post - bl_mean) / bl_std
    max_z = np.max(np.abs(z_response), axis=0)

    # Response latency: time to first z > 3
    latencies = np.full(connectome.num_regions, np.inf)
    for i in range(connectome.num_regions):
        above = np.where(np.abs(z_response[:, i]) > 3)[0]
        if len(above) > 0:
            latencies[i] = above[0] * result.dt

    # Compare by group
    print(f"\n  {'Group':<14} {'Mean |z|':>9} {'Max |z|':>9} {'Latency ms':>11} {'Responded':>10}")
    print("  " + "─" * 55)

    group_latencies = {}
    for gname, indices in sorted(groups.items()):
        if not indices:
            continue
        g_z = max_z[indices]
        g_lat = latencies[indices]
        finite_lat = g_lat[np.isfinite(g_lat)]
        responded = len(finite_lat)

        mean_z = float(np.mean(g_z))
        max_z_g = float(np.max(g_z))
        mean_lat = float(np.mean(finite_lat)) if responded > 0 else float("inf")

        print(f"  {gname:<14} {mean_z:>9.2f} {max_z_g:>9.2f} "
              f"{mean_lat:>11.1f} {responded:>5}/{len(indices)}")

        if responded > 0:
            group_latencies[gname] = list(finite_lat)

    # Test: sensory faster than others?
    if "sensory" in group_latencies:
        sens_lat = group_latencies["sensory"]
        other_lat = [l for g, lats in group_latencies.items()
                     if g != "sensory" for l in lats]
        if sens_lat and other_lat:
            stat, p = stats.mannwhitneyu(sens_lat, other_lat, alternative="less")
            print(f"\n  Sensory faster than others? p={p:.6f}")
            print(f"  Sensory mean latency: {np.mean(sens_lat):.1f} ms")
            print(f"  Other mean latency: {np.mean(other_lat):.1f} ms")
            if p < 0.05:
                print("  ✓ SUPPORTED: Sensory regions respond first")
            else:
                print("  ✗ Not significant")

    return max_z, latencies


def part2_hierarchy_map(connectome, groups, params):
    """Rank all 96 regions by dynamics — map the emergent hierarchy."""
    print(f"\n{'=' * 70}")
    print("PART 2: Full regional hierarchy at G=0.03")
    print("=" * 70)

    sim = BrainSimulator(connectome, global_coupling=0.03, params=params)
    result = sim.simulate(duration=8000, dt=0.1, transient=2000, seed=42)
    profiles = compute_regional_profiles(result)

    # Build full table sorted by variance
    rows = []
    for label, p in profiles.items():
        idx = p["index"]
        group = "other"
        for g, indices in groups.items():
            if idx in indices:
                group = g
                break
        rows.append({
            "label": label, "group": group,
            "variance": p["variance"],
            "tau": p["autocorrelation_tau"],
            "freq": p["peak_frequency"],
            "mean": p["mean_activity"],
        })

    rows.sort(key=lambda r: -r["variance"])

    print(f"\n  {'Rank':<5} {'Region':<14} {'Group':<14} {'Variance':>10} {'Tau ms':>8} {'Hz':>7} {'Mean':>7}")
    print("  " + "─" * 67)
    for i, r in enumerate(rows):
        marker = ""
        if r["variance"] > 0.1:
            marker = " ◆ oscillating"
        elif r["variance"] > 0.01:
            marker = " ◇ moderate"
        elif r["variance"] > 0.001:
            marker = " · weak"
        print(f"  {i+1:<5} {r['label']:<14} {r['group']:<14} "
              f"{r['variance']:>10.6f} {r['tau']:>8.1f} {r['freq']:>7.1f} "
              f"{r['mean']:>7.4f}{marker}")
        if i == 29:  # Show top 30
            remaining = len(rows) - 30
            silent = sum(1 for r in rows[30:] if r["variance"] < 0.001)
            print(f"  ... ({remaining} more regions, {silent} nearly silent)")
            break

    # Group summary
    print(f"\n  Group summary:")
    print(f"  {'Group':<14} {'Mean Var':>10} {'Mean Tau':>9} {'Mean Hz':>8} {'Status':<20}")
    print("  " + "─" * 63)
    for gname in ["basal_ganglia", "prefrontal", "hippocampus", "thalamus", "sensory", "motor", "other"]:
        indices = groups.get(gname, [])
        if not indices:
            continue
        gvars = [profiles[connectome.labels[i]]["variance"] for i in indices]
        gtaus = [profiles[connectome.labels[i]]["autocorrelation_tau"] for i in indices]
        gfreqs = [profiles[connectome.labels[i]]["peak_frequency"] for i in indices]
        mv = np.mean(gvars)
        status = "oscillating" if mv > 0.1 else "moderate" if mv > 0.01 else "weak" if mv > 0.001 else "silent"
        print(f"  {gname:<14} {mv:>10.6f} {np.mean(gtaus):>9.1f} {np.mean(gfreqs):>8.1f} {status:<20}")

    return profiles


def part3_null_model_per_prediction(connectome, groups, params):
    """Test: does random wiring also produce the PFC-is-slower effect?"""
    print(f"\n{'=' * 70}")
    print("PART 3: Does random wiring produce the same hierarchy?")
    print("=" * 70)

    G = 0.03
    n_random = 15

    # Real connectome — PFC tau vs sensory tau
    real_pfc_taus = []
    real_sens_taus = []
    real_bg_vars = []
    real_other_vars = []

    for seed in range(5):
        sim = BrainSimulator(connectome, global_coupling=G, params=params)
        r = sim.simulate(duration=5000, dt=0.1, transient=1000, seed=seed)
        profiles = compute_regional_profiles(r)
        pfc_idx = groups.get("prefrontal", [])
        sens_idx = groups.get("sensory", [])
        bg_idx = groups.get("basal_ganglia", [])

        for i in pfc_idx:
            real_pfc_taus.append(profiles[connectome.labels[i]]["autocorrelation_tau"])
        for i in sens_idx:
            real_sens_taus.append(profiles[connectome.labels[i]]["autocorrelation_tau"])
        for i in bg_idx:
            real_bg_vars.append(profiles[connectome.labels[i]]["variance"])
        for g, indices in groups.items():
            if g != "basal_ganglia":
                for i in indices:
                    real_other_vars.append(profiles[connectome.labels[i]]["variance"])

    # Random connectomes
    rand_pfc_taus = []
    rand_sens_taus = []
    rand_bg_vars = []
    rand_other_vars = []

    print(f"  Running {n_random} randomized connectomes...", end=" ", flush=True)
    t0 = time.time()
    for i in range(n_random):
        rc = _randomize_connectome(connectome, seed=200 + i)
        sim = BrainSimulator(rc, global_coupling=G, params=params)
        r = sim.simulate(duration=5000, dt=0.1, transient=1000, seed=42)
        profiles = compute_regional_profiles(r)
        pfc_idx = groups.get("prefrontal", [])
        sens_idx = groups.get("sensory", [])
        bg_idx = groups.get("basal_ganglia", [])

        for j in pfc_idx:
            rand_pfc_taus.append(profiles[connectome.labels[j]]["autocorrelation_tau"])
        for j in sens_idx:
            rand_sens_taus.append(profiles[connectome.labels[j]]["autocorrelation_tau"])
        for j in bg_idx:
            rand_bg_vars.append(profiles[connectome.labels[j]]["variance"])
        for g, indices in groups.items():
            if g != "basal_ganglia":
                for j in indices:
                    rand_other_vars.append(profiles[connectome.labels[j]]["variance"])
    print(f"{time.time() - t0:.1f}s")

    # Test 1: PFC tau ratio
    real_ratio = np.mean(real_pfc_taus) / (np.mean(real_sens_taus) + 1e-12)
    rand_ratio = np.mean(rand_pfc_taus) / (np.mean(rand_sens_taus) + 1e-12)
    print(f"\n  PFC/Sensory tau ratio:")
    print(f"    Real connectome:  {real_ratio:.2f}x (PFC={np.mean(real_pfc_taus):.1f}ms, Sens={np.mean(real_sens_taus):.1f}ms)")
    print(f"    Random wiring:    {rand_ratio:.2f}x (PFC={np.mean(rand_pfc_taus):.1f}ms, Sens={np.mean(rand_sens_taus):.1f}ms)")

    # Test 2: BG variance vs others
    real_bg_ratio = np.mean(real_bg_vars) / (np.mean(real_other_vars) + 1e-12)
    rand_bg_ratio = np.mean(rand_bg_vars) / (np.mean(rand_other_vars) + 1e-12)
    print(f"\n  Basal ganglia / other variance ratio:")
    print(f"    Real connectome:  {real_bg_ratio:.2f}x (BG={np.mean(real_bg_vars):.4f}, Other={np.mean(real_other_vars):.4f})")
    print(f"    Random wiring:    {rand_bg_ratio:.2f}x (BG={np.mean(rand_bg_vars):.4f}, Other={np.mean(rand_other_vars):.4f})")

    return {
        "pfc_tau_ratio_real": real_ratio,
        "pfc_tau_ratio_random": rand_ratio,
        "bg_var_ratio_real": real_bg_ratio,
        "bg_var_ratio_random": rand_bg_ratio,
    }


def part4_phase_transition(connectome, groups, params):
    """Characterize the phase transition: which regions silence first?"""
    print(f"\n{'=' * 70}")
    print("PART 4: Phase transition — which regions silence first?")
    print("=" * 70)

    couplings = [0.001, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]

    group_names = ["basal_ganglia", "prefrontal", "thalamus", "hippocampus", "sensory", "motor", "other"]

    print(f"\n  {'G':>7}", end="")
    for g in group_names:
        print(f"  {g[:10]:>10}", end="")
    print()
    print("  " + "─" * (7 + 12 * len(group_names)))

    silencing_order = {}

    for G in couplings:
        sim = BrainSimulator(connectome, global_coupling=G, params=params)
        r = sim.simulate(duration=3000, dt=0.1, transient=500, seed=42)

        print(f"  {G:>7.3f}", end="")
        for gname in group_names:
            indices = groups.get(gname, [])
            if not indices:
                print(f"  {'—':>10}", end="")
                continue
            mean_var = float(np.mean([np.var(r.E[:, i]) for i in indices]))
            if mean_var > 0.1:
                status = f"{mean_var:.4f}"
            elif mean_var > 0.01:
                status = f"{mean_var:.4f}*"
            elif mean_var > 0.001:
                status = f"{mean_var:.4f}."
            else:
                status = "silent"

            # Record first silencing
            if gname not in silencing_order and mean_var < 0.001:
                silencing_order[gname] = G

            print(f"  {status:>10}", end="")
        print()

    print(f"\n  Silencing order (first G where variance < 0.001):")
    for gname, g_val in sorted(silencing_order.items(), key=lambda x: x[1]):
        print(f"    {gname:<14} silenced at G={g_val:.3f}")

    not_silenced = [g for g in group_names if g not in silencing_order and groups.get(g)]
    for gname in not_silenced:
        print(f"    {gname:<14} NEVER silenced (resilient)")

    return silencing_order


def run_experiment():
    connectome = Connectome.from_bundled("tvb96")
    groups = _classify_tvb76_regions(connectome.labels)
    params = _oscillatory_params()

    max_z, latencies = part1_stimulus_response(connectome, groups, params)
    profiles = part2_hierarchy_map(connectome, groups, params)
    null_results = part3_null_model_per_prediction(connectome, groups, params)
    silencing = part4_phase_transition(connectome, groups, params)

    # Save
    results_dir = Path("results/exp03_deep_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "null_comparison": null_results,
        "silencing_order": silencing,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

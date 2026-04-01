"""Experiment 2: Comprehensive test of emergent functional roles.

Uses TVB96 (with thalamus and basal ganglia), sweeps coupling strengths,
and compares real connectome against randomized wiring.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from encephagen.connectome import Connectome
from encephagen.dynamics.brain_sim import BrainSimulator, StimulusEvent
from encephagen.dynamics.wilson_cowan import WilsonCowanParams
from encephagen.analysis.functional_roles import (
    run_all_predictions,
    compute_regional_profiles,
    _classify_tvb76_regions,
    test_thalamic_gating,
    test_prefrontal_sustained,
    test_frequency_differentiation,
    test_regional_differentiation_overall,
)


def _randomize_connectome(connectome: Connectome, seed: int) -> Connectome:
    """Degree-preserving rewiring of the connectome."""
    rng = np.random.default_rng(seed)
    w = connectome.weights.copy()
    n = w.shape[0]
    rows, cols = np.where(w > 0)
    edges = list(zip(rows.tolist(), cols.tolist()))
    weights_list = [float(w[r, c]) for r, c in edges]
    n_edges = len(edges)

    for _ in range(10 * n_edges):
        idx1, idx2 = rng.choice(n_edges, size=2, replace=False)
        a, b = edges[idx1]
        c, d = edges[idx2]
        if a == d or c == b or w[a, d] > 0 or w[c, b] > 0:
            continue
        w[a, b] = 0
        w[c, d] = 0
        w[a, d] = weights_list[idx1]
        w[c, b] = weights_list[idx2]
        edges[idx1] = (a, d)
        edges[idx2] = (c, b)

    return Connectome.from_numpy(w, list(connectome.labels))


def run_experiment():
    print("=" * 70)
    print("ENCEPHAGEN EXPERIMENT 2: COMPREHENSIVE")
    print("TVB96 (with thalamus + basal ganglia) — real vs random wiring")
    print("=" * 70)

    connectome = Connectome.from_bundled("tvb96")
    print(f"\nConnectome: {connectome}")

    groups = _classify_tvb76_regions(connectome.labels)
    print("\nRegion groups:")
    for group, indices in groups.items():
        labels = [connectome.labels[i] for i in indices[:5]]
        suffix = f"... (+{len(indices)-5} more)" if len(indices) > 5 else ""
        print(f"  {group:>14}: {len(indices):>2} regions — {labels}{suffix}")

    params = WilsonCowanParams(
        w_ee=16.0, w_ei=12.0, w_ie=15.0, w_ii=3.0,
        theta_e=2.0, theta_i=3.7, a_e=1.5, a_i=1.0,
        noise_sigma=0.01,
    )

    # --- Part 1: Coupling sweep on real connectome ---
    print("\n" + "=" * 70)
    print("PART 1: Coupling sweep — where does differentiation emerge?")
    print("=" * 70)

    coupling_values = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]

    for G in coupling_values:
        sim = BrainSimulator(connectome, global_coupling=G, params=params)
        print(f"\n  G={G:.3f}...", end=" ", flush=True)
        t0 = time.time()
        result = sim.simulate(duration=5000, dt=0.1, transient=1000, seed=42)
        elapsed = time.time() - t0

        # Quick group variance summary
        group_vars = {}
        for gname, indices in groups.items():
            if indices:
                vars_list = [float(np.var(result.E[:, i])) for i in indices]
                group_vars[gname] = np.mean(vars_list)

        var_vals = list(group_vars.values())
        var_cv = float(np.std(var_vals) / (np.mean(var_vals) + 1e-12))

        parts = [f"{g}={v:.4f}" for g, v in sorted(group_vars.items()) if v > 0.0001]
        print(f"{elapsed:.1f}s  var_cv={var_cv:.3f}  {' | '.join(parts)}")

    # --- Part 2: Best coupling — full prediction tests ---
    best_G = 0.03  # From sweep above
    print(f"\n{'=' * 70}")
    print(f"PART 2: Full prediction tests at G={best_G}")
    print(f"{'=' * 70}")

    sim = BrainSimulator(connectome, global_coupling=best_G, params=params)

    # Spontaneous
    print("\nRunning spontaneous simulation (10s)...", end=" ", flush=True)
    t0 = time.time()
    result_spont = sim.simulate(duration=12000, dt=0.1, transient=2000, seed=42)
    print(f"{time.time() - t0:.1f}s")

    # Stimulus (inject into sensory regions)
    sensory_idx = groups.get("sensory", list(range(4)))
    stimulus = StimulusEvent(
        region_indices=sensory_idx, onset=5000.0, duration=200.0, amplitude=3.0,
    )
    print("Running stimulus simulation (10s)...", end=" ", flush=True)
    t0 = time.time()
    result_stim = sim.simulate(
        duration=12000, dt=0.1, transient=2000, stimuli=[stimulus], seed=42,
    )
    print(f"{time.time() - t0:.1f}s")

    # Top 20 regions by variance
    print("\nTop 20 regions by variance (spontaneous):")
    profiles = compute_regional_profiles(result_spont)
    print(f"  {'Region':<14} {'Group':<14} {'Mean':>7} {'Var':>10} {'Tau ms':>8} {'Hz':>7}")
    print("  " + "─" * 62)
    sorted_profiles = sorted(profiles.items(), key=lambda x: -x[1]["variance"])
    for label, p in sorted_profiles[:20]:
        idx = p["index"]
        group = "?"
        for g, indices in groups.items():
            if idx in indices:
                group = g
                break
        print(f"  {label:<14} {group:<14} {p['mean_activity']:>7.4f} "
              f"{p['variance']:>10.6f} {p['autocorrelation_tau']:>8.1f} "
              f"{p['peak_frequency']:>7.1f}")

    # Predictions
    print(f"\n{'─' * 70}")
    print("PREDICTION TESTS (spontaneous)")
    print(f"{'─' * 70}")
    spont_preds = run_all_predictions(result_spont)
    for key, pred in spont_preds.items():
        if key == "region_groups":
            continue
        if not pred.get("testable", False):
            print(f"\n  {key}: NOT TESTABLE — {pred.get('reason', '')}")
            continue
        supported = "SUPPORTED" if pred.get("supported") else "NOT SUPPORTED"
        print(f"\n  {key}: {supported}")
        print(f"    {pred.get('prediction', '')}")
        for k, v in pred.items():
            if k in ("testable", "prediction", "supported", "group_frequencies"):
                continue
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
        if "group_frequencies" in pred:
            for g, s in pred["group_frequencies"].items():
                print(f"    {g}: mean={s['mean_freq']:.1f}Hz std={s['std_freq']:.1f} n={s['n']}")

    print(f"\n{'─' * 70}")
    print("PREDICTION TESTS (stimulus response)")
    print(f"{'─' * 70}")
    stim_preds = run_all_predictions(result_stim)
    p3 = stim_preds.get("P3_sensory_first", {})
    if p3.get("testable"):
        supported = "SUPPORTED" if p3.get("supported") else "NOT SUPPORTED"
        print(f"\n  P3_sensory_first: {supported}")
        print(f"    Sensory latency: {p3.get('sensory_mean_latency_ms', 'N/A')}")
        print(f"    Other latency: {p3.get('other_mean_latency_ms', 'N/A')}")
        print(f"    p-value: {p3.get('p_value', 'N/A')}")
    else:
        print(f"\n  P3_sensory_first: NOT TESTABLE — {p3.get('reason', '')}")

    # --- Part 3: Real vs Random comparison ---
    print(f"\n{'=' * 70}")
    print(f"PART 3: Real connectome vs randomized wiring (10 instances)")
    print(f"{'=' * 70}")

    n_random = 10
    real_var_cv = []
    random_var_cvs = []

    # Real (multiple seeds)
    for seed in range(5):
        sim = BrainSimulator(connectome, global_coupling=best_G, params=params)
        r = sim.simulate(duration=5000, dt=0.1, transient=1000, seed=seed)
        gvars = []
        for indices in groups.values():
            if indices:
                gvars.append(np.mean([np.var(r.E[:, i]) for i in indices]))
        real_var_cv.append(float(np.std(gvars) / (np.mean(gvars) + 1e-12)))

    print(f"\n  Real connectome var_cv: {np.mean(real_var_cv):.4f} ± {np.std(real_var_cv):.4f}")

    # Random
    print(f"  Running {n_random} randomized connectomes...", end=" ", flush=True)
    t0 = time.time()
    for i in range(n_random):
        rand_conn = _randomize_connectome(connectome, seed=100 + i)
        sim = BrainSimulator(rand_conn, global_coupling=best_G, params=params)
        r = sim.simulate(duration=5000, dt=0.1, transient=1000, seed=42)
        gvars = []
        for indices in groups.values():
            if indices:
                gvars.append(np.mean([np.var(r.E[:, i]) for i in indices]))
        random_var_cvs.append(float(np.std(gvars) / (np.mean(gvars) + 1e-12)))
    elapsed = time.time() - t0
    print(f"{elapsed:.1f}s")

    print(f"  Random connectome var_cv: {np.mean(random_var_cvs):.4f} ± {np.std(random_var_cvs):.4f}")

    # Statistical test
    stat, p = stats.mannwhitneyu(real_var_cv, random_var_cvs, alternative="greater")
    print(f"\n  Real > Random? p={p:.6f}")
    if p < 0.05:
        print("  ✓ SIGNIFICANT: Real connectome produces MORE differentiation than random")
    else:
        print("  ✗ Not significant at p<0.05")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")

    all_preds = {**spont_preds, "P3_stim": p3}
    supported = sum(1 for k, v in all_preds.items()
                    if isinstance(v, dict) and v.get("supported", False))
    testable = sum(1 for k, v in all_preds.items()
                   if isinstance(v, dict) and v.get("testable", False))
    print(f"\n  Predictions supported: {supported}/{testable}")
    print(f"  Real vs random differentiation p-value: {p:.6f}")

    # Save
    results_dir = Path("results/exp02_comprehensive")
    results_dir.mkdir(parents=True, exist_ok=True)

    def ser(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    summary = {
        "real_var_cv": real_var_cv,
        "random_var_cvs": random_var_cvs,
        "real_vs_random_p": float(p),
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=ser)
    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

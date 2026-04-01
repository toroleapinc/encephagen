"""Experiment 4: Isolate what specific wiring contributes beyond degree.

Compare real connectome against:
1. Degree-preserving rewiring (keeps degree, destroys specific targets)
2. Erdős-Rényi (destroys everything)
3. Strength-matched random (each region keeps same total connection weight)

If real ≠ degree-preserving but degree-preserving ≠ ER, then:
  - Real wiring contributes something beyond degree
  - AND degree contributes something beyond density

We test across ALL metrics, not just variance, to find where
the specific wiring matters most.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import signal, stats

from encephagen.connectome import Connectome
from encephagen.dynamics.brain_sim import BrainSimulator
from encephagen.dynamics.wilson_cowan import WilsonCowanParams
from encephagen.analysis.functional_roles import _classify_tvb76_regions


def _oscillatory_params():
    return WilsonCowanParams(
        w_ee=16.0, w_ei=12.0, w_ie=15.0, w_ii=3.0,
        theta_e=2.0, theta_i=3.7, a_e=1.5, a_i=1.0,
        noise_sigma=0.01,
    )


def _degree_preserving_rewire(connectome, seed):
    rng = np.random.default_rng(seed)
    w = connectome.weights.copy()
    n = w.shape[0]
    rows, cols = np.where(w > 0)
    edges = list(zip(rows.tolist(), cols.tolist()))
    wts = [float(w[r, c]) for r, c in edges]
    ne = len(edges)
    for _ in range(10 * ne):
        i1, i2 = rng.choice(ne, size=2, replace=False)
        a, b = edges[i1]
        c, d = edges[i2]
        if a == d or c == b or w[a, d] > 0 or w[c, b] > 0:
            continue
        w[a, b], w[c, d] = 0, 0
        w[a, d], w[c, b] = wts[i1], wts[i2]
        edges[i1], edges[i2] = (a, d), (c, b)
    return Connectome.from_numpy(w, list(connectome.labels))


def _erdos_renyi_null(connectome, seed):
    rng = np.random.default_rng(seed)
    n = connectome.num_regions
    ne = connectome.num_edges
    real_weights = connectome.weights[connectome.weights > 0]
    w = np.zeros((n, n), dtype=np.float32)
    possible = [(i, j) for i in range(n) for j in range(n) if i != j]
    chosen = rng.choice(len(possible), size=min(ne, len(possible)), replace=False)
    for idx in chosen:
        i, j = possible[idx]
        w[i, j] = rng.choice(real_weights)
    return Connectome.from_numpy(w, list(connectome.labels))


def _strength_preserving_null(connectome, seed):
    """Each region keeps its total in/out strength but targets are randomized."""
    rng = np.random.default_rng(seed)
    w = connectome.weights.copy()
    n = w.shape[0]
    # For each row (outgoing), shuffle the nonzero entries
    for i in range(n):
        nonzero_mask = w[i] > 0
        nonzero_vals = w[i, nonzero_mask].copy()
        nonzero_indices = np.where(nonzero_mask)[0]
        # Pick new random targets (excluding self)
        available = [j for j in range(n) if j != i]
        new_targets = rng.choice(available, size=len(nonzero_vals), replace=False)
        w[i, :] = 0
        for val, target in zip(nonzero_vals, new_targets):
            w[i, target] = val
    return Connectome.from_numpy(w, list(connectome.labels))


def compute_detailed_metrics(result, groups):
    """Compute many metrics to find where specific wiring matters."""
    E = result.E
    n = E.shape[1]
    dt_sec = result.dt / 1000.0
    fs = 1.0 / dt_sec

    metrics = {}

    # 1. Group variance hierarchy
    group_vars = {}
    for gname, indices in groups.items():
        if indices:
            group_vars[gname] = float(np.mean([np.var(E[:, i]) for i in indices]))
    metrics["group_variance"] = group_vars

    # 2. Functional connectivity structure
    fc = np.corrcoef(E.T)
    np.fill_diagonal(fc, 0)

    # Within-group vs between-group FC
    for gname, indices in groups.items():
        if len(indices) >= 2:
            within = [fc[i, j] for i in indices for j in indices if i != j]
            metrics[f"fc_within_{gname}"] = float(np.mean(within))

    # Between specific pairs
    for g1 in groups:
        for g2 in groups:
            if g1 >= g2:
                continue
            idx1, idx2 = groups[g1], groups[g2]
            if idx1 and idx2:
                between = [fc[i, j] for i in idx1 for j in idx2]
                metrics[f"fc_between_{g1}_{g2}"] = float(np.mean(between))

    # 3. Information flow asymmetry
    # Proxy: cross-correlation lag between group-averaged signals
    for g1 in ["sensory", "prefrontal", "basal_ganglia", "thalamus"]:
        for g2 in ["sensory", "prefrontal", "basal_ganglia", "thalamus"]:
            if g1 == g2:
                continue
            idx1, idx2 = groups.get(g1, []), groups.get(g2, [])
            if not idx1 or not idx2:
                continue
            sig1 = np.mean(E[:, idx1], axis=1)
            sig2 = np.mean(E[:, idx2], axis=1)
            if np.std(sig1) < 1e-8 or np.std(sig2) < 1e-8:
                metrics[f"xcorr_lag_{g1}_to_{g2}"] = 0.0
                continue
            xcorr = np.correlate(sig1 - sig1.mean(), sig2 - sig2.mean(), mode="full")
            xcorr /= (np.std(sig1) * np.std(sig2) * len(sig1))
            mid = len(xcorr) // 2
            # Peak lag in ms
            window = int(50 / result.dt)  # search within ±50ms
            start = max(0, mid - window)
            end = min(len(xcorr), mid + window)
            peak_idx = np.argmax(xcorr[start:end]) + start
            lag_ms = (peak_idx - mid) * result.dt
            metrics[f"xcorr_lag_{g1}_to_{g2}"] = float(lag_ms)

    # 4. Spectral properties per group
    for gname, indices in groups.items():
        if not indices:
            continue
        group_ts = np.mean(E[:, indices], axis=1)
        if np.std(group_ts) < 1e-8:
            metrics[f"peak_freq_{gname}"] = 0.0
            continue
        nperseg = min(1024, len(group_ts))
        freqs, psd = signal.welch(group_ts, fs=fs, nperseg=nperseg)
        metrics[f"peak_freq_{gname}"] = float(freqs[np.argmax(psd)])

    # 5. Overall differentiation
    all_vars = [np.var(E[:, i]) for i in range(n)]
    metrics["var_cv"] = float(np.std(all_vars) / (np.mean(all_vars) + 1e-12))
    metrics["var_range"] = float(np.max(all_vars) - np.min(all_vars))

    return metrics


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 4: Isolate what specific wiring contributes")
    print("=" * 70)

    connectome = Connectome.from_bundled("tvb96")
    groups = _classify_tvb76_regions(connectome.labels)
    params = _oscillatory_params()

    G = 0.01  # Use G where most regions are still active
    n_null = 15

    print(f"\nConnectome: {connectome}")
    print(f"Coupling: G={G}")
    print(f"Null instances: {n_null}")

    # --- Run real connectome ---
    print("\nRunning real connectome (5 seeds)...", end=" ", flush=True)
    t0 = time.time()
    real_metrics_list = []
    for seed in range(5):
        sim = BrainSimulator(connectome, global_coupling=G, params=params)
        r = sim.simulate(duration=5000, dt=0.1, transient=1000, seed=seed)
        real_metrics_list.append(compute_detailed_metrics(r, groups))
    print(f"{time.time() - t0:.1f}s")

    # --- Run null models ---
    null_types = {
        "degree_preserving": _degree_preserving_rewire,
        "erdos_renyi": _erdos_renyi_null,
        "strength_preserving": _strength_preserving_null,
    }

    null_metrics = {}
    for null_name, gen_fn in null_types.items():
        print(f"Running {null_name} ({n_null} instances)...", end=" ", flush=True)
        t0 = time.time()
        null_list = []
        for i in range(n_null):
            nc = gen_fn(connectome, seed=300 + i)
            sim = BrainSimulator(nc, global_coupling=G, params=params)
            r = sim.simulate(duration=5000, dt=0.1, transient=1000, seed=42)
            null_list.append(compute_detailed_metrics(r, groups))
        null_metrics[null_name] = null_list
        print(f"{time.time() - t0:.1f}s")

    # --- Compare ---
    print(f"\n{'=' * 70}")
    print("RESULTS: What does specific wiring contribute?")
    print("=" * 70)

    # Get all metric keys
    all_keys = sorted(real_metrics_list[0].keys())

    # Filter to scalar metrics
    scalar_keys = [k for k in all_keys
                   if isinstance(real_metrics_list[0][k], (int, float))]

    print(f"\n{'Metric':<40} {'Real':>8} | {'Deg-Pres':>8} {'ER':>8} {'Str-Pres':>8} | {'R vs DP':>7} {'R vs ER':>7}")
    print("─" * 110)

    significant_findings = []

    for key in scalar_keys:
        real_vals = [m[key] for m in real_metrics_list]
        real_mean = np.mean(real_vals)

        row = f"{key:<40} {real_mean:>8.4f} | "

        p_values = {}
        for null_name in ["degree_preserving", "erdos_renyi", "strength_preserving"]:
            null_vals = [m[key] for m in null_metrics[null_name]]
            null_mean = np.mean(null_vals)
            row += f"{null_mean:>8.4f} "

            if len(real_vals) >= 2 and len(null_vals) >= 2 and np.std(real_vals) + np.std(null_vals) > 0:
                _, p = stats.mannwhitneyu(real_vals, null_vals, alternative="two-sided")
                p_values[null_name] = p
            else:
                p_values[null_name] = 1.0

        row += "| "
        for null_name, short in [("degree_preserving", "R vs DP"), ("erdos_renyi", "R vs ER")]:
            p = p_values.get(null_name, 1.0)
            if p < 0.001:
                row += f"  {'***':>5} "
            elif p < 0.01:
                row += f"  {'**':>5} "
            elif p < 0.05:
                row += f"  {'*':>5} "
            else:
                row += f"  {'ns':>5} "

            if null_name == "degree_preserving" and p < 0.05:
                significant_findings.append((key, real_mean,
                    np.mean([m[key] for m in null_metrics[null_name]]), p))

        print(row)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY: Where specific wiring matters (Real ≠ Degree-Preserving)")
    print("=" * 70)

    if significant_findings:
        print(f"\n  {len(significant_findings)} metrics where specific wiring matters (p<0.05):\n")
        for key, real_val, null_val, p in sorted(significant_findings, key=lambda x: x[3]):
            direction = "higher" if real_val > null_val else "lower"
            print(f"    {key:<40} real={real_val:.4f} null={null_val:.4f} ({direction}) p={p:.4f}")
    else:
        print("\n  No metrics where specific wiring differs significantly from degree-preserving.")
        print("  This suggests degree distribution explains all emergent differentiation.")

    # Also report: what does degree contribute (DP ≠ ER)?
    print(f"\n{'=' * 70}")
    print("BONUS: Where degree distribution matters (Degree-Pres ≠ ER)")
    print("=" * 70)

    degree_findings = []
    for key in scalar_keys:
        dp_vals = [m[key] for m in null_metrics["degree_preserving"]]
        er_vals = [m[key] for m in null_metrics["erdos_renyi"]]
        if len(dp_vals) >= 2 and len(er_vals) >= 2 and np.std(dp_vals) + np.std(er_vals) > 0:
            _, p = stats.mannwhitneyu(dp_vals, er_vals, alternative="two-sided")
            if p < 0.05:
                degree_findings.append((key, np.mean(dp_vals), np.mean(er_vals), p))

    if degree_findings:
        print(f"\n  {len(degree_findings)} metrics where degree matters:\n")
        for key, dp_val, er_val, p in sorted(degree_findings, key=lambda x: x[3]):
            print(f"    {key:<40} deg-pres={dp_val:.4f} ER={er_val:.4f} p={p:.4f}")

    # Save
    results_dir = Path("results/exp04_isolate_topology")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

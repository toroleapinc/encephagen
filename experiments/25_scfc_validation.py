"""Experiment 25: SC-FC Validation.

Every Wilson-Cowan / spiking connectome paper validates simulated
functional connectivity (FC) against structural connectivity (SC)
and/or empirical fMRI FC. We haven't done this.

This experiment:
  1. Simulates the brain for 60 seconds (long enough for stable FC)
  2. Computes simulated FC (pairwise Pearson correlation of region activity)
  3. Compares simulated FC vs SC (upper triangle correlation)
  4. Compares connectome-simulated FC vs random-simulated FC
  5. Tests whether simulated FC captures known properties:
     - Homotopic connections (left-right homologs strongly correlated)
     - Distance dependence (nearby regions more correlated)
     - Default mode network-like resting state patterns

Benchmark: raw SC-FC Pearson r is typically 0.3-0.5 in the TVB literature.
A good model should produce higher SC-FC correlation than a random model.
"""

import time
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU


def classify_regions(labels):
    groups = {}
    for key, patterns in [
        ('visual', ['V1', 'V2', 'VAC']),
        ('prefrontal', ['PFC', 'FEF']),
        ('hippocampus', ['HC', 'PHC']),
        ('amygdala', ['AMYG']),
        ('thalamus', ['TM']),
        ('motor', ['M1', 'PMC']),
    ]:
        groups[key] = [i for i, l in enumerate(labels)
                       if any(p in l.upper() for p in patterns)]
    return groups


def randomize_connectome(connectome, seed):
    rng = np.random.default_rng(seed)
    w = connectome.weights.copy()
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


def simulate_fc(connectome, duration_ms=30000, device="cuda"):
    """Simulate brain and compute functional connectivity.

    Returns:
        fc: [n_regions, n_regions] correlation matrix
        region_ts: [n_timepoints, n_regions] time series
    """
    groups = classify_regions(connectome.labels)
    pfc_regions = groups.get('prefrontal', [])
    npr = 200
    n_regions = connectome.num_regions
    n_total = n_regions * npr

    brain = SpikingBrainGPU(
        connectome=connectome, neurons_per_region=npr,
        global_coupling=0.15, ext_rate_factor=3.5,
        tau_nmda=150.0, nmda_ratio=0.4,
        pfc_regions=pfc_regions, device=device,
        # No ALIF for FC validation — use the standard model
        use_delays=True,
        conduction_velocity=3.5,
        use_neuron_types=True,
        use_adaptation=False,
    )

    state = brain.init_state(batch_size=1)
    # Warmup
    with torch.no_grad():
        for _ in range(5000):
            state, _ = brain.step(state)

    # Simulate and record region rates every 10ms (100 steps at dt=0.1)
    record_interval = 100  # steps
    total_steps = int(duration_ms / brain.dt)
    n_timepoints = total_steps // record_interval

    region_ts = np.zeros((n_timepoints, n_regions))
    spike_acc = torch.zeros(n_regions, device=device)
    acc_count = 0
    tp = 0

    with torch.no_grad():
        for step in range(total_steps):
            state, spikes = brain.step(state)

            # Accumulate per-region spikes
            for r in range(n_regions):
                s = r * npr
                e = (r + 1) * npr
                spike_acc[r] += spikes[0, s:e].sum()
            acc_count += 1

            if acc_count >= record_interval:
                region_ts[tp] = (spike_acc / (npr * acc_count)).cpu().numpy()
                spike_acc.zero_()
                acc_count = 0
                tp += 1
                if tp >= n_timepoints:
                    break

    del brain
    torch.cuda.empty_cache()

    # Compute FC (Pearson correlation of region time series)
    fc = np.corrcoef(region_ts[:tp].T)
    return fc, region_ts[:tp]


def upper_triangle(mat):
    """Extract upper triangle (excluding diagonal) as flat vector."""
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]


def find_homotopic_pairs(labels):
    """Find left-right homotopic region pairs."""
    pairs = []
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if i >= j:
                continue
            # Check if one is left and other is right of same region
            li_clean = li.replace(' lh', '').replace(' rh', '').replace('_left', '').replace('_right', '')
            lj_clean = lj.replace(' lh', '').replace(' rh', '').replace('_left', '').replace('_right', '')
            if li_clean == lj_clean and li != lj:
                pairs.append((i, j))
    return pairs


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 25: SC-FC Validation")
    print("Does simulated FC correlate with structural connectivity?")
    print("=" * 70)

    connectome = Connectome.from_bundled("tvb96")
    sc = connectome.weights
    labels = connectome.labels
    n_regions = connectome.num_regions

    # Log-transform SC for correlation (standard in literature)
    sc_log = np.log1p(sc)

    N_RUNS = 3
    device = "cuda"
    duration_ms = 10000  # 10 seconds (100K steps — ~7 min per run)

    results = {"connectome": [], "random": []}
    t0 = time.time()

    for condition in ["connectome", "random"]:
        print(f"\n{'='*60}")
        print(f"CONDITION: {condition.upper()} ({N_RUNS} runs, {duration_ms/1000:.0f}s each)")
        print(f"{'='*60}")

        for run in range(N_RUNS):
            if condition == "connectome":
                conn = connectome
            else:
                conn = randomize_connectome(connectome, seed=1000 + run)

            print(f"  Run {run+1}/{N_RUNS}...", end=" ", flush=True)
            fc, ts = simulate_fc(conn, duration_ms=duration_ms, device=device)

            # 1. SC-FC correlation (upper triangle)
            sc_upper = upper_triangle(sc_log)
            fc_upper = upper_triangle(fc)
            # Remove NaN from FC (can happen if region is silent)
            valid = ~np.isnan(fc_upper) & ~np.isnan(sc_upper)
            if valid.sum() > 10:
                sc_fc_r, sc_fc_p = stats.pearsonr(sc_upper[valid], fc_upper[valid])
            else:
                sc_fc_r, sc_fc_p = 0.0, 1.0

            # 2. Homotopic correlation (left-right pairs)
            pairs = find_homotopic_pairs(labels)
            if pairs:
                homo_corrs = [fc[i, j] for i, j in pairs if not np.isnan(fc[i, j])]
                mean_homo = float(np.mean(homo_corrs)) if homo_corrs else 0.0
            else:
                mean_homo = 0.0

            # 3. Distance dependence (FC vs Euclidean distance)
            if connectome.positions is not None:
                dists = squareform(pdist(connectome.positions))
                dist_upper = upper_triangle(dists)
                valid_d = valid & ~np.isnan(dist_upper)
                if valid_d.sum() > 10:
                    dist_fc_r, _ = stats.pearsonr(dist_upper[valid_d], fc_upper[valid_d])
                else:
                    dist_fc_r = 0.0
            else:
                dist_fc_r = 0.0

            # 4. Mean FC (overall coupling)
            mean_fc = float(np.nanmean(fc_upper))

            elapsed = time.time() - t0
            print(f"SC-FC r={sc_fc_r:.3f}  homo={mean_homo:.3f}  "
                  f"dist-FC r={dist_fc_r:.3f}  mean_FC={mean_fc:.3f}  ({elapsed:.0f}s)")

            results[condition].append({
                "sc_fc_r": sc_fc_r,
                "homotopic": mean_homo,
                "dist_fc_r": dist_fc_r,
                "mean_fc": mean_fc,
            })

    # ========================================
    # Statistical comparison
    # ========================================
    print(f"\n{'='*70}")
    print("RESULTS: SC-FC Validation")
    print(f"{'='*70}")

    metrics = [
        ("SC-FC correlation (r)", "sc_fc_r", "higher = FC follows SC (benchmark: 0.3-0.5)"),
        ("Homotopic correlation", "homotopic", "higher = left-right pairs correlated"),
        ("Distance-FC correlation", "dist_fc_r", "negative = nearby regions more correlated"),
        ("Mean FC", "mean_fc", "overall functional coupling strength"),
    ]

    print(f"\n  {'Metric':<35} {'Connectome':>12} {'Random':>12} {'p-value':>10} {'Benchmark'}")
    print(f"  {'─'*90}")

    for label, key, benchmark in metrics:
        c_vals = np.array([r[key] for r in results["connectome"]])
        r_vals = np.array([r[key] for r in results["random"]])
        c_mean = c_vals.mean()
        r_mean = r_vals.mean()
        stat, p = stats.mannwhitneyu(c_vals, r_vals, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label:<35} {c_mean:>12.4f} {r_mean:>12.4f} {p:>9.4f}{sig:>1}  {benchmark}")

    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    c_scfc = np.mean([r['sc_fc_r'] for r in results["connectome"]])
    r_scfc = np.mean([r['sc_fc_r'] for r in results["random"]])
    c_homo = np.mean([r['homotopic'] for r in results["connectome"]])

    if c_scfc > 0.3:
        print(f"\n  SC-FC correlation: {c_scfc:.3f} — MEETS benchmark (0.3-0.5)")
        print(f"  Simulated FC reflects structural connectivity.")
    elif c_scfc > 0.1:
        print(f"\n  SC-FC correlation: {c_scfc:.3f} — BELOW benchmark (expect 0.3-0.5)")
        print(f"  Simulated FC partially reflects SC but model needs tuning.")
    else:
        print(f"\n  SC-FC correlation: {c_scfc:.3f} — FAILS benchmark")
        print(f"  Simulated FC does not reflect structural connectivity.")

    if c_scfc > r_scfc:
        print(f"  Connectome produces HIGHER SC-FC correlation than random ({c_scfc:.3f} vs {r_scfc:.3f})")
    else:
        print(f"  Random produces similar or higher SC-FC correlation ({r_scfc:.3f} vs {c_scfc:.3f})")

    if c_homo > 0.3:
        print(f"  Homotopic correlation: {c_homo:.3f} — left-right pairs are correlated (realistic)")
    else:
        print(f"  Homotopic correlation: {c_homo:.3f} — low (may need stronger interhemispheric coupling)")

    # Save
    results_dir = Path("results/exp25_scfc_validation")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

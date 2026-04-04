"""Experiment 26: SC-FC Parameter Tuning.

Grid search over global_coupling and ext_rate_factor to find parameters
where simulated FC correlates with SC at r > 0.3.

Current problem: SC-FC r = 0.074 (benchmark: 0.3-0.5).
The between-region coupling is too weak relative to background noise,
so regional activity is dominated by local dynamics rather than
inter-regional structure.

Strategy: sweep global_coupling from 0.05 to 2.0, ext_rate_factor
from 2.0 to 5.0. Use short simulations (5s) for speed. Record SC-FC
correlation for each parameter combination.
"""

import time
import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU


def classify_regions(labels):
    groups = {}
    for key, patterns in [
        ('prefrontal', ['PFC', 'FEF']),
    ]:
        groups[key] = [i for i, l in enumerate(labels)
                       if any(p in l.upper() for p in patterns)]
    return groups


def compute_scfc(connectome, global_coupling, ext_rate_factor,
                 duration_ms=5000, device="cuda"):
    """Run simulation and compute SC-FC correlation."""
    groups = classify_regions(connectome.labels)
    pfc_regions = groups.get('prefrontal', [])
    npr = 200
    n_regions = connectome.num_regions
    n_total = n_regions * npr

    try:
        brain = SpikingBrainGPU(
            connectome=connectome, neurons_per_region=npr,
            global_coupling=global_coupling,
            ext_rate_factor=ext_rate_factor,
            tau_nmda=150.0, nmda_ratio=0.4,
            pfc_regions=pfc_regions, device=device,
            use_delays=True,
            conduction_velocity=3.5,
            use_neuron_types=True,
            use_adaptation=False,
        )
    except Exception as e:
        return {"sc_fc_r": 0.0, "mean_rate": 0.0, "mean_fc": 0.0, "error": str(e)}

    state = brain.init_state(batch_size=1)

    # Warmup (shorter for speed)
    with torch.no_grad():
        for _ in range(2000):
            state, _ = brain.step(state)

    # Check if network is alive
    test_spikes = 0
    with torch.no_grad():
        for _ in range(1000):
            state, spikes = brain.step(state)
            test_spikes += spikes[0].sum().item()
    mean_rate = test_spikes / (n_total * 1000)

    if mean_rate < 1e-6 or mean_rate > 0.5:
        # Dead or exploding
        del brain
        torch.cuda.empty_cache()
        return {"sc_fc_r": 0.0, "mean_rate": mean_rate, "mean_fc": 0.0,
                "status": "dead" if mean_rate < 1e-6 else "exploding"}

    # Simulate and record
    record_interval = 100  # 10ms bins
    total_steps = int(duration_ms / brain.dt)
    n_timepoints = total_steps // record_interval

    region_ts = np.zeros((n_timepoints, n_regions))
    spike_acc = torch.zeros(n_regions, device=device)
    acc_count = 0
    tp = 0

    with torch.no_grad():
        for step in range(total_steps):
            state, spikes = brain.step(state)
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

    # Compute FC
    fc = np.corrcoef(region_ts[:tp].T)

    # SC-FC correlation
    sc_log = np.log1p(connectome.weights)
    idx = np.triu_indices_from(fc, k=1)
    fc_upper = fc[idx]
    sc_upper = sc_log[idx]
    valid = ~np.isnan(fc_upper) & ~np.isnan(sc_upper)
    if valid.sum() > 10:
        sc_fc_r, _ = stats.pearsonr(sc_upper[valid], fc_upper[valid])
    else:
        sc_fc_r = 0.0

    mean_fc = float(np.nanmean(fc_upper))

    return {"sc_fc_r": sc_fc_r, "mean_rate": mean_rate, "mean_fc": mean_fc}


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 26: SC-FC Parameter Tuning")
    print("Finding global_coupling × ext_rate_factor where SC-FC > 0.3")
    print("=" * 70)

    connectome = Connectome.from_bundled("tvb96")
    device = "cuda"

    # Parameter grid
    # global_coupling controls how much between-region weights scale
    # ext_rate_factor controls background noise level
    gc_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    erf_values = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

    print(f"\n  Grid: {len(gc_values)} × {len(erf_values)} = {len(gc_values)*len(erf_values)} combinations")
    print(f"  global_coupling: {gc_values}")
    print(f"  ext_rate_factor: {erf_values}")
    print(f"  Duration: 5s per run")

    t0 = time.time()
    results = []

    # Header
    print(f"\n  {'gc':>6} {'erf':>6} {'SC-FC r':>10} {'mean_rate':>10} {'mean_FC':>10} {'status':>10}")
    print(f"  {'─'*56}")

    best_r = -1
    best_params = None

    for gc in gc_values:
        for erf in erf_values:
            r = compute_scfc(connectome, gc, erf, duration_ms=5000, device=device)
            elapsed = time.time() - t0

            status = r.get("status", "ok")
            scfc = r["sc_fc_r"]
            rate = r["mean_rate"]
            mfc = r["mean_fc"]

            marker = ""
            if scfc > best_r and status == "ok":
                best_r = scfc
                best_params = (gc, erf)
                marker = " ★"

            if scfc > 0.3:
                marker = " ★★★ PASSES"

            print(f"  {gc:>6.2f} {erf:>6.1f} {scfc:>10.4f} {rate:>10.5f} {mfc:>10.4f} {status:>10}{marker}")

            results.append({
                "global_coupling": gc,
                "ext_rate_factor": erf,
                "sc_fc_r": scfc,
                "mean_rate": rate,
                "mean_fc": mfc,
                "status": status,
            })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Best SC-FC r: {best_r:.4f} at gc={best_params[0]}, erf={best_params[1]}")

    passing = [r for r in results if r["sc_fc_r"] > 0.3 and r.get("status", "ok") == "ok"]
    if passing:
        print(f"\n  PASSING combinations (SC-FC > 0.3):")
        for r in sorted(passing, key=lambda x: -x["sc_fc_r"]):
            print(f"    gc={r['global_coupling']:.2f}, erf={r['ext_rate_factor']:.1f}: "
                  f"SC-FC r={r['sc_fc_r']:.4f}, rate={r['mean_rate']:.5f}")
    else:
        print(f"\n  NO combinations pass the 0.3 threshold.")
        print(f"  Top 5:")
        top5 = sorted([r for r in results if r.get("status", "ok") == "ok"],
                       key=lambda x: -x["sc_fc_r"])[:5]
        for r in top5:
            print(f"    gc={r['global_coupling']:.2f}, erf={r['ext_rate_factor']:.1f}: "
                  f"SC-FC r={r['sc_fc_r']:.4f}, rate={r['mean_rate']:.5f}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save
    import json
    from pathlib import Path
    results_dir = Path("results/exp26_scfc_tuning")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

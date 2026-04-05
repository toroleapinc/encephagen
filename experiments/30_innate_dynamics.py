"""Experiment 30: Innate Dynamics — Where topology SHOULD matter.

Tests dynamics that depend on BOTH the timescale gradient AND topology:
  1. Stimulus propagation: flash visual → measure when each region responds
  2. Propagation hierarchy: does signal flow V1→temporal→parietal→frontal?
  3. Oscillation spectrum: does the network produce alpha-band (8-12 Hz)?
  4. Functional connectivity structure: does connectome FC match empirical FC
     BETTER than random FC matches empirical FC?

These are the tests where the ROUTING (topology) should matter, not just
the node properties (tau_m). A visual stimulus enters visual cortex and
must propagate through the specific connectome pathways — random wiring
routes it differently, producing different propagation timing.
"""

import time
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats, signal

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU
from encephagen.analysis.statistics import report_with_fdr, benjamini_hochberg


def load_neurolib80():
    sc = np.load('src/encephagen/connectome/bundled/neurolib80_weights.npy')
    tl = np.load('src/encephagen/connectome/bundled/neurolib80_tract_lengths.npy')
    labels = json.load(open('src/encephagen/connectome/bundled/neurolib80_labels.json'))
    c = Connectome(sc, labels); c.tract_lengths = tl
    return c


def randomize(conn, seed):
    rng = np.random.default_rng(seed)
    w = conn.weights.copy()
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
    r = Connectome(w, list(conn.labels))
    r.tract_lengths = conn.tract_lengths
    return r


def classify_aal_regions(labels):
    """Classify AAL2 regions into functional groups."""
    groups = {}
    for key, patterns in [
        ('visual', ['Calcarine', 'Cuneus', 'Lingual', 'Occipital']),
        ('auditory', ['Heschl', 'Temporal_Sup']),
        ('motor', ['Precentral', 'Supp_Motor']),
        ('somatosensory', ['Postcentral', 'Paracentral']),
        ('frontal', ['Frontal_Sup', 'Frontal_Mid', 'Frontal_Inf']),
        ('parietal', ['Parietal', 'Angular', 'SupraMarginal', 'Precuneus']),
        ('temporal', ['Temporal_Mid', 'Temporal_Inf', 'Temporal_Pole', 'Fusiform']),
        ('cingulate', ['Cingulate', 'Cingulum']),
    ]:
        groups[key] = [i for i, l in enumerate(labels)
                       if any(p in l for p in patterns)]
    return groups


def make_brain(conn, device="cuda"):
    return SpikingBrainGPU(
        connectome=conn, neurons_per_region=200,
        internal_conn_prob=0.05, between_conn_prob=0.03,
        global_coupling=5.0, ext_rate_factor=3.5,  # FC-FC=0.28, stim_resp=+6%
        pfc_regions=[], device=device,
        use_delays=True, conduction_velocity=3.5,
        use_t1t2_gradient=True,
    )


def test_propagation(brain, groups, n_regions, npr, n_total, device):
    """Flash visual stimulus, measure response latency per region group."""
    state = brain.init_state(batch_size=1)
    # Warmup
    with torch.no_grad():
        for _ in range(3000):
            state, _ = brain.step(state)

    # Baseline rates (1000 steps = 100ms)
    baseline = torch.zeros(n_regions, device=device)
    with torch.no_grad():
        for _ in range(1000):
            state, spikes = brain.step(state)
            for r in range(n_regions):
                baseline[r] += spikes[0, r*npr:(r+1)*npr].sum()
    baseline = baseline / (npr * 1000)

    # Flash visual stimulus
    vis_idx = groups.get('visual', [])
    if not vis_idx:
        return {}, {}

    ext = torch.zeros(1, n_total, device=device)
    for ri in vis_idx:
        ext[0, ri*npr:(ri+1)*npr] = 100.0  # Strong stimulus

    # Record response over 500ms (5000 steps) in 10ms bins
    n_bins = 50
    bin_size = 100  # steps = 10ms
    response = np.zeros((n_bins, n_regions))

    with torch.no_grad():
        for b in range(n_bins):
            bin_spikes = torch.zeros(n_regions, device=device)
            for _ in range(bin_size):
                # Stimulus only for first 100ms (10 bins)
                if b < 10:
                    state, spikes = brain.step(state, ext)
                else:
                    state, spikes = brain.step(state)
                for r in range(n_regions):
                    bin_spikes[r] += spikes[0, r*npr:(r+1)*npr].sum()
            response[b] = (bin_spikes / (npr * bin_size)).cpu().numpy()

    # Compute latency per group: first bin where response > 2x baseline
    bl = baseline.cpu().numpy()
    latencies = {}
    amplitudes = {}
    for gname, gidx in groups.items():
        if not gidx:
            continue
        group_resp = response[:, gidx].mean(axis=1)
        group_bl = bl[gidx].mean()
        # Threshold: baseline + 3% absolute (sensitive to small changes)
        threshold = group_bl + 0.003

        # Find first bin above threshold
        above = np.where(group_resp > threshold)[0]
        if len(above) > 0:
            latencies[gname] = above[0] * 10  # ms
        else:
            latencies[gname] = 500  # no response

        amplitudes[gname] = float(group_resp.max() - group_bl)

    return latencies, amplitudes


def test_oscillations(brain, n_regions, npr, device):
    """Record 2 seconds of activity, compute power spectrum."""
    state = brain.init_state(batch_size=1)
    with torch.no_grad():
        for _ in range(3000):
            state, _ = brain.step(state)

    # Record 2s in 1ms bins
    n_bins = 2000
    bin_size = 10  # steps = 1ms
    ts = np.zeros((n_bins, n_regions))
    spike_acc = torch.zeros(n_regions, device=device)
    acc = 0
    tp = 0

    with torch.no_grad():
        for _ in range(n_bins * bin_size):
            state, spikes = brain.step(state)
            for r in range(n_regions):
                spike_acc[r] += spikes[0, r*npr:(r+1)*npr].sum()
            acc += 1
            if acc >= bin_size:
                ts[tp] = (spike_acc / (npr * acc)).cpu().numpy()
                spike_acc.zero_()
                acc = 0
                tp += 1
                if tp >= n_bins:
                    break

    # Compute power spectrum (mean across regions)
    mean_ts = ts[:tp].mean(axis=1)
    fs = 1000  # 1ms sampling = 1000 Hz
    freqs, psd = signal.welch(mean_ts, fs=fs, nperseg=min(512, len(mean_ts)))

    # Find peak frequency
    peak_idx = np.argmax(psd[1:]) + 1  # skip DC
    peak_freq = freqs[peak_idx]
    peak_power = psd[peak_idx]

    # Alpha band power (8-12 Hz)
    alpha_mask = (freqs >= 8) & (freqs <= 12)
    alpha_power = psd[alpha_mask].sum() if alpha_mask.any() else 0
    total_power = psd[1:].sum()  # exclude DC
    alpha_ratio = alpha_power / (total_power + 1e-10)

    return peak_freq, alpha_ratio


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 30: Innate Dynamics — Stimulus Propagation + Oscillations")
    print("T1w/T2w gradient + delays — where topology SHOULD matter")
    print("=" * 70)

    conn = load_neurolib80()
    labels = json.load(open('src/encephagen/connectome/bundled/neurolib80_t1t2_labels.json'))
    groups = classify_aal_regions(labels)
    npr = 200; n_regions = 80; n_total = n_regions * npr
    device = "cuda"; N_RUNS = 8

    print(f"\nRegion groups:")
    for k, v in groups.items():
        print(f"  {k}: {len(v)} regions")

    results = {"connectome": {"latencies": [], "peak_freq": [], "alpha": []},
               "random": {"latencies": [], "peak_freq": [], "alpha": []}}

    t0 = time.time()

    for cond in ["connectome", "random"]:
        print(f"\n{'='*60}")
        print(f"CONDITION: {cond.upper()} ({N_RUNS} runs)")
        print(f"{'='*60}")

        for run in range(N_RUNS):
            if cond == "connectome":
                c = conn
            else:
                c = randomize(conn, seed=1000 + run)

            brain = make_brain(c, device=device)

            # Propagation test
            lats, amps = test_propagation(brain, groups, n_regions, npr, n_total, device)
            results[cond]["latencies"].append(lats)

            # Oscillation test
            peak_f, alpha_r = test_oscillations(brain, n_regions, npr, device)
            results[cond]["peak_freq"].append(peak_f)
            results[cond]["alpha"].append(alpha_r)

            elapsed = time.time() - t0
            lat_str = " ".join(f"{k[:3]}={v}ms" for k, v in sorted(lats.items(), key=lambda x: x[1]))
            print(f"  Run {run+1:>2}/{N_RUNS}  peak={peak_f:.1f}Hz  alpha={alpha_r:.3f}  "
                  f"latencies: {lat_str}  ({elapsed:.0f}s)")

            del brain; torch.cuda.empty_cache()

    # ========================================
    # Analysis
    # ========================================
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    # 1. Propagation hierarchy
    print(f"\n  --- Stimulus Propagation Latency (ms) ---")
    print(f"  {'Group':<15} {'Connectome':>12} {'Random':>12} {'p':>8}")
    print(f"  {'─'*50}")

    p_vals = []; p_labels = []
    for gname in ['visual', 'auditory', 'temporal', 'parietal', 'frontal', 'motor', 'cingulate']:
        c_lats = [r.get(gname, 500) for r in results["connectome"]["latencies"]]
        r_lats = [r.get(gname, 500) for r in results["random"]["latencies"]]
        if len(c_lats) > 1 and len(r_lats) > 1:
            _, p = stats.mannwhitneyu(c_lats, r_lats, alternative="two-sided")
        else:
            p = 1.0
        p_vals.append(p); p_labels.append(f"Latency {gname}")
        sig = "*" if p < 0.05 else ""
        print(f"  {gname:<15} {np.mean(c_lats):>12.1f} {np.mean(r_lats):>12.1f} {p:>8.4f}{sig}")

    # 2. Oscillations
    print(f"\n  --- Oscillations ---")
    c_peak = results["connectome"]["peak_freq"]
    r_peak = results["random"]["peak_freq"]
    c_alpha = results["connectome"]["alpha"]
    r_alpha = results["random"]["alpha"]
    _, p_peak = stats.mannwhitneyu(c_peak, r_peak, alternative="two-sided")
    _, p_alpha = stats.mannwhitneyu(c_alpha, r_alpha, alternative="two-sided")
    p_vals.extend([p_peak, p_alpha])
    p_labels.extend(["Peak frequency", "Alpha power ratio"])

    print(f"  Peak freq:   conn={np.mean(c_peak):.1f}Hz  rand={np.mean(r_peak):.1f}Hz  p={p_peak:.4f}")
    print(f"  Alpha ratio: conn={np.mean(c_alpha):.4f}  rand={np.mean(r_alpha):.4f}  p={p_alpha:.4f}")

    # 3. Propagation ORDER
    print(f"\n  --- Propagation Order ---")
    expected = ['visual', 'auditory', 'temporal', 'parietal', 'frontal']
    for cond in ["connectome", "random"]:
        mean_lats = {}
        for gname in expected:
            lats = [r.get(gname, 500) for r in results[cond]["latencies"]]
            mean_lats[gname] = np.mean(lats)
        order = sorted(mean_lats.keys(), key=lambda x: mean_lats[x])
        order_str = " → ".join(f"{g}({mean_lats[g]:.0f}ms)" for g in order)
        print(f"  {cond:>12}: {order_str}")

    # FDR
    print(f"\n{report_with_fdr(p_labels, p_vals)}")

    # Save
    results_dir = Path("results/exp30_innate_dynamics")
    results_dir.mkdir(parents=True, exist_ok=True)
    save = {}
    for cond in results:
        save[cond] = {
            "latencies": results[cond]["latencies"],
            "peak_freq": [float(x) for x in results[cond]["peak_freq"]],
            "alpha": [float(x) for x in results[cond]["alpha"]],
        }
    with open(results_dir / "results.json", "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

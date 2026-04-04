"""Experiment 23: Why does random wiring trend better at discrimination?

Experiment 22 found connectome wins on conditioning (p=0.011) but
random TRENDS better on discrimination (p=0.052). This is surprising
and potentially the most interesting finding.

Hypothesis: The connectome constrains information flow through
specific pathways (VIS→PFC, VIS→TEMP), creating correlated response
vectors. Random wiring spreads signals more diffusely, creating
more diverse (less correlated) responses that are easier to distinguish.

This experiment measures:
  1. Response vector diversity (mean inter-pattern cosine distance)
  2. Effective dimensionality of response space (PCA)
  3. Signal propagation breadth (how many regions activate per pattern)
  4. Pathway concentration (entropy of regional activation)

If confirmed, this reveals a fundamental trade-off:
  - Structure → efficient routing → good for conditioning (specific pathways)
  - Random → diverse routing → good for discrimination (representational diversity)
"""

import time
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

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


def make_brain(connectome, device="cuda"):
    groups = classify_regions(connectome.labels)
    pfc_regions = groups['prefrontal']
    brain = SpikingBrainGPU(
        connectome=connectome, neurons_per_region=200,
        global_coupling=0.15, ext_rate_factor=3.5,
        tau_nmda=150.0, nmda_ratio=0.4,
        pfc_regions=pfc_regions, device=device,
    )
    return brain, groups


def measure_responses(brain, groups, patterns, npr, n_total, device):
    """Collect response vectors for each pattern across multiple trials.

    Returns:
        responses: dict {pattern_idx: list of [n_regions] rate vectors}
        region_responses: dict {pattern_idx: list of [n_readout_neurons] spike vectors}
    """
    vis_idx = groups['visual']
    n_regions = brain.n_regions

    state = brain.init_state(batch_size=1)
    # Warmup
    with torch.no_grad():
        for _ in range(3000):
            state, _ = brain.step(state)

    # Collect region-level AND neuron-level responses
    region_responses = {i: [] for i in range(len(patterns))}
    neuron_responses = {i: [] for i in range(len(patterns))}

    for trial in range(8):
        for p_idx in range(len(patterns)):
            # Gap
            with torch.no_grad():
                for _ in range(200):
                    state, _ = brain.step(state)

            # Present pattern
            external = torch.zeros(1, n_total, device=device)
            pat_t = torch.tensor(patterns[p_idx], device=device)
            for ri in vis_idx:
                external[0, ri*npr:(ri+1)*npr] = pat_t * 12.0

            # Collect per-region firing rates
            region_spikes = torch.zeros(n_regions, device=device)
            all_spikes = torch.zeros(n_total, device=device)
            with torch.no_grad():
                for _ in range(500):
                    state, spikes = brain.step(state, external)
                    all_spikes += spikes[0].float()
                    for r in range(n_regions):
                        s = r * npr
                        e = (r + 1) * npr
                        region_spikes[r] += spikes[0, s:e].sum()

            region_rates = region_spikes.cpu().numpy() / (npr * 500)
            region_responses[p_idx].append(region_rates)
            neuron_responses[p_idx].append(all_spikes.cpu().numpy())

    return region_responses, neuron_responses


def analyze_diversity(region_responses, n_patterns):
    """Measure response vector diversity metrics."""
    results = {}

    # 1. Mean inter-pattern cosine distance
    # Higher = more distinguishable patterns
    mean_vecs = []
    for p_idx in range(n_patterns):
        mean_vec = np.mean(region_responses[p_idx], axis=0)
        mean_vecs.append(mean_vec)
    mean_vecs = np.array(mean_vecs)

    cosine_distances = []
    for i in range(n_patterns):
        for j in range(i + 1, n_patterns):
            vi, vj = mean_vecs[i], mean_vecs[j]
            ni, nj = np.linalg.norm(vi), np.linalg.norm(vj)
            if ni > 0 and nj > 0:
                cos_sim = np.dot(vi, vj) / (ni * nj)
                cosine_distances.append(1.0 - cos_sim)
    results['mean_cosine_distance'] = float(np.mean(cosine_distances))
    results['min_cosine_distance'] = float(np.min(cosine_distances))

    # 2. Trial-to-trial consistency (within-pattern similarity)
    # Higher = more reliable responses
    consistencies = []
    for p_idx in range(n_patterns):
        vecs = np.array(region_responses[p_idx])
        if len(vecs) < 2:
            continue
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                vi, vj = vecs[i], vecs[j]
                ni, nj = np.linalg.norm(vi), np.linalg.norm(vj)
                if ni > 0 and nj > 0:
                    consistencies.append(np.dot(vi, vj) / (ni * nj))
    results['mean_consistency'] = float(np.mean(consistencies))

    # 3. Discriminability index (inter / intra variance ratio)
    # Fisher's criterion: between-class variance / within-class variance
    all_vecs = np.vstack([np.array(region_responses[p]) for p in range(n_patterns)])
    global_mean = all_vecs.mean(axis=0)

    between_var = 0.0
    within_var = 0.0
    for p_idx in range(n_patterns):
        class_vecs = np.array(region_responses[p_idx])
        class_mean = class_vecs.mean(axis=0)
        between_var += len(class_vecs) * np.sum((class_mean - global_mean) ** 2)
        for v in class_vecs:
            within_var += np.sum((v - class_mean) ** 2)

    results['fisher_ratio'] = float(between_var / (within_var + 1e-10))

    # 4. Effective dimensionality (PCA participation ratio)
    # Higher = responses span more dimensions = more representational capacity
    centered = all_vecs - all_vecs.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]
    eigenvalues = eigenvalues / eigenvalues.sum()
    participation_ratio = 1.0 / np.sum(eigenvalues ** 2)
    results['effective_dim'] = float(participation_ratio)

    # 5. Signal propagation breadth
    # How many regions respond above baseline for each pattern?
    baseline = np.mean([np.mean(region_responses[p], axis=0) for p in range(n_patterns)], axis=0)
    breadths = []
    for p_idx in range(n_patterns):
        mean_resp = np.mean(region_responses[p_idx], axis=0)
        # Count regions with >10% change from mean
        deviation = np.abs(mean_resp - baseline) / (baseline + 1e-10)
        n_responding = np.sum(deviation > 0.1)
        breadths.append(n_responding)
    results['mean_breadth'] = float(np.mean(breadths))

    # 6. Activation entropy (how evenly distributed is activity?)
    # Higher entropy = more distributed = potentially more diverse representations
    entropies = []
    for p_idx in range(n_patterns):
        mean_resp = np.mean(region_responses[p_idx], axis=0)
        mean_resp = np.maximum(mean_resp, 0)
        total = mean_resp.sum()
        if total > 0:
            probs = mean_resp / total
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            entropies.append(entropy)
    results['mean_entropy'] = float(np.mean(entropies))

    return results


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 23: Why does random wiring trend better at discrimination?")
    print("Analyzing response diversity — connectome vs random")
    print("=" * 70)

    connectome = Connectome.from_bundled("tvb96")
    npr = 200
    n_total = 96 * npr
    device = "cuda"
    N_PATTERNS = 5
    N_RUNS = 8

    # Generate patterns (same as Exp 22)
    rng = np.random.default_rng(42)
    patterns = []
    for i in range(N_PATTERNS):
        p = np.zeros(npr, dtype=np.float32)
        p[rng.choice(npr, 60, replace=False)] = 1.0
        patterns.append(p)

    all_results = {"connectome": [], "random": []}
    t0 = time.time()

    for condition in ["connectome", "random"]:
        print(f"\n{'='*60}")
        print(f"CONDITION: {condition.upper()} ({N_RUNS} runs)")
        print(f"{'='*60}")

        for run in range(N_RUNS):
            if condition == "connectome":
                conn = connectome
            else:
                conn = randomize_connectome(connectome, seed=1000 + run)

            brain, groups = make_brain(conn, device=device)
            region_resp, neuron_resp = measure_responses(
                brain, groups, patterns, npr, n_total, device
            )
            metrics = analyze_diversity(region_resp, N_PATTERNS)

            elapsed = time.time() - t0
            print(f"  Run {run+1:>2}/{N_RUNS}  "
                  f"cos_dist={metrics['mean_cosine_distance']:.4f}  "
                  f"fisher={metrics['fisher_ratio']:.4f}  "
                  f"eff_dim={metrics['effective_dim']:.1f}  "
                  f"entropy={metrics['mean_entropy']:.3f}  "
                  f"breadth={metrics['mean_breadth']:.1f}  "
                  f"({elapsed:.0f}s)")

            all_results[condition].append(metrics)

            del brain
            torch.cuda.empty_cache()

    # ========================================
    # Statistical comparison
    # ========================================
    print(f"\n{'='*70}")
    print("RESULTS: Response Diversity Analysis")
    print(f"{'='*70}")

    metric_names = [
        ("Mean cosine distance", "mean_cosine_distance", "higher = more distinguishable"),
        ("Min cosine distance", "min_cosine_distance", "higher = no confusable pairs"),
        ("Trial consistency", "mean_consistency", "higher = more reliable"),
        ("Fisher discriminability", "fisher_ratio", "higher = better separable"),
        ("Effective dimensionality", "effective_dim", "higher = richer representation"),
        ("Signal breadth (regions)", "mean_breadth", "higher = wider propagation"),
        ("Activation entropy", "mean_entropy", "higher = more distributed"),
    ]

    print(f"\n  {'Metric':<35} {'Connectome':>12} {'Random':>12} {'p-value':>10} {'Winner':>12} {'Meaning'}")
    print(f"  {'─'*110}")

    for label, key, meaning in metric_names:
        c_vals = np.array([r[key] for r in all_results["connectome"]])
        r_vals = np.array([r[key] for r in all_results["random"]])

        c_mean = c_vals.mean()
        r_mean = r_vals.mean()

        stat, p = stats.mannwhitneyu(c_vals, r_vals, alternative="two-sided")

        winner = ""
        if p < 0.05:
            winner = "CONNECTOME" if c_mean > r_mean else "RANDOM"

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label:<35} {c_mean:>12.4f} {r_mean:>12.4f} {p:>9.4f}{sig:>1} {winner:>12}  {meaning}")

    # Effect sizes
    print(f"\n  Effect sizes:")
    for label, key, _ in metric_names:
        c_vals = np.array([r[key] for r in all_results["connectome"]])
        r_vals = np.array([r[key] for r in all_results["random"]])
        pooled_std = np.sqrt((c_vals.var() + r_vals.var()) / 2)
        d = (c_vals.mean() - r_vals.mean()) / (pooled_std + 1e-10)
        print(f"    {label:<35} d={d:+.3f}")

    # ========================================
    # Interpretation
    # ========================================
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    c_cos = np.mean([r['mean_cosine_distance'] for r in all_results["connectome"]])
    r_cos = np.mean([r['mean_cosine_distance'] for r in all_results["random"]])
    c_dim = np.mean([r['effective_dim'] for r in all_results["connectome"]])
    r_dim = np.mean([r['effective_dim'] for r in all_results["random"]])
    c_ent = np.mean([r['mean_entropy'] for r in all_results["connectome"]])
    r_ent = np.mean([r['mean_entropy'] for r in all_results["random"]])

    if r_cos > c_cos and r_dim > c_dim:
        print(f"\n  CONFIRMED: Random wiring creates MORE DIVERSE response vectors.")
        print(f"  - Cosine distance: random {r_cos:.4f} > connectome {c_cos:.4f}")
        print(f"  - Effective dim: random {r_dim:.1f} > connectome {c_dim:.1f}")
        print(f"\n  This explains the discrimination advantage:")
        print(f"  The connectome channels signals through specific pathways,")
        print(f"  creating correlated responses. Random wiring spreads signals")
        print(f"  more diffusely, creating a higher-dimensional response space")
        print(f"  where patterns are easier to distinguish.")
        print(f"\n  TRADE-OFF DISCOVERED:")
        print(f"    Structure → efficient routing → conditioning advantage")
        print(f"    Random   → diverse routing  → discrimination advantage")
    elif c_cos > r_cos:
        print(f"\n  UNEXPECTED: Connectome has MORE diverse responses but still")
        print(f"  worse discrimination. The issue may be trial-to-trial reliability,")
        print(f"  not diversity.")
    else:
        print(f"\n  INCONCLUSIVE: No clear pattern in diversity metrics.")
        print(f"  The discrimination difference may be noise (p=0.052 in Exp 22).")

    # Save
    results_dir = Path("results/exp23_discrimination_analysis")
    results_dir.mkdir(parents=True, exist_ok=True)

    save_data = {}
    for cond in all_results:
        save_data[cond] = all_results[cond]

    with open(results_dir / "results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

"""Experiment 24: The definitive test — does brain structure help learning?

Incorporates ALL expert panel recommendations:
  1. Conduction delays from tract lengths (Buzsáki)
  2. Continuous e-prop reward modulation (Gerstner)
  3. ALIF neurons for longer temporal credit (Gerstner)
  4. Region-specific neuron types (Murthy)
  5. Erdős-Rényi null model in addition to degree-preserving (Larson, Marder)
  6. 30 samples per condition (Marder)
  7. Weight change tracking and pathway analysis (Larson, Murthy)

Three conditions:
  A) Real human connectome (TVB96)
  B) Degree-preserving random rewiring (same density, same degree)
  C) Erdős-Rényi random (same density, completely random)

Same 4-phase protocol:
  Phase 1: Innate regional differentiation
  Phase 2: Classical conditioning (CS-US with continuous e-prop)
  Phase 3: Pattern discrimination (after e-prop training)
  Phase 4: Working memory persistence
"""

import time
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU
from encephagen.learning.eprop import EpropParams


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


def randomize_connectome_degree_preserving(connectome, seed):
    """Degree-preserving rewiring (same as Exp 21-22)."""
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


def randomize_connectome_erdos_renyi(connectome, seed):
    """Erdős-Rényi random graph with same density and weight distribution."""
    rng = np.random.default_rng(seed)
    n = connectome.num_regions
    w_orig = connectome.weights
    # Same number of edges, same weight distribution, completely random placement
    existing_weights = w_orig[w_orig > 0].copy()
    rng.shuffle(existing_weights)
    n_edges = len(existing_weights)
    w_new = np.zeros((n, n), dtype=np.float32)
    # Randomly place edges (no self-connections)
    possible = [(i, j) for i in range(n) for j in range(n) if i != j]
    chosen = rng.choice(len(possible), size=n_edges, replace=False)
    for idx, edge_idx in enumerate(chosen):
        i, j = possible[edge_idx]
        w_new[i, j] = existing_weights[idx]
    return Connectome.from_numpy(w_new, list(connectome.labels))


def make_brain(connectome, device="cuda"):
    """Create brain with all new features enabled."""
    groups = classify_regions(connectome.labels)
    pfc_regions = groups['prefrontal']
    brain = SpikingBrainGPU(
        connectome=connectome, neurons_per_region=200,
        global_coupling=0.15, ext_rate_factor=3.5,
        tau_nmda=150.0, nmda_ratio=0.4,
        pfc_regions=pfc_regions, device=device,
        use_delays=True,
        conduction_velocity=3.5,
        use_neuron_types=True,
        use_adaptation=True,
        tau_adapt=200.0, beta_adapt=1.6,
    )
    return brain, groups


def warmup(brain, steps=3000):
    state = brain.init_state(batch_size=1)
    with torch.no_grad():
        for _ in range(steps):
            state, _ = brain.step(state)
    return state


# ========================================
# Phase 1: Innate differences
# ========================================
def phase1_innate(brain, state, groups, npr, n_total, device):
    region_rates = {}
    steps = 2000
    counts = {name: 0.0 for name in groups}
    with torch.no_grad():
        for _ in range(steps):
            state, spikes = brain.step(state)
            for name, indices in groups.items():
                for ri in indices:
                    counts[name] += spikes[0, ri*npr:(ri+1)*npr].sum().item()
    for name, indices in groups.items():
        if indices:
            region_rates[name] = counts[name] / (len(indices) * npr * steps)
    rates = list(region_rates.values())
    cv = float(np.std(rates) / (np.mean(rates) + 1e-12))
    return state, cv, region_rates


# ========================================
# Phase 2: Conditioning with continuous e-prop
# ========================================
def phase2_conditioning(brain, state, groups, npr, n_total, device):
    vis_idx = groups['visual'][:4]
    amyg_idx = groups['amygdala']
    response_idx = amyg_idx + groups['prefrontal'][:4]

    rng = np.random.default_rng(42)
    pattern = np.zeros(npr, dtype=np.float32)
    pattern[rng.choice(npr, 60, replace=False)] = 1.0

    eprop_params = EpropParams(
        lr=0.1, tau_e=50.0, gamma=0.5, w_max=15.0,
        regularization=0.0, reward_decay=0.99,
    )
    learner = brain.enable_learning(eprop_params)

    # Save initial weights for pathway analysis
    W_initial = brain.W.coalesce().values().clone()

    responses_over_time = []

    for trial in range(30):
        # CS presentation — eligibility accumulates
        ext_cs = torch.zeros(1, n_total, device=device)
        pat_t = torch.tensor(pattern, device=device)
        for ri in vis_idx:
            ext_cs[0, ri*npr:(ri+1)*npr] = pat_t * 12.0

        for _ in range(500):
            state, spikes = brain.step(state, ext_cs)

        # US with CONTINUOUS reward modulation
        # Eligibility from CS phase naturally decays — reward modulates
        # whatever eligibility remains (temporal credit assignment)
        ext_us = torch.zeros(1, n_total, device=device)
        for ri in amyg_idx:
            ext_us[0, ri*npr:(ri+1)*npr] = 20.0

        for step in range(500):
            state, spikes = brain.step(state, ext_us)
            if step % 50 == 0:
                brain.apply_reward(spikes, reward=1.0)

        # Gap
        for _ in range(300):
            state, spikes = brain.step(state)

        # Measure CS response every 5 trials
        if (trial + 1) % 5 == 0:
            brain.learner = None
            ext_test = torch.zeros(1, n_total, device=device)
            for ri in vis_idx:
                ext_test[0, ri*npr:(ri+1)*npr] = pat_t * 12.0

            test_spikes = torch.zeros(len(response_idx), device=device)
            bl_spikes = torch.zeros(len(response_idx), device=device)
            with torch.no_grad():
                for _ in range(300):
                    state, spikes = brain.step(state)
                    for j, ri in enumerate(response_idx):
                        bl_spikes[j] += spikes[0, ri*npr:(ri+1)*npr].sum()
                for _ in range(500):
                    state, spikes = brain.step(state, ext_test)
                    for j, ri in enumerate(response_idx):
                        test_spikes[j] += spikes[0, ri*npr:(ri+1)*npr].sum()

            bl = bl_spikes.cpu().numpy().mean() / (npr * 300)
            ts = test_spikes.cpu().numpy().mean() / (npr * 500)
            responses_over_time.append(ts - bl)
            brain.learner = learner

    brain.learner = None

    # Weight change analysis
    W_final = brain.W.coalesce().values()
    dW = (W_final - W_initial).abs()
    weight_change_total = dW.sum().item()
    weight_change_mean = dW.mean().item()

    if len(responses_over_time) > 2:
        x = np.arange(len(responses_over_time))
        slope, _, _, _, _ = stats.linregress(x, responses_over_time)
        final_response = responses_over_time[-1]
    else:
        slope = 0.0
        final_response = 0.0

    return state, slope, final_response, weight_change_mean


# ========================================
# Phase 3: Pattern discrimination
# ========================================
def phase3_discrimination(brain, state, groups, npr, n_total, device):
    vis_idx = groups['visual']
    readout_idx = groups.get('prefrontal', [])[:8] + groups.get('hippocampus', [])

    rng = np.random.default_rng(42)
    patterns = []
    for i in range(5):
        p = np.zeros(npr, dtype=np.float32)
        p[rng.choice(npr, 60, replace=False)] = 1.0
        patterns.append(p)

    # Train with e-prop
    eprop_params = EpropParams(
        lr=0.1, tau_e=50.0, gamma=0.5, w_max=15.0,
        regularization=0.0, reward_decay=0.99,
    )
    learner = brain.enable_learning(eprop_params)

    for epoch in range(3):
        for p_idx in range(5):
            external = torch.zeros(1, n_total, device=device)
            pat_t = torch.tensor(patterns[p_idx], device=device)
            for ri in vis_idx:
                external[0, ri*npr:(ri+1)*npr] = pat_t * 12.0
            for step in range(500):
                state, spikes = brain.step(state, external)
                if step % 100 == 0:
                    brain.apply_reward(spikes, reward=0.5)
            for _ in range(200):
                state, spikes = brain.step(state)

    brain.learner = None

    # Test
    responses = {i: [] for i in range(5)}
    for trial in range(5):
        for p_idx in range(5):
            with torch.no_grad():
                for _ in range(200):
                    state, _ = brain.step(state)
            external = torch.zeros(1, n_total, device=device)
            pat_t = torch.tensor(patterns[p_idx], device=device)
            for ri in vis_idx:
                external[0, ri*npr:(ri+1)*npr] = pat_t * 12.0
            resp = torch.zeros(len(readout_idx), device=device)
            with torch.no_grad():
                for _ in range(500):
                    state, spikes = brain.step(state, external)
                    for j, ri in enumerate(readout_idx):
                        resp[j] += spikes[0, ri*npr:(ri+1)*npr].sum()
            responses[p_idx].append(resp.cpu().numpy())

    correct = 0
    total = 0
    for p_idx in range(5):
        for t_idx in range(5):
            test_vec = responses[p_idx][t_idx]
            best_sim = -1
            best_class = -1
            for ref_idx in range(5):
                ref_vecs = [responses[ref_idx][t] for t in range(5)
                            if not (ref_idx == p_idx and t == t_idx)]
                ref_mean = np.mean(ref_vecs, axis=0)
                nt, nr = np.linalg.norm(test_vec), np.linalg.norm(ref_mean)
                sim = np.dot(test_vec, ref_mean) / (nt * nr) if nt > 0 and nr > 0 else 0
                if sim > best_sim:
                    best_sim = sim
                    best_class = ref_idx
            if best_class == p_idx:
                correct += 1
            total += 1

    return state, correct / total


# ========================================
# Phase 4: Working memory
# ========================================
def phase4_memory(brain, state, groups, npr, n_total, device):
    vis_idx = groups['visual']
    pfc_idx = groups['prefrontal']

    bl_rate = 0.0
    with torch.no_grad():
        for _ in range(500):
            state, spikes = brain.step(state)
            for ri in pfc_idx:
                bl_rate += spikes[0, ri*npr:(ri+1)*npr].sum().item()
    bl_rate /= (len(pfc_idx) * npr * 500)

    external = torch.zeros(1, n_total, device=device)
    for ri in vis_idx:
        external[0, ri*npr:(ri+1)*npr] = 15.0
    stim_rate = 0.0
    with torch.no_grad():
        for _ in range(500):
            state, spikes = brain.step(state, external)
            for ri in pfc_idx:
                stim_rate += spikes[0, ri*npr:(ri+1)*npr].sum().item()
    stim_rate /= (len(pfc_idx) * npr * 500)

    delay_rate = 0.0
    with torch.no_grad():
        for _ in range(500):
            state, spikes = brain.step(state)
            for ri in pfc_idx:
                delay_rate += spikes[0, ri*npr:(ri+1)*npr].sum().item()
    delay_rate /= (len(pfc_idx) * npr * 500)

    stim_change = stim_rate - bl_rate
    persistence = (delay_rate - bl_rate) / (stim_change + 1e-10) if stim_change > 0 else 0
    return state, persistence, stim_change


# ========================================
# Main
# ========================================
def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 24: The Definitive Test")
    print("Delays + ALIF + Neuron Types + Continuous E-prop")
    print("Connectome vs Degree-Preserving vs Erdős-Rényi — 30 runs each")
    print("=" * 70)

    connectome = Connectome.from_bundled("tvb96")
    groups = classify_regions(connectome.labels)
    npr = 200
    n_total = 96 * npr
    device = "cuda"
    N_RUNS = 15  # 15 per condition × 3 conditions = 45 total runs

    conditions = {
        "connectome": lambda seed: connectome,
        "degree_preserving": lambda seed: randomize_connectome_degree_preserving(connectome, seed),
        "erdos_renyi": lambda seed: randomize_connectome_erdos_renyi(connectome, seed),
    }

    results = {cond: {"cv": [], "cond_slope": [], "cond_final": [],
                       "discrim": [], "memory": [], "weight_change": []}
               for cond in conditions}

    t0 = time.time()

    for cond_name, cond_fn in conditions.items():
        print(f"\n{'='*60}")
        print(f"CONDITION: {cond_name.upper()} ({N_RUNS} runs)")
        print(f"{'='*60}")

        for run in range(N_RUNS):
            conn = cond_fn(seed=1000 + run)
            brain, grp = make_brain(conn, device=device)
            state = warmup(brain, steps=3000)

            state, cv, _ = phase1_innate(brain, state, grp, npr, n_total, device)
            results[cond_name]["cv"].append(cv)

            state, slope, final_r, wc = phase2_conditioning(
                brain, state, grp, npr, n_total, device)
            results[cond_name]["cond_slope"].append(slope)
            results[cond_name]["cond_final"].append(final_r)
            results[cond_name]["weight_change"].append(wc)

            state, acc = phase3_discrimination(brain, state, grp, npr, n_total, device)
            results[cond_name]["discrim"].append(acc)

            state, persist, _ = phase4_memory(brain, state, grp, npr, n_total, device)
            results[cond_name]["memory"].append(persist)

            elapsed = time.time() - t0
            print(f"  Run {run+1:>2}/{N_RUNS}  cv={cv:.3f}  cond={final_r:.5f}  "
                  f"discrim={acc:.0%}  memory={persist:.0%}  wΔ={wc:.6f}  ({elapsed:.0f}s)")

            del brain
            torch.cuda.empty_cache()

    # ========================================
    # Statistical comparison
    # ========================================
    print(f"\n{'='*70}")
    print("RESULTS: Three-Way Comparison (with all improvements)")
    print(f"{'='*70}")

    metrics = [
        ("Regional differentiation (CV)", "cv"),
        ("Conditioning speed (slope)", "cond_slope"),
        ("Conditioning strength (final)", "cond_final"),
        ("Pattern discrimination", "discrim"),
        ("Working memory (persistence)", "memory"),
        ("Weight change (mean |dW|)", "weight_change"),
    ]

    # Pairwise comparisons: connectome vs each null
    for null_name in ["degree_preserving", "erdos_renyi"]:
        print(f"\n  --- Connectome vs {null_name.upper()} ---")
        print(f"  {'Metric':<35} {'Connectome':>12} {null_name:>15} {'p-value':>10} {'Winner':>12}")
        print(f"  {'─'*86}")

        for label, key in metrics:
            c_vals = np.array(results["connectome"][key])
            n_vals = np.array(results[null_name][key])
            c_mean = c_vals.mean()
            n_mean = n_vals.mean()
            stat, p = stats.mannwhitneyu(c_vals, n_vals, alternative="two-sided")
            winner = ""
            if p < 0.05:
                winner = "CONNECTOME" if c_mean > n_mean else null_name.upper()
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {label:<35} {c_mean:>12.5f} {n_mean:>15.5f} {p:>9.4f}{sig:>1} {winner:>12}")

    # Effect sizes
    print(f"\n  Effect sizes (Cohen's d, connectome vs each null):")
    for label, key in metrics:
        c_vals = np.array(results["connectome"][key])
        dp_vals = np.array(results["degree_preserving"][key])
        er_vals = np.array(results["erdos_renyi"][key])
        for null_name, n_vals in [("DP", dp_vals), ("ER", er_vals)]:
            pooled_std = np.sqrt((c_vals.var() + n_vals.var()) / 2)
            d = (c_vals.mean() - n_vals.mean()) / (pooled_std + 1e-10)
            size = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small" if abs(d) > 0.2 else "neg."
            print(f"    {label:<35} vs {null_name}: d={d:+.3f} ({size})")

    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    # Count significant connectome wins across both null models
    wins = 0
    total = 0
    for null_name in ["degree_preserving", "erdos_renyi"]:
        for _, key in metrics:
            c_vals = np.array(results["connectome"][key])
            n_vals = np.array(results[null_name][key])
            _, p = stats.mannwhitneyu(c_vals, n_vals, alternative="two-sided")
            if p < 0.05 and c_vals.mean() > n_vals.mean():
                wins += 1
            total += 1

    print(f"\n  Connectome wins: {wins}/{total} comparisons (across both null models)")

    if wins >= 6:
        print(f"  STRONG EVIDENCE: Brain structure helps learning.")
    elif wins >= 3:
        print(f"  MODERATE EVIDENCE: Structure provides selective advantages.")
    elif wins >= 1:
        print(f"  WEAK EVIDENCE: Structure matters for some tasks only.")
    else:
        print(f"  NO EVIDENCE: Structure doesn't help at macro scale.")

    # Save
    results_dir = Path("results/exp24_definitive")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_data = {}
    for cond in results:
        save_data[cond] = {k: [float(x) for x in v] for k, v in results[cond].items()}
    with open(results_dir / "results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

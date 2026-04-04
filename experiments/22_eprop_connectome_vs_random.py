"""Experiment 22: Does brain structure help LEARNING?

The definitive re-test of Experiment 21, now with a REAL learning rule.

Experiment 21 used Hebbian outer product (torch.outer(cs_active, us_active))
which is not a real learning rule — no spike timing, no eligibility traces,
no temporal credit assignment. Expert reviewers flagged this.

This experiment uses e-prop (Bellec et al. 2020):
  - Eligibility traces per synapse (causal influence of weight on spiking)
  - Surrogate gradient (approximates gradient through spiking nonlinearity)
  - Reward-modulated three-factor rule (pre × post_surrogate × reward)
  - Temporal credit assignment (eligibility carries information ~100ms)

Same protocol as Exp 21:
  - 10 connectome brains vs 10 random brains
  - Phase 1: Innate regional differentiation
  - Phase 2: Conditioning (CS-US pairing with e-prop learning)
  - Phase 3: Pattern discrimination (after e-prop training)
  - Phase 4: Working memory persistence

The critical question: does the connectome's organization help the brain
LEARN better, not just organize differently?
"""

import time
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU
from encephagen.learning.eprop import EpropLearner, EpropParams


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
    """Degree-preserving rewiring."""
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


def make_brain(connectome, seed, device="cuda"):
    """Create a brain with e-prop learning enabled."""
    groups = classify_regions(connectome.labels)
    pfc_regions = groups['prefrontal']
    brain = SpikingBrainGPU(
        connectome=connectome, neurons_per_region=200,
        global_coupling=0.15, ext_rate_factor=3.5,
        tau_nmda=150.0, nmda_ratio=0.4,
        pfc_regions=pfc_regions, device=device,
    )
    return brain, groups


def warmup(brain, steps=3000):
    """Warm up brain to steady state."""
    state = brain.init_state(batch_size=1)
    with torch.no_grad():
        for _ in range(steps):
            state, _ = brain.step(state)
    return state


# ========================================
# Phase 1: Innate differences (same as Exp 21)
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
# Phase 2: Conditioning with E-PROP
# ========================================
def phase2_conditioning_eprop(brain, state, groups, npr, n_total, device):
    """Classical conditioning using e-prop learning.

    Key difference from Exp 21: instead of injecting Hebbian outer product,
    we use e-prop eligibility traces + reward signal. The brain learns
    through its own dynamics — eligibility traces track which synapses
    causally contributed to activity, and reward strengthens those.
    """
    vis_idx = groups['visual'][:4]
    amyg_idx = groups['amygdala']
    response_idx = amyg_idx + groups['prefrontal'][:4]

    rng = np.random.default_rng(42)
    pattern = np.zeros(npr, dtype=np.float32)
    pattern[rng.choice(npr, 60, replace=False)] = 1.0

    # Enable e-prop learning
    eprop_params = EpropParams(
        lr=0.1,
        tau_e=50.0,        # Eligibility filter ~50ms
        gamma=0.5,         # Surrogate gradient dampening
        w_max=15.0,
        regularization=0.0,
        target_rate=0.02,
        reward_decay=0.99,  # Slow baseline tracking
    )
    learner = brain.enable_learning(eprop_params)

    responses_over_time = []

    for trial in range(30):
        # === CS presentation (500 steps) ===
        # Eligibility traces accumulate, tracking which synapses
        # causally contributed to the CS response
        ext_cs = torch.zeros(1, n_total, device=device)
        pat_t = torch.tensor(pattern, device=device)
        for ri in vis_idx:
            ext_cs[0, ri*npr:(ri+1)*npr] = pat_t * 12.0

        for _ in range(500):
            state, spikes = brain.step(state, ext_cs)

        # Snapshot eligibility at CS offset
        learner.snapshot_eligibility()

        # === US presentation (reward) ===
        # Reward modulates the SNAPSHOTTED eligibility from CS phase
        ext_us = torch.zeros(1, n_total, device=device)
        for ri in amyg_idx:
            ext_us[0, ri*npr:(ri+1)*npr] = 20.0

        for step in range(500):
            state, spikes = brain.step(state, ext_us)

        # Apply reward ONCE per trial using CS-phase eligibility snapshot
        brain.apply_reward(spikes, reward=1.0)

        # === Gap ===
        for _ in range(300):
            state, spikes = brain.step(state)

        # === Measure CS response every 5 trials ===
        if (trial + 1) % 5 == 0:
            # Disable learning for test
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

            # Re-enable learning
            brain.learner = learner

    # Disable learning after conditioning
    brain.learner = None

    if len(responses_over_time) > 2:
        x = np.arange(len(responses_over_time))
        slope, _, r_value, _, _ = stats.linregress(x, responses_over_time)
        final_response = responses_over_time[-1]
    else:
        slope = 0.0
        final_response = 0.0

    return state, slope, final_response, responses_over_time


# ========================================
# Phase 3: Pattern discrimination (after e-prop training)
# ========================================
def phase3_discrimination_eprop(brain, state, groups, npr, n_total, device):
    """Pattern discrimination WITH e-prop training.

    Train the brain to discriminate 5 patterns using e-prop,
    then test discrimination accuracy.
    """
    vis_idx = groups['visual']
    readout_idx = groups.get('prefrontal', [])[:8] + groups.get('hippocampus', [])
    amyg_idx = groups['amygdala']

    rng = np.random.default_rng(42)
    patterns = []
    for i in range(5):
        p = np.zeros(npr, dtype=np.float32)
        p[rng.choice(npr, 60, replace=False)] = 1.0
        patterns.append(p)

    # === Training phase: expose to each pattern with reward ===
    eprop_params = EpropParams(
        lr=0.1,
        tau_e=50.0,
        gamma=0.5,
        w_max=15.0,
        regularization=0.0,
        target_rate=0.02,
        reward_decay=0.99,
    )
    learner = brain.enable_learning(eprop_params)

    for epoch in range(3):
        for p_idx in range(5):
            # Present pattern — accumulate eligibility
            external = torch.zeros(1, n_total, device=device)
            pat_t = torch.tensor(patterns[p_idx], device=device)
            for ri in vis_idx:
                external[0, ri*npr:(ri+1)*npr] = pat_t * 12.0

            for _ in range(500):
                state, spikes = brain.step(state, external)

            # Snapshot and apply reward
            learner.snapshot_eligibility()
            brain.apply_reward(spikes, reward=1.0)

            # Gap between patterns
            for _ in range(200):
                state, spikes = brain.step(state)

    # Disable learning for testing
    brain.learner = None

    # === Test phase: measure responses ===
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

    # Classification accuracy (leave-one-out nearest centroid)
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

    accuracy = correct / total
    return state, accuracy


# ========================================
# Phase 4: Working memory (same as Exp 21)
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
# Main experiment
# ========================================
def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 22: Does brain structure help LEARNING?")
    print("E-prop learning (Bellec 2020) — Connectome vs Random — 10 runs each")
    print("=" * 70)
    print()
    print("Key upgrade from Exp 21:")
    print("  OLD: Hebbian outer product (torch.outer, no credit assignment)")
    print("  NEW: E-prop (eligibility traces + surrogate gradient + reward)")
    print()

    connectome = Connectome.from_bundled("tvb96")
    groups = classify_regions(connectome.labels)
    npr = 200
    n_total = 96 * npr
    device = "cuda"

    N_RUNS = 10

    results = {
        "connectome": {"cv": [], "cond_slope": [], "cond_final": [],
                       "discrim": [], "memory": []},
        "random": {"cv": [], "cond_slope": [], "cond_final": [],
                   "discrim": [], "memory": []},
    }

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

            brain, grp = make_brain(conn, seed=run, device=device)
            state = warmup(brain, steps=3000)

            # Phase 1: Innate differentiation
            state, cv, _ = phase1_innate(brain, state, grp, npr, n_total, device)
            results[condition]["cv"].append(cv)

            # Phase 2: Conditioning with E-PROP
            state, slope, final_r, _ = phase2_conditioning_eprop(
                brain, state, grp, npr, n_total, device
            )
            results[condition]["cond_slope"].append(slope)
            results[condition]["cond_final"].append(final_r)

            # Phase 3: Discrimination with E-PROP training
            state, acc = phase3_discrimination_eprop(
                brain, state, grp, npr, n_total, device
            )
            results[condition]["discrim"].append(acc)

            # Phase 4: Memory
            state, persist, _ = phase4_memory(brain, state, grp, npr, n_total, device)
            results[condition]["memory"].append(persist)

            elapsed = time.time() - t0
            print(f"  Run {run+1:>2}/{N_RUNS}  cv={cv:.3f}  cond={final_r:.5f}  "
                  f"discrim={acc:.0%}  memory={persist:.0%}  ({elapsed:.0f}s)")

            del brain
            torch.cuda.empty_cache()

    # ========================================
    # Statistical comparison
    # ========================================
    print(f"\n{'='*70}")
    print("RESULTS: Connectome vs Random (with E-prop learning)")
    print(f"{'='*70}")

    metrics = [
        ("Regional differentiation (CV)", "cv"),
        ("Conditioning speed (slope)", "cond_slope"),
        ("Conditioning strength (final)", "cond_final"),
        ("Pattern discrimination (accuracy)", "discrim"),
        ("Working memory (persistence)", "memory"),
    ]

    significant_wins = 0
    connectome_wins = 0
    total_tests = 0

    print(f"\n  {'Metric':<40} {'Connectome':>12} {'Random':>12} {'p-value':>10} {'Winner':>10}")
    print(f"  {'─'*86}")

    for label, key in metrics:
        c_vals = np.array(results["connectome"][key])
        r_vals = np.array(results["random"][key])

        c_mean = c_vals.mean()
        r_mean = r_vals.mean()

        stat, p = stats.mannwhitneyu(c_vals, r_vals, alternative="two-sided")

        winner = ""
        if p < 0.05:
            winner = "CONNECTOME" if c_mean > r_mean else "RANDOM"
            significant_wins += 1
            if winner == "CONNECTOME":
                connectome_wins += 1
        total_tests += 1

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label:<40} {c_mean:>12.5f} {r_mean:>12.5f} {p:>9.4f}{sig:>1} {winner:>10}")

    # Effect sizes
    print(f"\n  Effect sizes (Cohen's d):")
    for label, key in metrics:
        c_vals = np.array(results["connectome"][key])
        r_vals = np.array(results["random"][key])
        pooled_std = np.sqrt((c_vals.var() + r_vals.var()) / 2)
        d = (c_vals.mean() - r_vals.mean()) / (pooled_std + 1e-10)
        size = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small" if abs(d) > 0.2 else "negligible"
        print(f"    {label:<40} d={d:+.3f} ({size})")

    # Compare with Exp 21
    print(f"\n{'='*70}")
    print("COMPARISON WITH EXPERIMENT 21 (Hebbian)")
    print(f"{'='*70}")
    print(f"\n  Exp 21 (Hebbian outer product):")
    print(f"    - Structure creates organization (CV: p=0.0002)")
    print(f"    - But NO cognitive advantage (conditioning, discrimination, memory: ns)")
    print(f"\n  Exp 22 (E-prop eligibility propagation):")
    print(f"    - Significant differences: {significant_wins}/{total_tests}")
    print(f"    - Connectome wins: {connectome_wins}/{total_tests}")

    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    if connectome_wins >= 3:
        print(f"\n  BRAIN STRUCTURE HELPS LEARNING.")
        print(f"  With a proper learning rule (e-prop), the human connectome")
        print(f"  provides a genuine advantage on {connectome_wins}/{total_tests} cognitive measures.")
        print(f"  This was NOT visible with the crude Hebbian rule in Exp 21.")
        print(f"  The 先天 (innate structure) × 后天 (learned calibration) hypothesis is SUPPORTED.")
    elif connectome_wins >= 1:
        print(f"\n  PARTIAL EVIDENCE that structure helps learning.")
        print(f"  {connectome_wins}/{total_tests} measures show connectome advantage with e-prop.")
    elif significant_wins >= 1:
        print(f"\n  STRUCTURE AFFECTS BUT DOESN'T HELP learning.")
        print(f"  Significant differences exist, but random wiring sometimes wins.")
    else:
        print(f"\n  LEARNING RULE DOESN'T UNLOCK STRUCTURE ADVANTAGE.")
        print(f"  Even with e-prop, random wiring performs equally well.")
        print(f"  The macro-scale connectome may be too coarse for learning benefits.")
        print(f"  Need synaptic-resolution data (like FlyWire) for structure to matter.")

    # Save
    results_dir = Path("results/exp22_eprop_connectome_vs_random")
    results_dir.mkdir(parents=True, exist_ok=True)

    save_data = {}
    for cond in results:
        save_data[cond] = {k: [float(x) for x in v] for k, v in results[cond].items()}

    with open(results_dir / "results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

"""Experiment 27: Connectome vs Random — WITH VALIDATED PARAMETERS.

This is the re-run of Experiment 22 using parameters that pass SC-FC
validation (gc=0.20, erf=3.5, SC-FC r=0.388).

All previous experiments (21-24) used gc=0.15 which produced unrealistic
dynamics (SC-FC r=0.074). Those results are suspect. This experiment
uses validated parameters where simulated FC actually reflects structural
connectivity.

Protocol:
  - 15 connectome brains vs 15 degree-preserving random
  - Phase 1: Innate regional differentiation
  - Phase 2: Classical conditioning (e-prop, continuous reward)
  - Phase 3: Pattern discrimination (after e-prop training)
  - Phase 4: Working memory persistence
  - FDR correction on all multi-comparison tests
  - No ALIF (known to interact badly — separate experiment)
  - Conduction delays + neuron types enabled

Parameters: gc=0.20, erf=3.5 (SC-FC validated r=0.388)
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
from encephagen.analysis.statistics import benjamini_hochberg, report_with_fdr


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
    """Create brain with SC-FC validated parameters."""
    groups = classify_regions(connectome.labels)
    pfc_regions = groups['prefrontal']
    brain = SpikingBrainGPU(
        connectome=connectome, neurons_per_region=200,
        global_coupling=0.20,       # SC-FC validated (was 0.15)
        ext_rate_factor=3.5,        # SC-FC validated
        tau_nmda=150.0, nmda_ratio=0.4,
        pfc_regions=pfc_regions, device=device,
        use_delays=True,
        conduction_velocity=3.5,
        use_neuron_types=True,
        use_adaptation=False,       # No ALIF — known interaction issue
    )
    return brain, groups


def warmup(brain, steps=3000):
    state = brain.init_state(batch_size=1)
    with torch.no_grad():
        for _ in range(steps):
            state, _ = brain.step(state)
    return state


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
    responses_over_time = []

    for trial in range(30):
        ext_cs = torch.zeros(1, n_total, device=device)
        pat_t = torch.tensor(pattern, device=device)
        for ri in vis_idx:
            ext_cs[0, ri*npr:(ri+1)*npr] = pat_t * 12.0

        for _ in range(500):
            state, spikes = brain.step(state, ext_cs)

        # US with continuous reward
        ext_us = torch.zeros(1, n_total, device=device)
        for ri in amyg_idx:
            ext_us[0, ri*npr:(ri+1)*npr] = 20.0
        for step in range(500):
            state, spikes = brain.step(state, ext_us)
            if step % 50 == 0:
                brain.apply_reward(spikes, reward=1.0)

        for _ in range(300):
            state, spikes = brain.step(state)

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

    if len(responses_over_time) > 2:
        x = np.arange(len(responses_over_time))
        slope, _, _, _, _ = stats.linregress(x, responses_over_time)
        final_response = responses_over_time[-1]
    else:
        slope = 0.0
        final_response = 0.0
    return state, slope, final_response


def phase3_discrimination(brain, state, groups, npr, n_total, device):
    vis_idx = groups['visual']
    readout_idx = groups.get('prefrontal', [])[:8] + groups.get('hippocampus', [])

    rng = np.random.default_rng(42)
    patterns = []
    for i in range(5):
        p = np.zeros(npr, dtype=np.float32)
        p[rng.choice(npr, 60, replace=False)] = 1.0
        patterns.append(p)

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


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 27: Validated Connectome vs Random")
    print("gc=0.20, erf=3.5 (SC-FC r=0.388) — delays + neuron types — no ALIF")
    print("15 runs per condition — FDR corrected")
    print("=" * 70)

    connectome = Connectome.from_bundled("tvb96")
    npr = 200
    n_total = 96 * npr
    device = "cuda"
    N_RUNS = 15

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

            brain, grp = make_brain(conn, device=device)
            state = warmup(brain, steps=3000)

            state, cv, _ = phase1_innate(brain, state, grp, npr, n_total, device)
            results[condition]["cv"].append(cv)

            state, slope, final_r = phase2_conditioning(
                brain, state, grp, npr, n_total, device)
            results[condition]["cond_slope"].append(slope)
            results[condition]["cond_final"].append(final_r)

            state, acc = phase3_discrimination(brain, state, grp, npr, n_total, device)
            results[condition]["discrim"].append(acc)

            state, persist, _ = phase4_memory(brain, state, grp, npr, n_total, device)
            results[condition]["memory"].append(persist)

            elapsed = time.time() - t0
            print(f"  Run {run+1:>2}/{N_RUNS}  cv={cv:.3f}  cond={final_r:.5f}  "
                  f"discrim={acc:.0%}  memory={persist:.0%}  ({elapsed:.0f}s)")

            del brain
            torch.cuda.empty_cache()

    # ========================================
    # Statistical comparison with FDR
    # ========================================
    print(f"\n{'='*70}")
    print("RESULTS: Connectome vs Random (SC-FC Validated)")
    print(f"{'='*70}")

    metrics = [
        ("Regional differentiation (CV)", "cv"),
        ("Conditioning speed (slope)", "cond_slope"),
        ("Conditioning strength (final)", "cond_final"),
        ("Pattern discrimination", "discrim"),
        ("Working memory (persistence)", "memory"),
    ]

    p_values = []
    labels_for_fdr = []

    print(f"\n  {'Metric':<40} {'Connectome':>12} {'Random':>12} {'p-value':>10} {'Winner':>12}")
    print(f"  {'─'*86}")

    for label, key in metrics:
        c_vals = np.array(results["connectome"][key])
        r_vals = np.array(results["random"][key])
        c_mean = c_vals.mean()
        r_mean = r_vals.mean()
        stat, p = stats.mannwhitneyu(c_vals, r_vals, alternative="two-sided")
        p_values.append(p)
        labels_for_fdr.append(label)

        winner = ""
        if p < 0.05:
            winner = "CONNECTOME" if c_mean > r_mean else "RANDOM"
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label:<40} {c_mean:>12.5f} {r_mean:>12.5f} {p:>9.4f}{sig:>1} {winner:>12}")

        # Effect size
        pooled_std = np.sqrt((c_vals.var() + r_vals.var()) / 2)
        d = (c_mean - r_mean) / (pooled_std + 1e-10)
        size = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small" if abs(d) > 0.2 else "neg."
        print(f"  {'':40} {'Cohen d':>12} = {d:+.3f} ({size})")

    # FDR correction
    print(f"\n{report_with_fdr(labels_for_fdr, p_values)}")

    # Verdict
    survives = benjamini_hochberg(p_values)
    connectome_fdr_wins = 0
    for i, (label, key) in enumerate(metrics):
        if survives[i]:
            c_mean = np.mean(results["connectome"][key])
            r_mean = np.mean(results["random"][key])
            if c_mean > r_mean:
                connectome_fdr_wins += 1

    print(f"\n{'='*70}")
    print("VERDICT (FDR-corrected)")
    print(f"{'='*70}")
    print(f"\n  Connectome FDR-significant wins: {connectome_fdr_wins}/{len(metrics)}")

    if connectome_fdr_wins >= 3:
        print(f"  STRONG: Structure helps learning with validated dynamics.")
    elif connectome_fdr_wins >= 1:
        print(f"  PARTIAL: Structure provides selective advantages.")
    else:
        print(f"  NONE: No FDR-surviving connectome advantage.")

    # Save
    results_dir = Path("results/exp27_validated")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_data = {}
    for cond in results:
        save_data[cond] = {k: [float(x) for x in v] for k, v in results[cond].items()}
    with open(results_dir / "results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

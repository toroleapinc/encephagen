"""Experiment 29: Connectome vs Random — on VALIDATED human brain dynamics.

This is the real test. Everything before this was on bad data (tvb96: 3 weight values)
or unrealistic parameters (SC-FC < 0.1). Now we have:
  - Neurolib 80-region HCP data (6,220 unique continuous weights, 42M x dynamic range)
  - Real tract lengths for conduction delays
  - Empirical fMRI FC for validation (FC-FC = 0.42 at gc=10)
  - Connectome-dominant architecture (long-range boost, inhibitory BG pathways)
  - FDR correction on all comparisons

The question remains: does human brain structure help cognition?
With validated dynamics that actually match real fMRI, the answer matters.
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


def load_neurolib80():
    """Load neurolib 80-region connectome with tract lengths."""
    sc = np.load('src/encephagen/connectome/bundled/neurolib80_weights.npy')
    tl = np.load('src/encephagen/connectome/bundled/neurolib80_tract_lengths.npy')
    labels = json.load(open('src/encephagen/connectome/bundled/neurolib80_labels.json'))
    c = Connectome(sc, labels)
    c.tract_lengths = tl
    return c


def randomize_connectome(connectome, seed):
    """Degree-preserving rewiring."""
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
    result = Connectome(w, list(connectome.labels))
    result.tract_lengths = connectome.tract_lengths
    return result


def make_brain(connectome, device="cuda"):
    """Create brain with validated parameters."""
    brain = SpikingBrainGPU(
        connectome=connectome, neurons_per_region=200,
        internal_conn_prob=0.05,
        between_conn_prob=0.03,
        global_coupling=10.0,       # FC-FC(emp) = 0.42 at this value
        ext_rate_factor=3.5,
        pfc_regions=[], device=device,
        use_delays=True,
        conduction_velocity=3.5,
        use_neuron_types=False,      # AAL2 labels don't classify well
        use_adaptation=False,        # Known to interact badly
    )
    return brain


def warmup(brain, steps=3000):
    state = brain.init_state(batch_size=1)
    with torch.no_grad():
        for _ in range(steps):
            state, _ = brain.step(state)
    return state


def phase1_innate(brain, state, n_regions, npr, n_total, device):
    """Regional differentiation."""
    steps = 2000
    region_spikes = torch.zeros(n_regions, device=device)
    with torch.no_grad():
        for _ in range(steps):
            state, spikes = brain.step(state)
            for r in range(n_regions):
                region_spikes[r] += spikes[0, r*npr:(r+1)*npr].sum()
    rates = (region_spikes / (npr * steps)).cpu().numpy()
    cv = float(np.std(rates) / (np.mean(rates) + 1e-12))
    return state, cv


def phase2_conditioning(brain, state, n_regions, npr, n_total, device):
    """Classical conditioning with e-prop."""
    # Use first 4 regions as "visual", region 40 as "reward"
    vis_idx = list(range(4))
    reward_idx = [40, 41]
    response_idx = reward_idx + list(range(20, 24))

    rng = np.random.default_rng(42)
    pattern = np.zeros(npr, dtype=np.float32)
    pattern[rng.choice(npr, 60, replace=False)] = 1.0

    eprop_params = EpropParams(
        lr=0.1, tau_e=50.0, gamma=0.5, w_max=15.0,
        regularization=0.0, reward_decay=0.99,
    )
    learner = brain.enable_learning(eprop_params)
    responses = []

    for trial in range(30):
        # CS
        ext_cs = torch.zeros(1, n_total, device=device)
        pat_t = torch.tensor(pattern, device=device)
        for ri in vis_idx:
            ext_cs[0, ri*npr:(ri+1)*npr] = pat_t * 12.0
        for _ in range(500):
            state, spikes = brain.step(state, ext_cs)

        # US with continuous reward
        ext_us = torch.zeros(1, n_total, device=device)
        for ri in reward_idx:
            ext_us[0, ri*npr:(ri+1)*npr] = 20.0
        for step in range(500):
            state, spikes = brain.step(state, ext_us)
            if step % 50 == 0:
                brain.apply_reward(spikes, reward=1.0)

        # Gap
        for _ in range(300):
            state, _ = brain.step(state)

        # Test every 10 trials
        if (trial + 1) % 10 == 0:
            brain.learner = None
            ext_test = torch.zeros(1, n_total, device=device)
            for ri in vis_idx:
                ext_test[0, ri*npr:(ri+1)*npr] = pat_t * 12.0
            bl = torch.zeros(len(response_idx), device=device)
            ts = torch.zeros(len(response_idx), device=device)
            with torch.no_grad():
                for _ in range(300):
                    state, spikes = brain.step(state)
                    for j, ri in enumerate(response_idx):
                        bl[j] += spikes[0, ri*npr:(ri+1)*npr].sum()
                for _ in range(500):
                    state, spikes = brain.step(state, ext_test)
                    for j, ri in enumerate(response_idx):
                        ts[j] += spikes[0, ri*npr:(ri+1)*npr].sum()
            responses.append(ts.cpu().numpy().mean()/(npr*500) - bl.cpu().numpy().mean()/(npr*300))
            brain.learner = learner

    brain.learner = None
    final = responses[-1] if responses else 0.0
    return state, final


def phase3_discrimination(brain, state, n_regions, npr, n_total, device):
    """Pattern discrimination with e-prop training."""
    vis_idx = list(range(8))
    readout_idx = list(range(20, 30))

    rng = np.random.default_rng(42)
    patterns = []
    for i in range(5):
        p = np.zeros(npr, dtype=np.float32)
        p[rng.choice(npr, 60, replace=False)] = 1.0
        patterns.append(p)

    # Train
    eprop_params = EpropParams(lr=0.1, tau_e=50.0, gamma=0.5, w_max=15.0,
                               regularization=0.0, reward_decay=0.99)
    learner = brain.enable_learning(eprop_params)
    for epoch in range(3):
        for p_idx in range(5):
            ext = torch.zeros(1, n_total, device=device)
            pat_t = torch.tensor(patterns[p_idx], device=device)
            for ri in vis_idx:
                ext[0, ri*npr:(ri+1)*npr] = pat_t * 12.0
            for step in range(500):
                state, spikes = brain.step(state, ext)
                if step % 100 == 0:
                    brain.apply_reward(spikes, reward=0.5)
            for _ in range(200):
                state, _ = brain.step(state)
    brain.learner = None

    # Test
    responses = {i: [] for i in range(5)}
    for trial in range(5):
        for p_idx in range(5):
            with torch.no_grad():
                for _ in range(200):
                    state, _ = brain.step(state)
            ext = torch.zeros(1, n_total, device=device)
            pat_t = torch.tensor(patterns[p_idx], device=device)
            for ri in vis_idx:
                ext[0, ri*npr:(ri+1)*npr] = pat_t * 12.0
            resp = torch.zeros(len(readout_idx), device=device)
            with torch.no_grad():
                for _ in range(500):
                    state, spikes = brain.step(state, ext)
                    for j, ri in enumerate(readout_idx):
                        resp[j] += spikes[0, ri*npr:(ri+1)*npr].sum()
            responses[p_idx].append(resp.cpu().numpy())

    correct = 0; total = 0
    for p in range(5):
        for t in range(5):
            tv = responses[p][t]
            best_s, best_c = -1, -1
            for r in range(5):
                rv = [responses[r][tt] for tt in range(5) if not (r == p and tt == t)]
                rm = np.mean(rv, axis=0)
                nt, nr = np.linalg.norm(tv), np.linalg.norm(rm)
                s = np.dot(tv, rm)/(nt*nr) if nt > 0 and nr > 0 else 0
                if s > best_s: best_s, best_c = s, r
            if best_c == p: correct += 1
            total += 1
    return state, correct / total


def phase4_memory(brain, state, n_regions, npr, n_total, device):
    """Working memory (NMDA persistence)."""
    vis_idx = list(range(8))
    pfc_idx = list(range(20, 30))

    bl = 0.0
    with torch.no_grad():
        for _ in range(500):
            state, spikes = brain.step(state)
            for ri in pfc_idx:
                bl += spikes[0, ri*npr:(ri+1)*npr].sum().item()
    bl /= (len(pfc_idx) * npr * 500)

    ext = torch.zeros(1, n_total, device=device)
    for ri in vis_idx:
        ext[0, ri*npr:(ri+1)*npr] = 15.0
    stim = 0.0
    with torch.no_grad():
        for _ in range(500):
            state, spikes = brain.step(state, ext)
            for ri in pfc_idx:
                stim += spikes[0, ri*npr:(ri+1)*npr].sum().item()
    stim /= (len(pfc_idx) * npr * 500)

    delay = 0.0
    with torch.no_grad():
        for _ in range(500):
            state, spikes = brain.step(state)
            for ri in pfc_idx:
                delay += spikes[0, ri*npr:(ri+1)*npr].sum().item()
    delay /= (len(pfc_idx) * npr * 500)

    sc = stim - bl
    persist = (delay - bl) / (sc + 1e-10) if sc > 0 else 0
    return state, persist


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 29: Connectome vs Random — Empirically Validated")
    print("Neurolib 80 regions, continuous weights, FC-FC(emp)=0.42")
    print("15 runs per condition, FDR corrected")
    print("=" * 70)

    connectome = load_neurolib80()
    npr = 200; n_regions = 80; n_total = n_regions * npr
    device = "cuda"; N_RUNS = 15

    results = {cond: {"cv": [], "cond": [], "discrim": [], "memory": []}
               for cond in ["connectome", "random"]}
    t0 = time.time()

    for cond in ["connectome", "random"]:
        print(f"\n{'='*60}")
        print(f"CONDITION: {cond.upper()} ({N_RUNS} runs)")
        print(f"{'='*60}")

        for run in range(N_RUNS):
            if cond == "connectome":
                conn = connectome
            else:
                conn = randomize_connectome(connectome, seed=1000 + run)

            brain = make_brain(conn, device=device)
            state = warmup(brain, steps=3000)

            state, cv = phase1_innate(brain, state, n_regions, npr, n_total, device)
            results[cond]["cv"].append(cv)

            state, cond_r = phase2_conditioning(brain, state, n_regions, npr, n_total, device)
            results[cond]["cond"].append(cond_r)

            state, acc = phase3_discrimination(brain, state, n_regions, npr, n_total, device)
            results[cond]["discrim"].append(acc)

            state, persist = phase4_memory(brain, state, n_regions, npr, n_total, device)
            results[cond]["memory"].append(persist)

            elapsed = time.time() - t0
            print(f"  Run {run+1:>2}/{N_RUNS}  cv={cv:.3f}  cond={cond_r:.5f}  "
                  f"discrim={acc:.0%}  memory={persist:.0%}  ({elapsed:.0f}s)")

            del brain; torch.cuda.empty_cache()

    # Statistics
    print(f"\n{'='*70}")
    print("RESULTS: Connectome vs Random (Empirically Validated)")
    print(f"{'='*70}")

    metrics = [
        ("Regional differentiation (CV)", "cv"),
        ("Conditioning strength", "cond"),
        ("Pattern discrimination", "discrim"),
        ("Working memory", "memory"),
    ]

    p_values = []; labels_fdr = []
    print(f"\n  {'Metric':<35} {'Connectome':>12} {'Random':>12} {'p-value':>10} {'d':>8} {'Winner':>12}")
    print(f"  {'─'*86}")

    for label, key in metrics:
        c = np.array(results["connectome"][key])
        r = np.array(results["random"][key])
        _, p = stats.mannwhitneyu(c, r, alternative="two-sided")
        p_values.append(p); labels_fdr.append(label)
        pooled = np.sqrt((c.var() + r.var()) / 2)
        d = (c.mean() - r.mean()) / (pooled + 1e-10)
        winner = ""
        if p < 0.05:
            winner = "CONNECTOME" if c.mean() > r.mean() else "RANDOM"
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label:<35} {c.mean():>12.5f} {r.mean():>12.5f} {p:>9.4f}{sig:>1} {d:>+8.3f} {winner:>12}")

    print(f"\n{report_with_fdr(labels_fdr, p_values)}")

    survives = benjamini_hochberg(p_values)
    c_wins = sum(1 for i, (_, k) in enumerate(metrics)
                 if survives[i] and np.mean(results["connectome"][k]) > np.mean(results["random"][k]))

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    print(f"\n  FDR-corrected connectome wins: {c_wins}/{len(metrics)}")
    if c_wins >= 3:
        print("  STRONG: Structure helps with empirically validated dynamics.")
    elif c_wins >= 1:
        print("  PARTIAL: Structure helps selectively.")
    else:
        print("  NONE: No structural advantage even with validated dynamics.")

    # Save
    results_dir = Path("results/exp29_neurolib80")
    results_dir.mkdir(parents=True, exist_ok=True)
    save = {c: {k: [float(x) for x in v] for k, v in d.items()} for c, d in results.items()}
    with open(results_dir / "results.json", "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

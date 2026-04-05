"""Experiment 31: Does brain structure help LEARNING?

The 先天 × 后天 test: two identical brains (same neurons, same learning rule,
same experiences) — one wired by the human connectome, one wired randomly.
Train both on the same task. Which learns faster?

This is fundamentally different from Experiments 21-29 which tested INNATE
performance. The connectome's role might not be to compute better at birth,
but to provide a better SCAFFOLD for learning.

Task: Stimulus-action mapping with delayed reward
  - 3 visual patterns (A, B, C) presented to visual cortex
  - Correct response: A→motor region 1 active, B→region 2, C→region 3
  - Reward delivered 500ms after action (delayed — needs temporal credit)
  - E-prop eligibility traces carry credit through the delay
  - Measure: learning curve (accuracy over training episodes)

If connectome brain learns faster → structure provides learning scaffold.
If random learns equally fast → structure doesn't help learning either.
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


def make_brain(conn, device="cuda"):
    return SpikingBrainGPU(
        connectome=conn, neurons_per_region=200,
        internal_conn_prob=0.05, between_conn_prob=0.03,
        global_coupling=5.0, ext_rate_factor=3.5,
        pfc_regions=[], device=device,
        use_delays=True, conduction_velocity=3.5,
        use_t1t2_gradient=True,
    )


def train_and_evaluate(brain, n_episodes=60, device="cuda"):
    """Train on stimulus-action mapping task, return learning curve.

    Task: 3 patterns → 3 motor outputs
    Each episode: present pattern, let brain respond, reward correct output
    """
    npr = 200; n_regions = 80; n_total = n_regions * npr
    rng = np.random.default_rng(42)

    # Define stimulus patterns (3 patterns in "visual" regions 0-7)
    patterns = []
    for i in range(3):
        p = np.zeros(npr, dtype=np.float32)
        p[rng.choice(npr, 80, replace=False)] = 1.0
        patterns.append(p)

    vis_regions = list(range(8))          # first 8 regions = "visual"
    motor_regions = [70, 72, 74]          # 3 motor output regions
    readout_regions = motor_regions

    # Enable learning
    eprop_params = EpropParams(
        lr=0.1, tau_e=50.0, gamma=0.5, w_max=15.0,
        regularization=0.0, reward_decay=0.99,
    )
    learner = brain.enable_learning(eprop_params)

    state = brain.init_state(batch_size=1)
    # Warmup
    with torch.no_grad():
        for _ in range(2000):
            state, _ = brain.step(state)

    learning_curve = []
    correct_count = 0
    total_count = 0

    for episode in range(n_episodes):
        # Pick random pattern
        p_idx = rng.integers(3)
        correct_motor = motor_regions[p_idx]

        # Present stimulus (500 steps = 50ms)
        ext = torch.zeros(1, n_total, device=device)
        pat_t = torch.tensor(patterns[p_idx], device=device)
        for ri in vis_regions:
            ext[0, ri*npr:(ri+1)*npr] = pat_t * 50.0

        for _ in range(500):
            state, spikes = brain.step(state, ext)

        # Read motor output (which motor region is most active?)
        motor_activity = []
        for ri in motor_regions:
            motor_activity.append(spikes[0, ri*npr:(ri+1)*npr].sum().item())

        chosen = motor_regions[np.argmax(motor_activity)]
        is_correct = (chosen == correct_motor)
        if is_correct:
            correct_count += 1
        total_count += 1

        # Reward (positive if correct, negative if wrong)
        reward = 1.0 if is_correct else -0.5

        # Delay period (300 steps = 30ms) — eligibility traces carry credit
        for step in range(300):
            state, spikes = brain.step(state)

        # Apply reward
        brain.apply_reward(spikes, reward)

        # Gap between episodes
        for _ in range(200):
            state, _ = brain.step(state)

        # Record accuracy every 10 episodes
        if (episode + 1) % 10 == 0:
            acc = correct_count / total_count
            learning_curve.append(acc)
            correct_count = 0
            total_count = 0

    brain.learner = None
    return learning_curve


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 31: Does brain structure help LEARNING?")
    print("先天 × 后天: Connectome as learning scaffold")
    print("Stimulus-action mapping, e-prop, 60 episodes")
    print("=" * 70)

    conn = load_neurolib80()
    device = "cuda"
    N_RUNS = 10
    N_EPISODES = 60

    results = {"connectome": [], "random": []}
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
            curve = train_and_evaluate(brain, n_episodes=N_EPISODES, device=device)
            results[cond].append(curve)

            elapsed = time.time() - t0
            final_acc = curve[-1] if curve else 0
            print(f"  Run {run+1:>2}/{N_RUNS}  final_acc={final_acc:.0%}  "
                  f"curve={[f'{x:.0%}' for x in curve]}  ({elapsed:.0f}s)")

            del brain; torch.cuda.empty_cache()

    # ========================================
    # Analysis
    # ========================================
    print(f"\n{'='*70}")
    print("RESULTS: Learning Speed Comparison")
    print(f"{'='*70}")

    # Compare at each checkpoint
    n_checkpoints = len(results["connectome"][0])
    print(f"\n  {'Checkpoint':>12} {'Connectome':>12} {'Random':>12} {'p':>8} {'Winner':>10}")
    print(f"  {'─'*56}")

    p_vals = []; p_labels = []
    for cp in range(n_checkpoints):
        c_accs = [r[cp] for r in results["connectome"]]
        r_accs = [r[cp] for r in results["random"]]
        c_mean = np.mean(c_accs)
        r_mean = np.mean(r_accs)
        _, p = stats.mannwhitneyu(c_accs, r_accs, alternative="two-sided")
        p_vals.append(p); p_labels.append(f"Episode {(cp+1)*10}")
        winner = ""
        if p < 0.05:
            winner = "CONNECTOME" if c_mean > r_mean else "RANDOM"
        sig = "*" if p < 0.05 else ""
        print(f"  Episode {(cp+1)*10:>3}  {c_mean:>12.1%} {r_mean:>12.1%} {p:>8.4f}{sig} {winner:>10}")

    # Overall learning speed (area under learning curve)
    c_auc = [np.mean(r) for r in results["connectome"]]
    r_auc = [np.mean(r) for r in results["random"]]
    _, p_auc = stats.mannwhitneyu(c_auc, r_auc, alternative="two-sided")

    print(f"\n  Mean accuracy (AUC): conn={np.mean(c_auc):.1%} vs rand={np.mean(r_auc):.1%}, p={p_auc:.4f}")

    # FDR
    p_vals.append(p_auc)
    p_labels.append("AUC (overall)")
    print(f"\n{report_with_fdr(p_labels, p_vals)}")

    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT: 先天 × 后天")
    print(f"{'='*70}")
    if np.mean(c_auc) > np.mean(r_auc) and p_auc < 0.05:
        print(f"\n  CONNECTOME LEARNS FASTER.")
        print(f"  The human brain structure provides a learning scaffold.")
    elif np.mean(r_auc) > np.mean(c_auc) and p_auc < 0.05:
        print(f"\n  RANDOM LEARNS FASTER.")
        print(f"  Structure constrains learning, doesn't help it.")
    else:
        print(f"\n  NO DIFFERENCE in learning speed (p={p_auc:.4f}).")

    # Save
    results_dir = Path("results/exp31_learning_scaffold")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f)
    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

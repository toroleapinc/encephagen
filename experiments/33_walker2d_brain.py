"""Experiment 33: Brain-controlled Walker2d — does structure help motor learning?

MuJoCo Walker2d is properly unstable — falls at step ~119 with zero action.
The brain must ACTIVELY control it. This is where structure should matter:
the connectome routes sensory→motor through specific pathways with specific
timing (fast sensory tau=10ms, medium motor tau=20ms).

Architecture:
  Walker2d obs (17-dim) → somatosensory regions (fast)
  Brain processes through connectome pathways
  Motor cortex (6 groups) → 6 joint torques
  E-prop learns from reward (staying upright + moving forward)

BT-SNN approach (Zhao et al. 2024): connectome as RL architecture.
Compare learning curves: connectome vs random with same T1w/T2w gradient.
"""

import time
import json
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
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


def classify_regions():
    tau_labels = json.load(open(
        'src/encephagen/connectome/bundled/neurolib80_t1t2_labels.json'))
    groups = {}
    for key, patterns in [
        ('somatosensory', ['Postcentral', 'Paracentral']),
        ('motor', ['Precentral', 'Supp_Motor']),
    ]:
        groups[key] = [i for i, l in enumerate(tau_labels)
                       if any(p in l for p in patterns)]
    return groups


def run_walker_episode(brain, state, groups, env, n_regions, npr, device,
                       enable_learning=False, learner=None):
    """Run one Walker2d episode with brain control.

    Maps:
      obs[0:6] (positions) → somatosensory regions (proportional current)
      obs[6:17] (velocities) → somatosensory regions (proportional current)
      motor cortex firing (6 groups) → 6 joint torques [-1, 1]
    """
    n_total = n_regions * npr
    soma_idx = groups['somatosensory']
    motor_idx = groups['motor']

    # Precompute motor neuron index tensors for vectorized readout
    all_motor_neurons = []
    for ri in motor_idx:
        all_motor_neurons.extend(range(ri * npr, (ri + 1) * npr))
    n_motor = len(all_motor_neurons)
    neurons_per_action = n_motor // 6
    motor_tensor_idx = torch.tensor(all_motor_neurons, device=device, dtype=torch.long)

    # Precompute soma input indices
    soma_neuron_starts = [ri * npr for ri in soma_idx]

    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0

    brain_steps = 20  # 2ms brain time per env step

    for env_step in range(500):
        # === SENSE: observation → somatosensory (vectorized) ===
        ext = torch.zeros(1, n_total, device=device)
        obs_norm = np.clip(obs / 5.0, -1, 1)
        obs_abs = np.abs(obs_norm).astype(np.float32)
        mean_signal = float(obs_abs.mean()) * 30.0
        for s_start in soma_neuron_starts:
            ext[0, s_start:s_start + npr] = mean_signal

        # === PROCESS: brain runs (vectorized motor readout) ===
        motor_acc = torch.zeros(n_motor, device=device)
        with torch.no_grad():
            for _ in range(brain_steps):
                state, spikes = brain.step(state, ext)
                motor_acc += spikes[0, motor_tensor_idx].float()

        # === ACT: motor spikes → 6 joint torques ===
        action = np.zeros(6, dtype=np.float32)
        motor_np = motor_acc.cpu().numpy()
        for a in range(6):
            chunk = motor_np[a * neurons_per_action:(a + 1) * neurons_per_action]
            rate = chunk.sum() / (neurons_per_action * brain_steps)
            action[a] = np.clip((rate - 0.04) * 50.0, -1.0, 1.0)

        # === STEP environment ===
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # === LEARN: reward modulates eligibility ===
        if enable_learning and learner is not None:
            # Reward: positive for staying upright and moving forward
            brain.apply_reward(spikes, reward=reward * 0.1)

        if terminated or truncated:
            break

    return steps, total_reward


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 33: Brain-Controlled Walker2d")
    print("Does connectome help learn motor control?")
    print("BT-SNN approach: connectome as RL architecture")
    print("=" * 70)

    conn = load_neurolib80()
    groups = classify_regions()
    n_regions = 80; npr = 200; device = "cuda"

    N_RUNS = 5
    N_EPISODES = 20  # episodes per run (learning over episodes)

    results = {"connectome": [], "random": []}
    t0 = time.time()

    for cond in ["connectome", "random"]:
        print(f"\n{'='*60}")
        print(f"CONDITION: {cond.upper()} ({N_RUNS} runs × {N_EPISODES} episodes)")
        print(f"{'='*60}")

        for run in range(N_RUNS):
            if cond == "connectome":
                c = conn
            else:
                c = randomize(conn, seed=1000 + run)

            brain = SpikingBrainGPU(
                connectome=c, neurons_per_region=npr,
                internal_conn_prob=0.05, between_conn_prob=0.03,
                global_coupling=5.0, ext_rate_factor=3.5,
                pfc_regions=[], device=device,
                use_delays=True, conduction_velocity=3.5,
                use_t1t2_gradient=True,
            )

            # Enable learning
            eprop_params = EpropParams(
                lr=0.05, tau_e=50.0, gamma=0.5, w_max=15.0,
                regularization=0.0, reward_decay=0.99,
            )
            learner = brain.enable_learning(eprop_params)

            state = brain.init_state(batch_size=1)
            # Warmup brain
            with torch.no_grad():
                for _ in range(2000):
                    state, _ = brain.step(state)

            env = gym.make('Walker2d-v5')
            episode_steps = []
            episode_rewards = []

            for ep in range(N_EPISODES):
                steps, reward = run_walker_episode(
                    brain, state, groups, env, n_regions, npr, device,
                    enable_learning=True, learner=learner,
                )
                episode_steps.append(steps)
                episode_rewards.append(reward)

            env.close()

            # Summary
            early = np.mean(episode_steps[:5])
            late = np.mean(episode_steps[-5:])
            improvement = late - early

            results[cond].append({
                'steps': episode_steps,
                'rewards': episode_rewards,
                'early_steps': early,
                'late_steps': late,
                'improvement': improvement,
                'mean_steps': np.mean(episode_steps),
                'mean_reward': np.mean(episode_rewards),
            })

            elapsed = time.time() - t0
            print(f"  Run {run+1:>2}/{N_RUNS}  "
                  f"early={early:.0f} late={late:.0f} improve={improvement:+.0f}  "
                  f"mean_steps={np.mean(episode_steps):.0f}  "
                  f"mean_reward={np.mean(episode_rewards):.1f}  ({elapsed:.0f}s)")

            del brain; torch.cuda.empty_cache()

    # ========================================
    # Analysis
    # ========================================
    print(f"\n{'='*70}")
    print("RESULTS: Walker2d Motor Learning")
    print(f"{'='*70}")

    metrics = [
        ("Mean survival (steps)", "mean_steps"),
        ("Mean reward", "mean_reward"),
        ("Early performance (steps)", "early_steps"),
        ("Late performance (steps)", "late_steps"),
        ("Improvement (late-early)", "improvement"),
    ]

    p_vals = []; p_labels = []
    print(f"\n  {'Metric':<30} {'Connectome':>12} {'Random':>12} {'p':>8} {'d':>8}")
    print(f"  {'─'*72}")

    for label, key in metrics:
        c_vals = [r[key] for r in results["connectome"]]
        r_vals = [r[key] for r in results["random"]]
        c_mean = np.mean(c_vals)
        r_mean = np.mean(r_vals)
        _, p = stats.mannwhitneyu(c_vals, r_vals, alternative="two-sided")
        p_vals.append(p); p_labels.append(label)
        pooled = np.sqrt((np.var(c_vals) + np.var(r_vals)) / 2)
        d = (c_mean - r_mean) / (pooled + 1e-10)
        sig = "*" if p < 0.05 else ""
        print(f"  {label:<30} {c_mean:>12.1f} {r_mean:>12.1f} {p:>8.4f} {d:>+8.3f}{sig}")

    print(f"\n{report_with_fdr(p_labels, p_vals)}")

    # Learning curves
    print(f"\n  Learning curves (mean steps per episode):")
    print(f"  {'Episode':>8} {'Connectome':>12} {'Random':>12}")
    for ep in range(N_EPISODES):
        c_ep = np.mean([r['steps'][ep] for r in results["connectome"]])
        r_ep = np.mean([r['steps'][ep] for r in results["random"]])
        print(f"  {ep+1:>8} {c_ep:>12.1f} {r_ep:>12.1f}")

    # Zero-action baseline
    print(f"\n  Baseline (zero action): ~119 steps before falling")

    # Verdict
    c_mean = np.mean([r['mean_steps'] for r in results["connectome"]])
    r_mean = np.mean([r['mean_steps'] for r in results["random"]])
    _, p_main = stats.mannwhitneyu(
        [r['mean_steps'] for r in results["connectome"]],
        [r['mean_steps'] for r in results["random"]],
        alternative="two-sided"
    )

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    if c_mean > 119 and c_mean > r_mean and p_main < 0.05:
        print(f"  CONNECTOME HELPS MOTOR CONTROL.")
        print(f"  Brain-controlled Walker2d survives {c_mean:.0f} steps (vs {r_mean:.0f} random, baseline 119)")
    elif c_mean > 119:
        print(f"  Brain helps ({c_mean:.0f} > 119 baseline) but no structural advantage (p={p_main:.4f})")
    else:
        print(f"  Brain doesn't help (conn={c_mean:.0f}, rand={r_mean:.0f}, baseline=119)")

    # Save
    results_dir = Path("results/exp33_walker2d")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f)
    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

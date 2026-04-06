"""Experiment 34: Pure Brain Controller — no cheating.

Remove ALL hand-coded reflexes. No CPG. No PD controller.
The brain's spiking activity IS the motor command.

Architecture:
  Walker2d obs (17-dim) → somatosensory regions → [brain connectome] → motor cortex
  Motor cortex firing rates → 6 joint torques directly

If the brain can keep Walker2d alive longer than 119 steps (zero-action baseline),
the 16K spiking neurons are producing useful motor control from neural dynamics alone.

Then compare: connectome vs random. Does the wiring matter when the brain
IS the controller?
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


class PureBrainController:
    """Brain directly controls body. No CPG. No reflexes. No cheating."""

    def __init__(self, conn, device="cuda"):
        tau_labels = json.load(open(
            'src/encephagen/connectome/bundled/neurolib80_t1t2_labels.json'))

        self.npr = 200
        self.n_regions = 80
        self.n_total = self.n_regions * self.npr
        self.device = device

        # Somatosensory regions (receive body state)
        soma_regions = [i for i, l in enumerate(tau_labels)
                        if 'Postcentral' in l or 'Paracentral' in l]
        self.soma_starts = [ri * self.npr for ri in soma_regions]

        # Motor regions (produce joint torques)
        motor_regions = [i for i, l in enumerate(tau_labels)
                         if 'Precentral' in l or 'Supp_Motor' in l]
        all_motor = []
        for ri in motor_regions:
            all_motor.extend(range(ri * self.npr, (ri + 1) * self.npr))
        self.motor_idx = torch.tensor(all_motor, device=device, dtype=torch.long)
        self.neurons_per_action = len(all_motor) // 6

        # Build brain
        self.brain = SpikingBrainGPU(
            connectome=conn, neurons_per_region=self.npr,
            internal_conn_prob=0.05, between_conn_prob=0.03,
            global_coupling=5.0, ext_rate_factor=3.5,
            pfc_regions=[], device=device,
            use_delays=True, conduction_velocity=3.5,
            use_t1t2_gradient=True,
        )

        # Warmup
        self.state = self.brain.init_state(batch_size=1)
        with torch.no_grad():
            for _ in range(2000):
                self.state, _ = self.brain.step(self.state)

        # Calibrate: measure baseline motor cortex rate
        motor_total = 0
        with torch.no_grad():
            for _ in range(500):
                self.state, spikes = self.brain.step(self.state)
                motor_total += spikes[0, self.motor_idx].float().sum().item()
        self.baseline_rate = motor_total / (len(all_motor) * 500)

    def step(self, obs, brain_steps=20):
        """Sense → Think → Act. Pure brain, no tricks.

        obs: Walker2d 17-dim observation
        Returns: 6-dim action [-1, 1]
        """
        # === SENSE ===
        # Encode body state as current to somatosensory
        # Each obs dimension maps to a proportion of somatosensory neurons
        ext = torch.zeros(1, self.n_total, device=self.device)

        # Normalize obs roughly
        obs_scaled = np.clip(obs / 3.0, -1, 1)

        # Distribute all 17 obs dimensions across somatosensory neurons
        for s_start in self.soma_starts:
            for j, val in enumerate(obs_scaled):
                n_start = s_start + (j * self.npr // len(obs_scaled))
                n_end = s_start + ((j + 1) * self.npr // len(obs_scaled))
                # Positive obs → positive current. Negative obs → negative (inhibition).
                # This preserves the SIGN of the observation.
                ext[0, n_start:n_end] = float(val) * 30.0

        # === THINK ===
        motor_acc = torch.zeros(len(self.motor_idx), device=self.device)
        with torch.no_grad():
            for _ in range(brain_steps):
                self.state, spikes = self.brain.step(self.state, ext)
                motor_acc += spikes[0, self.motor_idx].float()

        # === ACT ===
        # Motor cortex firing rate → joint torque
        # Rate above baseline → positive torque
        # Rate below baseline → negative torque
        action = np.zeros(6, dtype=np.float32)
        motor_np = motor_acc.cpu().numpy()
        for a in range(6):
            chunk = motor_np[a * self.neurons_per_action:(a + 1) * self.neurons_per_action]
            rate = chunk.sum() / (self.neurons_per_action * brain_steps)
            # Map: baseline → 0, higher → positive, lower → negative
            action[a] = np.clip((rate - self.baseline_rate) * 40.0, -1.0, 1.0)

        return action


def run_episode(controller, max_steps=500):
    """Run one Walker2d episode with pure brain control."""
    env = gym.make('Walker2d-v5')
    obs, _ = env.reset()

    total_reward = 0.0
    for step in range(max_steps):
        action = controller.step(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    env.close()
    return step + 1, total_reward


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 34: Pure Brain Controller")
    print("NO CPG. NO reflexes. NO hand-coded anything.")
    print("Brain spiking → joint torques directly.")
    print("=" * 70)
    print(f"\n  Baseline (zero action): ~119 steps")

    conn = load_neurolib80()
    N_RUNS = 8

    results = {"connectome": [], "random": [], "zero": []}
    t0 = time.time()

    # Zero-action baseline
    print(f"\n  Zero-action baseline:")
    env = gym.make('Walker2d-v5')
    for run in range(5):
        obs, _ = env.reset()
        for step in range(500):
            obs, _, term, trunc, _ = env.step([0]*6)
            if term or trunc: break
        results["zero"].append({"steps": step+1})
        print(f"    Run {run+1}: {step+1} steps")
    env.close()
    zero_mean = np.mean([r["steps"] for r in results["zero"]])
    print(f"    Mean: {zero_mean:.0f} steps")

    for cond in ["connectome", "random"]:
        print(f"\n{'='*50}")
        print(f"  {cond.upper()} ({N_RUNS} runs)")
        print(f"{'='*50}")

        for run in range(N_RUNS):
            if cond == "connectome":
                c = conn
            else:
                c = randomize(conn, seed=1000 + run)

            controller = PureBrainController(c)
            steps, reward = run_episode(controller, max_steps=500)
            results[cond].append({"steps": steps, "reward": reward})

            elapsed = time.time() - t0
            vs = "BETTER" if steps > zero_mean else "WORSE"
            print(f"    Run {run+1:>2}/{N_RUNS}  steps={steps:>4}  "
                  f"reward={reward:>6.1f}  {vs} than zero  ({elapsed:.0f}s)")

            del controller
            torch.cuda.empty_cache()

    # ========================================
    # Analysis
    # ========================================
    print(f"\n{'='*70}")
    print("RESULTS: Pure Brain Controller")
    print(f"{'='*70}")

    c_steps = [r["steps"] for r in results["connectome"]]
    r_steps = [r["steps"] for r in results["random"]]
    c_reward = [r["reward"] for r in results["connectome"]]
    r_reward = [r["reward"] for r in results["random"]]

    _, p_steps = stats.mannwhitneyu(c_steps, r_steps, alternative="two-sided")
    _, p_reward = stats.mannwhitneyu(c_reward, r_reward, alternative="two-sided")

    # vs zero baseline
    _, p_c_vs_zero = stats.mannwhitneyu(c_steps, [r["steps"] for r in results["zero"]], alternative="two-sided")
    _, p_r_vs_zero = stats.mannwhitneyu(r_steps, [r["steps"] for r in results["zero"]], alternative="two-sided")

    print(f"\n  {'':>20} {'Steps':>8} {'Reward':>8}")
    print(f"  {'─'*38}")
    print(f"  {'Zero action':<20} {zero_mean:>8.0f}")
    print(f"  {'Connectome brain':<20} {np.mean(c_steps):>8.1f} {np.mean(c_reward):>8.1f}")
    print(f"  {'Random brain':<20} {np.mean(r_steps):>8.1f} {np.mean(r_reward):>8.1f}")
    print(f"\n  Connectome vs Zero: p={p_c_vs_zero:.4f}")
    print(f"  Random vs Zero:    p={p_r_vs_zero:.4f}")
    print(f"  Connectome vs Random: p={p_steps:.4f}")

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    c_mean = np.mean(c_steps)
    r_mean = np.mean(r_steps)

    if c_mean > zero_mean * 1.2 and p_c_vs_zero < 0.05:
        print(f"\n  The BRAIN (pure spiking, no reflexes) CONTROLS THE BODY.")
        print(f"  {c_mean:.0f} steps vs {zero_mean:.0f} zero-action = {c_mean/zero_mean:.1f}x")
        if p_steps < 0.05 and c_mean > r_mean:
            print(f"  AND the connectome helps! (p={p_steps:.4f})")
        else:
            print(f"  But connectome vs random: no difference (p={p_steps:.4f})")
    elif c_mean > zero_mean:
        print(f"\n  Brain slightly helps ({c_mean:.0f} vs {zero_mean:.0f}) but not significant.")
    else:
        print(f"\n  Brain does NOT help. Pure spiking can't control Walker2d.")
        print(f"  {c_mean:.0f} steps vs {zero_mean:.0f} zero-action.")

    # Save
    results_dir = Path("results/exp34_pure_brain")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f)
    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

"""Experiment 32: The Newborn — Innate behavior from structure alone.

A newborn human has NO learned behavior. But it has:
  - Spontaneous movement (general movements, fidgety)
  - Startle reflex (sudden stimulus → whole-body jerk)
  - Righting reflex (tilted → corrective movement)
  - Stepping reflex (feet touch surface → alternating legs)

The worm crawls. The fly forages. The newborn MOVES.

This experiment wires the brain to a body in a closed loop:
  Body state → somatosensory regions → [brain connectome] → motor regions → CPG → body

NO learning. Purely structure-driven. The question:
  Does the connectome-wired brain produce MORE ORGANIZED innate behavior
  than a random-wired brain?

We measure:
  1. Spontaneous movement complexity (entropy of motor output)
  2. Righting reflex (corrective response to perturbation)
  3. Movement diversity (different limbs move differently)
  4. Survival time (how long before falling)
"""

import time
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU
from encephagen.spinal.cpg import SpinalCPG, CPGParams
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


def classify_regions(labels):
    tau_labels = json.load(open(
        'src/encephagen/connectome/bundled/neurolib80_t1t2_labels.json'))
    groups = {}
    for key, patterns in [
        ('somatosensory', ['Postcentral', 'Paracentral']),
        ('motor', ['Precentral', 'Supp_Motor']),
        ('visual', ['Calcarine', 'Cuneus']),
        ('auditory', ['Heschl', 'Temporal_Sup']),
    ]:
        groups[key] = [i for i, l in enumerate(tau_labels)
                       if any(p in l for p in patterns)]
    return groups


def run_newborn(conn, duration_ms=5000, device="cuda"):
    """Run a newborn brain connected to a body via CPG.

    The body is simulated as a simple state:
      - tilt angle (perturbed randomly)
      - 4 joint positions (from CPG)

    The brain receives:
      - Body tilt → somatosensory regions (proprioception)
      - Joint positions → somatosensory regions

    The brain outputs:
      - Motor cortex firing rate → CPG drive modulation

    Returns metrics about the quality of innate behavior.
    """
    groups = classify_regions(conn.labels)
    npr = 200; n_regions = 80; n_total = n_regions * npr

    brain = SpikingBrainGPU(
        connectome=conn, neurons_per_region=npr,
        internal_conn_prob=0.05, between_conn_prob=0.03,
        global_coupling=5.0, ext_rate_factor=3.5,
        pfc_regions=[], device=device,
        use_delays=True, conduction_velocity=3.5,
        use_t1t2_gradient=True,
    )

    cpg = SpinalCPG(CPGParams(
        tau=50.0, tau_adapt=500.0, drive=1.0,
        w_mutual=2.5, w_crossed=1.5, beta=2.5,
    ))

    state = brain.init_state(batch_size=1)
    # Warmup brain AND CPG
    with torch.no_grad():
        for _ in range(2000):
            state, _ = brain.step(state)
    for _ in range(5000):
        cpg.step(0.1, brain_drive=0.0)

    soma_idx = groups['somatosensory']
    motor_idx = groups['motor']

    # Simulated body state
    body_tilt = 0.0         # radians
    body_velocity = 0.0
    rng = np.random.default_rng(42)

    # Recording
    motor_outputs = []
    tilt_history = []
    torque_history = []

    # Physics: 10ms body timestep (100 brain steps of 0.1ms)
    body_dt = 0.01  # seconds (10ms)
    brain_steps_per_body = 100

    n_body_steps = int(duration_ms / (body_dt * 1000))

    for body_step in range(n_body_steps):
        # === SENSE: body state → brain ===
        ext = torch.zeros(1, n_total, device=device)
        # Strong proprioceptive signal proportional to tilt
        tilt_signal = min(abs(body_tilt) * 500.0, 200.0)
        vel_signal = min(abs(body_velocity) * 200.0, 100.0)
        for ri in soma_idx:
            ext[0, ri*npr:(ri+1)*npr] = tilt_signal + vel_signal

        # === PROCESS: brain runs for 100 steps (10ms) ===
        motor_spikes = 0
        with torch.no_grad():
            for _ in range(brain_steps_per_body):
                state, spikes = brain.step(state, ext)
                for ri in motor_idx:
                    motor_spikes += spikes[0, ri*npr:(ri+1)*npr].sum().item()

        motor_rate = motor_spikes / (len(motor_idx) * npr * brain_steps_per_body)

        # === ACT: motor output → CPG → torques ===
        brain_drive = (motor_rate - 0.04) * 30.0
        torques = cpg.step(body_dt * 1000, brain_drive=brain_drive)

        # === BODY PHYSICS (inverted pendulum) ===
        gravity = -9.8 * np.sin(body_tilt)
        cpg_force = (torques[0] + torques[2]) * 0.5 * 5.0  # hip torques, amplified

        # Random perturbation every ~500ms
        perturbation = 0.0
        if body_step % 50 == 25:
            perturbation = rng.normal(0, 5.0)

        net = gravity + cpg_force + perturbation
        body_velocity += net * body_dt
        body_tilt += body_velocity * body_dt
        body_velocity *= 0.98  # damping

        # Record
        motor_outputs.append(motor_rate)
        tilt_history.append(body_tilt)
        torque_history.append(torques.copy())

        # Check if fallen
        if abs(body_tilt) > 1.5:  # ~86 degrees
            break

    motor_outputs = np.array(motor_outputs)
    tilt_history = np.array(tilt_history)
    torque_history = np.array(torque_history)
    survival_time = len(motor_outputs) * body_dt * 1000  # ms

    del brain; torch.cuda.empty_cache()

    # === METRICS ===

    # 1. Survival time (how long before falling)
    survival_ms = survival_time

    # 2. Motor complexity (entropy of motor output)
    if len(motor_outputs) > 10:
        # Binned entropy
        hist, _ = np.histogram(motor_outputs, bins=20, density=True)
        hist = hist[hist > 0]
        motor_entropy = -np.sum(hist * np.log(hist + 1e-10))
    else:
        motor_entropy = 0.0

    # 3. Tilt stability (std of tilt — lower = more stable)
    tilt_std = float(np.std(tilt_history)) if len(tilt_history) > 0 else 99.0

    # 4. Movement diversity (std across joint torques — higher = more diverse)
    if len(torque_history) > 10:
        torque_std = float(np.std(torque_history, axis=0).mean())
    else:
        torque_std = 0.0

    # 5. Righting response (correlation between tilt and corrective torque)
    if len(tilt_history) > 20 and len(torque_history) > 20:
        hip_torques = (torque_history[:, 0] + torque_history[:, 2]) / 2
        # Negative correlation = corrective (tilt right → push left)
        r_right, _ = stats.pearsonr(tilt_history[:len(hip_torques)], hip_torques)
    else:
        r_right = 0.0

    return {
        'survival_ms': survival_ms,
        'motor_entropy': motor_entropy,
        'tilt_std': tilt_std,
        'torque_std': torque_std,
        'righting_r': r_right,
    }


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 32: The Newborn")
    print("Innate behavior from structure alone — no learning")
    print("Brain → CPG → Body → Brain (closed loop)")
    print("=" * 70)

    conn = load_neurolib80()
    N_RUNS = 10

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

            metrics = run_newborn(c, duration_ms=10000)
            results[cond].append(metrics)

            elapsed = time.time() - t0
            print(f"  Run {run+1:>2}/{N_RUNS}  "
                  f"survive={metrics['survival_ms']:.0f}ms  "
                  f"tilt_std={metrics['tilt_std']:.3f}  "
                  f"righting={metrics['righting_r']:.3f}  "
                  f"entropy={metrics['motor_entropy']:.2f}  "
                  f"({elapsed:.0f}s)")

    # ========================================
    # Statistics
    # ========================================
    print(f"\n{'='*70}")
    print("RESULTS: Newborn Behavior — Connectome vs Random")
    print(f"{'='*70}")

    metrics_list = [
        ("Survival time (ms)", "survival_ms", "higher = more stable"),
        ("Tilt stability (std)", "tilt_std", "lower = more stable"),
        ("Righting reflex (r)", "righting_r", "negative = corrective"),
        ("Motor entropy", "motor_entropy", "moderate = organized movement"),
        ("Torque diversity", "torque_std", "higher = different limb patterns"),
    ]

    p_vals = []; p_labels = []
    print(f"\n  {'Metric':<30} {'Connectome':>12} {'Random':>12} {'p':>8} {'d':>8}")
    print(f"  {'─'*72}")

    for label, key, meaning in metrics_list:
        c_vals = [r[key] for r in results["connectome"]]
        r_vals = [r[key] for r in results["random"]]
        c_mean = np.mean(c_vals)
        r_mean = np.mean(r_vals)
        _, p = stats.mannwhitneyu(c_vals, r_vals, alternative="two-sided")
        p_vals.append(p); p_labels.append(label)
        pooled = np.sqrt((np.var(c_vals) + np.var(r_vals)) / 2)
        d = (c_mean - r_mean) / (pooled + 1e-10)
        sig = "*" if p < 0.05 else ""
        print(f"  {label:<30} {c_mean:>12.3f} {r_mean:>12.3f} {p:>8.4f} {d:>+8.3f}{sig}")

    print(f"\n{report_with_fdr(p_labels, p_vals)}")

    # Save
    results_dir = Path("results/exp32_newborn")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f)
    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

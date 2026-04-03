"""Experiment 19: Brain learns to calibrate CPG through experience.

走路 = 先天程序 (CPG) × 后天学习 (brain calibration)

The CPG already produces alternating gait (innate).
The brain learns WHAT DRIVE LEVEL keeps the body upright.

Protocol:
  1. Body stands with CPG active
  2. Brain receives proprioceptive feedback (tilt, height)
  3. When upright: reward signal → brain strengthens current state
  4. When falling: no reward → brain adjusts
  5. Over trials, brain learns the right drive level

The brain doesn't learn to walk. It learns to CALIBRATE.
Like a baby falling, adjusting, falling, adjusting.
"""

import time
import numpy as np
import torch

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU
from encephagen.body.simple_body import SimpleBody


def classify_regions(labels):
    groups = {}
    for key, patterns in [
        ('visual', ['V1', 'V2', 'VAC']),
        ('somatosensory', ['S1', 'S2']),
        ('prefrontal', ['PFC', 'FEF']),
        ('hippocampus', ['HC', 'PHC']),
        ('amygdala', ['AMYG']),
        ('basal_ganglia', ['BG']),
        ('thalamus', ['TM']),
        ('motor', ['M1', 'PMC']),
        ('cingulate', ['CC']),
    ]:
        groups[key] = [i for i, l in enumerate(labels)
                       if any(p in l.upper() for p in patterns)]
    return groups


def run_cpg_step(x, v, drive, tau=50, tau_a=500, w_m=2.5, w_c=1.5, beta=2.5, dt=0.5):
    """Matsuoka CPG step. Drive controls speed/amplitude."""
    y = np.maximum(0, x)
    inh = np.zeros(4)
    inh[0] = -w_m*y[1] - w_c*y[2]
    inh[1] = -w_m*y[0] - w_c*y[3]
    inh[2] = -w_m*y[3] - w_c*y[0]
    inh[3] = -w_m*y[2] - w_c*y[1]
    drives = np.full(4, drive)
    ddx = (-x + drives + inh - beta*v) / tau
    dv = (-v + y) / tau_a
    x += dt * ddx
    v += dt * dv
    lt = y[1] - y[0]
    rt = y[3] - y[2]
    scale = 0.3
    torques = np.clip(
        np.array([rt*scale, rt*scale*0.5, lt*scale, lt*scale*0.5]),
        -1, 1
    )
    return x, v, torques


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 19: Brain learns to calibrate walking")
    print("先天程序 (CPG) × 后天学习 (brain calibration)")
    print("=" * 70)

    connectome = Connectome.from_bundled("tvb96")
    groups = classify_regions(connectome.labels)
    npr = 200
    n_total = 96 * npr
    device = "cuda"

    pfc_regions = groups['prefrontal']
    motor_idx = groups['motor']
    somato_idx = groups['somatosensory']
    amyg_idx = groups['amygdala']

    brain = SpikingBrainGPU(
        connectome=connectome, neurons_per_region=npr,
        global_coupling=0.15, ext_rate_factor=3.5,
        tau_nmda=150.0, nmda_ratio=0.4,
        pfc_regions=pfc_regions, device=device,
    )

    # Warm up brain
    print("\n  Warming up brain...")
    state = brain.init_state(batch_size=1)
    with torch.no_grad():
        for _ in range(5000):
            state, _ = brain.step(state)

    # Get motor cortex baseline rate
    motor_counts = torch.zeros(len(motor_idx) * npr, device=device)
    with torch.no_grad():
        for _ in range(2000):
            state, spikes = brain.step(state)
            for j, ri in enumerate(motor_idx):
                motor_counts[j*npr:(j+1)*npr] += spikes[0, ri*npr:(ri+1)*npr]
    motor_baseline = motor_counts.mean().item() / 2000
    print(f"  Motor baseline rate: {motor_baseline:.4f}")

    # Learning weights
    W_dense = brain.W.to_dense()
    learning_rate = 0.002

    # === PHASE 1: Baseline (no learning) ===
    print(f"\n{'='*60}")
    print("PHASE 1: Baseline — CPG + body, brain observes but doesn't learn")
    print(f"{'='*60}")

    baseline_survivals = []
    for trial in range(5):
        body = SimpleBody()
        body.reset()
        cpg_x = np.array([0.5, -0.3, -0.3, 0.5])
        cpg_v = np.zeros(4)

        survived = 0
        for step in range(5000):  # 2.5 seconds max
            # Brain reads body state
            body_obs = body.get_sensory_input()
            tilt = float(np.clip(body_obs[10], -1, 1))
            height = float(body_obs[8])

            # Sensory → brain
            external = torch.zeros(1, n_total, device=device)
            for ri in somato_idx:
                external[0, ri*npr:(ri+1)*npr] = (abs(tilt) + 1) * 5.0
            # Height danger signal
            danger = max(0, 0.8 - height)
            for ri in somato_idx[:2]:
                external[0, ri*npr:(ri+1)*npr] += danger * 15.0

            # Brain step (5 steps per CPG step)
            motor_spikes = 0
            with torch.no_grad():
                for _ in range(5):
                    state, spikes = brain.step(state, external)
                    for ri in motor_idx:
                        motor_spikes += spikes[0, ri*npr:(ri+1)*npr].sum().item()

            # Motor cortex output → CPG drive
            motor_rate = motor_spikes / (len(motor_idx) * npr * 5)
            brain_drive = (motor_rate - motor_baseline) * 50
            cpg_drive = 2.0 + np.clip(brain_drive, -0.5, 0.5)

            # CPG step
            cpg_x, cpg_v, torques = run_cpg_step(cpg_x, cpg_v, cpg_drive)

            # Body step
            body.step_n(torques, n=1)
            survived = step + 1

            if body.get_state().is_fallen:
                break

        survival_sec = survived * 0.5 / 1000
        baseline_survivals.append(survival_sec)

    print(f"  Baseline survival: {baseline_survivals}")
    print(f"  Mean: {np.mean(baseline_survivals):.2f}s")

    # === PHASE 2: Learning through experience ===
    print(f"\n{'='*60}")
    print("PHASE 2: Brain learns from experience (30 episodes)")
    print("  Upright → reward signal → strengthen current brain state")
    print("  Falling → no reward → brain state shifts")
    print(f"{'='*60}")

    learning_survivals = []
    t0 = time.time()

    for episode in range(30):
        body = SimpleBody()
        body.reset()
        cpg_x = np.array([0.5, -0.3, -0.3, 0.5])
        cpg_v = np.zeros(4)

        survived = 0
        episode_reward = 0

        for step in range(5000):
            body_obs = body.get_sensory_input()
            tilt = float(np.clip(body_obs[10], -1, 1))
            height = float(body_obs[8])
            upright = height > 0.3 and abs(tilt) < 0.5

            # Sensory → brain (tilt and height)
            external = torch.zeros(1, n_total, device=device)
            for ri in somato_idx:
                external[0, ri*npr:(ri+1)*npr] = (abs(tilt) + 1) * 5.0
            danger = max(0, 0.8 - height)
            for ri in somato_idx[:2]:
                external[0, ri*npr:(ri+1)*npr] += danger * 15.0

            # If upright → reward signal to amygdala
            if upright:
                for ri in amyg_idx:
                    external[0, ri*npr:(ri+1)*npr] += 10.0
                episode_reward += 1

            # Brain step
            motor_spikes = 0
            brain_activity = torch.zeros(n_total, device=device)
            with torch.no_grad():
                for _ in range(5):
                    state, spikes = brain.step(state, external)
                    brain_activity += spikes[0]
                    for ri in motor_idx:
                        motor_spikes += spikes[0, ri*npr:(ri+1)*npr].sum().item()

            # Three-factor learning: when upright (reward present)
            # strengthen connections that are currently active
            if upright and step % 20 == 0:
                active = (brain_activity > 1).float()
                # Strengthen: active sensory → active motor pathways
                somato_active = torch.zeros(n_total, device=device)
                motor_active = torch.zeros(n_total, device=device)
                for ri in somato_idx:
                    somato_active[ri*npr:(ri+1)*npr] = active[ri*npr:(ri+1)*npr]
                for ri in motor_idx:
                    motor_active[ri*npr:(ri+1)*npr] = active[ri*npr:(ri+1)*npr]
                # Also strengthen amygdala connections (reward pathway)
                amyg_active = torch.zeros(n_total, device=device)
                for ri in amyg_idx:
                    amyg_active[ri*npr:(ri+1)*npr] = active[ri*npr:(ri+1)*npr]

                dW = learning_rate * torch.outer(somato_active + amyg_active, motor_active)
                mask = (W_dense != 0).float()
                W_dense = torch.clamp(W_dense + dW * mask, -20, 20)

            # Motor → CPG drive
            motor_rate = motor_spikes / (len(motor_idx) * npr * 5)
            brain_drive = (motor_rate - motor_baseline) * 50
            cpg_drive = 2.0 + np.clip(brain_drive, -0.5, 0.5)

            cpg_x, cpg_v, torques = run_cpg_step(cpg_x, cpg_v, cpg_drive)
            body.step_n(torques, n=1)
            survived = step + 1

            if body.get_state().is_fallen:
                break

        survival_sec = survived * 0.5 / 1000
        learning_survivals.append(survival_sec)

        # Update brain weights periodically
        if (episode + 1) % 5 == 0:
            indices = brain.W.coalesce().indices()
            new_vals = W_dense[indices[0], indices[1]]
            brain.W = torch.sparse_coo_tensor(
                indices, new_vals, brain.W.shape
            ).coalesce()

            elapsed = time.time() - t0
            recent = learning_survivals[-5:]
            print(f"  Ep {episode+1:>2}: survival={recent} "
                  f"mean={np.mean(recent):.2f}s reward={episode_reward} ({elapsed:.0f}s)")

    # === PHASE 3: Test retention ===
    print(f"\n{'='*60}")
    print("PHASE 3: Test — does learned calibration persist?")
    print(f"{'='*60}")

    # Final weight update
    indices = brain.W.coalesce().indices()
    new_vals = W_dense[indices[0], indices[1]]
    brain.W = torch.sparse_coo_tensor(indices, new_vals, brain.W.shape).coalesce()

    test_survivals = []
    for trial in range(5):
        body = SimpleBody()
        body.reset()
        cpg_x = np.array([0.5, -0.3, -0.3, 0.5])
        cpg_v = np.zeros(4)

        survived = 0
        for step in range(5000):
            body_obs = body.get_sensory_input()
            tilt = float(np.clip(body_obs[10], -1, 1))
            height = float(body_obs[8])

            external = torch.zeros(1, n_total, device=device)
            for ri in somato_idx:
                external[0, ri*npr:(ri+1)*npr] = (abs(tilt) + 1) * 5.0
            danger = max(0, 0.8 - height)
            for ri in somato_idx[:2]:
                external[0, ri*npr:(ri+1)*npr] += danger * 15.0

            motor_spikes = 0
            with torch.no_grad():
                for _ in range(5):
                    state, spikes = brain.step(state, external)
                    for ri in motor_idx:
                        motor_spikes += spikes[0, ri*npr:(ri+1)*npr].sum().item()

            motor_rate = motor_spikes / (len(motor_idx) * npr * 5)
            brain_drive = (motor_rate - motor_baseline) * 50
            cpg_drive = 2.0 + np.clip(brain_drive, -0.5, 0.5)

            cpg_x, cpg_v, torques = run_cpg_step(cpg_x, cpg_v, cpg_drive)
            body.step_n(torques, n=1)
            survived = step + 1

            if body.get_state().is_fallen:
                break

        test_survivals.append(survived * 0.5 / 1000)

    print(f"  Test survival: {test_survivals}")
    print(f"  Mean: {np.mean(test_survivals):.2f}s")

    # === RESULTS ===
    print(f"\n{'='*60}")
    print("RESULTS: Did the brain learn to calibrate walking?")
    print(f"{'='*60}")

    bl_mean = np.mean(baseline_survivals)
    learn_early = np.mean(learning_survivals[:5])
    learn_late = np.mean(learning_survivals[-5:])
    test_mean = np.mean(test_survivals)

    print(f"\n  Baseline (no learning):     {bl_mean:.2f}s")
    print(f"  Early learning (ep 1-5):    {learn_early:.2f}s")
    print(f"  Late learning (ep 26-30):   {learn_late:.2f}s")
    print(f"  Test (retention):           {test_mean:.2f}s")

    improvement = test_mean - bl_mean
    print(f"\n  Improvement: {improvement:+.2f}s")

    if improvement > 0.3:
        print(f"\n  The brain LEARNED to calibrate the CPG!")
        print(f"  Like a baby: fell, adjusted, fell, adjusted → walks better.")
    elif improvement > 0.1:
        print(f"\n  Marginal improvement — brain partially calibrated.")
    else:
        print(f"\n  No improvement — the learning signal may be too weak.")

    # Learning curve
    print(f"\n  Learning curve (5-episode windows):")
    for i in range(0, len(learning_survivals), 5):
        chunk = learning_survivals[i:i+5]
        print(f"    Episodes {i+1:>2}-{i+5:>2}: {np.mean(chunk):.2f}s")


if __name__ == "__main__":
    run_experiment()

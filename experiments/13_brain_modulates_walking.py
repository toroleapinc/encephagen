"""Experiment 13: Brain modulates walking speed through CPG.

The biologically correct architecture:
  Sensory input → BRAIN (19K spiking neurons, connectome)
                    ↓
               motor cortex firing rate → tonic drive
                    ↓
               SPINAL CPG (Matsuoka oscillator)
                    ↓
               alternating joint torques → BODY
                    ↓
               proprioceptive feedback → brain sensory regions

The brain doesn't generate the walking rhythm.
It modulates the CPG: more motor cortex activity → faster stepping.
Sensory events (touch, obstacle) change brain activity → change walking.

Test:
  Phase 1: CPG alone (baseline walking)
  Phase 2: Brain + CPG (brain modulates speed)
  Phase 3: Perturbation — inject stimulus midway, observe response
"""

import time
import numpy as np
import torch

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU
from encephagen.body.simple_body import SimpleBody
from encephagen.analysis.functional_roles import _classify_tvb76_regions


def run_cpg_step(x, v, drives, tau, tau_a, w_m, w_c, beta, dt):
    """One Matsuoka CPG step. Returns (x, v, torques)."""
    y = np.maximum(0, x)
    inh = np.zeros(4)
    inh[0] = -w_m*y[1] - w_c*y[2]
    inh[1] = -w_m*y[0] - w_c*y[3]
    inh[2] = -w_m*y[3] - w_c*y[0]
    inh[3] = -w_m*y[2] - w_c*y[1]
    dx = (-x + drives + inh - beta*v) / tau
    dv = (-v + y) / tau_a
    x = x + dt * dx
    v = v + dt * dv
    lt = y[1] - y[0]
    rt = y[3] - y[2]
    scale = 0.5
    torques = np.clip(
        np.array([rt*scale, rt*scale*0.5, lt*scale, lt*scale*0.5]),
        -1, 1
    )
    return x, v, torques


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 13: Brain modulates walking speed")
    print("Brain → tonic drive → CPG → alternating torques → body")
    print("=" * 70)

    # CPG parameters (proven to oscillate)
    tau, tau_a, c_base, w_m, w_c, beta, cpg_dt = 50, 500, 2.0, 2.5, 1.5, 2.5, 0.5

    # --- Phase 1: CPG alone (10s baseline) ---
    print("\nPhase 1: CPG alone (baseline)")
    x = np.array([0.5, -0.3, -0.3, 0.5])
    v = np.zeros(4)
    body = SimpleBody()
    body.reset()

    baseline_heights = []
    for step in range(20000):
        drives = np.full(4, c_base)
        x, v, torques = run_cpg_step(x, v, drives, tau, tau_a, w_m, w_c, beta, cpg_dt)
        body.step_n(torques, n=1)
        baseline_heights.append(body.get_state().torso_height)
        if body.get_state().is_fallen:
            print(f"  Fell at {step*0.5/1000:.1f}s")
            break

    baseline_time = len(baseline_heights) * 0.5 / 1000
    print(f"  Baseline: {baseline_time:.1f}s, height={np.mean(baseline_heights):.3f}")

    # --- Phase 2: Brain + CPG ---
    print(f"\nPhase 2: Brain modulates CPG (20s)")

    connectome = Connectome.from_bundled("tvb96")
    groups = _classify_tvb76_regions(connectome.labels)
    motor_idx = groups.get("motor", [])
    sensory_idx = groups.get("sensory", [])
    npr = 200
    n_total = connectome.num_regions * npr

    brain = SpikingBrainGPU(
        connectome=connectome, neurons_per_region=npr,
        global_coupling=0.15, ext_rate_factor=3.5, device="cuda",
    )

    # Warm up brain
    print("  Warming up brain...")
    brain_state = brain.init_state(batch_size=1)
    motor_baseline = 0.0
    with torch.no_grad():
        mc = torch.zeros(npr, device="cuda")
        for _ in range(5000):
            brain_state, spikes = brain.step(brain_state)
            if motor_idx:
                mc += spikes[0, motor_idx[0]*npr:(motor_idx[0]+1)*npr]
        motor_baseline = mc.mean().item() / 5000
    print(f"  Motor baseline: {motor_baseline:.4f}")

    # Reset CPG and body
    x = np.array([0.5, -0.3, -0.3, 0.5])
    v = np.zeros(4)
    body.reset()

    BRAIN_STEPS_PER_CPG = 5  # 0.5ms CPG = 5 × 0.1ms brain steps

    heights, drives_log, torques_log = [], [], []
    stimulus_active = False

    print("  Running brain + CPG + body...")
    t0 = time.time()

    for step in range(40000):  # 20 seconds
        sim_time = step * 0.5 / 1000  # seconds

        # Body sensory → brain
        body_obs = body.get_sensory_input()
        external = torch.zeros(1, n_total, device="cuda")

        # Basic sensory encoding
        for i, ri in enumerate(sensory_idx[:8]):
            if i < len(body_obs):
                s = ri * npr
                external[0, s:s+npr] = (float(body_obs[i]) + 1) * 5.0

        # PERTURBATION: at 8-10 seconds, inject strong stimulus
        # (simulates unexpected touch or obstacle)
        if 8.0 <= sim_time < 10.0:
            if not stimulus_active:
                print(f"  >>> STIMULUS ON at {sim_time:.1f}s")
                stimulus_active = True
            # Strong input to first 4 sensory regions
            for ri in sensory_idx[:4]:
                s = ri * npr
                external[0, s:s+npr] += 20.0
        elif stimulus_active and sim_time >= 10.0:
            print(f"  >>> STIMULUS OFF at {sim_time:.1f}s")
            stimulus_active = False

        # Brain steps
        motor_spikes = 0
        with torch.no_grad():
            for _ in range(BRAIN_STEPS_PER_CPG):
                brain_state, spikes = brain.step(brain_state, external)
                if motor_idx:
                    s = motor_idx[0] * npr
                    motor_spikes += spikes[0, s:s+npr].sum().item()

        # Brain → CPG tonic drive
        motor_rate = motor_spikes / (npr * BRAIN_STEPS_PER_CPG)
        brain_drive = (motor_rate - motor_baseline) * 50
        brain_drive = np.clip(brain_drive, -1.0, 1.0)

        # CPG step with brain-modulated drive
        total_drive = c_base + brain_drive
        drives = np.full(4, total_drive)
        x, v, torques = run_cpg_step(x, v, drives, tau, tau_a, w_m, w_c, beta, cpg_dt)

        # Body
        body.step_n(torques, n=1)
        bs = body.get_state()
        heights.append(bs.torso_height)
        drives_log.append(total_drive)
        torques_log.append(torques.copy())

        if bs.is_fallen:
            print(f"  Fell at {sim_time:.2f}s")
            break

        if (step + 1) % 8000 == 0:
            elapsed = time.time() - t0
            ta = np.array(torques_log[-8000:])
            print(f"  {sim_time:.1f}s  h={bs.torso_height:.3f}  "
                  f"drive={total_drive:.2f}  torque_std={ta[:,0].std():.3f}  "
                  f"({elapsed:.0f}s)")

    elapsed = time.time() - t0
    sim_time_total = len(heights) * 0.5 / 1000
    ha = np.array(heights)
    da = np.array(drives_log)
    ta = np.array(torques_log)

    # --- Results ---
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    print(f"\n  Survived: {sim_time_total:.1f}s ({elapsed:.0f}s wall)")
    print(f"  Height: mean={ha.mean():.3f} std={ha.std():.3f}")

    # Drive modulation by brain
    pre_stim = da[:16000]   # 0-8s
    during_stim = da[16000:20000] if len(da) > 20000 else da[16000:]  # 8-10s
    post_stim = da[20000:] if len(da) > 20000 else []  # 10-20s

    print(f"\n  Brain tonic drive:")
    print(f"    Pre-stimulus (0-8s):    mean={pre_stim.mean():.3f} std={pre_stim.std():.3f}")
    if len(during_stim) > 0:
        print(f"    During stimulus (8-10s): mean={during_stim.mean():.3f} std={during_stim.std():.3f}")
    if len(post_stim) > 0:
        print(f"    Post-stimulus (10-20s):  mean={np.array(post_stim).mean():.3f}")

    # Did the stimulus change the drive?
    if len(during_stim) > 100 and len(pre_stim) > 100:
        drive_change = during_stim.mean() - pre_stim.mean()
        print(f"\n  Stimulus effect on drive: {drive_change:+.4f}")
        if abs(drive_change) > 0.01:
            print(f"    Brain RESPONDED to stimulus! Drive changed by {drive_change:+.3f}")
            if drive_change > 0:
                print(f"    → CPG would speed up (more tonic excitation)")
            else:
                print(f"    → CPG would slow down (less tonic excitation)")
        else:
            print(f"    Brain did not significantly change drive")

    # Torque oscillation
    from scipy import signal
    if ta[:,0].std() > 0.01 and len(ta) > 2000:
        f, ps = signal.welch(ta[2000:,0], fs=2000, nperseg=min(2048, len(ta)-2000))
        print(f"\n  Stepping frequency: {f[np.argmax(ps)]:.2f} Hz")
        lr = np.corrcoef(ta[2000:,0], ta[2000:,2])[0,1]
        print(f"  L-R alternation: {lr:+.3f}")


if __name__ == "__main__":
    run_experiment()

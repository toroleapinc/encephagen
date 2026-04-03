"""Experiment 12: Brain + Spinal CPG + Body

The biologically correct architecture:
  BRAIN (19K neurons, connectome) → tonic drive signal
                                       ↓
  SPINAL CPG (200 adaptive neurons) → rhythmic motor pattern
                                       ↓
  BODY (MuJoCo biped) ← proprioceptive feedback → CPG

The brain doesn't generate the walking rhythm.
The spinal cord does. The brain modulates speed and direction.
"""

import time
import numpy as np

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU
from encephagen.spinal.cpg import SpinalCPG, CPGParams
from encephagen.body.simple_body import SimpleBody
from encephagen.analysis.functional_roles import _classify_tvb76_regions


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 12: Brain + Spinal CPG + Body")
    print("Brain modulates, CPG generates rhythm, body moves.")
    print("=" * 70)

    # --- Phase 1: Test CPG alone (no brain, no body) ---
    print("\nPhase 1: CPG alone (does it oscillate?)")
    cpg = SpinalCPG()
    dt = 1.0  # 1ms timesteps for CPG

    rates = {"lf": [], "le": [], "rf": [], "re": []}
    for step in range(5000):  # 5 seconds
        torques = cpg.step(dt, brain_drive=0.5)
        state = cpg.get_state()
        rates["lf"].append(state["left_flex"])
        rates["le"].append(state["left_ext"])
        rates["rf"].append(state["right_flex"])
        rates["re"].append(state["right_ext"])

    # Check for oscillation
    lf = np.array(rates["lf"])
    rf = np.array(rates["rf"])
    print(f"  Left flex rate:  mean={lf.mean():.3f} std={lf.std():.3f}")
    print(f"  Right flex rate: mean={rf.mean():.3f} std={rf.std():.3f}")

    # Check left-right alternation
    if len(lf) > 100:
        corr = np.corrcoef(lf[100:], rf[100:])[0, 1]
        print(f"  Left-Right correlation: {corr:+.3f} (should be negative for alternation)")

    # Check for rhythmic oscillation
    from scipy import signal
    if lf.std() > 0.01:
        freqs, psd = signal.welch(lf, fs=1000.0, nperseg=min(512, len(lf)))
        peak_freq = freqs[np.argmax(psd)]
        print(f"  Oscillation frequency: {peak_freq:.1f} Hz (target: 0.5-3 Hz)")
    else:
        print(f"  WARNING: No oscillation detected (std too low)")

    # --- Phase 2: CPG + Body (no brain) ---
    print(f"\nPhase 2: CPG + Body (no brain, just CPG driving body)")
    cpg.reset()
    body = SimpleBody()
    body_state = body.reset()

    heights, torques_log = [], []
    cpg_states = {"lf": [], "le": [], "rf": [], "re": []}

    for step in range(10000):  # 10 seconds at 1ms
        # Proprioceptive feedback: joint angles
        joint_angles = body_state.joint_angles
        # Left leg extension signal (positive = extended)
        left_extension = (joint_angles[2] + joint_angles[3]) / 2  # left hip + knee
        right_extension = (joint_angles[0] + joint_angles[1]) / 2

        # CPG step with proprioception
        torques = cpg.step(dt, brain_drive=0.5,
                           proprio_left=left_extension,
                           proprio_right=right_extension)

        # Physics: multiple steps per CPG step for stability
        body_state = body.step_n(torques, n=1)  # 5ms physics per 1ms CPG

        heights.append(body_state.torso_height)
        torques_log.append(torques.copy())
        state = cpg.get_state()
        for k in cpg_states:
            label = {"lf": "left_flex", "le": "left_ext",
                     "rf": "right_flex", "re": "right_ext"}[k]
            cpg_states[k].append(state[label])

        if body_state.is_fallen:
            print(f"  Fell at {step/1000:.2f}s")
            break

        if (step + 1) % 2000 == 0:
            ta = np.array(torques_log[-2000:])
            print(f"  {step/1000:.1f}s  h={body_state.torso_height:.3f}  "
                  f"T=[{ta[:,0].mean():+.2f},{ta[:,1].mean():+.2f},"
                  f"{ta[:,2].mean():+.2f},{ta[:,3].mean():+.2f}]  "
                  f"T_std=[{ta[:,0].std():.2f},{ta[:,1].std():.2f},"
                  f"{ta[:,2].std():.2f},{ta[:,3].std():.2f}]")

    sim_time = len(heights) / 1000
    ha = np.array(heights)
    ta = np.array(torques_log)

    print(f"\n  Survived: {sim_time:.2f}s")
    print(f"  Height: mean={ha.mean():.3f} std={ha.std():.3f}")

    names = ["r_hip", "r_knee", "l_hip", "l_knee"]
    print(f"  Torques:")
    for j in range(4):
        print(f"    {names[j]:<8} mean={ta[:,j].mean():+.3f} std={ta[:,j].std():.3f}")

    # Left-right alternation in torques?
    if len(ta) > 200:
        lr_corr = np.corrcoef(ta[200:, 0], ta[200:, 2])[0, 1]
        print(f"  Left-Right hip torque correlation: {lr_corr:+.3f}")
        if lr_corr < -0.3:
            print(f"    ALTERNATING! (anti-phase between legs)")

    # Frequency of torque oscillation
    if ta[:, 0].std() > 0.01 and len(ta) > 200:
        freqs, psd = signal.welch(ta[:, 0], fs=1000.0, nperseg=min(512, len(ta)))
        peak_freq = freqs[np.argmax(psd)]
        print(f"  Torque oscillation: {peak_freq:.2f} Hz")

    # --- Phase 3: Brain + CPG + Body ---
    print(f"\nPhase 3: Full system (Brain → CPG → Body)")

    connectome = Connectome.from_bundled("tvb96")
    groups = _classify_tvb76_regions(connectome.labels)
    motor_idx = groups.get("motor", [])
    sensory_idx = groups.get("sensory", [])
    npr = 200
    n_total = connectome.num_regions * npr

    brain = SpikingBrainGPU(connectome=connectome, neurons_per_region=npr,
        global_coupling=0.15, ext_rate_factor=3.5, device="cuda")

    import torch
    brain_state = brain.init_state(batch_size=1)

    # Warm up brain
    print("  Warming up brain...")
    motor_baseline = 0.0
    with torch.no_grad():
        motor_counts = torch.zeros(npr, device="cuda")
        for _ in range(5000):
            brain_state, spikes = brain.step(brain_state)
            if motor_idx:
                s = motor_idx[0] * npr
                motor_counts += spikes[0, s:s+npr]
        motor_baseline = motor_counts.mean().item() / 5000

    print(f"  Motor baseline rate: {motor_baseline:.4f}")

    cpg.reset()
    body_state = body.reset()
    heights2, torques_log2 = [], []

    BRAIN_STEPS_PER_CPG = 10  # 1ms CPG = 10 × 0.1ms brain steps

    for step in range(10000):  # 10 seconds
        # Sensory → Brain
        body_obs = body.get_sensory_input()
        external = torch.zeros(1, n_total, device="cuda")
        for i, ri in enumerate(sensory_idx[:8]):
            if i < len(body_obs):
                s = ri * npr
                external[0, s:s+npr] = (float(body_obs[i]) + 1) * 5.0

        # Brain steps
        motor_spikes = 0
        with torch.no_grad():
            for _ in range(BRAIN_STEPS_PER_CPG):
                brain_state, spikes = brain.step(brain_state, external)
                if motor_idx:
                    s = motor_idx[0] * npr
                    motor_spikes += spikes[0, s:s+npr].sum().item()

        # Brain motor output → CPG tonic drive
        motor_rate = motor_spikes / (npr * BRAIN_STEPS_PER_CPG)
        brain_drive = (motor_rate - motor_baseline) * 100  # Scale deviation to drive
        brain_drive = np.clip(brain_drive, -0.5, 1.0)

        # Proprioception
        ja = body_state.joint_angles
        left_ext = (ja[2] + ja[3]) / 2
        right_ext = (ja[0] + ja[1]) / 2

        # CPG step
        torques = cpg.step(dt, brain_drive=brain_drive,
                           proprio_left=left_ext, proprio_right=right_ext)

        # Body
        body_state = body.step_n(torques, n=1)
        heights2.append(body_state.torso_height)
        torques_log2.append(torques.copy())

        if body_state.is_fallen:
            print(f"  Fell at {step/1000:.2f}s")
            break

        if (step + 1) % 2000 == 0:
            ta2 = np.array(torques_log2[-2000:])
            print(f"  {step/1000:.1f}s  h={body_state.torso_height:.3f}  "
                  f"drive={brain_drive:+.2f}  "
                  f"T_std=[{ta2[:,0].std():.2f},{ta2[:,1].std():.2f},"
                  f"{ta2[:,2].std():.2f},{ta2[:,3].std():.2f}]")

    sim_time2 = len(heights2) / 1000
    ha2 = np.array(heights2)
    ta2 = np.array(torques_log2)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"\n  CPG alone:        survived {sim_time:.2f}s")
    print(f"  Brain + CPG + Body: survived {sim_time2:.2f}s")
    print(f"\n  Height (full system): mean={ha2.mean():.3f} std={ha2.std():.3f}")

    if len(ta2) > 200:
        lr = np.corrcoef(ta2[200:, 0], ta2[200:, 2])[0, 1]
        print(f"  Left-Right alternation: {lr:+.3f} {'YES!' if lr < -0.3 else 'no'}")

        freqs, psd = signal.welch(ta2[:, 0], fs=1000.0, nperseg=min(512, len(ta2)))
        pf = freqs[np.argmax(psd)]
        print(f"  Stepping frequency: {pf:.2f} Hz")

    if sim_time2 > 3.0:
        print(f"\n  The brain-controlled body survived {sim_time2:.1f} seconds!")
    elif sim_time2 > sim_time:
        print(f"\n  Brain improved survival: {sim_time:.2f}s → {sim_time2:.2f}s")


if __name__ == "__main__":
    run_experiment()

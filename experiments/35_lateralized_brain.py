"""Experiment 35: Lateralized Brain Control — making the brain contribution REAL.

The problem with the hybrid demo: brain adds only 3% because sensory input
loses sign information (abs(tilt)). Brain sees same activity for tilt-left
and tilt-right → can't produce corrective commands.

Fix: LATERALIZED encoding. Tilt right → excite LEFT somatosensory (contralateral).
Brain's hemispheric asymmetry → asymmetric motor output → correction.
This is how real brains work.

Architecture:
  Tilt RIGHT → Postcentral_L (left somatosensory) activated
              → brain routes through connectome
              → Precentral_L (left motor cortex) more active
              → LEFT leg produces more torque → corrective push

Then measure:
  1. Brain-only control (lateralized, no CPG, no hand-coded reflex)
  2. Brain + CPG (brain replaces the PD righting reflex)
  3. Compare: is the brain producing REAL corrective control?
"""

import numpy as np
import torch
import json
import gymnasium as gym
from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU


def load_brain():
    sc = np.load('src/encephagen/connectome/bundled/neurolib80_weights.npy')
    tl = np.load('src/encephagen/connectome/bundled/neurolib80_tract_lengths.npy')
    labels = json.load(open('src/encephagen/connectome/bundled/neurolib80_labels.json'))
    tau_labels = json.load(open('src/encephagen/connectome/bundled/neurolib80_t1t2_labels.json'))
    c = Connectome(sc, labels); c.tract_lengths = tl
    return c, tau_labels


class LateralizedBrain:
    """Brain with lateralized sensory-motor pathways."""

    def __init__(self, conn, tau_labels, device="cuda"):
        self.npr = 200
        self.n_regions = 80
        self.n_total = self.n_regions * self.npr
        self.device = device

        # Left/right somatosensory
        self.soma_L = [i for i, l in enumerate(tau_labels)
                       if ('Postcentral' in l or 'Paracentral' in l) and l.endswith('_L')]
        self.soma_R = [i for i, l in enumerate(tau_labels)
                       if ('Postcentral' in l or 'Paracentral' in l) and l.endswith('_R')]

        # Left/right motor
        self.motor_L = [i for i, l in enumerate(tau_labels)
                        if ('Precentral' in l or 'Supp_Motor' in l) and l.endswith('_L')]
        self.motor_R = [i for i, l in enumerate(tau_labels)
                        if ('Precentral' in l or 'Supp_Motor' in l) and l.endswith('_R')]

        # Motor neuron indices for fast readout
        motor_L_neurons = []
        for ri in self.motor_L:
            motor_L_neurons.extend(range(ri * self.npr, (ri + 1) * self.npr))
        motor_R_neurons = []
        for ri in self.motor_R:
            motor_R_neurons.extend(range(ri * self.npr, (ri + 1) * self.npr))
        self.motor_L_idx = torch.tensor(motor_L_neurons, device=device, dtype=torch.long)
        self.motor_R_idx = torch.tensor(motor_R_neurons, device=device, dtype=torch.long)

        self.brain = SpikingBrainGPU(
            connectome=conn, neurons_per_region=self.npr,
            internal_conn_prob=0.05, between_conn_prob=0.03,
            global_coupling=5.0, ext_rate_factor=3.5,
            pfc_regions=[], device=device,
            use_delays=True, conduction_velocity=3.5,
            use_t1t2_gradient=True,
        )

        self.state = self.brain.init_state(batch_size=1)
        with torch.no_grad():
            for _ in range(2000):
                self.state, _ = self.brain.step(self.state)

        # Calibrate baseline rates per hemisphere
        L_total, R_total = 0, 0
        with torch.no_grad():
            for _ in range(500):
                self.state, spikes = self.brain.step(self.state)
                L_total += spikes[0, self.motor_L_idx].float().sum().item()
                R_total += spikes[0, self.motor_R_idx].float().sum().item()
        self.baseline_L = L_total / (len(motor_L_neurons) * 500)
        self.baseline_R = R_total / (len(motor_R_neurons) * 500)

    def step(self, obs, brain_steps=20):
        """Lateralized sense → think → act.

        Key: tilt is encoded CONTRALATERALLY.
        Tilt right → left somatosensory excited
        → left motor more active → left leg corrects
        """
        ext = torch.zeros(1, self.n_total, device=self.device)

        tilt = obs[1]           # body angle (positive = right)
        height = obs[0]         # body height
        h_vel = obs[8]          # horizontal velocity
        v_vel = obs[9]          # vertical velocity (angular)

        # Base proprioceptive signal to both sides
        base_signal = (abs(h_vel) * 20.0 + abs(v_vel) * 20.0 +
                        max(0, (1.25 - height)) * 50.0)

        # LATERALIZED tilt encoding (contralateral)
        # Tilt RIGHT → LEFT somatosensory gets MORE input
        tilt_signal = tilt * 80.0  # amplified, preserves sign

        for ri in self.soma_L:
            ext[0, ri*self.npr:(ri+1)*self.npr] = base_signal + tilt_signal
        for ri in self.soma_R:
            ext[0, ri*self.npr:(ri+1)*self.npr] = base_signal - tilt_signal

        # Clamp to non-negative (neurons can't receive negative current easily)
        ext = torch.clamp(ext, min=0.0)

        # Run brain
        L_acc = torch.zeros(len(self.motor_L_idx), device=self.device)
        R_acc = torch.zeros(len(self.motor_R_idx), device=self.device)
        with torch.no_grad():
            for _ in range(brain_steps):
                self.state, spikes = self.brain.step(self.state, ext)
                L_acc += spikes[0, self.motor_L_idx].float()
                R_acc += spikes[0, self.motor_R_idx].float()

        # Motor rates per hemisphere
        L_rate = L_acc.sum().item() / (len(self.motor_L_idx) * brain_steps)
        R_rate = R_acc.sum().item() / (len(self.motor_R_idx) * brain_steps)

        # Brain-derived corrective signal
        # If left motor > baseline → left leg should push more
        L_drive = (L_rate - self.baseline_L) * 30.0
        R_drive = (R_rate - self.baseline_R) * 30.0

        return L_drive, R_drive, L_rate, R_rate


def run_test(mode="brain+cpg", n_runs=10, max_steps=500):
    """Test different control modes."""
    from encephagen.spinal.cpg import SpinalCPG, CPGParams

    conn, tau_labels = load_brain()
    env = gym.make('Walker2d-v5')

    results = []
    for run in range(n_runs):
        brain = LateralizedBrain(conn, tau_labels) if mode != "cpg_only" else None
        cpg = SpinalCPG(CPGParams(tau=50, tau_adapt=500, drive=1.0,
                                   w_mutual=2.5, w_crossed=1.5, beta=2.5))
        for _ in range(5000):
            cpg.step(0.1)

        obs, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = np.zeros(6, dtype=np.float32)

            if mode == "brain_only":
                # PURE brain control (lateralized)
                L_drive, R_drive, _, _ = brain.step(obs)
                # Left brain → left body joints [3,4,5]
                # Right brain → right body joints [0,1,2]
                action[0] = np.clip(R_drive * 0.5, -1, 1)  # right thigh
                action[1] = np.clip(R_drive * 0.3, -1, 1)  # right leg
                action[3] = np.clip(L_drive * 0.5, -1, 1)  # left thigh
                action[4] = np.clip(L_drive * 0.3, -1, 1)  # left leg

            elif mode == "cpg_only":
                # CPG + hand-coded righting (no brain)
                tilt = obs[1]
                ang_vel = obs[9]
                righting = -tilt * 3.0 - ang_vel * 1.0
                torques = cpg.step(2.0, brain_drive=0.2)
                action[0] = torques[0] * 0.3 + 0.08 + righting * 0.3
                action[1] = torques[1] * 0.2 + 0.1
                action[2] = torques[1] * 0.1
                action[3] = torques[2] * 0.3 + 0.08 + righting * 0.3
                action[4] = torques[3] * 0.2 + 0.1
                action[5] = torques[3] * 0.1

            elif mode == "brain+cpg":
                # Brain REPLACES the hand-coded righting reflex
                L_drive, R_drive, _, _ = brain.step(obs)
                torques = cpg.step(2.0, brain_drive=0.2)

                # CPG provides rhythm
                action[0] = torques[0] * 0.3 + 0.08
                action[1] = torques[1] * 0.2 + 0.1
                action[2] = torques[1] * 0.1
                action[3] = torques[2] * 0.3 + 0.08
                action[4] = torques[3] * 0.2 + 0.1
                action[5] = torques[3] * 0.1

                # Brain provides LATERALIZED righting (replaces PD controller)
                action[0] += np.clip(R_drive * 0.3, -0.5, 0.5)
                action[3] += np.clip(L_drive * 0.3, -0.5, 0.5)

            action = np.clip(action, -1, 1)
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            if term or trunc:
                break

        results.append((step + 1, total_reward))
        if brain:
            del brain
            torch.cuda.empty_cache()

    env.close()
    return results


def main():
    print("=" * 60)
    print("  LATERALIZED BRAIN CONTROL")
    print("  Does the brain produce REAL corrective control?")
    print("=" * 60)

    modes = {
        "zero": "Zero action (baseline)",
        "cpg_only": "CPG + hand-coded righting (no brain)",
        "brain_only": "Pure lateralized brain (no CPG, no reflexes)",
        "brain+cpg": "CPG + brain righting (brain REPLACES PD reflex)",
    }

    all_results = {}
    for mode, desc in modes.items():
        print(f"\n  {desc}:")
        if mode == "zero":
            env = gym.make('Walker2d-v5')
            results = []
            for _ in range(10):
                obs, _ = env.reset()
                for s in range(500):
                    obs, _, t, tr, _ = env.step([0]*6)
                    if t or tr: break
                results.append((s+1, 0))
            env.close()
        else:
            results = run_test(mode=mode, n_runs=8)

        steps = [r[0] for r in results]
        rewards = [r[1] for r in results]
        print(f"    Steps: {np.mean(steps):.0f} ± {np.std(steps):.0f}  "
              f"(range {min(steps)}-{max(steps)})")
        print(f"    Reward: {np.mean(rewards):.0f}")
        all_results[mode] = steps

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY: Who's actually controlling the body?")
    print(f"{'='*60}")
    print(f"\n  {'Mode':<45} {'Steps':>8}")
    print(f"  {'─'*55}")
    for mode, desc in modes.items():
        steps = all_results[mode]
        bar = "█" * int(np.mean(steps) / 5)
        print(f"  {desc:<45} {np.mean(steps):>6.0f}  {bar}")

    zero_mean = np.mean(all_results["zero"])
    cpg_mean = np.mean(all_results["cpg_only"])
    brain_only_mean = np.mean(all_results["brain_only"])
    brain_cpg_mean = np.mean(all_results["brain+cpg"])

    print(f"\n  Brain-only vs zero: {brain_only_mean:.0f} vs {zero_mean:.0f} "
          f"({'HELPS' if brain_only_mean > zero_mean else 'HURTS'})")
    print(f"  Brain+CPG vs CPG-only: {brain_cpg_mean:.0f} vs {cpg_mean:.0f} "
          f"({'BRAIN HELPS' if brain_cpg_mean > cpg_mean else 'BRAIN HURTS'})")

    if brain_cpg_mean > cpg_mean * 1.1:
        print(f"\n  The brain provides REAL corrective control (+{(brain_cpg_mean/cpg_mean - 1)*100:.0f}%)")
    elif brain_cpg_mean > cpg_mean * 0.9:
        print(f"\n  Brain contribution is marginal ({(brain_cpg_mean/cpg_mean - 1)*100:+.0f}%)")
    else:
        print(f"\n  Brain HURTS performance ({(brain_cpg_mean/cpg_mean - 1)*100:+.0f}%)")


if __name__ == "__main__":
    main()

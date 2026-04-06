"""Visualize the newborn brain controlling a 3D Humanoid body.

Full 3D humanoid with torso, arms, legs, head — controlled by
16,000 spiking neurons through lateralized brain pathways + CPG.

Usage:
  python visualize_humanoid.py
  python visualize_humanoid.py --duration 20
"""

import argparse
import json
import time

import cv2
import numpy as np
import torch
import gymnasium as gym

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU
from encephagen.spinal.cpg import SpinalCPG, CPGParams


class HumanoidBrain:
    """Brain controlling a 3D Humanoid."""

    def __init__(self, device="cuda"):
        sc = np.load('src/encephagen/connectome/bundled/neurolib80_weights.npy')
        tl = np.load('src/encephagen/connectome/bundled/neurolib80_tract_lengths.npy')
        labels = json.load(open('src/encephagen/connectome/bundled/neurolib80_labels.json'))
        tau_labels = json.load(open(
            'src/encephagen/connectome/bundled/neurolib80_t1t2_labels.json'))
        self.tau_m = np.load('src/encephagen/connectome/bundled/neurolib80_tau_m.npy')

        c = Connectome(sc, labels); c.tract_lengths = tl
        self.npr = 200
        self.n_regions = 80
        self.n_total = self.n_regions * self.npr
        self.device = device
        self.tau_labels = tau_labels

        # Region groups
        self.display_groups = {}
        for key, patterns in [
            ('Somatosens L', ['Postcentral_L', 'Paracentral_L']),
            ('Somatosens R', ['Postcentral_R', 'Paracentral_R']),
            ('Motor L', ['Precentral_L', 'Supp_Motor_L']),
            ('Motor R', ['Precentral_R', 'Supp_Motor_R']),
            ('Visual L', ['Calcarine_L', 'Cuneus_L']),
            ('Visual R', ['Calcarine_R', 'Cuneus_R']),
            ('Frontal L', ['Frontal_Sup_L']),
            ('Frontal R', ['Frontal_Sup_R']),
        ]:
            self.display_groups[key] = [i for i, l in enumerate(tau_labels)
                                         if l in patterns]

        # Lateralized indices
        soma_L = [i for i, l in enumerate(tau_labels)
                  if ('Postcentral' in l or 'Paracentral' in l) and '_L' in l]
        soma_R = [i for i, l in enumerate(tau_labels)
                  if ('Postcentral' in l or 'Paracentral' in l) and '_R' in l]
        motor_all = [i for i, l in enumerate(tau_labels)
                     if 'Precentral' in l or 'Supp_Motor' in l]

        self.soma_L_starts = [ri * self.npr for ri in soma_L]
        self.soma_R_starts = [ri * self.npr for ri in soma_R]

        # Motor neurons — divide into 17 groups for 17 actions
        all_motor = []
        for ri in motor_all:
            all_motor.extend(range(ri * self.npr, (ri + 1) * self.npr))
        self.motor_idx = torch.tensor(all_motor, device=device, dtype=torch.long)
        self.n_motor = len(all_motor)
        self.neurons_per_action = self.n_motor // 17

        # Build brain
        self.brain = SpikingBrainGPU(
            connectome=c, neurons_per_region=self.npr,
            internal_conn_prob=0.05, between_conn_prob=0.03,
            global_coupling=5.0, ext_rate_factor=3.5,
            pfc_regions=[], device=device,
            use_delays=True, conduction_velocity=3.5,
            use_t1t2_gradient=True,
        )

        # CPG for legs
        self.cpg = SpinalCPG(CPGParams(tau=50, tau_adapt=500, drive=1.0,
                                        w_mutual=2.5, w_crossed=1.5, beta=2.5))

        # Warmup
        self.state = self.brain.init_state(batch_size=1)
        with torch.no_grad():
            for _ in range(2000):
                self.state, _ = self.brain.step(self.state)
        for _ in range(5000):
            self.cpg.step(0.1)

        # Calibrate baseline
        motor_total = 0
        with torch.no_grad():
            for _ in range(500):
                self.state, sp = self.brain.step(self.state)
                motor_total += sp[0, self.motor_idx].float().sum().item()
        self.baseline = motor_total / (self.n_motor * 500)

        self.region_rates = {}

    def step(self, obs):
        """Sense → Think → Act for Humanoid.

        Humanoid obs (348-dim): lots of joint positions, velocities, forces.
        Key indices:
          obs[0]: z-position (height)
          obs[1]: x-tilt (forward/back)
          obs[2]: y-tilt (left/right)
          obs[22:39]: joint positions
          obs[39:56]: joint velocities
        """
        ext = torch.zeros(1, self.n_total, device=self.device)

        # Extract key body state
        height = obs[0] if len(obs) > 0 else 1.3
        tilt_fb = obs[1] if len(obs) > 1 else 0  # forward/back
        tilt_lr = obs[2] if len(obs) > 2 else 0  # left/right

        # Base proprioceptive signal
        base = max(0, (1.4 - height)) * 80.0 + abs(tilt_fb) * 40.0

        # Lateralized encoding: left-right tilt
        lr_signal = tilt_lr * 60.0

        for s in self.soma_L_starts:
            ext[0, s:s + self.npr] = max(0, base + lr_signal)
        for s in self.soma_R_starts:
            ext[0, s:s + self.npr] = max(0, base - lr_signal)

        # Brain step
        motor_acc = torch.zeros(self.n_motor, device=self.device)
        brain_steps = 15
        with torch.no_grad():
            for _ in range(brain_steps):
                self.state, spikes = self.brain.step(self.state, ext)
                motor_acc += spikes[0, self.motor_idx].float()

        # Update display rates
        for gname, gidx in self.display_groups.items():
            if gidx:
                total = sum(spikes[0, ri*self.npr:(ri+1)*self.npr].sum().item()
                            for ri in gidx)
                self.region_rates[gname] = total / (len(gidx) * self.npr)

        # Motor output → 17 joint torques
        motor_np = motor_acc.cpu().numpy()
        brain_action = np.zeros(17, dtype=np.float32)
        for a in range(17):
            start = a * self.neurons_per_action
            end = start + self.neurons_per_action
            if end <= len(motor_np):
                rate = motor_np[start:end].sum() / (self.neurons_per_action * brain_steps)
                brain_action[a] = (rate - self.baseline) * 20.0

        # CPG for leg rhythm
        cpg_torques = self.cpg.step(1.5, brain_drive=0.2)

        # Compose action
        action = np.zeros(17, dtype=np.float32)

        # Humanoid-v5 action mapping (approximate):
        # 0: abdomen_z, 1: abdomen_y, 2: abdomen_x (torso)
        # 3-5: right_hip (x,z,y), 6: right_knee
        # 7-9: left_hip (x,z,y), 10: left_knee
        # 11-12: right_shoulder, 13: right_elbow
        # 14-15: left_shoulder, 16: left_elbow

        # Torso stabilization (righting reflex via brain lateralization)
        action[0] = np.clip(-tilt_fb * 2.0 + brain_action[0] * 0.3, -0.4, 0.4)
        action[1] = np.clip(-tilt_lr * 2.0 + brain_action[1] * 0.3, -0.4, 0.4)
        action[2] = np.clip(brain_action[2] * 0.2, -0.4, 0.4)

        # Right leg — CPG rhythm + brain
        action[3] = np.clip(cpg_torques[0] * 0.15 + 0.05 + brain_action[3] * 0.1, -0.4, 0.4)
        action[4] = np.clip(brain_action[4] * 0.1, -0.4, 0.4)
        action[5] = np.clip(cpg_torques[1] * 0.1 + brain_action[5] * 0.1, -0.4, 0.4)
        action[6] = np.clip(cpg_torques[1] * 0.1 + 0.05, -0.4, 0.4)

        # Left leg — anti-phase CPG + brain
        action[7] = np.clip(cpg_torques[2] * 0.15 + 0.05 + brain_action[7] * 0.1, -0.4, 0.4)
        action[8] = np.clip(brain_action[8] * 0.1, -0.4, 0.4)
        action[9] = np.clip(cpg_torques[3] * 0.1 + brain_action[9] * 0.1, -0.4, 0.4)
        action[10] = np.clip(cpg_torques[3] * 0.1 + 0.05, -0.4, 0.4)

        # Arms — gentle brain modulation (natural arm swing)
        for a in range(11, 17):
            action[a] = np.clip(brain_action[a] * 0.15, -0.4, 0.4)

        return action

    def draw_brain_panel(self, width=400, height=480):
        """Draw brain activity panel."""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (25, 25, 35)

        cv2.putText(panel, "BRAIN ACTIVITY", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(panel, "16,000 neurons | 80 regions | HCP connectome", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)

        y = 80
        bar_max_w = width - 160

        for gname in self.display_groups:
            rate = self.region_rates.get(gname, 0)
            gidx = self.display_groups[gname]
            tau_mean = np.mean([self.tau_m[i] for i in gidx]) if gidx else 0

            is_left = 'L' in gname
            color = (100, 200, 255) if is_left else (255, 150, 100)

            cv2.putText(panel, gname, (10, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            bar_w = int(min(rate * bar_max_w * 20, bar_max_w))
            cv2.rectangle(panel, (140, y), (140 + max(bar_w, 1), y + 16), color, -1)

            cv2.putText(panel, f"{rate:.3f}", (140 + bar_w + 5, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

            if tau_mean > 0:
                cv2.putText(panel, f"{tau_mean:.0f}ms", (width - 45, y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

            y += 28
            if not is_left:
                y += 10
                cv2.line(panel, (10, y), (width - 10, y), (50, 50, 60), 1)
                y += 10

        # Legend
        y += 20
        cv2.putText(panel, "Left hemisphere", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
        cv2.putText(panel, "Right hemisphere", (200, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 100), 1)

        return panel


def record_video(duration_s=15, output="humanoid_brain.mp4"):
    """Record humanoid brain-body video."""
    print(f"Building brain + humanoid...")
    brain = HumanoidBrain()
    env = gym.make('Humanoid-v5', render_mode='rgb_array')
    obs, _ = env.reset()

    max_steps = int(duration_s * 50)
    fps = 25

    body_frame = env.render().copy()
    brain_panel = brain.draw_brain_panel()
    combined_w = body_frame.shape[1] + brain_panel.shape[1]
    combined_h = max(body_frame.shape[0], brain_panel.shape[0])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (combined_w, combined_h))

    total_reward = 0
    t0 = time.time()
    print(f"Recording {duration_s}s... (baseline: ~40 steps = 0.8s)")

    for step in range(max_steps):
        action = brain.step(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if step % 2 == 0:
            body_frame = env.render().copy()
            brain_panel = brain.draw_brain_panel()

            # Status overlay
            cv2.putText(body_frame, f"Step: {step}  ({step/50:.1f}s)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(body_frame, f"Height: {obs[0]:.2f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(body_frame, f"Reward: {total_reward:.0f}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
            cv2.putText(body_frame, "Brain + CPG + Reflexes", (10, 465),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 255), 1)

            combined = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
            combined[:body_frame.shape[0], :body_frame.shape[1]] = body_frame
            combined[:brain_panel.shape[0], body_frame.shape[1]:] = brain_panel
            out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        if terminated or truncated:
            fell = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
            fell[:] = (25, 25, 35)
            cv2.putText(fell, f"FELL at step {step} ({step/50:.1f}s)",
                        (combined_w // 5, combined_h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 3)
            cv2.putText(fell, f"Baseline: ~40 steps (0.8s)",
                        (combined_w // 4, combined_h // 2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
            cv2.putText(fell, f"Improvement: {step/40:.1f}x",
                        (combined_w // 4, combined_h // 2 + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
            for _ in range(fps * 3):
                out.write(cv2.cvtColor(fell, cv2.COLOR_RGB2BGR))
            break

    out.release()
    env.close()

    elapsed = time.time() - t0
    print(f"\nSaved: {output}")
    print(f"  Survived: {step+1} steps ({(step+1)/50:.1f}s)")
    print(f"  Baseline: ~40 steps (0.8s)")
    print(f"  Improvement: {(step+1)/40:.1f}x")
    print(f"  Reward: {total_reward:.0f}")
    print(f"  Wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=15)
    parser.add_argument("--output", default="humanoid_brain.mp4")
    args = parser.parse_args()
    record_video(duration_s=args.duration, output=args.output)

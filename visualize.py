"""Visualize the newborn brain controlling a Walker2d body.

Generates an MP4 video showing:
  Left panel: Walker2d body in MuJoCo
  Right panel: Brain region activity (live bar chart)
  Bottom: Status text (steps, tilt, brain mode)

Usage:
  python visualize.py                    # 15s video, lateralized brain + CPG
  python visualize.py --mode brain_only  # Pure brain control (spoiler: falls fast)
  python visualize.py --mode cpg_only    # CPG + PD only (no brain)
  python visualize.py --duration 30      # 30 second video
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


class BrainVisualizer:
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

        # Region groups for display
        self.display_groups = {
            'Somatosens L': [i for i, l in enumerate(tau_labels)
                             if ('Postcentral' in l or 'Paracentral' in l) and '_L' in l],
            'Somatosens R': [i for i, l in enumerate(tau_labels)
                             if ('Postcentral' in l or 'Paracentral' in l) and '_R' in l],
            'Motor L': [i for i, l in enumerate(tau_labels)
                        if ('Precentral' in l or 'Supp_Motor' in l) and '_L' in l],
            'Motor R': [i for i, l in enumerate(tau_labels)
                        if ('Precentral' in l or 'Supp_Motor' in l) and '_R' in l],
            'Visual L': [i for i, l in enumerate(tau_labels)
                         if ('Calcarine' in l or 'Cuneus' in l) and '_L' in l],
            'Visual R': [i for i, l in enumerate(tau_labels)
                         if ('Calcarine' in l or 'Cuneus' in l) and '_R' in l],
            'Frontal L': [i for i, l in enumerate(tau_labels)
                          if 'Frontal_Sup' in l and '_L' in l],
            'Frontal R': [i for i, l in enumerate(tau_labels)
                          if 'Frontal_Sup' in l and '_R' in l],
            'Temporal L': [i for i, l in enumerate(tau_labels)
                           if 'Temporal_Mid' in l and '_L' in l],
            'Temporal R': [i for i, l in enumerate(tau_labels)
                           if 'Temporal_Mid' in l and '_R' in l],
        }

        # Build brain
        self.brain = SpikingBrainGPU(
            connectome=c, neurons_per_region=self.npr,
            internal_conn_prob=0.05, between_conn_prob=0.03,
            global_coupling=5.0, ext_rate_factor=3.5,
            pfc_regions=[], device=device,
            use_delays=True, conduction_velocity=3.5,
            use_t1t2_gradient=True,
        )

        # Motor indices
        soma_L = [i for i, l in enumerate(tau_labels)
                  if ('Postcentral' in l or 'Paracentral' in l) and '_L' in l]
        soma_R = [i for i, l in enumerate(tau_labels)
                  if ('Postcentral' in l or 'Paracentral' in l) and '_R' in l]
        motor_L = [i for i, l in enumerate(tau_labels)
                   if ('Precentral' in l or 'Supp_Motor' in l) and '_L' in l]
        motor_R = [i for i, l in enumerate(tau_labels)
                   if ('Precentral' in l or 'Supp_Motor' in l) and '_R' in l]

        self.soma_L_starts = [ri * self.npr for ri in soma_L]
        self.soma_R_starts = [ri * self.npr for ri in soma_R]

        motor_L_neurons = []
        for ri in motor_L:
            motor_L_neurons.extend(range(ri * self.npr, (ri + 1) * self.npr))
        motor_R_neurons = []
        for ri in motor_R:
            motor_R_neurons.extend(range(ri * self.npr, (ri + 1) * self.npr))
        self.motor_L_idx = torch.tensor(motor_L_neurons, device=device, dtype=torch.long)
        self.motor_R_idx = torch.tensor(motor_R_neurons, device=device, dtype=torch.long)

        # CPG
        self.cpg = SpinalCPG(CPGParams(tau=50, tau_adapt=500, drive=1.0,
                                        w_mutual=2.5, w_crossed=1.5, beta=2.5))

        # Warmup
        self.state = self.brain.init_state(batch_size=1)
        with torch.no_grad():
            for _ in range(2000):
                self.state, _ = self.brain.step(self.state)
        for _ in range(5000):
            self.cpg.step(0.1)

        # Calibrate baselines
        L_t, R_t = 0, 0
        with torch.no_grad():
            for _ in range(500):
                self.state, sp = self.brain.step(self.state)
                L_t += sp[0, self.motor_L_idx].float().sum().item()
                R_t += sp[0, self.motor_R_idx].float().sum().item()
        self.bl_L = L_t / (len(motor_L_neurons) * 500)
        self.bl_R = R_t / (len(motor_R_neurons) * 500)

        self.region_rates = {}
        self.last_spikes = None

    def step(self, obs, mode="brain+cpg"):
        """One step: sense → think → act. Returns action and brain activity."""
        ext = torch.zeros(1, self.n_total, device=self.device)
        tilt = obs[1]
        h_vel = obs[8]
        v_vel = obs[9] if len(obs) > 9 else 0
        height = obs[0]

        base = abs(h_vel) * 20 + abs(v_vel) * 20 + max(0, (1.25 - height)) * 50
        tilt_signal = tilt * 80.0

        for s in self.soma_L_starts:
            ext[0, s:s + self.npr] = max(0, base + tilt_signal)
        for s in self.soma_R_starts:
            ext[0, s:s + self.npr] = max(0, base - tilt_signal)

        # Brain step
        L_acc = torch.zeros(len(self.motor_L_idx), device=self.device)
        R_acc = torch.zeros(len(self.motor_R_idx), device=self.device)
        brain_steps = 20
        with torch.no_grad():
            for _ in range(brain_steps):
                self.state, spikes = self.brain.step(self.state, ext)
                L_acc += spikes[0, self.motor_L_idx].float()
                R_acc += spikes[0, self.motor_R_idx].float()

        self.last_spikes = spikes

        # Compute region rates for display
        for gname, gidx in self.display_groups.items():
            total = 0
            for ri in gidx:
                total += spikes[0, ri*self.npr:(ri+1)*self.npr].sum().item()
            self.region_rates[gname] = total / (len(gidx) * self.npr)

        L_rate = L_acc.sum().item() / (len(self.motor_L_idx) * brain_steps)
        R_rate = R_acc.sum().item() / (len(self.motor_R_idx) * brain_steps)
        L_drive = (L_rate - self.bl_L) * 30.0
        R_drive = (R_rate - self.bl_R) * 30.0

        action = np.zeros(6, dtype=np.float32)

        if mode == "brain+cpg":
            torques = self.cpg.step(2.0, brain_drive=0.2)
            action[0] = torques[0] * 0.3 + 0.08 + np.clip(R_drive * 0.3, -0.5, 0.5)
            action[1] = torques[1] * 0.2 + 0.1
            action[2] = torques[1] * 0.1
            action[3] = torques[2] * 0.3 + 0.08 + np.clip(L_drive * 0.3, -0.5, 0.5)
            action[4] = torques[3] * 0.2 + 0.1
            action[5] = torques[3] * 0.1
        elif mode == "brain_only":
            action[0] = np.clip(R_drive * 0.5, -1, 1)
            action[1] = np.clip(R_drive * 0.3, -1, 1)
            action[3] = np.clip(L_drive * 0.5, -1, 1)
            action[4] = np.clip(L_drive * 0.3, -1, 1)
        elif mode == "cpg_only":
            righting = -tilt * 3.0 - v_vel * 1.0
            torques = self.cpg.step(2.0, brain_drive=0.2)
            action[0] = torques[0] * 0.3 + 0.08 + righting * 0.3
            action[1] = torques[1] * 0.2 + 0.1
            action[2] = torques[1] * 0.1
            action[3] = torques[2] * 0.3 + 0.08 + righting * 0.3
            action[4] = torques[3] * 0.2 + 0.1
            action[5] = torques[3] * 0.1

        return np.clip(action, -1, 1)

    def draw_brain_panel(self, width=400, height=480):
        """Draw brain activity as a side panel."""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 40)  # dark background

        # Title
        cv2.putText(panel, "BRAIN ACTIVITY", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(panel, "16,000 neurons | 80 regions", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Draw bars for each group
        y = 80
        bar_max_w = width - 160
        colors = {
            'L': (100, 200, 255),   # blue for left
            'R': (255, 150, 100),   # orange for right
        }

        for gname, gidx in self.display_groups.items():
            rate = self.region_rates.get(gname, 0)
            tau_mean = np.mean([self.tau_m[i] for i in gidx])
            side = 'L' if '_L' in gname or ' L' in gname else 'R'
            color = colors[side]

            # Label
            cv2.putText(panel, gname, (10, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            # Bar
            bar_w = int(min(rate * bar_max_w * 20, bar_max_w))
            cv2.rectangle(panel, (140, y), (140 + bar_w, y + 16), color, -1)

            # Rate text
            cv2.putText(panel, f"{rate:.3f}", (140 + bar_w + 5, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

            # Tau label
            cv2.putText(panel, f"{tau_mean:.0f}ms", (width - 40, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

            y += 28

            # Separator between L/R pairs
            if side == 'R' and gname != list(self.display_groups.keys())[-1]:
                y += 8
                cv2.line(panel, (10, y), (width - 10, y), (60, 60, 70), 1)
                y += 8

        return panel


def record_video(mode="brain+cpg", duration_s=15, output="newborn_brain.mp4"):
    """Record video of brain-controlled Walker2d."""
    print(f"Recording {duration_s}s video: {output}")
    print(f"Mode: {mode}")

    viz = BrainVisualizer()
    env = gym.make('Walker2d-v5', render_mode='rgb_array')
    obs, _ = env.reset()

    max_steps = int(duration_s * 50)
    fps = 25  # video fps (env runs at 50Hz, we record every 2 steps)

    # Setup video writer
    body_frame = env.render()
    brain_panel = viz.draw_brain_panel()
    combined_w = body_frame.shape[1] + brain_panel.shape[1]
    combined_h = max(body_frame.shape[0], brain_panel.shape[0])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (combined_w, combined_h))

    total_reward = 0
    t0 = time.time()

    for step in range(max_steps):
        action = viz.step(obs, mode=mode)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        # Record every 2nd frame (25 fps from 50Hz sim)
        if step % 2 == 0:
            body_frame = env.render().copy()  # copy — MuJoCo returns read-only
            brain_panel = viz.draw_brain_panel()

            # Add status text to body frame
            cv2.putText(body_frame, f"Step: {step}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(body_frame, f"Tilt: {obs[1]:.3f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(body_frame, f"Height: {obs[0]:.2f}", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(body_frame, f"Reward: {total_reward:.0f}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            cv2.putText(body_frame, f"Mode: {mode}", (10, 465),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)

            # Combine
            combined = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
            combined[:body_frame.shape[0], :body_frame.shape[1]] = body_frame
            combined[:brain_panel.shape[0], body_frame.shape[1]:] = brain_panel

            # Convert RGB to BGR for cv2
            out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        if terminated or truncated:
            # Add "FELL" frame
            fell_frame = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
            fell_frame[:] = (30, 30, 40)
            cv2.putText(fell_frame, f"FELL at step {step}", (combined_w // 4, combined_h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(fell_frame, f"Survived {step/50:.1f}s  |  Baseline: 2.4s",
                        (combined_w // 6, combined_h // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            for _ in range(fps * 2):  # 2 seconds of "FELL" screen
                out.write(cv2.cvtColor(fell_frame, cv2.COLOR_RGB2BGR))
            break

    out.release()
    env.close()

    elapsed = time.time() - t0
    print(f"\nDone! Saved to {output}")
    print(f"  Survived: {step+1} steps ({(step+1)/50:.1f}s)")
    print(f"  Baseline: ~119 steps (2.4s)")
    print(f"  Reward: {total_reward:.0f}")
    print(f"  Wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="brain+cpg",
                        choices=["brain+cpg", "brain_only", "cpg_only"])
    parser.add_argument("--duration", type=int, default=15)
    parser.add_argument("--output", default="newborn_brain.mp4")
    args = parser.parse_args()

    record_video(mode=args.mode, duration_s=args.duration, output=args.output)

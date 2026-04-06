"""Newborn Brain Demo — Fly-inspired architecture.

A miniature human brain controlling a Walker2d body.
Same architecture as Eon Systems' fly: CPG handles locomotion,
brain provides high-level modulation through descending commands,
hardwired reflex arcs handle specific stimulus-response patterns.

The brain doesn't "learn to walk" — it modulates an innate walking
pattern (CPG) and triggers reflexes in response to sensory events.
Just like a real newborn.

Reflexes:
  - Righting: tilt detected → corrective CPG modulation
  - Startle: sudden sensory change → brief motor burst
  - Stepping: rhythmic CPG drive modulated by brain state
  - Orienting: asymmetric sensory input → asymmetric motor output

Usage:
  python newborn_demo.py              # Run 30-second demo with stats
  python newborn_demo.py --render     # Render Walker2d visually
  python newborn_demo.py --interactive # Interactive mode
"""

import argparse
import json
import sys
import time

import numpy as np
import torch
import gymnasium as gym

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU
from encephagen.spinal.cpg import SpinalCPG, CPGParams


class NewbornBrain:
    """A newborn human brain connected to a body.

    Architecture (fly-inspired):
      Sensory input → Brain (16K neurons, 80 regions) → Descending commands
      Descending commands → Reflex/CPG layer → Joint torques → Walker2d body
    """

    def __init__(self, device="cuda"):
        print("=" * 60)
        print("  NEWBORN BRAIN — 16,000 neurons controlling a body")
        print("=" * 60)

        # Load connectome
        sc = np.load('src/encephagen/connectome/bundled/neurolib80_weights.npy')
        tl = np.load('src/encephagen/connectome/bundled/neurolib80_tract_lengths.npy')
        labels = json.load(open('src/encephagen/connectome/bundled/neurolib80_labels.json'))
        tau_labels = json.load(open(
            'src/encephagen/connectome/bundled/neurolib80_t1t2_labels.json'))
        self.tau_m = np.load('src/encephagen/connectome/bundled/neurolib80_tau_m.npy')

        c = Connectome(sc, labels)
        c.tract_lengths = tl

        self.npr = 200
        self.n_regions = 80
        self.n_total = self.n_regions * self.npr
        self.device = device

        # Region groups
        self.groups = {}
        for key, patterns in [
            ('somatosensory', ['Postcentral', 'Paracentral']),
            ('motor', ['Precentral', 'Supp_Motor']),
            ('visual', ['Calcarine', 'Cuneus', 'Lingual', 'Occipital']),
            ('auditory', ['Heschl', 'Temporal_Sup']),
            ('frontal', ['Frontal_Sup', 'Frontal_Mid', 'Frontal_Inf']),
            ('parietal', ['Parietal', 'Angular', 'SupraMarginal']),
            ('temporal', ['Temporal_Mid', 'Temporal_Inf', 'Fusiform']),
            ('cingulate', ['Cingulate']),
        ]  :
            self.groups[key] = [i for i, l in enumerate(tau_labels)
                                if any(p in l for p in patterns)]

        # Build brain
        print("  Building brain...", flush=True)
        self.brain = SpikingBrainGPU(
            connectome=c, neurons_per_region=self.npr,
            internal_conn_prob=0.05, between_conn_prob=0.03,
            global_coupling=5.0, ext_rate_factor=3.5,
            pfc_regions=[], device=device,
            use_delays=True, conduction_velocity=3.5,
            use_t1t2_gradient=True,
        )

        # Precompute motor neuron indices for fast readout
        motor_idx = self.groups['motor']
        all_motor = []
        for ri in motor_idx:
            all_motor.extend(range(ri * self.npr, (ri + 1) * self.npr))
        self.n_motor = len(all_motor)
        self.motor_tensor = torch.tensor(all_motor, device=device, dtype=torch.long)
        self.neurons_per_action = self.n_motor // 6

        # Somatosensory input indices
        self.soma_starts = [ri * self.npr for ri in self.groups['somatosensory']]

        # CPG for walking
        self.cpg = SpinalCPG(CPGParams(
            tau=50.0, tau_adapt=500.0, drive=1.0,
            w_mutual=2.5, w_crossed=1.5, beta=2.5,
            w_hip_knee=0.7,
        ))

        # Warmup
        print("  Warming up brain + CPG...", flush=True)
        self.state = self.brain.init_state(batch_size=1)
        with torch.no_grad():
            for _ in range(2000):
                self.state, _ = self.brain.step(self.state)
        for _ in range(5000):
            self.cpg.step(0.1)

        # Reflex state
        self.prev_obs = None
        self.startle_timer = 0

        print("  Ready.\n")

    def sense(self, obs):
        """Convert Walker2d observation to brain input current.

        obs (17-dim):
          [0]: height
          [1]: angle (tilt)
          [2-7]: joint angles
          [8]: horizontal velocity
          [9]: vertical velocity
          [10-16]: joint velocities
        """
        ext = torch.zeros(1, self.n_total, device=self.device)

        # Proprioception → somatosensory regions
        tilt = abs(obs[1]) * 200.0          # strong tilt signal
        velocity = abs(obs[8]) * 50.0       # horizontal velocity
        height = max(0, (1.25 - obs[0])) * 100.0  # lower height = stronger signal
        joint_signal = np.abs(obs[2:8]).mean() * 30.0

        total_signal = min(tilt + velocity + height + joint_signal, 200.0)

        for s_start in self.soma_starts:
            ext[0, s_start:s_start + self.npr] = total_signal

        # Startle detection: sudden large change in observation
        if self.prev_obs is not None:
            delta = np.abs(obs - self.prev_obs).sum()
            if delta > 5.0:  # sudden change
                self.startle_timer = 5  # 5 brain cycles of startle
                # Flash ALL sensory regions
                for gname in ['somatosensory', 'visual', 'auditory']:
                    for ri in self.groups[gname]:
                        ext[0, ri*self.npr:(ri+1)*self.npr] = 150.0

        self.prev_obs = obs.copy()
        return ext

    def think(self, ext, brain_steps=20):
        """Run brain for brain_steps and extract motor commands."""
        motor_acc = torch.zeros(self.n_motor, device=self.device)
        with torch.no_grad():
            for _ in range(brain_steps):
                self.state, spikes = self.brain.step(self.state, ext)
                motor_acc += spikes[0, self.motor_tensor].float()

        # Extract 6 motor channel rates
        motor_rates = np.zeros(6)
        motor_np = motor_acc.cpu().numpy()
        for a in range(6):
            chunk = motor_np[a * self.neurons_per_action:(a + 1) * self.neurons_per_action]
            motor_rates[a] = chunk.sum() / (self.neurons_per_action * brain_steps)

        return motor_rates

    def act(self, motor_rates, obs):
        """Convert brain motor output + reflexes to joint torques.

        This is the reflex/CPG layer — analogous to the fly's
        descending neuron → pre-trained controller mapping.
        """
        # === Brain modulates CPG drive ===
        mean_motor = motor_rates.mean()
        brain_drive = (mean_motor - 0.04) * 30.0

        # === Righting reflex: tilt → corrective torque ===
        tilt = obs[1]  # body angle
        angular_vel = obs[9] if len(obs) > 9 else 0  # angular velocity
        # Gentle PD control — enough to correct, not enough to oscillate
        righting = -tilt * 3.0 - angular_vel * 1.0

        # === Startle: brief motor burst ===
        startle_boost = 0.0
        if self.startle_timer > 0:
            startle_boost = 2.0
            self.startle_timer -= 1

        # === CPG step ===
        cpg_drive = brain_drive + startle_boost
        torques = self.cpg.step(2.0, brain_drive=cpg_drive)  # 2ms CPG step

        # === Compose final action ===
        action = np.zeros(6, dtype=np.float32)

        # Torques from CPG (walking rhythm)
        action[0] = torques[2] * 0.4   # left hip (thigh)
        action[1] = torques[3] * 0.3   # left knee (leg)
        action[2] = 0.0                # left ankle (minimal)
        action[3] = torques[0] * 0.4   # right hip (thigh)
        action[4] = torques[1] * 0.3   # right knee (leg)
        action[5] = 0.0                # right ankle (minimal)

        # Righting reflex — hips push against tilt
        action[0] += righting * 0.3
        action[3] += righting * 0.3

        # Knee stabilization — keep knees slightly bent (prevents hyperextension)
        action[1] += 0.1  # slight constant knee flexion
        action[4] += 0.1

        # Brain motor modulation
        for a in range(6):
            action[a] += (motor_rates[a] - 0.04) * 5.0

        # Clip to valid range
        action = np.clip(action, -1.0, 1.0)

        return action

    def get_brain_status(self):
        """Get current brain region activity for display."""
        rates = {}
        with torch.no_grad():
            for _ in range(100):
                self.state, spikes = self.brain.step(self.state)
            for gname, gidx in self.groups.items():
                total = 0
                for ri in gidx:
                    total += spikes[0, ri*self.npr:(ri+1)*self.npr].sum().item()
                rates[gname] = total / (len(gidx) * self.npr)
        return rates


def run_demo(render=False, duration_s=30):
    """Run the newborn brain-body demo."""
    newborn = NewbornBrain()

    render_mode = "human" if render else None
    env = gym.make('Walker2d-v5', render_mode=render_mode)
    obs, _ = env.reset()

    total_reward = 0.0
    steps = 0
    max_steps = int(duration_s * 50)  # Walker2d runs at ~50Hz

    # Stats tracking
    heights = []
    tilts = []
    rewards = []

    print(f"  Running for {duration_s}s ({max_steps} steps)...")
    print(f"  Baseline (no brain): ~119 steps before falling")
    print()

    t0 = time.time()

    for step in range(max_steps):
        # Sense → Think → Act
        ext = newborn.sense(obs)
        motor_rates = newborn.think(ext, brain_steps=20)
        action = newborn.act(motor_rates, obs)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Track stats
        heights.append(obs[0])
        tilts.append(abs(obs[1]))
        rewards.append(reward)

        # Print status every 5 seconds (250 steps)
        if step > 0 and step % 250 == 0:
            elapsed = time.time() - t0
            avg_height = np.mean(heights[-250:])
            avg_tilt = np.mean(tilts[-250:])
            avg_reward = np.mean(rewards[-250:])
            brain_rates = newborn.get_brain_status()

            print(f"  t={step/50:.0f}s  height={avg_height:.2f}  "
                  f"tilt={avg_tilt:.3f}  reward={avg_reward:.1f}  "
                  f"steps={step}")

            # Brain activity bars
            for gname in ['somatosensory', 'motor', 'visual', 'frontal']:
                r = brain_rates.get(gname, 0)
                bar = "█" * int(r * 200)
                tau = np.mean([newborn.tau_m[i] for i in newborn.groups[gname]])
                print(f"    {gname:<15} {bar:<12} ({r:.3f}, tau={tau:.0f}ms)")
            print()

        if terminated or truncated:
            print(f"  FELL at step {step} ({step/50:.1f}s)")
            break

    elapsed = time.time() - t0
    env.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Survived: {steps} steps ({steps/50:.1f}s)")
    print(f"  Baseline: ~119 steps (2.4s)")
    print(f"  Improvement: {steps/119:.1f}x baseline")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Mean height: {np.mean(heights):.3f}")
    print(f"  Mean tilt: {np.mean(tilts):.3f}")
    print(f"  Wall time: {elapsed:.1f}s")
    print(f"  Brain speed: {steps*20/(elapsed):.0f} neural steps/s")

    return steps, total_reward


def run_interactive():
    """Interactive mode — type commands while brain controls body."""
    newborn = NewbornBrain()
    env = gym.make('Walker2d-v5')
    obs, _ = env.reset()

    print("  Interactive mode. Brain is controlling the body.")
    print("  Commands: status, startle, reset, run <N>, quit")
    print()

    total_steps = 0
    episode_steps = 0

    while True:
        try:
            cmd = input("newborn> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue
        elif cmd == 'quit':
            break
        elif cmd == 'status':
            rates = newborn.get_brain_status()
            print(f"  Steps alive: {episode_steps} (total: {total_steps})")
            for gn, r in sorted(rates.items()):
                bar = "█" * int(r * 200)
                print(f"    {gn:<15} {bar:<12} ({r:.3f})")
        elif cmd == 'startle':
            newborn.startle_timer = 10
            print("  STARTLE triggered!")
        elif cmd == 'reset':
            obs, _ = env.reset()
            episode_steps = 0
            newborn.cpg.reset()
            for _ in range(500):
                newborn.cpg.step(0.1)
            print("  Environment reset.")
        elif cmd.startswith('run'):
            parts = cmd.split()
            n = int(parts[1]) if len(parts) > 1 else 100
            for _ in range(n):
                ext = newborn.sense(obs)
                motor = newborn.think(ext, brain_steps=20)
                action = newborn.act(motor, obs)
                obs, reward, term, trunc, info = env.step(action)
                episode_steps += 1
                total_steps += 1
                if term or trunc:
                    print(f"  FELL at step {episode_steps}")
                    obs, _ = env.reset()
                    episode_steps = 0
                    break
            else:
                print(f"  Ran {n} steps. Height={obs[0]:.2f}, tilt={obs[1]:.3f}")
        else:
            print(f"  Unknown: {cmd}")

    env.close()
    print(f"\n  Total steps: {total_steps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render Walker2d")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    else:
        run_demo(render=args.render, duration_s=args.duration)

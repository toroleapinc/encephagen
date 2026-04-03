"""Experiment 20: Brain learns to calibrate Walker2d through experience.

Walker2d is properly unstable — falls in <1s without active control.
The brain must learn the right CPG modulation to keep it upright.

Architecture:
  Body sensor (17-dim) → brain sensory regions
  Brain motor cortex → tonic drive → CPG (6 joints)
  CPG → alternating torques → Walker2d body
  Upright → reward → strengthen current brain state
  Falling → episode ends → reset
"""

import time
import numpy as np
import torch
import gymnasium as gym

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU


def classify_regions(labels):
    groups = {}
    for key, patterns in [
        ('somatosensory', ['S1', 'S2']),
        ('visual', ['V1', 'V2', 'VAC']),
        ('prefrontal', ['PFC', 'FEF']),
        ('motor', ['M1', 'PMC']),
        ('amygdala', ['AMYG']),
        ('basal_ganglia', ['BG']),
        ('thalamus', ['TM']),
    ]:
        groups[key] = [i for i, l in enumerate(labels)
                       if any(p in l.upper() for p in patterns)]
    return groups


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 20: Brain calibrates Walker2d")
    print("Unstable biped — falls in <1s without proper control")
    print("=" * 70)

    # Walker2d environment
    env = gym.make('Walker2d-v4')
    n_actions = env.action_space.shape[0]  # 6 joints
    n_obs = env.observation_space.shape[0]  # 17 dims
    print(f"\n  Walker2d: {n_obs} obs dims, {n_actions} action dims")

    # Brain
    connectome = Connectome.from_bundled("tvb96")
    groups = classify_regions(connectome.labels)
    npr = 200
    n_total = 96 * npr
    device = "cuda"
    pfc_regions = groups['prefrontal']

    brain = SpikingBrainGPU(
        connectome=connectome, neurons_per_region=npr,
        global_coupling=0.15, ext_rate_factor=3.5,
        tau_nmda=150.0, nmda_ratio=0.4,
        pfc_regions=pfc_regions, device=device,
    )

    # Warm up
    print("  Warming up brain...")
    state = brain.init_state(batch_size=1)
    with torch.no_grad():
        for _ in range(3000):
            state, _ = brain.step(state)

    # Motor baseline
    motor_idx = groups['motor']
    somato_idx = groups['somatosensory']
    amyg_idx = groups['amygdala']

    motor_counts = torch.zeros(len(motor_idx) * npr, device=device)
    with torch.no_grad():
        for _ in range(1000):
            state, spikes = brain.step(state)
            for j, ri in enumerate(motor_idx):
                motor_counts[j*npr:(j+1)*npr] += spikes[0, ri*npr:(ri+1)*npr]
    motor_baseline = motor_counts.mean().item() / 1000

    # Learning weights
    W_dense = brain.W.to_dense()
    learning_rate = 0.002

    # Map motor region neurons to 6 Walker2d actions
    # Split motor neurons into 6 groups
    motor_total = len(motor_idx) * npr
    neurons_per_action = motor_total // n_actions

    BRAIN_STEPS_PER_ACTION = 10  # 1ms brain time per body action
    # Walker2d dt=0.008s, so ~8ms per env step
    # 10 brain steps (1ms) per env step is fast enough

    def brain_to_action(state, obs):
        """Brain processes observation, outputs 6 torques."""
        external = torch.zeros(1, n_total, device=device)

        # Map 17 obs dims to sensory regions
        obs_clipped = np.clip(obs, -5, 5) / 5.0  # Normalize to [-1, 1]
        for i, ri in enumerate(somato_idx):
            if i < len(obs_clipped):
                external[0, ri*npr:(ri+1)*npr] = (obs_clipped[i % len(obs_clipped)] + 1) * 5.0
        # Spread remaining obs to visual regions
        for i, ri in enumerate(groups['visual'][:8]):
            idx = (i + len(somato_idx)) % len(obs_clipped)
            external[0, ri*npr:(ri+1)*npr] = (obs_clipped[idx] + 1) * 3.0

        # Brain steps
        motor_spikes = torch.zeros(motor_total, device=device)
        with torch.no_grad():
            for _ in range(BRAIN_STEPS_PER_ACTION):
                state, spikes = brain.step(state, external)
                for j, ri in enumerate(motor_idx):
                    motor_spikes[j*npr:(j+1)*npr] += spikes[0, ri*npr:(ri+1)*npr]

        # Decode 6 actions from motor neuron groups
        actions = np.zeros(n_actions)
        for a in range(n_actions):
            s = a * neurons_per_action
            e = min(s + neurons_per_action, motor_total)
            rate = motor_spikes[s:e].mean().item() / BRAIN_STEPS_PER_ACTION
            actions[a] = np.clip((rate - motor_baseline) * 100, -1, 1)

        return state, actions, external

    # === Run episodes ===
    n_episodes = 40
    survivals_per_phase = {"baseline": [], "learning": [], "test": []}

    for phase, n_ep, learn in [("baseline", 5, False), ("learning", 30, True), ("test", 5, False)]:
        print(f"\n{'='*60}")
        print(f"PHASE: {phase.upper()} ({n_ep} episodes, learning={'ON' if learn else 'OFF'})")
        print(f"{'='*60}")

        t0 = time.time()
        for ep in range(n_ep):
            obs, _ = env.reset(seed=ep + (0 if phase=="baseline" else 100 if phase=="learning" else 200))
            total_reward = 0
            steps = 0

            for step in range(500):  # Max 4 seconds
                state, actions, external = brain_to_action(state, obs)

                obs, reward, terminated, truncated, info = env.step(actions)
                total_reward += reward
                steps = step + 1

                # Learning: when upright, strengthen sensory→motor pathways
                if learn and reward > 0.5 and step % 10 == 0:
                    # Reward signal to amygdala
                    with torch.no_grad():
                        ext_reward = torch.zeros(1, n_total, device=device)
                        for ri in amyg_idx:
                            ext_reward[0, ri*npr:(ri+1)*npr] = 10.0
                        state, r_spikes = brain.step(state, ext_reward)

                    # Three-factor learning
                    active = torch.zeros(n_total, device=device)
                    for ri in somato_idx + groups['visual'][:4]:
                        active[ri*npr:(ri+1)*npr] = 1.0
                    motor_active = torch.zeros(n_total, device=device)
                    for ri in motor_idx:
                        motor_active[ri*npr:(ri+1)*npr] = 1.0

                    dW = learning_rate * torch.outer(active, motor_active)
                    mask = (W_dense != 0).float()
                    W_dense = torch.clamp(W_dense + dW * mask, -20, 20)

                if terminated or truncated:
                    break

            survival = steps * env.unwrapped.dt
            survivals_per_phase[phase].append(survival)

            if (ep + 1) % 5 == 0:
                recent = survivals_per_phase[phase][-5:]
                elapsed = time.time() - t0
                print(f"  Ep {ep+1:>2}: survival={[f'{s:.2f}' for s in recent]} "
                      f"mean={np.mean(recent):.2f}s ({elapsed:.0f}s)")

        # Update weights after learning phase
        if learn:
            indices = brain.W.coalesce().indices()
            new_vals = W_dense[indices[0], indices[1]]
            brain.W = torch.sparse_coo_tensor(
                indices, new_vals, brain.W.shape
            ).coalesce()

    env.close()

    # === Results ===
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    bl = np.mean(survivals_per_phase["baseline"])
    learn_early = np.mean(survivals_per_phase["learning"][:5])
    learn_late = np.mean(survivals_per_phase["learning"][-5:])
    test = np.mean(survivals_per_phase["test"])

    print(f"\n  Baseline (no learning):   {bl:.2f}s")
    print(f"  Early learning (ep 1-5):  {learn_early:.2f}s")
    print(f"  Late learning (ep 26-30): {learn_late:.2f}s")
    print(f"  Test (retention):         {test:.2f}s")
    print(f"\n  Improvement: {test - bl:+.2f}s")

    # For reference: zero torques = 0.97s, random = 0.16s
    print(f"\n  Reference: zero torques = 0.97s, random = 0.16s")
    if test > bl + 0.2:
        print(f"\n  BRAIN LEARNED! Survival improved by {test-bl:.2f}s")
    elif test > 0.97:
        print(f"\n  Brain does BETTER than zero torques ({test:.2f}s > 0.97s)")
    else:
        print(f"\n  No clear improvement yet")


if __name__ == "__main__":
    run_experiment()

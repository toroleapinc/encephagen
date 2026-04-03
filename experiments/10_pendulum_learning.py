"""Experiment 10: Can 1000 spiking neurons learn to balance a pendulum?

The simplest possible motor learning task. If this doesn't work,
nothing more complex will.

Architecture:
  2 sensory neurons (angle, velocity) -> 1000 LIF neurons -> 1 motor output
  All on GPU. Surrogate gradient training with REINFORCE.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn


class SimplePendulum:
    """Inverted pendulum: keep a pole balanced upright."""

    def __init__(self, dt=0.02):
        self.dt = dt
        self.gravity = 9.81
        self.length = 1.0
        self.mass = 1.0
        self.max_torque = 2.0
        self.angle = 0.0
        self.velocity = 0.0
        self.steps = 0

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        self.angle = rng.uniform(-0.1, 0.1)
        self.velocity = rng.uniform(-0.1, 0.1)
        self.steps = 0
        return np.array([self.angle, self.velocity], dtype=np.float32)

    def step(self, action):
        torque = np.clip(action, -1, 1) * self.max_torque
        acc = (self.gravity * np.sin(self.angle) + torque) / (self.mass * self.length ** 2)
        self.velocity += acc * self.dt
        self.angle += self.velocity * self.dt
        self.steps += 1
        upright = abs(self.angle) < 0.2
        reward = 1.0 if upright else -1.0
        reward -= 0.01 * action ** 2
        done = abs(self.angle) > 1.0 or self.steps >= 500
        return np.array([self.angle, self.velocity], dtype=np.float32), float(reward), done


class SurrogateSpike(torch.autograd.Function):
    """Surrogate gradient: Heaviside forward, sigmoid backward."""
    @staticmethod
    def forward(ctx, v, threshold):
        ctx.save_for_backward(v, torch.tensor(threshold))
        return (v >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        v, threshold = ctx.saved_tensors
        sigmoid = torch.sigmoid(5.0 * (v - threshold))
        grad = sigmoid * (1 - sigmoid) * 5.0
        return grad_output * grad, None


class SpikingPendulumBrain(nn.Module):
    """1000-neuron spiking network for pendulum control."""

    def __init__(self, n_neurons=1000, device="cuda"):
        super().__init__()
        self.n = n_neurons
        self.dt = 0.1
        self.v_threshold = 20.0
        self.tau_m = 20.0
        self.tau_syn = 5.0
        self.device = torch.device(device)

        self.W_in = nn.Linear(2, n_neurons, bias=False)
        self.W_rec = nn.Linear(n_neurons, n_neurons, bias=False)
        with torch.no_grad():
            mask = torch.rand(n_neurons, n_neurons) < 0.1
            mask.fill_diagonal_(False)
            self.W_rec.weight *= mask.float() * 0.5
        self.W_out = nn.Linear(n_neurons, 1, bias=False)
        self.to(self.device)

    def init_state(self):
        return {
            "v": torch.zeros(1, self.n, device=self.device),
            "i_syn": torch.zeros(1, self.n, device=self.device),
            "last_spikes": torch.zeros(1, self.n, device=self.device),
        }

    def step(self, state, sensory_input):
        v = state["v"]
        i_syn = state["i_syn"]

        i_in = self.W_in(sensory_input) * 5.0
        i_rec = self.W_rec(state["last_spikes"])
        i_syn = i_syn * np.exp(-self.dt / self.tau_syn) + i_in + i_rec

        dv = (-v + i_syn) / self.tau_m
        v = v + self.dt * dv

        spikes = SurrogateSpike.apply(v, self.v_threshold)
        v = v * (1 - spikes)

        motor = torch.tanh(self.W_out(spikes))

        new_state = {
            "v": v,
            "i_syn": i_syn,
            "last_spikes": spikes.detach(),
        }
        return new_state, spikes, motor

    def run_episode(self, env, n_brain_steps=50):
        obs = env.reset()
        state = self.init_state()
        total_reward = 0.0
        actions = []
        while True:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            motor_acc = torch.zeros(1, 1, device=self.device)
            for _ in range(n_brain_steps):
                state, spikes, motor = self.step(state, obs_t)
                motor_acc += motor
            action = (motor_acc / n_brain_steps).squeeze().item()
            obs, reward, done = env.step(action)
            total_reward += reward
            actions.append(action)
            if done:
                break
        return total_reward, actions


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 10: Can 1000 spiking neurons balance a pendulum?")
    print("Surrogate gradient + REINFORCE on GPU")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    env = SimplePendulum(dt=0.02)
    brain = SpikingPendulumBrain(n_neurons=1000, device=device)
    optimizer = torch.optim.Adam(brain.parameters(), lr=1e-3)

    n_episodes = 200
    n_brain_steps = 50
    best_reward = -float("inf")
    reward_history = []

    print(f"\nTraining for {n_episodes} episodes...")
    print(f"Brain: 1000 LIF neurons, surrogate gradient")
    print(f"Task: keep pendulum upright (|angle| < 0.2 rad)")
    print()

    t0 = time.time()

    for ep in range(n_episodes):
        brain.train()
        obs = env.reset(seed=ep)
        state = brain.init_state()
        total_reward = 0.0
        log_probs = []
        rewards = []

        while True:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            motor_acc = torch.zeros(1, 1, device=device)
            for _ in range(n_brain_steps):
                state, spikes, motor = brain.step(state, obs_t)
                motor_acc += motor
            action_mean = motor_acc / n_brain_steps
            action_noise = action_mean + torch.randn_like(action_mean) * 0.3
            action = torch.tanh(action_noise).squeeze().item()
            log_prob = -0.5 * ((action_noise - action_mean) ** 2).sum()
            log_probs.append(log_prob)
            obs, reward, done = env.step(action)
            rewards.append(reward)
            total_reward += reward
            if done:
                break

        reward_history.append(total_reward)

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.tensor(returns, device=device)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = sum(-lp * ret for lp, ret in zip(log_probs, returns)) / len(returns)

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimizer.step()

        if total_reward > best_reward:
            best_reward = total_reward

        if (ep + 1) % 20 == 0:
            recent = reward_history[-20:]
            elapsed = time.time() - t0
            print(f"  Ep {ep+1:>4}/{n_episodes}  "
                  f"mean_reward={np.mean(recent):>+7.1f}  "
                  f"max={max(recent):>+7.1f}  "
                  f"best={best_reward:>+7.1f}  "
                  f"({elapsed:.0f}s)")

    # Results
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    baseline = reward_history[:20]
    final = reward_history[-20:]
    print(f"\n  First 20 episodes:  mean={np.mean(baseline):>+7.1f}  max={max(baseline):>+7.1f}")
    print(f"  Last 20 episodes:   mean={np.mean(final):>+7.1f}  max={max(final):>+7.1f}")
    print(f"  Improvement:        {np.mean(final) - np.mean(baseline):>+7.1f}")

    if np.mean(final) > np.mean(baseline) + 20:
        print(f"\n  LEARNING DETECTED: Spiking network improved at pendulum control")
    elif np.mean(final) > np.mean(baseline) + 5:
        print(f"\n  MARGINAL IMPROVEMENT")
    else:
        print(f"\n  NO LEARNING: Spiking network did not improve")

    # Test
    print(f"\n  Testing learned policy (no noise)...")
    brain.eval()
    test_rewards = []
    test_steps = []
    with torch.no_grad():
        for i in range(10):
            r, actions = brain.run_episode(env, n_brain_steps)
            test_rewards.append(r)
            test_steps.append(len(actions))

    print(f"  Test: mean_reward={np.mean(test_rewards):>+7.1f}  "
          f"mean_steps={np.mean(test_steps):.0f}  "
          f"max_steps={max(test_steps)}")

    if np.mean(test_steps) > 200:
        print(f"\n  SUCCESS: Spiking brain balances pendulum for {np.mean(test_steps):.0f} steps")
    elif np.mean(test_steps) > 50:
        print(f"\n  PARTIAL: Balances for {np.mean(test_steps):.0f} steps (target: 200+)")
    else:
        print(f"\n  FAILED: Only {np.mean(test_steps):.0f} steps")


if __name__ == "__main__":
    run_experiment()

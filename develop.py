"""Developmental Runner: 先天 → 后天 transition.

Simulates months of infant development:
  - Runs the integrated brain + body through episodes
  - All 7 learning rules active simultaneously
  - Takes snapshots at milestones
  - Tracks what capabilities emerge

Usage:
  python develop.py --months 4 --video
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

from encephagen.brain import IntegratedBrain
from encephagen.learning.plasticity import DistributedPlasticity
from encephagen.snapshot import snapshot_brain, compare_snapshots


def run_episode(brain, plasticity, env, max_steps=300):
    """Run one episode with all learning active."""
    obs, _ = env.reset()
    total_reward = 0
    steps = 0

    for step in range(max_steps):
        sensory = {
            'somatosensory': min(abs(obs[1]) * 5, 1.0),
            'height': obs[0],
            'tilt_fb': obs[1],
            'angular_vel': obs[9] if len(obs) > 9 else 0,
            'visual': 0.1,
            'threat': min(abs(obs[1]) * 2, 1.0) if abs(obs[1]) > 0.3 else 0,
            'reward': max(0, 1.0 - abs(obs[1]) * 5),  # reward for being upright
        }

        output = brain.step(sensory)

        # Apply all learning rules
        plasticity.step(output, sensory)

        # Motor output
        action = np.zeros(6, dtype=np.float32)
        reflexes = output['reflexes']
        r = reflexes.get('righting', 0)
        drive = reflexes.get('stepping_drive', 0)

        action[0] = np.clip(output['cpg_right'] * drive * 0.15 + 0.08 + r * 0.3, -1, 1)
        action[1] = 0.1
        action[3] = np.clip(output['cpg_left'] * drive * 0.15 + 0.08 + r * 0.3, -1, 1)
        action[4] = 0.1

        if output['fear_level'] > 0.3:
            action *= 0.5

        obs, reward, term, trunc, _ = env.step(np.clip(action, -1, 1))
        total_reward += reward
        steps += 1

        if term or trunc:
            break

    return steps, total_reward


def run_development(n_months=4, episodes_per_month=50, video_months=None):
    """Run developmental simulation."""
    print("=" * 60)
    print("  DEVELOPMENTAL SIMULATION: 先天 → 后天")
    print(f"  {n_months} months × {episodes_per_month} episodes")
    print("=" * 60)

    brain = IntegratedBrain()
    plasticity = DistributedPlasticity(brain)

    # Warmup
    for _ in range(500):
        brain.step({'height': 1.25, 'tilt_fb': 0.0})

    # Innate baseline
    print("\n  Capturing innate baseline...")
    env = gym.make('Walker2d-v5')
    baseline_steps = []
    for _ in range(5):
        s, _ = run_episode(brain, plasticity, env, max_steps=300)
        baseline_steps.append(s)
    innate_mean = np.mean(baseline_steps)
    print(f"  Innate survival: {baseline_steps} mean={innate_mean:.0f}")

    # Development log
    dev_log = {
        'innate_baseline': float(innate_mean),
        'months': [],
    }

    t0 = time.time()

    for month in range(1, n_months + 1):
        print(f"\n{'='*50}")
        print(f"  MONTH {month}")
        print(f"{'='*50}")

        month_steps = []
        month_rewards = []

        for ep in range(episodes_per_month):
            steps, reward = run_episode(brain, plasticity, env, max_steps=300)
            month_steps.append(steps)
            month_rewards.append(reward)

            if (ep + 1) % 10 == 0:
                recent_mean = np.mean(month_steps[-10:])
                stats = plasticity.get_learning_stats()
                print(f"    Ep {ep+1:>3}: mean={recent_mean:.0f} "
                      f"DA={brain.bg.dopamine:.2f} "
                      f"fear={brain.amygdala.fear_level:.2f}", flush=True)

        month_mean = np.mean(month_steps)
        improvement = month_mean / innate_mean

        # Snapshot at end of month
        snapshot_brain(brain, name=f"month_{month}",
                       description=f"After {month} month(s) of development. "
                                   f"Survival: {month_mean:.0f} steps ({improvement:.1f}x innate)")

        dev_log['months'].append({
            'month': month,
            'mean_survival': float(month_mean),
            'std_survival': float(np.std(month_steps)),
            'improvement_vs_innate': float(improvement),
            'mean_reward': float(np.mean(month_rewards)),
            'learning_stats': plasticity.get_learning_stats(),
        })

        elapsed = time.time() - t0
        print(f"\n  Month {month} summary:")
        print(f"    Survival: {month_mean:.0f} ± {np.std(month_steps):.0f} steps")
        print(f"    vs innate: {improvement:.2f}x")
        print(f"    Time: {elapsed:.0f}s")

    env.close()

    # Final comparison
    print(f"\n{'='*60}")
    print(f"  DEVELOPMENT COMPLETE: {n_months} months")
    print(f"{'='*60}")
    print(f"\n  {'Month':<10} {'Survival':>10} {'vs Innate':>12}")
    print(f"  {'─'*35}")
    print(f"  {'Innate':<10} {innate_mean:>10.0f} {'1.0x':>12}")
    for m in dev_log['months']:
        print(f"  Month {m['month']:<5} {m['mean_survival']:>10.0f} "
              f"{m['improvement_vs_innate']:>11.2f}x")

    # Compare innate vs final
    print(f"\n  Comparing innate → month {n_months}...")
    compare_snapshots(
        f"snapshots/innate_baseline.json",
        f"snapshots/month_{n_months}.json"
    )

    # Save dev log
    with open("snapshots/development_log.json", "w") as f:
        json.dump(dev_log, f, indent=2)
    print(f"\n  Development log: snapshots/development_log.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()
    run_development(n_months=args.months, episodes_per_month=args.episodes)

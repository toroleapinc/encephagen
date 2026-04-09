"""Developmental Runner with Sleep + Sensorimotor Delays.

The key insight from the expert panel: motor learning requires REST.
The cortex consolidates motor skills during sleep, not during training.

Protocol:
  For each "day":
    1. AWAKE phase: 20 Walker2d episodes with sensorimotor delays
    2. SLEEP phase: 5000 steps of offline consolidation
    3. Measure: M1-cerebellum coherence + survival improvement
"""

import numpy as np
import torch
import json, time
import gymnasium as gym

from encephagen.brain import IntegratedBrain
from encephagen.learning.plasticity import DistributedPlasticity
from encephagen.learning.sleep import SleepConsolidation
from encephagen.loop.sensorimotor_delay import SensorimotorDelay
from encephagen.snapshot import snapshot_brain, compare_snapshots


def run_episode(brain, plasticity, delay, env, max_steps=300):
    obs, _ = env.reset()
    delay.init(obs.shape, (6,))
    total_reward = 0

    for step in range(max_steps):
        # DELAYED sensory input (afferent delay)
        delayed_obs = delay.delay_sensory(obs)

        sensory = {
            'somatosensory': min(abs(delayed_obs[1]) * 5, 1.0),
            'height': delayed_obs[0],
            'tilt_fb': delayed_obs[1],
            'angular_vel': delayed_obs[9] if len(delayed_obs) > 9 else 0,
            'visual': 0.1,
            'threat': min(abs(delayed_obs[1]) * 2, 1.0) if abs(delayed_obs[1]) > 0.3 else 0,
            'reward': max(0, 1.0 - abs(delayed_obs[1]) * 5),
        }

        output = brain.step(sensory)
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

        # DELAYED motor output (efferent delay)
        delayed_action = delay.delay_motor(action)

        obs, reward, term, trunc, _ = env.step(np.clip(delayed_action, -1, 1))
        total_reward += reward

        if term or trunc:
            break

    return step + 1, total_reward


def main():
    print("=" * 60)
    print("  DEVELOPMENT WITH SLEEP + SENSORIMOTOR DELAYS")
    print("  The brain learns during training, consolidates during sleep")
    print("=" * 60)

    brain = IntegratedBrain()
    plasticity = DistributedPlasticity(brain)
    sleep = SleepConsolidation(brain)
    delay = SensorimotorDelay(sensory_delay_ms=30, motor_delay_ms=20)

    # Warmup
    for _ in range(500):
        brain.step({'height': 1.25, 'tilt_fb': 0.0})

    env = gym.make('Walker2d-v5')

    # Baseline (no delays, no sleep)
    baseline_steps = []
    delay_off = SensorimotorDelay(sensory_delay_ms=0, motor_delay_ms=0)
    for _ in range(5):
        s, _ = run_episode(brain, plasticity, delay_off, env)
        baseline_steps.append(s)
    innate_mean = np.mean(baseline_steps)
    print(f"\n  Innate baseline: {baseline_steps} mean={innate_mean:.0f}")

    # Development: days with awake + sleep cycles
    N_DAYS = 30
    EPISODES_PER_DAY = 10
    SLEEP_STEPS = 3000

    dev_log = {
        'innate': float(innate_mean),
        'days': [],
    }
    t0 = time.time()

    for day in range(1, N_DAYS + 1):
        print(f"\n{'='*50}")
        print(f"  DAY {day}")
        print(f"{'='*50}")

        # AWAKE: training episodes with sensorimotor delays
        print(f"  AWAKE: {EPISODES_PER_DAY} episodes (with 30ms sensory + 20ms motor delay)")
        day_steps = []
        for ep in range(EPISODES_PER_DAY):
            s, r = run_episode(brain, plasticity, delay, env, max_steps=300)
            day_steps.append(s)

        awake_mean = np.mean(day_steps)
        print(f"    Mean survival: {awake_mean:.0f} steps ({awake_mean/innate_mean:.2f}x innate)")

        # SLEEP: offline consolidation
        print(f"  SLEEP: {SLEEP_STEPS} steps of offline consolidation...")
        sleep_result = sleep.sleep_episode(duration_steps=SLEEP_STEPS)
        coherence = sleep_result['coherence_3_6hz']
        consolidation = sleep_result['consolidation_events']
        print(f"    Consolidation events: {consolidation}")
        print(f"    M1-Cerebellum coherence: {coherence:.4f} "
              f"({'emerging!' if coherence > 0.1 else 'not yet'})")

        # POST-SLEEP test (does sleep help?)
        print(f"  POST-SLEEP TEST:")
        post_steps = []
        for _ in range(5):
            s, _ = run_episode(brain, plasticity, delay, env, max_steps=300)
            post_steps.append(s)
        post_mean = np.mean(post_steps)
        print(f"    Post-sleep survival: {post_mean:.0f} ({post_mean/innate_mean:.2f}x innate)")

        elapsed = time.time() - t0
        dev_log['days'].append({
            'day': day,
            'awake_mean': float(awake_mean),
            'post_sleep_mean': float(post_mean),
            'improvement': float(post_mean / innate_mean),
            'coherence': float(coherence),
            'consolidation_events': consolidation,
            'elapsed_s': elapsed,
        })

        print(f"    Time: {elapsed:.0f}s")

    env.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"  DEVELOPMENT SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Day':>5} {'Awake':>8} {'Post-Sleep':>12} {'vs Innate':>10} {'Coherence':>10}")
    print(f"  {'─'*48}")
    print(f"  {'Innate':>5} {innate_mean:>8.0f}")
    for d in dev_log['days']:
        print(f"  {d['day']:>5} {d['awake_mean']:>8.0f} {d['post_sleep_mean']:>12.0f} "
              f"{d['improvement']:>9.2f}x {d['coherence']:>10.4f}")

    # Did sleep help?
    awake_all = [d['awake_mean'] for d in dev_log['days']]
    postsleep_all = [d['post_sleep_mean'] for d in dev_log['days']]
    if len(awake_all) > 3:
        from scipy import stats
        _, p = stats.mannwhitneyu(postsleep_all[-3:], awake_all[:3], alternative='greater')
        print(f"\n  Post-sleep (last 3 days) vs awake (first 3 days): p={p:.4f}")

    # Coherence trend
    coherences = [d['coherence'] for d in dev_log['days']]
    if coherences[-1] > coherences[0] + 0.01:
        print(f"  M1-Cerebellum coherence: INCREASING ({coherences[0]:.4f} → {coherences[-1]:.4f})")
        print(f"  → Motor skill consolidation IS occurring!")
    else:
        print(f"  M1-Cerebellum coherence: flat ({coherences[0]:.4f} → {coherences[-1]:.4f})")

    # Snapshot
    snapshot_brain(brain, name="post_sleep_training",
                   description=f"After {N_DAYS} days of awake training + sleep consolidation. "
                               f"Sensorimotor delays: 30ms afferent + 20ms efferent.")

    # Compare to innate
    print(f"\n  Comparing to innate baseline...")
    compare_snapshots("snapshots/innate_baseline.json", "snapshots/post_sleep_training.json")

    with open("snapshots/sleep_development_log.json", "w") as f:
        json.dump(dev_log, f, indent=2)


if __name__ == "__main__":
    main()

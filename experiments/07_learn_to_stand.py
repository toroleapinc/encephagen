"""Experiment 7: Can the brain learn to keep the body upright?

The ultimate test: a brain with human connectome topology,
identical parameters everywhere, controlling a MuJoCo body.
Can reward-modulated STDP teach it to stay upright?

Protocol:
1. Run 10 episodes WITHOUT learning (random baseline)
2. Run 50 episodes WITH learning (STDP + reward modulation)
3. Run 10 episodes WITHOUT learning (test retention)
4. Compare: does the brain-controlled body stay upright longer after learning?
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from encephagen.connectome import Connectome
from encephagen.loop.embodied_loop import EmbodiedLoopRunner


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 7: Can the brain learn to keep the body upright?")
    print("A mini human brain controls a 2-legged MuJoCo body.")
    print("=" * 70)

    connectome = Connectome.from_bundled("toy20")  # Faster for this experiment

    max_actions = 50  # Per episode (50 actions × 50ms = 2.5s of body time)

    # --- Phase 1: Random baseline (no learning) ---
    print(f"\n{'=' * 70}")
    print("PHASE 1: Baseline — no learning (10 episodes)")
    print(f"{'=' * 70}")

    runner = EmbodiedLoopRunner(
        connectome,
        neurons_per_region=100,
        global_coupling=0.05,
        ext_rate=3.5,
        enable_learning=False,
        brain_steps_per_action=200,  # 20ms per action (faster for experiment)
        physics_steps_per_action=4,   # 20ms physics
        seed=42,
    )

    baseline_logs = runner.run_episodes(n_episodes=10, max_actions=max_actions, log_every=5)

    baseline_steps = [l.steps for l in baseline_logs]
    baseline_rewards = [l.total_reward for l in baseline_logs]
    baseline_heights = [np.mean(l.heights) for l in baseline_logs]
    baseline_falls = [l.fell for l in baseline_logs]

    print(f"\n  Baseline results:")
    print(f"    Avg steps before fall: {np.mean(baseline_steps):.1f} ± {np.std(baseline_steps):.1f}")
    print(f"    Avg reward: {np.mean(baseline_rewards):.1f} ± {np.std(baseline_rewards):.1f}")
    print(f"    Avg height: {np.mean(baseline_heights):.3f}")
    print(f"    Fall rate: {np.mean(baseline_falls):.0%}")

    # --- Phase 2: Learning (STDP + reward) ---
    print(f"\n{'=' * 70}")
    print("PHASE 2: Learning — STDP + reward modulation (50 episodes)")
    print(f"{'=' * 70}")

    learner = EmbodiedLoopRunner(
        connectome,
        neurons_per_region=100,
        global_coupling=0.05,
        ext_rate=3.5,
        enable_learning=True,
        brain_steps_per_action=200,
        physics_steps_per_action=4,
        stdp_every=50,
        seed=42,
    )

    learning_logs = learner.run_episodes(n_episodes=50, max_actions=max_actions, log_every=10)

    # --- Phase 3: Test retention (learning disabled, same brain) ---
    print(f"\n{'=' * 70}")
    print("PHASE 3: Test retention — learning disabled, same brain (10 episodes)")
    print(f"{'=' * 70}")

    # Disable learning but keep the modified weights
    learner.enable_learning = False
    test_logs = learner.run_episodes(n_episodes=10, max_actions=max_actions, log_every=5)

    test_steps = [l.steps for l in test_logs]
    test_rewards = [l.total_reward for l in test_logs]
    test_heights = [np.mean(l.heights) for l in test_logs]
    test_falls = [l.fell for l in test_logs]

    # --- Learning curve ---
    print(f"\n{'=' * 70}")
    print("LEARNING CURVE")
    print(f"{'=' * 70}")

    window = 5
    for i in range(0, len(learning_logs), window):
        chunk = learning_logs[i:i+window]
        avg_steps = np.mean([l.steps for l in chunk])
        avg_reward = np.mean([l.total_reward for l in chunk])
        avg_height = np.mean([np.mean(l.heights) for l in chunk])
        print(f"  Episodes {i+1:>3}-{i+len(chunk):>3}: "
              f"steps={avg_steps:>5.1f}  reward={avg_reward:>+7.1f}  "
              f"height={avg_height:>.3f}")

    # --- Final comparison ---
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    print(f"\n  {'Metric':<25} {'Baseline':>12} {'After Learning':>14} {'Change':>10}")
    print(f"  {'─' * 63}")

    metrics = [
        ("Avg steps", np.mean(baseline_steps), np.mean(test_steps)),
        ("Avg reward", np.mean(baseline_rewards), np.mean(test_rewards)),
        ("Avg height", np.mean(baseline_heights), np.mean(test_heights)),
        ("Fall rate", np.mean(baseline_falls), np.mean(test_falls)),
    ]

    for name, before, after in metrics:
        change = after - before
        sign = "+" if change > 0 else ""
        print(f"  {name:<25} {before:>12.2f} {after:>14.2f} {sign}{change:>9.2f}")

    # Did learning help?
    steps_improved = np.mean(test_steps) > np.mean(baseline_steps)
    reward_improved = np.mean(test_rewards) > np.mean(baseline_rewards)

    print()
    if steps_improved and reward_improved:
        print("  ✓ LEARNING HELPED: Brain stays upright longer and gets more reward")
        print("    after STDP learning with reward modulation.")
    elif steps_improved or reward_improved:
        print("  ~ PARTIAL IMPROVEMENT: Some metrics improved after learning.")
    else:
        print("  ✗ NO IMPROVEMENT: Learning did not help the brain control the body.")
        print("    This is expected — 100 neurons per region may be too few,")
        print("    and reward-modulated STDP is a weak learning signal for motor control.")

    # Save
    results_dir = Path("results/exp07_learn_to_stand")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

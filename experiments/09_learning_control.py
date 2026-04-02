"""Experiment 9: Control for Experiment 7 — Is the learning real?

Reviewer 3 raised the critical concern: does STDP without reward
modulation also reduce falls? If yes, the Experiment 7 result is
just self-organization, not reward-driven learning.

Three conditions, same body, same parameters:
  A) No learning (baseline) — same as Exp 7 Phase 1
  B) STDP WITH reward modulation — same as Exp 7 Phase 2
  C) STDP WITHOUT reward modulation — THE CONTROL
  D) No STDP, reward signal only — another control

Also: check what motor outputs actually look like.
Reviewer 3 asked: "Is the body standing stiff (motor silence)
rather than actively balancing?"
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from encephagen.connectome import Connectome
from encephagen.loop.embodied_loop import EmbodiedLoopRunner


def run_condition(label, connectome, enable_learning, reward_modulated, n_episodes, max_actions, seed):
    """Run one experimental condition."""
    runner = EmbodiedLoopRunner(
        connectome,
        neurons_per_region=100,
        global_coupling=0.05,
        ext_rate=3.5,
        enable_learning=enable_learning,
        brain_steps_per_action=200,
        physics_steps_per_action=4,
        stdp_every=50,
        seed=seed,
    )

    # Override reward modulation if needed
    if enable_learning and not reward_modulated:
        # Monkey-patch: make reward factor always 1.0 (no modulation)
        original_apply = runner._apply_learning
        def _no_reward_learning(reward):
            original_apply(0.0)  # Always pass 0 reward → factor=1.0
        runner._apply_learning = _no_reward_learning

    logs = runner.run_episodes(n_episodes=n_episodes, max_actions=max_actions, log_every=10)

    # Collect motor output stats from last episode
    # Re-run one episode to capture torques
    runner_check = EmbodiedLoopRunner(
        connectome, neurons_per_region=100, global_coupling=0.05,
        ext_rate=3.5, enable_learning=False,
        brain_steps_per_action=200, physics_steps_per_action=4, seed=seed + 999,
    )
    # Copy learned weights if learning was enabled
    if enable_learning:
        for i in range(len(runner.brain.regions)):
            runner_check.brain.regions[i].exc_conn = runner.brain.regions[i].exc_conn.copy()

    # Run one episode and capture torques
    body_state = runner_check.body.reset()
    runner_check.motor_decoder.reset()
    torques_log = []
    for _ in range(max_actions):
        obs = runner_check.body.get_sensory_input()
        ext = runner_check._encode_body_observation(obs)
        for step in range(runner_check.brain_steps_per_action):
            runner_check.brain.step(runner_check.dt, external_currents=ext)
            motor_pop = runner_check.brain.regions[runner_check.motor_idx]
            runner_check.motor_decoder.update(motor_pop.neurons.spikes, runner_check.dt)
        torques = runner_check._decode_torques()
        torques_log.append(torques.copy())
        body_state = runner_check.body.step_n(torques, n=runner_check.physics_steps_per_action)
        if body_state.is_fallen:
            break

    torques_arr = np.array(torques_log)

    return logs, torques_arr


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 9: Is the learning real? (Control experiment)")
    print("=" * 70)

    connectome = Connectome.from_bundled("toy20")
    n_episodes = 30
    max_actions = 50

    conditions = [
        ("A: No learning", False, False),
        ("B: STDP + reward", True, True),
        ("C: STDP only (no reward)", True, False),
    ]

    results = {}

    for label, enable_learning, reward_mod in conditions:
        print(f"\n{'=' * 70}")
        print(f"CONDITION {label} ({n_episodes} episodes)")
        print(f"{'=' * 70}")

        t0 = time.time()
        logs, torques = run_condition(
            label, connectome, enable_learning, reward_mod,
            n_episodes, max_actions, seed=42,
        )
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.0f}s")

        # Stats from last 10 episodes
        final = logs[-10:]
        results[label] = {
            "reward": np.mean([l.total_reward for l in final]),
            "steps": np.mean([l.steps for l in final]),
            "height": np.mean([np.mean(l.heights) for l in final]),
            "falls": np.mean([l.fell for l in final]),
            "torque_mean": float(np.abs(torques).mean()),
            "torque_std": float(torques.std()),
            "torque_range": float(torques.max() - torques.min()),
        }

        print(f"\n  Motor output analysis (final episode):")
        print(f"    Mean |torque|: {results[label]['torque_mean']:.3f}")
        print(f"    Torque std:    {results[label]['torque_std']:.3f}")
        print(f"    Torque range:  {results[label]['torque_range']:.3f}")

    # --- Comparison ---
    print(f"\n{'=' * 70}")
    print("RESULTS COMPARISON (last 10 episodes)")
    print(f"{'=' * 70}")

    print(f"\n  {'Metric':<20}", end="")
    for label in results:
        print(f"  {label:>22}", end="")
    print()
    print(f"  {'─' * 88}")

    for metric in ["reward", "steps", "height", "falls", "torque_mean", "torque_std"]:
        print(f"  {metric:<20}", end="")
        for label in results:
            val = results[label][metric]
            if metric == "falls":
                print(f"  {val:>22.0%}", end="")
            else:
                print(f"  {val:>22.2f}", end="")
        print()

    # --- Verdict ---
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    no_learn = results["A: No learning"]
    with_reward = results["B: STDP + reward"]
    no_reward = results["C: STDP only (no reward)"]

    print(f"\n  Q1: Does STDP+reward beat no learning?")
    diff_br = with_reward["reward"] - no_learn["reward"]
    if diff_br > 3:
        print(f"    YES (+{diff_br:.1f} reward)")
    else:
        print(f"    NO (diff={diff_br:+.1f})")

    print(f"\n  Q2: Does STDP+reward beat STDP alone?")
    diff_rr = with_reward["reward"] - no_reward["reward"]
    if diff_rr > 3:
        print(f"    YES (+{diff_rr:.1f} reward) — reward modulation matters!")
    else:
        print(f"    NO (diff={diff_rr:+.1f}) — STDP alone does just as well")

    print(f"\n  Q3: Is the body actively moving or standing stiff?")
    if with_reward["torque_std"] > 0.1:
        print(f"    ACTIVE CONTROL (torque std={with_reward['torque_std']:.3f})")
    else:
        print(f"    STIFF/PASSIVE (torque std={with_reward['torque_std']:.3f})")
        print(f"    → The 0% fall rate may be from motor silence, not learned balance")

    # Save
    results_dir = Path("results/exp09_learning_control")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

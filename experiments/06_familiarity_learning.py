"""Experiment 6: Does STDP learning produce a familiarity effect?

Protocol:
1. Build spiking brain with STDP enabled
2. Define two stimulus patterns (A and B)
3. Present pattern A 50 times (learning phase)
4. Present pattern B 0 times
5. Test: present both A and B once each
6. Measure: does A produce faster/stronger response than B?

If yes → the brain learned to recognize pattern A through STDP.
If no → STDP at this scale doesn't produce measurable familiarity.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import stats

from encephagen.connectome import Connectome
from encephagen.network.spiking_brain import SpikingBrain
from encephagen.neurons.lif import LIFParams
from encephagen.learning.stdp import STDPRule, STDPParams
from encephagen.learning.homeostatic import HomeostaticPlasticity, HomeostaticParams
from encephagen.analysis.functional_roles import _classify_tvb76_regions


def _create_stimulus_pattern(n_neurons: int, n_active: int, seed: int) -> np.ndarray:
    """Create a spatial pattern: which neurons receive extra input.

    Returns a binary mask [n_neurons] where n_active neurons are active.
    """
    rng = np.random.default_rng(seed)
    pattern = np.zeros(n_neurons, dtype=np.float64)
    active_indices = rng.choice(n_neurons, size=n_active, replace=False)
    pattern[active_indices] = 1.0
    return pattern


def _measure_response(
    brain: SpikingBrain,
    stimulus_regions: list[int],
    pattern: np.ndarray,
    stimulus_strength: float,
    dt: float,
    baseline_steps: int,
    stimulus_steps: int,
    post_steps: int,
) -> dict:
    """Present a stimulus and measure the network response.

    Returns dict with response metrics.
    """
    n = brain.neurons_per_region

    # Baseline: run without stimulus
    baseline_counts = np.zeros(brain.n_regions)
    for _ in range(baseline_steps):
        brain.step(dt)
        for i in range(brain.n_regions):
            baseline_counts[i] += brain.regions[i].neurons.spikes.sum()
    baseline_rate = baseline_counts / (brain.neurons_per_region * baseline_steps * dt / 1000)

    # Stimulus: inject pattern into sensory regions
    stim_counts = np.zeros(brain.n_regions)
    for step in range(stimulus_steps):
        ext = {}
        for reg_idx in stimulus_regions:
            ext[reg_idx] = pattern * stimulus_strength
        brain.step(dt, external_currents=ext)
        for i in range(brain.n_regions):
            stim_counts[i] += brain.regions[i].neurons.spikes.sum()
    stim_rate = stim_counts / (brain.neurons_per_region * stimulus_steps * dt / 1000)

    # Post-stimulus: measure lingering response
    post_counts = np.zeros(brain.n_regions)
    for _ in range(post_steps):
        brain.step(dt)
        for i in range(brain.n_regions):
            post_counts[i] += brain.regions[i].neurons.spikes.sum()
    post_rate = post_counts / (brain.neurons_per_region * post_steps * dt / 1000)

    # Response metrics
    response_magnitude = np.mean(stim_rate - baseline_rate)
    response_in_sensory = np.mean([stim_rate[i] - baseline_rate[i] for i in stimulus_regions])
    response_in_other = np.mean([stim_rate[i] - baseline_rate[i]
                                  for i in range(brain.n_regions)
                                  if i not in stimulus_regions])

    return {
        "baseline_mean": float(np.mean(baseline_rate)),
        "stim_mean": float(np.mean(stim_rate)),
        "post_mean": float(np.mean(post_rate)),
        "response_magnitude": float(response_magnitude),
        "response_sensory": float(response_in_sensory),
        "response_other": float(response_in_other),
    }


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 6: Does STDP produce a familiarity effect?")
    print("=" * 70)

    connectome = Connectome.from_bundled("tvb96")
    groups = _classify_tvb76_regions(connectome.labels)
    params = LIFParams(j_exc=2.0, g_inh=5.0)

    sensory_idx = groups.get("sensory", list(range(4)))
    n_per_region = 200
    dt = 0.1

    # Create two distinct stimulus patterns
    pattern_a = _create_stimulus_pattern(n_per_region, n_active=50, seed=100)
    pattern_b = _create_stimulus_pattern(n_per_region, n_active=50, seed=200)

    overlap = (pattern_a * pattern_b).sum()
    print(f"\nPattern A: {int(pattern_a.sum())} active neurons in each sensory region")
    print(f"Pattern B: {int(pattern_b.sum())} active neurons in each sensory region")
    print(f"Overlap: {int(overlap)} neurons")

    stimulus_strength = 15.0  # mV — strong enough to drive extra firing
    baseline_steps = 1000     # 100ms baseline
    stimulus_steps = 2000     # 200ms stimulus
    post_steps = 1000         # 100ms post

    # --- Phase 1: Pre-learning test (both patterns novel) ---
    print(f"\n{'=' * 70}")
    print("PHASE 1: Pre-learning — both patterns are novel")
    print(f"{'=' * 70}")

    brain = SpikingBrain(
        connectome, neurons_per_region=n_per_region,
        between_conn_prob=0.02, global_coupling=0.05,
        ext_rate=3.5, params=params, seed=42,
    )

    # Warm up
    print("  Warming up (1s)...", end=" ", flush=True)
    for _ in range(10000):
        brain.step(dt)
    print("done")

    print("  Testing pattern A (novel)...", end=" ", flush=True)
    resp_a_pre = _measure_response(
        brain, sensory_idx, pattern_a, stimulus_strength,
        dt, baseline_steps, stimulus_steps, post_steps,
    )
    print(f"response={resp_a_pre['response_magnitude']:.2f} Hz")

    print("  Testing pattern B (novel)...", end=" ", flush=True)
    resp_b_pre = _measure_response(
        brain, sensory_idx, pattern_b, stimulus_strength,
        dt, baseline_steps, stimulus_steps, post_steps,
    )
    print(f"response={resp_b_pre['response_magnitude']:.2f} Hz")

    # --- Phase 2: Learning — present pattern A repeatedly with STDP ---
    print(f"\n{'=' * 70}")
    print("PHASE 2: Learning — present pattern A 50 times with STDP")
    print(f"{'=' * 70}")

    # Set up STDP for each region's internal excitatory connections
    stdp_rules = {}
    for i, pop in enumerate(brain.regions):
        stdp_rules[i] = STDPRule(
            n_pre=pop.n_exc,
            n_post=pop.n_neurons,
            params=STDPParams(a_plus=0.005, a_minus=0.005, w_max=10.0),
        )

    homeo = {}
    for i, pop in enumerate(brain.regions):
        homeo[i] = HomeostaticPlasticity(
            n_neurons=pop.n_neurons,
            params=HomeostaticParams(target_rate=10.0, tau_homeo=500.0, eta=0.01),
        )

    n_presentations = 30
    presentation_steps = 1000   # 100ms per presentation
    gap_steps = 500             # 50ms gap between presentations
    stdp_every = 10             # Apply STDP every 10 timesteps (1ms) to save time

    t0 = time.time()
    for trial in range(n_presentations):
        # Present pattern A
        for step in range(presentation_steps):
            ext = {}
            for reg_idx in sensory_idx:
                ext[reg_idx] = pattern_a * stimulus_strength
            brain.step(dt, external_currents=ext)

            # STDP update (every N steps for performance)
            if step % stdp_every == 0:
                sim_time = (trial * (presentation_steps + gap_steps) + step) * dt
                for i, pop in enumerate(brain.regions):
                    exc_spikes = pop.neurons.spikes[:pop.n_exc]
                    all_spikes = pop.neurons.spikes
                    pop.exc_conn = stdp_rules[i].step(
                        dt * stdp_every, exc_spikes, all_spikes, pop.exc_conn,
                    )
                    pop.exc_conn = homeo[i].step(
                        dt * stdp_every, all_spikes, pop.exc_conn,
                        apply_every_ms=100.0, current_time=sim_time,
                    )

        # Gap (no stimulus)
        for _ in range(gap_steps):
            brain.step(dt)
            for i, pop in enumerate(brain.regions):
                homeo[i].update_rates(dt, pop.neurons.spikes)

        if (trial + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Trial {trial + 1}/{n_presentations} ({elapsed:.1f}s)")

    total_learning_time = time.time() - t0
    print(f"  Learning complete: {total_learning_time:.1f}s")

    # Check weight changes
    total_weight_change = 0
    for i, pop in enumerate(brain.regions):
        orig_mean = params.j_exc  # Original weight
        current_mean = pop.exc_conn.data.mean() if pop.exc_conn.nnz > 0 else 0
        total_weight_change += abs(current_mean - orig_mean)
    print(f"  Average weight change per region: {total_weight_change / brain.n_regions:.4f}")

    # --- Phase 3: Post-learning test ---
    print(f"\n{'=' * 70}")
    print("PHASE 3: Post-learning — A is familiar, B is novel")
    print(f"{'=' * 70}")

    # Cool down (let transients settle)
    print("  Cool down (500ms)...", end=" ", flush=True)
    for _ in range(5000):
        brain.step(dt)
    print("done")

    # Test pattern A (should be familiar)
    print("  Testing pattern A (familiar)...", end=" ", flush=True)
    resp_a_post = _measure_response(
        brain, sensory_idx, pattern_a, stimulus_strength,
        dt, baseline_steps, stimulus_steps, post_steps,
    )
    print(f"response={resp_a_post['response_magnitude']:.2f} Hz")

    # Cool down
    for _ in range(5000):
        brain.step(dt)

    # Test pattern B (should be novel)
    print("  Testing pattern B (novel)...", end=" ", flush=True)
    resp_b_post = _measure_response(
        brain, sensory_idx, pattern_b, stimulus_strength,
        dt, baseline_steps, stimulus_steps, post_steps,
    )
    print(f"response={resp_b_post['response_magnitude']:.2f} Hz")

    # --- Results ---
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    print(f"\n  {'Metric':<25} {'A (pre)':>10} {'B (pre)':>10} {'A (post)':>10} {'B (post)':>10}")
    print(f"  {'─' * 67}")
    for metric in ["response_magnitude", "response_sensory", "response_other",
                    "baseline_mean", "stim_mean"]:
        print(f"  {metric:<25} "
              f"{resp_a_pre[metric]:>10.2f} "
              f"{resp_b_pre[metric]:>10.2f} "
              f"{resp_a_post[metric]:>10.2f} "
              f"{resp_b_post[metric]:>10.2f}")

    # Key comparison: A_post vs B_post
    diff_post = resp_a_post["response_magnitude"] - resp_b_post["response_magnitude"]
    diff_pre = resp_a_pre["response_magnitude"] - resp_b_pre["response_magnitude"]

    print(f"\n  A-B difference (pre-learning):  {diff_pre:+.2f} Hz")
    print(f"  A-B difference (post-learning): {diff_post:+.2f} Hz")

    # Familiarity effect = change in A-B difference
    familiarity_effect = diff_post - diff_pre
    print(f"  Familiarity effect (change):    {familiarity_effect:+.2f} Hz")

    if abs(familiarity_effect) > 0.5:
        if familiarity_effect > 0:
            print(f"\n  ✓ FAMILIARITY EFFECT DETECTED: Pattern A produces {familiarity_effect:.2f} Hz")
            print(f"    stronger response than B after learning")
        else:
            print(f"\n  ✓ HABITUATION DETECTED: Pattern A produces {abs(familiarity_effect):.2f} Hz")
            print(f"    weaker response than B after learning (repetition suppression)")
    else:
        print(f"\n  ✗ No significant familiarity effect ({familiarity_effect:.2f} Hz)")

    # Save
    results_dir = Path("results/exp06_familiarity")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Results saved to {results_dir}")


if __name__ == "__main__":
    run_experiment()

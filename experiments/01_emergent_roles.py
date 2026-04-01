"""Experiment 1: Do region-specific functional roles emerge from connectome topology?

Every region gets IDENTICAL Wilson-Cowan parameters. The only difference
between regions is their position in the connectome graph. We test whether
this alone causes regions to develop distinct functional behaviors matching
their known biological roles.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from encephagen.connectome import Connectome
from encephagen.dynamics.brain_sim import BrainSimulator, StimulusEvent
from encephagen.dynamics.wilson_cowan import WilsonCowanParams
from encephagen.analysis.functional_roles import (
    run_all_predictions,
    compute_regional_profiles,
    _classify_tvb76_regions,
)


def run_experiment():
    print("=" * 70)
    print("ENCEPHAGEN EXPERIMENT 1")
    print("Do region-specific functional roles emerge from topology alone?")
    print("=" * 70)

    connectome = Connectome.from_bundled("tvb76")
    print(f"\nConnectome: {connectome}")

    groups = _classify_tvb76_regions(connectome.labels)
    print("\nRegion groups:")
    for group, indices in groups.items():
        labels = [connectome.labels[i] for i in indices[:5]]
        suffix = f"... (+{len(indices)-5} more)" if len(indices) > 5 else ""
        print(f"  {group:>12}: {len(indices)} regions — {labels}{suffix}")

    # Parameters: oscillatory regime found in conntopo parameter sweep
    params = WilsonCowanParams(
        w_ee=16.0, w_ei=12.0, w_ie=15.0, w_ii=3.0,
        theta_e=2.0, theta_i=3.7, a_e=1.5, a_i=1.0,
        noise_sigma=0.01,
    )

    # Identify sensory regions for stimulus injection
    sensory_idx = groups.get("sensory", [])
    if not sensory_idx:
        # Fallback: use first few regions
        sensory_idx = list(range(min(4, connectome.num_regions)))

    # --- Phase 1: Spontaneous dynamics (no stimulus) ---
    print("\n" + "─" * 70)
    print("PHASE 1: Spontaneous dynamics (no stimulus)")
    print("─" * 70)

    sim = BrainSimulator(connectome, global_coupling=0.03, params=params)

    print("Simulating 10s of spontaneous activity...", end=" ", flush=True)
    t0 = time.time()
    result_spont = sim.simulate(
        duration=12000, dt=0.1, transient=2000, seed=42,
    )
    elapsed = time.time() - t0
    print(f"{elapsed:.1f}s")

    print("\nRegional profiles (spontaneous):")
    profiles = compute_regional_profiles(result_spont)
    print(f"  {'Region':<12} {'Mean Act':>9} {'Variance':>10} {'Tau (ms)':>10} {'Peak Hz':>9}")
    print("  " + "─" * 52)
    for label, p in sorted(profiles.items(), key=lambda x: -x[1]["variance"])[:15]:
        print(f"  {label:<12} {p['mean_activity']:>9.4f} {p['variance']:>10.6f} "
              f"{p['autocorrelation_tau']:>10.1f} {p['peak_frequency']:>9.1f}")

    # --- Phase 2: Stimulus response ---
    print("\n" + "─" * 70)
    print("PHASE 2: Stimulus response (pulse to sensory regions)")
    print("─" * 70)

    stimulus = StimulusEvent(
        region_indices=sensory_idx,
        onset=5000.0,  # 5s into recording (after 2s transient = 7s total)
        duration=200.0,  # 200ms pulse
        amplitude=2.0,
    )

    print(f"Stimulus: {len(sensory_idx)} sensory regions, onset=5000ms, "
          f"duration=200ms, amplitude=2.0")
    print("Simulating...", end=" ", flush=True)
    t0 = time.time()
    result_stim = sim.simulate(
        duration=12000, dt=0.1, transient=2000, stimuli=[stimulus], seed=42,
    )
    elapsed = time.time() - t0
    print(f"{elapsed:.1f}s")

    # --- Run all predictions ---
    print("\n" + "=" * 70)
    print("PREDICTION TESTS")
    print("=" * 70)

    # Spontaneous predictions
    print("\n--- Spontaneous dynamics predictions ---")
    spont_predictions = run_all_predictions(result_spont)

    for key, pred in spont_predictions.items():
        if key == "region_groups":
            continue
        if not pred.get("testable", False):
            print(f"\n  {key}: NOT TESTABLE — {pred.get('reason', 'unknown')}")
            continue

        supported = pred.get("supported", False)
        marker = "SUPPORTED" if supported else "NOT SUPPORTED"
        print(f"\n  {key}: {marker}")
        print(f"    Prediction: {pred.get('prediction', '')}")
        for k, v in pred.items():
            if k in ("testable", "prediction", "supported", "group_frequencies"):
                continue
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")
        if "group_frequencies" in pred:
            for group, stats in pred["group_frequencies"].items():
                print(f"    {group}: mean={stats['mean_freq']:.1f} Hz, "
                      f"std={stats['std_freq']:.1f}, n={stats['n']}")

    # Stimulus predictions
    print("\n--- Stimulus response predictions ---")
    stim_predictions = run_all_predictions(result_stim)

    p3 = stim_predictions.get("P3_sensory_first", {})
    if p3.get("testable", False):
        supported = p3.get("supported", False)
        marker = "SUPPORTED" if supported else "NOT SUPPORTED"
        print(f"\n  P3_sensory_first: {marker}")
        print(f"    Prediction: {p3.get('prediction', '')}")
        print(f"    Sensory mean latency: {p3.get('sensory_mean_latency_ms', 'N/A'):.1f} ms")
        print(f"    Other mean latency: {p3.get('other_mean_latency_ms', 'N/A'):.1f} ms")
        print(f"    p-value: {p3.get('p_value', 'N/A')}")
    else:
        print(f"\n  P3_sensory_first: NOT TESTABLE — {p3.get('reason', '')}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_preds = {**spont_predictions, **{"P3_sensory_first_stim": stim_predictions.get("P3_sensory_first", {})}}
    supported_count = sum(
        1 for k, v in all_preds.items()
        if isinstance(v, dict) and v.get("supported", False)
    )
    testable_count = sum(
        1 for k, v in all_preds.items()
        if isinstance(v, dict) and v.get("testable", False)
    )

    print(f"\n  Predictions supported: {supported_count}/{testable_count}")

    if supported_count >= 3:
        print("\n  STRONG EVIDENCE: Region-specific functional roles emerge")
        print("  from human connectome topology with identical local dynamics.")
    elif supported_count >= 1:
        print("\n  PARTIAL EVIDENCE: Some regional differentiation emerges")
        print("  from topology, but not all predicted roles are observed.")
    else:
        print("\n  NO EVIDENCE: Topology alone does not produce recognizable")
        print("  region-specific functional roles at this scale/parameterization.")
        print("  This is also a meaningful finding.")

    # Save results
    results_dir = Path("results/exp01_emergent_roles")
    results_dir.mkdir(parents=True, exist_ok=True)

    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Save only scalar predictions (avoid circular refs from numpy arrays)
    def _clean_pred(d):
        return {k: v for k, v in d.items()
                if isinstance(v, (str, int, float, bool, dict, list, type(None)))}

    all_results = {
        "spontaneous_predictions": {k: _clean_pred(v) if isinstance(v, dict) else v
                                    for k, v in spont_predictions.items()},
        "stimulus_predictions": {k: _clean_pred(v) if isinstance(v, dict) else v
                                 for k, v in stim_predictions.items()},
    }
    serializable = json.loads(json.dumps(all_results, default=to_serializable))
    with open(results_dir / "summary.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {results_dir / 'summary.json'}")


if __name__ == "__main__":
    run_experiment()

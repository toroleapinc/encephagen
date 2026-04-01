"""Experiment 5: Does the spiking network preserve the hierarchy findings?

The Wilson-Cowan model showed that basal ganglia oscillate while other
regions silence. Does the same hierarchy emerge with 96,000 individual
spiking neurons?
"""

import time
import numpy as np

from encephagen.connectome import Connectome
from encephagen.network.spiking_brain import SpikingBrain
from encephagen.neurons.lif import LIFParams
from encephagen.analysis.functional_roles import _classify_tvb76_regions


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 5: Spiking Network Hierarchy Test")
    print("Do 96K spiking neurons reproduce the Wilson-Cowan hierarchy?")
    print("=" * 70)

    connectome = Connectome.from_bundled("tvb96")
    groups = _classify_tvb76_regions(connectome.labels)

    print(f"\nConnectome: {connectome}")
    print(f"Neurons per region: 1000")
    print(f"Total neurons: {96 * 1000:,}")

    params = LIFParams()

    # Test multiple coupling strengths
    for G in [0.1, 0.5, 1.0, 2.0, 5.0]:
        print(f"\n{'─' * 70}")
        print(f"Global coupling G = {G}")
        print(f"{'─' * 70}")

        brain = SpikingBrain(
            connectome,
            neurons_per_region=1000,
            between_conn_prob=0.02,
            global_coupling=G,
            params=params,
            seed=42,
        )

        result = brain.simulate(
            duration=3000, dt=0.1, transient=500,
            record_interval=10.0,
        )

        # Group mean firing rates
        print(f"\n  {'Group':<14} {'Mean Rate Hz':>12} {'Std Rate':>10} {'Status':<15}")
        print("  " + "─" * 53)

        group_rates = {}
        for gname in ["basal_ganglia", "prefrontal", "thalamus",
                       "hippocampus", "sensory", "motor", "other"]:
            indices = groups.get(gname, [])
            if not indices:
                continue
            rates = [result.firing_rates[:, i].mean() for i in indices]
            mean_rate = np.mean(rates)
            std_rate = np.std(rates)
            group_rates[gname] = mean_rate

            if mean_rate > 20:
                status = "ACTIVE"
            elif mean_rate > 5:
                status = "moderate"
            elif mean_rate > 1:
                status = "low"
            else:
                status = "silent"

            print(f"  {gname:<14} {mean_rate:>12.2f} {std_rate:>10.2f} {status:<15}")

        # Check hierarchy
        if group_rates:
            ranked = sorted(group_rates.items(), key=lambda x: -x[1])
            print(f"\n  Hierarchy (most → least active):")
            for i, (g, r) in enumerate(ranked):
                print(f"    {i+1}. {g:<14} {r:.2f} Hz")

    print(f"\n{'=' * 70}")
    print("Compare with Wilson-Cowan hierarchy:")
    print("  WC silencing order: thalamus/motor → hippo/sensory → other → PFC → BG(never)")
    print("  Does spiking network show the same pattern?")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()

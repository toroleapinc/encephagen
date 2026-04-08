"""Experiment 43: Tractography threshold sweep.

Does FC-FC improve when we remove weak (potentially false) connections?
Maier-Hein (2017): tractography produces systematic false positives.
If FC-FC improves at higher thresholds, weak connections are noise.
"""

import numpy as np
import json
from scipy import stats


def main():
    sc = np.load('src/encephagen/connectome/bundled/neurolib80_weights.npy')
    fc_emp = np.load('src/encephagen/connectome/bundled/neurolib80_fc_empirical.npy')

    print("TRACTOGRAPHY THRESHOLD SWEEP")
    print(f"{'Threshold':>12} {'Edges kept':>12} {'SC-FC(raw)':>12}")
    print("-" * 40)

    for pct in [0, 10, 20, 30, 40, 50, 60]:
        sc_t = sc.copy()
        if pct > 0:
            threshold = np.percentile(sc[sc > 0], pct)
            sc_t[sc_t < threshold] = 0

        n_edges = (sc_t > 0).sum()
        idx = np.triu_indices(80, k=1)
        sc_log = np.log1p(sc_t)
        valid = ~np.isnan(fc_emp[idx])
        r, _ = stats.pearsonr(sc_log[idx][valid], fc_emp[idx][valid])
        print(f"  {pct:>3}%tile     {n_edges:>8}     {r:>10.4f}")

    print("\nIf SC-FC improves with thresholding → weak connections are noise.")
    print("If SC-FC degrades → even weak connections carry signal.")


if __name__ == "__main__":
    main()

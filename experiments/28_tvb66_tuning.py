"""Quick SC-FC parameter sweep for tvb66 with continuous weights."""

import numpy as np
import torch
from scipy import stats
from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU


def test_params(connectome, int_prob, bet_prob, gc, erf, npr=200):
    n_regions = connectome.num_regions
    n_total = n_regions * npr
    try:
        brain = SpikingBrainGPU(
            connectome=connectome, neurons_per_region=npr,
            internal_conn_prob=int_prob, between_conn_prob=bet_prob,
            global_coupling=gc, ext_rate_factor=erf,
            tau_nmda=150.0, nmda_ratio=0.4,
            pfc_regions=[], device='cuda',
            use_delays=True, conduction_velocity=3.5,
            use_neuron_types=False, use_adaptation=False,
        )
    except Exception as e:
        return 0.0, 0.0

    state = brain.init_state(batch_size=1)
    with torch.no_grad():
        for _ in range(2000):
            state, _ = brain.step(state)

    test_spikes = 0
    with torch.no_grad():
        for _ in range(500):
            state, spikes = brain.step(state)
            test_spikes += spikes[0].sum().item()
    rate = test_spikes / (n_total * 500)
    if rate < 1e-6 or rate > 0.5:
        del brain; torch.cuda.empty_cache()
        return 0.0, rate

    record_interval = 100
    total_steps = 30000
    n_tp = total_steps // record_interval
    region_ts = np.zeros((n_tp, n_regions))
    spike_acc = torch.zeros(n_regions, device='cuda')
    acc = 0; tp = 0
    with torch.no_grad():
        for _ in range(total_steps):
            state, spikes = brain.step(state)
            for r in range(n_regions):
                spike_acc[r] += spikes[0, r*npr:(r+1)*npr].sum()
            acc += 1
            if acc >= record_interval:
                region_ts[tp] = (spike_acc / (npr * acc)).cpu().numpy()
                spike_acc.zero_(); acc = 0; tp += 1
                if tp >= n_tp: break

    fc = np.corrcoef(region_ts[:tp].T)
    sc_log = np.log1p(connectome.weights)
    idx = np.triu_indices_from(fc, k=1)
    valid = ~np.isnan(fc[idx]) & ~np.isnan(sc_log[idx])
    if valid.sum() > 10:
        r_val, _ = stats.pearsonr(sc_log[idx][valid], fc[idx][valid])
    else:
        r_val = 0.0
    del brain; torch.cuda.empty_cache()
    return r_val, rate


if __name__ == "__main__":
    c = Connectome.from_bundled("tvb66")
    print(f"tvb66: {c.num_regions} regions, {c.num_edges} edges, "
          f"weight range {c.weights[c.weights>0].min():.6f}-{c.weights[c.weights>0].max():.6f}")

    print(f"\n{'int':>5} {'bet':>5} {'gc':>6} {'erf':>5} {'SC-FC':>8} {'rate':>8}")
    print("-" * 50)

    for int_p in [0.03, 0.05, 0.08]:
        for bet_p in [0.05, 0.10, 0.20]:
            for gc in [2.0, 5.0, 10.0, 20.0]:
                for erf in [3.0, 3.5, 4.0]:
                    r, rate = test_params(c, int_p, bet_p, gc, erf)
                    marker = " ★★★" if r > 0.3 else " ★" if r > 0.2 else ""
                    status = "dead" if rate < 1e-6 else "boom" if rate > 0.5 else ""
                    print(f"{int_p:>5.2f} {bet_p:>5.2f} {gc:>6.1f} {erf:>5.1f} {r:>8.4f} {rate:>8.5f} {status}{marker}", flush=True)

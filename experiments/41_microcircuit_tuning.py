"""Experiment 41: Tune microcircuit brain + test connectome vs random.

Step 1: Sweep tonic drive and coupling to find SC-FC > 0.3
Step 2: At best params, compare connectome vs random (10 seeds)
"""

import numpy as np
import torch
import json
import time
from scipy import stats

from encephagen.connectome import Connectome
from encephagen.cortex.microcircuit import (
    build_microcircuit_connectivity,
    build_between_region_connectivity,
    CellTypeConfig,
)


def load_neurolib80():
    sc = np.load('src/encephagen/connectome/bundled/neurolib80_weights.npy')
    tl = np.load('src/encephagen/connectome/bundled/neurolib80_tract_lengths.npy')
    labels = json.load(open('src/encephagen/connectome/bundled/neurolib80_labels.json'))
    c = Connectome(sc, labels); c.tract_lengths = tl
    return c


def randomize(conn, seed):
    rng = np.random.default_rng(seed)
    w = conn.weights.copy()
    rows, cols = np.where(w > 0)
    edges = list(zip(rows.tolist(), cols.tolist()))
    wts = [float(w[r, c]) for r, c in edges]
    ne = len(edges)
    for _ in range(10 * ne):
        i1, i2 = rng.choice(ne, size=2, replace=False)
        a, b = edges[i1]
        c, d = edges[i2]
        if a == d or c == b or w[a, d] > 0 or w[c, b] > 0:
            continue
        w[a, b], w[c, d] = 0, 0
        w[a, d], w[c, b] = wts[i1], wts[i2]
        edges[i1], edges[i2] = (a, d), (c, b)
    r = Connectome(w, list(conn.labels))
    r.tract_lengths = conn.tract_lengths
    return r


def build_and_run(connectome, tonic_base=7.0, tonic_L4=9.0, gc=5.0,
                  device="cuda", sim_steps=20000):
    """Build microcircuit brain and compute SC-FC."""
    npr = 200; n_regions = connectome.num_regions; n_total = n_regions * npr
    config = CellTypeConfig()
    rng = np.random.default_rng(42)

    (int_r, int_c, int_v), tau_m, cell_map = \
        build_microcircuit_connectivity(npr, n_regions, config, rng)
    bet_r, bet_c, bet_v = build_between_region_connectivity(
        connectome.weights, cell_map, n_regions, npr, gc, rng)

    from scipy import sparse
    all_r = np.concatenate([int_r, bet_r])
    all_c = np.concatenate([int_c, bet_c])
    all_v = np.concatenate([int_v, bet_v])
    W = sparse.csr_matrix((all_v, (all_r, all_c)), shape=(n_total, n_total))

    indices = torch.tensor(np.array(W.nonzero()), dtype=torch.long)
    values = torch.tensor(W[W.nonzero()].A1, dtype=torch.float32)
    W_t = torch.sparse_coo_tensor(indices, values, (n_total, n_total)).coalesce().to(device)

    tonic = torch.full((n_total,), tonic_base, device=device)
    for r in range(n_regions):
        s, e = cell_map.get((r, 'L4'), (0, 0))
        tonic[s:e] = tonic_L4

    tau_m_d = torch.tensor(tau_m, dtype=torch.float32, device=device)
    dt = 0.1; v_thr = 8.0

    v = torch.rand(n_total, device=device) * 5.0
    refrac = torch.zeros(n_total, device=device)
    i_syn = torch.zeros(n_total, device=device)

    # Warmup
    for _ in range(3000):
        noise = torch.randn(n_total, device=device) * 0.5
        i_total = i_syn + tonic + noise
        active = refrac <= 0; dv = (-v + i_total) / tau_m_d
        v = v + dt * dv * active.float()
        spikes = (v >= v_thr) & active
        v = torch.where(spikes, torch.zeros_like(v), v)
        refrac = torch.where(spikes, torch.full_like(refrac, 1.5), refrac)
        refrac = torch.clamp(refrac - dt, min=0)
        syn_input = torch.sparse.mm(W_t.t(), spikes.float().unsqueeze(1)).squeeze(1)
        i_syn = i_syn * np.exp(-dt / 5.0) + syn_input

    # Record
    region_ts = []; spike_acc = torch.zeros(n_regions, device=device); acc = 0
    for _ in range(sim_steps):
        noise = torch.randn(n_total, device=device) * 0.5
        i_total = i_syn + tonic + noise
        active = refrac <= 0; dv = (-v + i_total) / tau_m_d
        v = v + dt * dv * active.float()
        spikes = (v >= v_thr) & active
        v = torch.where(spikes, torch.zeros_like(v), v)
        refrac = torch.where(spikes, torch.full_like(refrac, 1.5), refrac)
        refrac = torch.clamp(refrac - dt, min=0)
        syn_input = torch.sparse.mm(W_t.t(), spikes.float().unsqueeze(1)).squeeze(1)
        i_syn = i_syn * np.exp(-dt / 5.0) + syn_input
        for r in range(n_regions):
            spike_acc[r] += spikes[r*npr:(r+1)*npr].sum()
        acc += 1
        if acc >= 100:
            region_ts.append((spike_acc / (npr * acc)).cpu().numpy())
            spike_acc.zero_(); acc = 0

    mean_rate = spikes.float().mean().item()

    # SC-FC
    fc_emp = np.load('src/encephagen/connectome/bundled/neurolib80_fc_empirical.npy')
    ts = np.array(region_ts)
    if len(ts) < 10:
        return 0.0, mean_rate, 0.0

    fc_sim = np.corrcoef(ts.T)
    sc_log = np.log1p(connectome.weights)
    idx = np.triu_indices(n_regions, k=1)
    valid = ~np.isnan(fc_sim[idx]) & ~np.isnan(fc_emp[idx])
    if valid.sum() > 10:
        r_fcfc, _ = stats.pearsonr(fc_emp[idx][valid], fc_sim[idx][valid])
        r_scfc, _ = stats.pearsonr(sc_log[idx][valid], fc_sim[idx][valid])
    else:
        r_fcfc = r_scfc = 0.0

    # Regional CV
    region_rates = ts.mean(axis=0)
    cv = np.std(region_rates) / (np.mean(region_rates) + 1e-10)

    del W_t; torch.cuda.empty_cache()
    return r_fcfc, mean_rate, cv


def main():
    print("=" * 60)
    print("  MICROCIRCUIT SC-FC TUNING + CONNECTOME VS RANDOM")
    print("=" * 60)

    conn = load_neurolib80()
    t0 = time.time()

    # Step 1: Parameter sweep
    print(f"\n  STEP 1: SC-FC tuning sweep")
    print(f"  {'tonic':>6} {'L4':>5} {'gc':>5} {'FC-FC':>8} {'rate':>8} {'CV':>8}")
    print(f"  {'─'*45}")

    best_r = -1; best_params = None
    for tonic in [7.0, 8.0, 9.0, 10.0]:
        for l4_boost in [1.0, 2.0, 3.0]:
            for gc in [3.0, 5.0, 8.0, 12.0]:
                r, rate, cv = build_and_run(conn, tonic, tonic + l4_boost, gc,
                                             sim_steps=15000)
                m = ' ***' if r > 0.3 else ' *' if r > 0.15 else ''
                if r > best_r:
                    best_r = r; best_params = (tonic, tonic + l4_boost, gc)
                print(f"  {tonic:>6.1f} {tonic+l4_boost:>5.1f} {gc:>5.1f} "
                      f"{r:>8.4f} {rate:>8.5f} {cv:>8.4f}{m}", flush=True)

    print(f"\n  Best: FC-FC={best_r:.4f} at tonic={best_params[0]}, "
          f"L4={best_params[1]}, gc={best_params[2]}")

    # Step 2: Connectome vs Random at best params
    print(f"\n  STEP 2: Connectome vs Random (10 seeds each)")
    tonic, l4, gc = best_params

    results = {'connectome': [], 'random': []}
    for cond in ['connectome', 'random']:
        print(f"    {cond.upper()}:")
        for seed in range(10):
            if cond == 'connectome':
                c = conn
            else:
                c = randomize(conn, seed=3000 + seed)
            r, rate, cv = build_and_run(c, tonic, l4, gc, sim_steps=20000)
            results[cond].append({'fc_fc': r, 'rate': rate, 'cv': cv})
            elapsed = time.time() - t0
            print(f"      Seed {seed+1:>2}: FC-FC={r:.4f} CV={cv:.4f} ({elapsed:.0f}s)",
                  flush=True)

    # Analysis
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")

    c_fcfc = [r['fc_fc'] for r in results['connectome']]
    r_fcfc = [r['fc_fc'] for r in results['random']]
    c_cv = [r['cv'] for r in results['connectome']]
    r_cv = [r['cv'] for r in results['random']]

    _, p_fcfc = stats.mannwhitneyu(c_fcfc, r_fcfc, alternative='two-sided')
    _, p_cv = stats.mannwhitneyu(c_cv, r_cv, alternative='two-sided')

    print(f"\n  FC-FC(emp): Connectome {np.mean(c_fcfc):.4f} vs Random {np.mean(r_fcfc):.4f}  p={p_fcfc:.4f}")
    print(f"  Regional CV: Connectome {np.mean(c_cv):.4f} vs Random {np.mean(r_cv):.4f}  p={p_cv:.4f}")

    if p_fcfc < 0.05 and np.mean(c_fcfc) > np.mean(r_fcfc):
        print(f"\n  ✓ CONNECTOME PRODUCES BETTER FC-FC WITH MICROCIRCUITS!")
        print(f"  The canonical microcircuit + real connectome routing =")
        print(f"  realistic functional connectivity that random cannot match.")
    elif p_cv < 0.05 and np.mean(c_cv) > np.mean(r_cv):
        print(f"\n  ✓ CONNECTOME PRODUCES MORE DIFFERENTIATED ACTIVITY!")
    else:
        print(f"\n  ✗ Still no significant difference with microcircuits.")

    # Save
    from pathlib import Path
    results_dir = Path("results/exp41_microcircuit")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump({'best_params': list(best_params), 'results': results}, f)
    print(f"  Saved to {results_dir}")


if __name__ == "__main__":
    main()

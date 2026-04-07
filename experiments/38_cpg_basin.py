"""Experiment 38: CPG Parameter Basin Analysis.

THE key 先天 test: does the connectome-constrained CPG architecture
oscillate in a LARGER parameter region than a random-wired CPG?

Method:
  1. Start from CMA-ES optimized parameters (the working point)
  2. Perturb EACH parameter independently by ±10%, ±20%, ±30%, ±50%
  3. Measure fitness (oscillation quality) at each perturbation
  4. Count how many perturbations still produce oscillation
  5. Repeat for RANDOM-wired CPGs (same size, same E/I ratio, random connectivity)

If the structured CPG's basin is LARGER:
  → Structure makes rhythmic behavior more robust to parameter variation
  → This is a genuine innate architectural advantage (先天)
  → Evolution calibrated these circuits, and the structure helps them STAY calibrated

If basins are EQUAL:
  → Structure is irrelevant — any wiring works equally well
  → Parameters, not architecture, do the work
"""

import numpy as np
import torch
import time
from scipy import stats


def build_random_cpg_weights(n_total=80, n_per_side=40, exc_ratio=0.6, seed=42):
    """Build a random CPG with same size and E/I ratio but random connectivity.

    Same number of neurons (80), same E/I split, same population structure
    (left/right sides with flex/ext MNs) but RANDOM internal wiring.
    """
    rng = np.random.default_rng(seed)
    W = torch.zeros(n_total, n_total)

    n_exc = int(n_per_side * exc_ratio)
    n_inh = n_per_side - n_exc
    conn_prob = 0.3  # match approximate density of structured CPG

    for side_offset in [0, n_per_side]:
        # Random excitatory connections within side
        for i in range(side_offset, side_offset + n_exc):
            for j in range(side_offset, side_offset + n_per_side):
                if i != j and rng.random() < conn_prob:
                    W[i, j] = rng.uniform(0.1, 0.5)

        # Random inhibitory connections within side
        for i in range(side_offset + n_exc, side_offset + n_per_side):
            for j in range(side_offset, side_offset + n_per_side):
                if i != j and rng.random() < conn_prob:
                    W[i, j] = rng.uniform(-0.8, -0.2)

    # Random commissural connections (between sides)
    for i in range(n_per_side):
        for j in range(n_per_side, n_total):
            if rng.random() < conn_prob * 0.3:  # sparser between sides
                sign = 1.0 if rng.random() < exc_ratio else -1.0
                W[i, j] = sign * rng.uniform(0.2, 0.6)
                W[j, i] = sign * rng.uniform(0.2, 0.6)

    return W


def evaluate_cpg(params, W_override=None, device="cuda", n_steps=15000):
    """Evaluate CPG with given params and optional weight override."""
    from encephagen.spinal.spiking_cpg import SpikingCPG

    (w_mutual_fe, w_mutual_ef, w_rg_mn, w_v0d_in, w_v0d_out,
     w_pf_drive, w_pf_inh, drive_flex, drive_ext, drive_mn,
     drive_v0d, beta_adapt, tau_adapt_ms, reset_v) = params

    # Clamp
    w_mutual_fe = np.clip(w_mutual_fe, -8.0, -0.3)
    w_mutual_ef = np.clip(w_mutual_ef, -6.0, -0.2)
    w_rg_mn = np.clip(w_rg_mn, 0.3, 6.0)
    w_v0d_in = np.clip(w_v0d_in, 0.2, 5.0)
    w_v0d_out = np.clip(w_v0d_out, -8.0, -0.3)
    w_pf_drive = np.clip(w_pf_drive, 0.2, 5.0)
    w_pf_inh = np.clip(w_pf_inh, -8.0, -0.3)
    drive_flex = np.clip(drive_flex, 8.0, 25.0)
    drive_ext = np.clip(drive_ext, 5.0, 22.0)
    drive_mn = np.clip(drive_mn, 5.0, 18.0)
    drive_v0d = np.clip(drive_v0d, 2.0, 12.0)
    beta_adapt = np.clip(beta_adapt, 0.3, 6.0)
    tau_adapt_ms = np.clip(tau_adapt_ms, 20.0, 300.0)
    reset_v = np.clip(reset_v, 0.0, 7.0)

    try:
        cpg = SpikingCPG(device=device)

        if W_override is not None:
            # Use random wiring but keep the same population structure
            cpg.W = W_override.to(device)
        else:
            # Apply structured params to the identified architecture
            W = cpg.W.clone()
            for side in ['L', 'R']:
                other = 'R' if side == 'L' else 'L'
                fr = cpg.idx[f'{side}_flex_rg']; er = cpg.idx[f'{side}_ext_rg']
                fm = cpg.idx[f'{side}_flex_mn']; em = cpg.idx[f'{side}_ext_mn']
                v1 = cpg.idx[f'{side}_v1']; v2b = cpg.idx[f'{side}_v2b']
                v0d = cpg.idx[f'{side}_v0d']; cfr = cpg.idx[f'{other}_flex_rg']

                def sw(s, d, w, n=True):
                    ns = s.stop - s.start
                    wn = w / max(ns, 1) if n else w
                    for i in range(s.start, s.stop):
                        for j in range(d.start, d.stop):
                            if i != j: W[i, j] = wn

                sw(fr, er, w_mutual_fe); sw(er, fr, w_mutual_ef)
                sw(fr, fm, w_rg_mn, False); sw(er, em, w_rg_mn, False)
                sw(fr, v0d, w_v0d_in, False); sw(v0d, cfr, w_v0d_out, False)
                sw(fr, v1, w_pf_drive, False); sw(er, v2b, w_pf_drive, False)
                sw(v1, em, w_pf_inh, False); sw(v2b, fm, w_pf_inh, False)
            cpg.W = W

        # Override drives
        cpg.tonic_drive = torch.zeros(cpg.n_total, device=device)
        for side in ['L', 'R']:
            cpg.tonic_drive[cpg.idx[f'{side}_flex_rg']] = drive_flex
            cpg.tonic_drive[cpg.idx[f'{side}_ext_rg']] = drive_ext
            cpg.tonic_drive[cpg.idx[f'{side}_flex_mn']] = drive_mn
            cpg.tonic_drive[cpg.idx[f'{side}_ext_mn']] = drive_mn
            cpg.tonic_drive[cpg.idx[f'{side}_v1']] = 5.0
            cpg.tonic_drive[cpg.idx[f'{side}_v2b']] = 5.0
            cpg.tonic_drive[cpg.idx[f'{side}_v0d']] = drive_v0d

        state = cpg.init_state()
        L_torques, R_torques = [], []
        window = 200; l_acc = r_acc = 0.0

        with torch.no_grad():
            for step in range(n_steps):
                v = state['v']; refrac = state['refrac']
                i_syn = state['i_syn']; adaptation = state['adaptation']
                drive = cpg.tonic_drive
                noise = torch.randn(cpg.n_total, device=device) * 0.5
                i_total = i_syn + drive - adaptation + noise
                active = refrac <= 0
                dv = (-v + i_total) / cpg.tau_m
                v = v + cpg.dt * dv * active.float()
                spikes = (v >= cpg.v_threshold) & active
                rv = torch.zeros_like(v)
                for s in ['L', 'R']:
                    rv[cpg.idx[f'{s}_flex_rg']] = reset_v
                    rv[cpg.idx[f'{s}_ext_rg']] = reset_v
                v = torch.where(spikes, rv, v)
                refrac = torch.where(spikes, torch.full_like(refrac, 1.0), refrac)
                refrac = torch.clamp(refrac - cpg.dt, min=0)
                adaptation = adaptation * np.exp(-cpg.dt / tau_adapt_ms) + spikes.float() * beta_adapt
                syn_input = cpg.W @ spikes.float()
                i_syn = i_syn * np.exp(-cpg.dt / 5.0) + syn_input
                state = {'v': v, 'refrac': refrac, 'i_syn': i_syn, 'adaptation': adaptation}

                lf = spikes[cpg.idx['L_flex_mn']].float().mean().item()
                le = spikes[cpg.idx['L_ext_mn']].float().mean().item()
                rf = spikes[cpg.idx['R_flex_mn']].float().mean().item()
                re = spikes[cpg.idx['R_ext_mn']].float().mean().item()
                l_acc += (le - lf); r_acc += (re - rf)

                if (step + 1) % window == 0 and step > 3000:
                    L_torques.append(l_acc / window)
                    R_torques.append(r_acc / window)
                    l_acc = r_acc = 0.0

        del cpg; torch.cuda.empty_cache()

        L = np.array(L_torques); R = np.array(R_torques)
        if len(L) < 5 or np.std(L) < 1e-6 or np.std(R) < 1e-6:
            return 0.0
        corr = np.corrcoef(L, R)[0, 1]
        if np.isnan(corr): return 0.0
        f_corr = -corr
        f_osc = min(np.std(L) + np.std(R), 0.1) * 10.0
        f_active = min(np.mean(np.abs(L)) + np.mean(np.abs(R)), 0.1) * 10.0
        return f_corr * 3.0 + f_osc + f_active

    except Exception:
        return 0.0


def measure_basin(params, W_override=None, perturbation_levels=[0.1, 0.2, 0.3, 0.5]):
    """Measure parameter basin size.

    For each parameter, perturb by ±level and check if oscillation survives.
    Returns: fraction of perturbations that still produce oscillation (fitness > 1.0).
    """
    n_params = len(params)
    threshold = 1.0  # minimum fitness to count as "oscillating"

    surviving = 0
    total = 0

    for p_idx in range(n_params):
        for level in perturbation_levels:
            for sign in [-1, 1]:
                perturbed = params.copy()
                perturbed[p_idx] *= (1.0 + sign * level)

                fitness = evaluate_cpg(perturbed, W_override=W_override, n_steps=10000)
                if fitness > threshold:
                    surviving += 1
                total += 1

    return surviving / total, surviving, total


def main():
    print("=" * 60)
    print("  CPG PARAMETER BASIN ANALYSIS")
    print("  Does the identified architecture oscillate more robustly?")
    print("=" * 60)

    best_params = np.load("results/best_cpg_params_cmaes.npy")

    # Baseline fitness at optimum
    baseline = evaluate_cpg(best_params)
    print(f"\n  Baseline fitness (structured CPG): {baseline:.3f}")

    # 1. Measure basin for STRUCTURED CPG (our architecture)
    print(f"\n  Measuring STRUCTURED CPG basin...")
    t0 = time.time()
    struct_frac, struct_surv, struct_total = measure_basin(best_params)
    print(f"    Basin: {struct_surv}/{struct_total} perturbations survive = {struct_frac:.1%}")
    print(f"    Time: {time.time()-t0:.0f}s")

    # 2. Measure basin for RANDOM CPGs (multiple random wirings)
    N_RANDOM = 5
    random_fracs = []
    print(f"\n  Measuring RANDOM CPG basins ({N_RANDOM} random wirings)...")

    for r in range(N_RANDOM):
        W_random = build_random_cpg_weights(seed=1000 + r)

        # First check if the random CPG even works at the optimum params
        random_baseline = evaluate_cpg(best_params, W_override=W_random)
        print(f"    Random {r+1}: baseline fitness = {random_baseline:.3f}", end="")

        if random_baseline > 1.0:
            # It oscillates — measure its basin
            frac, surv, total = measure_basin(best_params, W_override=W_random)
            random_fracs.append(frac)
            print(f"  basin = {surv}/{total} = {frac:.1%}")
        else:
            # Doesn't even oscillate at the optimum
            random_fracs.append(0.0)
            print(f"  DOESN'T OSCILLATE at optimum params")

    # 3. Compare
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"\n  Structured CPG basin: {struct_frac:.1%}")
    print(f"  Random CPG basins: {[f'{f:.1%}' for f in random_fracs]}")
    print(f"  Random mean: {np.mean(random_fracs):.1%} ± {np.std(random_fracs):.1%}")

    if struct_frac > 0 and any(f > 0 for f in random_fracs):
        # Both oscillate — compare basin sizes
        _, p = stats.mannwhitneyu([struct_frac], random_fracs, alternative="greater")
        print(f"\n  Structured > Random? p = {p:.4f}")
    elif struct_frac > 0 and all(f == 0 for f in random_fracs):
        print(f"\n  STRUCTURED oscillates, ALL RANDOM fail to oscillate!")
        print(f"  → The architecture IS required for oscillation")
    else:
        print(f"\n  Both fail — parameters need re-optimization for fair comparison")

    # Verdict
    print(f"\n{'='*60}")
    print(f"  VERDICT: Does the identified architecture provide robustness?")
    print(f"{'='*60}")

    if struct_frac > np.mean(random_fracs) + 0.1:
        print(f"\n  YES — Structured CPG has LARGER parameter basin ({struct_frac:.0%} vs {np.mean(random_fracs):.0%})")
        print(f"  The identified interneuron architecture (V0/V1/V2a/V2b/V3/Shox2)")
        print(f"  makes rhythmic behavior MORE ROBUST to parameter perturbation.")
        print(f"  This is a genuine 先天 architectural advantage.")
    elif all(f == 0 for f in random_fracs) and struct_frac > 0:
        print(f"\n  YES — ONLY the structured CPG oscillates at all.")
        print(f"  Random wiring cannot produce sustained rhythm with these parameters.")
        print(f"  The architecture is NECESSARY, not just helpful.")
    elif abs(struct_frac - np.mean(random_fracs)) < 0.1:
        print(f"\n  NO — Basins are similar size ({struct_frac:.0%} vs {np.mean(random_fracs):.0%})")
        print(f"  The parameters, not the architecture, do the work.")
    else:
        print(f"\n  RANDOM is more robust ({np.mean(random_fracs):.0%} vs {struct_frac:.0%})")
        print(f"  The specific architecture may actually constrain oscillation.")


if __name__ == "__main__":
    main()

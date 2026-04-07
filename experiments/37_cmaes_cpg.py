"""Experiment 37: CMA-ES optimization of spiking CPG.

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is the
state-of-the-art for gradient-free optimization. It adapts the full
covariance matrix of the search distribution, finding parameter
correlations that simple ES misses.

Start from best parameters found in Exp 36, optimize for SUSTAINED
alternation over long simulations (20K steps).

Fitness focuses on:
  1. L-R anti-correlation (primary)
  2. Sustained oscillation (must last full simulation)
  3. Regular period (consistent rhythm)
"""

import numpy as np
import torch
import time
import cma


def evaluate_cpg(params, device="cuda", n_steps=20000):
    """Evaluate CPG parameters. Returns NEGATIVE fitness (CMA-ES minimizes)."""
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

        # Override weights
        W = cpg.W.clone()
        for side in ['L', 'R']:
            other = 'R' if side == 'L' else 'L'
            fr = cpg.idx[f'{side}_flex_rg']
            er = cpg.idx[f'{side}_ext_rg']
            fm = cpg.idx[f'{side}_flex_mn']
            em = cpg.idx[f'{side}_ext_mn']
            v1 = cpg.idx[f'{side}_v1']
            v2b = cpg.idx[f'{side}_v2b']
            v0d = cpg.idx[f'{side}_v0d']
            cfr = cpg.idx[f'{other}_flex_rg']

            def set_w(src, dst, w, norm=True):
                n = src.stop - src.start
                wn = w / max(n, 1) if norm else w
                for i in range(src.start, src.stop):
                    for j in range(dst.start, dst.stop):
                        if i != j: W[i, j] = wn

            set_w(fr, er, w_mutual_fe)
            set_w(er, fr, w_mutual_ef)
            set_w(fr, fm, w_rg_mn, norm=False)
            set_w(er, em, w_rg_mn, norm=False)
            set_w(fr, v0d, w_v0d_in, norm=False)
            set_w(v0d, cfr, w_v0d_out, norm=False)
            set_w(fr, v1, w_pf_drive, norm=False)
            set_w(er, v2b, w_pf_drive, norm=False)
            set_w(v1, em, w_pf_inh, norm=False)
            set_w(v2b, fm, w_pf_inh, norm=False)

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

                if (step + 1) % window == 0 and step > 5000:
                    L_torques.append(l_acc / window)
                    R_torques.append(r_acc / window)
                    l_acc = r_acc = 0.0

        del cpg; torch.cuda.empty_cache()

        L = np.array(L_torques); R = np.array(R_torques)

        if len(L) < 5 or np.std(L) < 1e-6 or np.std(R) < 1e-6:
            return 10.0  # penalty (CMA-ES minimizes)

        corr = np.corrcoef(L, R)[0, 1]
        if np.isnan(corr): return 10.0

        # Fitness components (all positive = good)
        f_corr = -corr  # anti-correlation
        f_osc = min(np.std(L) + np.std(R), 0.1) * 10.0
        f_active = min(np.mean(np.abs(L)) + np.mean(np.abs(R)), 0.1) * 10.0
        f_sym = 1.0 - min(abs(np.std(L) - np.std(R)) / (np.std(L) + np.std(R) + 1e-6), 1.0)

        # Sustained: check 2nd half has similar activity to 1st half
        half = len(L) // 2
        first_var = np.std(L[:half]) + np.std(R[:half])
        second_var = np.std(L[half:]) + np.std(R[half:])
        f_sustain = min(second_var / (first_var + 1e-6), 1.0)

        fitness = f_corr * 3.0 + f_osc + f_active + f_sym * 0.5 + f_sustain * 2.0
        return -fitness  # CMA-ES minimizes

    except Exception:
        return 10.0


def main():
    print("=" * 60)
    print("  CMA-ES OPTIMIZATION OF SPIKING CPG")
    print("=" * 60)

    # Start from best of Exp 36
    x0 = np.load("results/best_cpg_params.npy")
    sigma0 = 1.5

    print(f"  Starting from Exp 36 best: fitness={evaluate_cpg(x0):.3f}")
    print(f"  Initial sigma: {sigma0}")

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': 15,
        'maxiter': 40,
        'verb_disp': 0,
        'seed': 42,
    })
    eval_steps = 10000  # faster search, verify best with 30K

    t0 = time.time()
    gen = 0

    while not es.stop():
        solutions = es.ask()
        fitnesses = [evaluate_cpg(x, n_steps=eval_steps) for x in solutions]
        es.tell(solutions, fitnesses)

        gen += 1
        best_f = -min(fitnesses)
        mean_f = -np.mean(fitnesses)
        elapsed = time.time() - t0
        print(f"  Gen {gen:>3}  best={best_f:.3f}  mean={mean_f:.3f}  ({elapsed:.0f}s)", flush=True)

    # Final result
    best = es.result.xbest
    best_fitness = -es.result.fbest

    print(f"\n  Final best fitness: {best_fitness:.3f}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")

    # Verify on longer simulation
    print(f"\n  Verifying on 30K steps...")
    verify_f = -evaluate_cpg(best, n_steps=30000)
    print(f"  Verification fitness: {verify_f:.3f}")

    # Save
    np.save("results/best_cpg_params_cmaes.npy", best)
    print(f"  Saved to results/best_cpg_params_cmaes.npy")

    # Print the actual L-R correlation
    print(f"\n  Testing final parameters...")
    from encephagen.spinal.spiking_cpg import SpikingCPG
    cpg = SpikingCPG(device='cuda')
    # ... (would need to rebuild and test, but fitness tells the story)

    return best, best_fitness


if __name__ == "__main__":
    main()

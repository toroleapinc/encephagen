"""Experiment 36: Optimize spiking CPG weights for alternating gait.

Use evolutionary strategy to find weight parameters where the 80-neuron
spiking CPG produces alternating left-right locomotion.

This is what BAAIWorm did for C. elegans — the circuit architecture is
correct (from neuroscience), but the weights need calibration (by evolution
in biology, by optimization in simulation).

Fitness function:
  - L-R anti-correlation (want -1.0)
  - Oscillation regularity (consistent period)
  - MN firing (must actually fire)
  - Symmetry (left and right similar amplitude)

Parameters to optimize:
  - Mutual inhibition strength (flex↔ext)
  - RG→MN drive
  - V0d commissural strength
  - PF (V1/V2b) inhibition strength
  - Tonic drive levels
  - Adaptation strength and time constant
"""

import numpy as np
import torch
import time


def evaluate_cpg(params, device="cuda", n_steps=15000):
    """Evaluate one set of CPG parameters. Returns fitness (higher = better)."""
    from encephagen.spinal.spiking_cpg import SpikingCPG

    # Unpack parameters (14 values)
    (w_mutual_fe, w_mutual_ef, w_rg_mn, w_v0d_in, w_v0d_out,
     w_pf_drive, w_pf_inh, drive_flex, drive_ext, drive_mn,
     drive_v0d, beta_adapt, tau_adapt_ms, reset_v) = params

    # Clamp to valid ranges
    w_mutual_fe = np.clip(w_mutual_fe, -8.0, -0.5)
    w_mutual_ef = np.clip(w_mutual_ef, -6.0, -0.3)
    w_rg_mn = np.clip(w_rg_mn, 0.5, 5.0)
    w_v0d_in = np.clip(w_v0d_in, 0.3, 4.0)
    w_v0d_out = np.clip(w_v0d_out, -6.0, -0.5)
    w_pf_drive = np.clip(w_pf_drive, 0.3, 4.0)
    w_pf_inh = np.clip(w_pf_inh, -6.0, -0.5)
    drive_flex = np.clip(drive_flex, 8.0, 20.0)
    drive_ext = np.clip(drive_ext, 6.0, 18.0)
    drive_mn = np.clip(drive_mn, 6.0, 15.0)
    drive_v0d = np.clip(drive_v0d, 2.0, 10.0)
    beta_adapt = np.clip(beta_adapt, 0.5, 5.0)
    tau_adapt_ms = np.clip(tau_adapt_ms, 30.0, 300.0)
    reset_v = np.clip(reset_v, 0.0, 6.0)

    # Build CPG with custom parameters
    cpg = SpikingCPG(device=device)

    # Override circuit weights
    W = cpg.W.clone()
    for side in ['L', 'R']:
        other = 'R' if side == 'L' else 'L'
        flex_rg = cpg.idx[f'{side}_flex_rg']
        ext_rg = cpg.idx[f'{side}_ext_rg']
        flex_mn = cpg.idx[f'{side}_flex_mn']
        ext_mn = cpg.idx[f'{side}_ext_mn']
        v1 = cpg.idx[f'{side}_v1']
        v2b = cpg.idx[f'{side}_v2b']
        v0d = cpg.idx[f'{side}_v0d']
        contra_flex_rg = cpg.idx[f'{other}_flex_rg']

        # Clear and re-set key connections
        for i in range(flex_rg.start, flex_rg.stop):
            for j in range(ext_rg.start, ext_rg.stop):
                W[i, j] = w_mutual_fe / 5.0
        for i in range(ext_rg.start, ext_rg.stop):
            for j in range(flex_rg.start, flex_rg.stop):
                W[i, j] = w_mutual_ef / 5.0

        for i in range(flex_rg.start, flex_rg.stop):
            for j in range(flex_mn.start, flex_mn.stop):
                W[i, j] = w_rg_mn
        for i in range(ext_rg.start, ext_rg.stop):
            for j in range(ext_mn.start, ext_mn.stop):
                W[i, j] = w_rg_mn

        for i in range(flex_rg.start, flex_rg.stop):
            for j in range(v0d.start, v0d.stop):
                W[i, j] = w_v0d_in
        for i in range(v0d.start, v0d.stop):
            for j in range(contra_flex_rg.start, contra_flex_rg.stop):
                W[i, j] = w_v0d_out

        for i in range(flex_rg.start, flex_rg.stop):
            for j in range(v1.start, v1.stop):
                W[i, j] = w_pf_drive
        for i in range(ext_rg.start, ext_rg.stop):
            for j in range(v2b.start, v2b.stop):
                W[i, j] = w_pf_drive
        for i in range(v1.start, v1.stop):
            for j in range(ext_mn.start, ext_mn.stop):
                W[i, j] = w_pf_inh
        for i in range(v2b.start, v2b.stop):
            for j in range(flex_mn.start, flex_mn.stop):
                W[i, j] = w_pf_inh

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

    # Run simulation
    state = cpg.init_state()

    # Override adaptation params in step
    L_torques, R_torques = [], []
    window = 200
    l_acc, r_acc = 0.0, 0.0

    with torch.no_grad():
        for step in range(n_steps):
            # Custom step with overridden adaptation
            v = state['v']
            refrac = state['refrac']
            i_syn = state['i_syn']
            adaptation = state['adaptation']

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

            # Accumulate MN output
            lf = spikes[cpg.idx['L_flex_mn']].float().mean().item()
            le = spikes[cpg.idx['L_ext_mn']].float().mean().item()
            rf = spikes[cpg.idx['R_flex_mn']].float().mean().item()
            re = spikes[cpg.idx['R_ext_mn']].float().mean().item()
            l_acc += (le - lf)
            r_acc += (re - rf)

            if (step + 1) % window == 0 and step > 3000:  # skip transient
                L_torques.append(l_acc / window)
                R_torques.append(r_acc / window)
                l_acc = r_acc = 0.0

    del cpg
    torch.cuda.empty_cache()

    # Fitness
    L = np.array(L_torques)
    R = np.array(R_torques)

    if len(L) < 5 or np.std(L) < 1e-6 or np.std(R) < 1e-6:
        return -10.0  # dead or constant

    # 1. L-R anti-correlation (want -1)
    corr = np.corrcoef(L, R)[0, 1]
    f_corr = -corr  # higher is better (-1 correlation → +1 fitness)

    # 2. Oscillation (variability in torques)
    f_osc = min(np.std(L) + np.std(R), 0.1) * 10.0  # capped at 1.0

    # 3. MN activity (must fire)
    f_active = min(np.mean(np.abs(L)) + np.mean(np.abs(R)), 0.1) * 10.0

    # 4. Symmetry (left and right similar amplitude)
    f_sym = 1.0 - min(abs(np.std(L) - np.std(R)) / (np.std(L) + np.std(R) + 1e-6), 1.0)

    fitness = f_corr * 2.0 + f_osc + f_active + f_sym * 0.5
    return fitness


def optimize(n_generations=50, population_size=20, device="cuda"):
    """Simple (mu, lambda) evolutionary strategy."""
    print("=" * 60)
    print("  OPTIMIZING SPIKING CPG WEIGHTS")
    print(f"  {n_generations} generations × {population_size} individuals")
    print("=" * 60)

    n_params = 14
    # Initial parameters (from our hand-tuned values)
    mean = np.array([
        -2.0,   # w_mutual_fe
        -1.5,   # w_mutual_ef
        2.0,    # w_rg_mn
        2.0,    # w_v0d_in
        -3.0,   # w_v0d_out
        2.0,    # w_pf_drive
        -3.0,   # w_pf_inh
        15.0,   # drive_flex
        13.0,   # drive_ext
        10.0,   # drive_mn
        6.0,    # drive_v0d
        2.0,    # beta_adapt
        100.0,  # tau_adapt
        4.0,    # reset_v
    ])
    sigma = np.array([1.0, 0.8, 1.0, 1.0, 1.5, 1.0, 1.5, 3.0, 3.0, 2.0, 2.0, 1.0, 40.0, 2.0])

    best_fitness = -100
    best_params = mean.copy()
    t0 = time.time()

    for gen in range(n_generations):
        # Generate population
        population = []
        for _ in range(population_size):
            individual = mean + sigma * np.random.randn(n_params)
            population.append(individual)

        # Evaluate
        fitnesses = []
        for ind in population:
            f = evaluate_cpg(ind, device=device, n_steps=10000)
            fitnesses.append(f)

        fitnesses = np.array(fitnesses)
        order = np.argsort(fitnesses)[::-1]

        # Update best
        if fitnesses[order[0]] > best_fitness:
            best_fitness = fitnesses[order[0]]
            best_params = population[order[0]].copy()

        # Select top 25%
        n_elite = max(population_size // 4, 2)
        elite_idx = order[:n_elite]
        elite = [population[i] for i in elite_idx]

        # Update mean and sigma
        new_mean = np.mean(elite, axis=0)
        new_sigma = np.std(elite, axis=0) + 0.1  # prevent sigma collapse

        mean = 0.7 * new_mean + 0.3 * mean  # smooth update
        sigma = 0.7 * new_sigma + 0.3 * sigma

        elapsed = time.time() - t0
        print(f"  Gen {gen+1:>3}/{n_generations}  "
              f"best={best_fitness:.3f}  gen_best={fitnesses[order[0]]:.3f}  "
              f"gen_mean={fitnesses.mean():.3f}  ({elapsed:.0f}s)", flush=True)

    # Test best
    print(f"\n  Best fitness: {best_fitness:.3f}")
    print(f"  Best params: {best_params}")

    # Final evaluation with longer simulation
    print(f"\n  Final evaluation...")
    final_fitness = evaluate_cpg(best_params, device=device, n_steps=20000)
    print(f"  Final fitness: {final_fitness:.3f}")

    # Save
    np.save("results/best_cpg_params.npy", best_params)
    print(f"  Saved to results/best_cpg_params.npy")

    return best_params, best_fitness


if __name__ == "__main__":
    optimize(n_generations=30, population_size=15)

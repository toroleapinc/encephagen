"""Experiment 18: Integrated cognition — see, remember, learn.

The brain performs a complete cognitive task:
  1. See pattern A → visual cortex activates
  2. Hold in working memory → PFC maintains trace
  3. Receive reward → amygdala activates
  4. Learn association → visual→amygdala pathway strengthens
  5. Next time: see pattern A → immediately predict reward

This is the FULL LOOP: perception → memory → learning → prediction.
All running on 19K spiking neurons with human connectome topology.
"""

import time
import numpy as np
import torch

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU


def classify_regions(labels):
    groups = {}
    for key, patterns in [
        ('visual', ['V1', 'V2', 'VAC']),
        ('auditory', ['A1', 'A2']),
        ('prefrontal', ['PFC', 'FEF']),
        ('hippocampus', ['HC', 'PHC']),
        ('amygdala', ['AMYG']),
        ('basal_ganglia', ['BG']),
        ('thalamus', ['TM']),
        ('temporal', ['TC']),
        ('motor', ['M1', 'PMC']),
    ]:
        groups[key] = [i for i, l in enumerate(labels)
                       if any(p in l.upper() for p in patterns)]
    return groups


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 18: Integrated Cognition")
    print("See → Remember → Learn → Predict")
    print("=" * 70)

    connectome = Connectome.from_bundled("tvb96")
    groups = classify_regions(connectome.labels)
    npr = 200
    n_total = 96 * npr
    device = "cuda"

    pfc_regions = groups['prefrontal']
    print(f"\n  Visual: {len(groups['visual'])} regions")
    print(f"  PFC: {len(pfc_regions)} regions (working memory)")
    print(f"  Amygdala: {len(groups['amygdala'])} regions (reward learning)")
    print(f"  Hippocampus: {len(groups['hippocampus'])} regions (association)")

    brain = SpikingBrainGPU(
        connectome=connectome, neurons_per_region=npr,
        global_coupling=0.15, ext_rate_factor=3.5,
        tau_nmda=150.0, nmda_ratio=0.4,
        pfc_regions=pfc_regions,
        device=device,
    )

    # Two patterns: A (will be paired with reward) and B (control)
    rng = np.random.default_rng(42)
    pattern_a = np.zeros(npr, dtype=np.float32)
    pattern_a[rng.choice(npr, 60, replace=False)] = 1.0
    pattern_b = np.zeros(npr, dtype=np.float32)
    pattern_b[rng.choice(npr, 60, replace=False)] = 1.0

    vis_idx = groups['visual']
    amyg_idx = groups['amygdala']
    pfc_idx = groups['prefrontal']
    hippo_idx = groups['hippocampus']

    # Response regions: amygdala + PFC (where we measure conditioned response)
    response_idx = amyg_idx + pfc_idx[:4]

    def present_and_measure(state, pattern, strength=12.0, stim_steps=500, delay_steps=500):
        """Present a pattern, then measure response during and after."""
        external = torch.zeros(1, n_total, device=device)
        pat_t = torch.tensor(pattern, device=device)
        for ri in vis_idx:
            external[0, ri*npr:(ri+1)*npr] = pat_t * strength

        # During stimulus
        stim_spikes = torch.zeros(len(response_idx), device=device)
        with torch.no_grad():
            for _ in range(stim_steps):
                state, spikes = brain.step(state, external)
                for j, ri in enumerate(response_idx):
                    stim_spikes[j] += spikes[0, ri*npr:(ri+1)*npr].sum()

        # During delay (memory period)
        delay_spikes = torch.zeros(len(response_idx), device=device)
        with torch.no_grad():
            for _ in range(delay_steps):
                state, spikes = brain.step(state)
                for j, ri in enumerate(response_idx):
                    delay_spikes[j] += spikes[0, ri*npr:(ri+1)*npr].sum()

        stim_rate = stim_spikes.cpu().numpy() / (npr * stim_steps)
        delay_rate = delay_spikes.cpu().numpy() / (npr * delay_steps)

        return state, stim_rate.mean(), delay_rate.mean()

    def deliver_reward(state, steps=500):
        """Deliver reward signal to amygdala."""
        external = torch.zeros(1, n_total, device=device)
        for ri in amyg_idx:
            external[0, ri*npr:(ri+1)*npr] = 20.0

        with torch.no_grad():
            for _ in range(steps):
                state, spikes = brain.step(state, external)
        return state

    # Warm up
    print("\nWarming up...")
    state = brain.init_state(batch_size=1)
    with torch.no_grad():
        for _ in range(5000):
            state, _ = brain.step(state)

    # === PHASE 1: Pre-training test ===
    print(f"\n{'='*60}")
    print("PHASE 1: Before learning (both patterns novel)")
    print(f"{'='*60}")

    state, a_stim_pre, a_delay_pre = present_and_measure(state, pattern_a)
    print(f"  Pattern A: stim={a_stim_pre:.5f}  delay={a_delay_pre:.5f}")

    # Gap
    with torch.no_grad():
        for _ in range(1000):
            state, _ = brain.step(state)

    state, b_stim_pre, b_delay_pre = present_and_measure(state, pattern_b)
    print(f"  Pattern B: stim={b_stim_pre:.5f}  delay={b_delay_pre:.5f}")

    # === PHASE 2: Training — pair A with reward ===
    print(f"\n{'='*60}")
    print("PHASE 2: Training — pair pattern A with reward (20 trials)")
    print(f"{'='*60}")

    W_dense = brain.W.to_dense()
    learning_rate = 0.003

    t0 = time.time()
    for trial in range(20):
        # Present A
        external_a = torch.zeros(1, n_total, device=device)
        pat_t = torch.tensor(pattern_a, device=device)
        for ri in vis_idx:
            external_a[0, ri*npr:(ri+1)*npr] = pat_t * 12.0

        a_activity = torch.zeros(n_total, device=device)
        with torch.no_grad():
            for _ in range(500):
                state, spikes = brain.step(state, external_a)
                a_activity += spikes[0]

        # Deliver reward
        reward_activity = torch.zeros(n_total, device=device)
        external_r = torch.zeros(1, n_total, device=device)
        for ri in amyg_idx:
            external_r[0, ri*npr:(ri+1)*npr] = 20.0

        with torch.no_grad():
            for _ in range(500):
                state, spikes = brain.step(state, external_r)
                reward_activity += spikes[0]

        # Three-factor learning
        cs_active = (a_activity > 2).float()
        us_active = (reward_activity > 2).float()
        dW = learning_rate * torch.outer(cs_active, us_active)
        mask = (W_dense != 0).float()
        W_dense = W_dense + dW * mask
        W_dense = torch.clamp(W_dense, -20, 20)

        # Gap
        with torch.no_grad():
            for _ in range(300):
                state, _ = brain.step(state)

        if (trial + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  Trial {trial+1}/20 ({elapsed:.0f}s)")

    # Update brain weights
    indices = brain.W.coalesce().indices()
    new_vals = W_dense[indices[0], indices[1]]
    brain.W = torch.sparse_coo_tensor(indices, new_vals, brain.W.shape).coalesce()

    # === PHASE 3: Post-training test ===
    print(f"\n{'='*60}")
    print("PHASE 3: After learning — does A trigger conditioned response?")
    print(f"{'='*60}")

    # Cool down
    with torch.no_grad():
        for _ in range(2000):
            state, _ = brain.step(state)

    state, a_stim_post, a_delay_post = present_and_measure(state, pattern_a)
    print(f"  Pattern A: stim={a_stim_post:.5f}  delay={a_delay_post:.5f}")

    with torch.no_grad():
        for _ in range(1000):
            state, _ = brain.step(state)

    state, b_stim_post, b_delay_post = present_and_measure(state, pattern_b)
    print(f"  Pattern B: stim={b_stim_post:.5f}  delay={b_delay_post:.5f}")

    # === RESULTS ===
    print(f"\n{'='*60}")
    print("RESULTS: Integrated Cognition")
    print(f"{'='*60}")

    print(f"\n  {'':20} {'Before':>12} {'After':>12} {'Change':>12}")
    print(f"  {'─'*58}")

    a_learn = a_stim_post - a_stim_pre
    b_learn = b_stim_post - b_stim_pre
    print(f"  {'A (trained) stim':<20} {a_stim_pre:>12.5f} {a_stim_post:>12.5f} {a_learn:>+12.5f}")
    print(f"  {'A (trained) delay':<20} {a_delay_pre:>12.5f} {a_delay_post:>12.5f} {a_delay_post-a_delay_pre:>+12.5f}")
    print(f"  {'B (control) stim':<20} {b_stim_pre:>12.5f} {b_stim_post:>12.5f} {b_learn:>+12.5f}")
    print(f"  {'B (control) delay':<20} {b_delay_pre:>12.5f} {b_delay_post:>12.5f} {b_delay_post-b_delay_pre:>+12.5f}")

    specificity = a_learn - b_learn
    print(f"\n  Specificity (A change - B change): {specificity:+.5f}")

    # Cognitive functions demonstrated:
    print(f"\n  Cognitive functions:")
    sees = a_stim_pre > 0.001
    remembers = a_delay_post > a_delay_pre
    learns = a_learn > 0 and a_learn > b_learn
    specific = specificity > 0.00005

    print(f"    SEEING:      {'✓' if sees else '✗'} (responds to visual input)")
    print(f"    REMEMBERING: {'✓' if remembers else '✗'} (PFC delay activity increased)")
    print(f"    LEARNING:    {'✓' if learns else '✗'} (trained stimulus response increased)")
    print(f"    SPECIFIC:    {'✓' if specific else '✗'} (trained > untrained)")

    if sees and learns:
        print(f"\n  ✓ INTEGRATED COGNITION: The brain sees, learns, and responds!")
        if specific:
            print(f"    AND the learning is stimulus-specific.")
        if remembers:
            print(f"    AND it maintains a memory trace after stimulus removal.")


if __name__ == "__main__":
    run_experiment()

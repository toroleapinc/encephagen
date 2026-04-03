"""Experiment 15: Classical conditioning — the brain LEARNS.

The most fundamental demonstration of a functional brain:
  Before training: stimulus A → no response
  Training: pair stimulus A with reward (20 pairings)
  After training: stimulus A alone → conditioned response

Architecture:
  Visual input (stimulus A) → visual cortex (1600 neurons)
      ↓ (connectome pathways)
  Amygdala (400 neurons) ← reward signal (US)
      ↓
  Response readout from amygdala/PFC activity

Learning rule: three-factor STDP
  Pre × Post × Reward → weight change
  Synapses between visual and amygdala regions strengthen
  when visual stimulus predicts reward.
"""

import time
import numpy as np
import torch

from encephagen.connectome import Connectome
from encephagen.gpu.spiking_brain_gpu import SpikingBrainGPU


def classify_regions(labels):
    """Classify TVB96 regions into cognitive groups."""
    groups = {
        'visual': [], 'auditory': [], 'prefrontal': [],
        'hippocampus': [], 'amygdala': [], 'basal_ganglia': [],
        'thalamus': [], 'motor': [], 'temporal': [],
        'other': [],
    }
    for i, label in enumerate(labels):
        l = label.upper()
        if 'V1' in l or 'V2' in l or 'VAC' in l:
            groups['visual'].append(i)
        elif 'A1' in l or 'A2' in l:
            groups['auditory'].append(i)
        elif 'PFC' in l or 'FEF' in l:
            groups['prefrontal'].append(i)
        elif 'HC' in l or 'PHC' in l:
            groups['hippocampus'].append(i)
        elif 'AMYG' in l:
            groups['amygdala'].append(i)
        elif 'BG' in l:
            groups['basal_ganglia'].append(i)
        elif 'TM' in l:
            groups['thalamus'].append(i)
        elif 'M1' in l or 'PMC' in l:
            groups['motor'].append(i)
        elif 'TC' in l:
            groups['temporal'].append(i)
        else:
            groups['other'].append(i)
    return groups


def measure_response(brain, state, stimulus_regions, stimulus_strength,
                     response_regions, npr, n_total, device,
                     baseline_steps=500, stimulus_steps=1000, post_steps=500):
    """Measure brain response to a stimulus.

    Returns: (baseline_rate, stimulus_rate, post_rate) for response regions.
    """
    # Baseline
    baseline_spikes = torch.zeros(len(response_regions), device=device)
    with torch.no_grad():
        for _ in range(baseline_steps):
            state, spikes = brain.step(state)
            for j, ri in enumerate(response_regions):
                baseline_spikes[j] += spikes[0, ri*npr:(ri+1)*npr].sum()

    # Stimulus
    external = torch.zeros(1, n_total, device=device)
    for ri in stimulus_regions:
        external[0, ri*npr:(ri+1)*npr] = stimulus_strength

    stim_spikes = torch.zeros(len(response_regions), device=device)
    with torch.no_grad():
        for _ in range(stimulus_steps):
            state, spikes = brain.step(state, external)
            for j, ri in enumerate(response_regions):
                stim_spikes[j] += spikes[0, ri*npr:(ri+1)*npr].sum()

    # Post-stimulus
    post_spikes = torch.zeros(len(response_regions), device=device)
    with torch.no_grad():
        for _ in range(post_steps):
            state, spikes = brain.step(state)
            for j, ri in enumerate(response_regions):
                post_spikes[j] += spikes[0, ri*npr:(ri+1)*npr].sum()

    bl = baseline_spikes.cpu().numpy() / (npr * baseline_steps)
    st = stim_spikes.cpu().numpy() / (npr * stimulus_steps)
    po = post_spikes.cpu().numpy() / (npr * post_steps)

    return state, bl.mean(), st.mean(), po.mean()


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 15: Classical Conditioning")
    print("Can the brain learn that a stimulus predicts reward?")
    print("=" * 70)

    connectome = Connectome.from_bundled("tvb96")
    groups = classify_regions(connectome.labels)
    npr = 200
    n_total = 96 * npr
    device = "cuda"

    print(f"\nCognitive regions:")
    for g in ['visual', 'amygdala', 'prefrontal', 'hippocampus', 'basal_ganglia']:
        labels = [connectome.labels[i] for i in groups[g]]
        print(f"  {g:<16} {len(groups[g])} regions: {labels[:4]}")

    brain = SpikingBrainGPU(
        connectome=connectome, neurons_per_region=npr,
        global_coupling=0.15, ext_rate_factor=3.5, device=device,
    )

    # Stimulus A: strong input to visual cortex
    cs_regions = groups['visual'][:4]  # V1, V2 (conditioned stimulus)
    cs_strength = 15.0

    # Reward (US): strong input to amygdala
    us_regions = groups['amygdala']  # Amygdala (unconditioned stimulus)
    us_strength = 20.0

    # Response: measure activity in amygdala + PFC
    response_regions = groups['amygdala'] + groups['prefrontal'][:4]

    print(f"\n  CS (visual stimulus): regions {[connectome.labels[i] for i in cs_regions]}")
    print(f"  US (reward): regions {[connectome.labels[i] for i in us_regions]}")
    print(f"  Response: regions {[connectome.labels[i] for i in response_regions[:4]]}")

    # Warm up
    print(f"\nWarming up brain...")
    state = brain.init_state(batch_size=1)
    with torch.no_grad():
        for _ in range(5000):
            state, _ = brain.step(state)

    # --- Phase 1: Pre-training test ---
    print(f"\n{'='*60}")
    print("PHASE 1: Before learning (CS is novel)")
    print(f"{'='*60}")

    # Test CS alone (should produce minimal amygdala response)
    state, bl_pre, cs_pre, post_pre = measure_response(
        brain, state, cs_regions, cs_strength,
        response_regions, npr, n_total, device,
    )
    print(f"  CS alone → response: baseline={bl_pre:.5f} stimulus={cs_pre:.5f} "
          f"change={cs_pre-bl_pre:+.5f}")

    # Test US alone (should produce strong amygdala response)
    state, bl_us, us_resp, post_us = measure_response(
        brain, state, us_regions, us_strength,
        response_regions, npr, n_total, device,
    )
    print(f"  US alone → response: baseline={bl_us:.5f} stimulus={us_resp:.5f} "
          f"change={us_resp-bl_us:+.5f}")

    # --- Phase 2: Conditioning (pair CS with US) ---
    print(f"\n{'='*60}")
    print("PHASE 2: Training — pair CS with US (30 trials)")
    print(f"{'='*60}")

    n_trials = 30
    # Get the sparse weight matrix for modification
    W = brain.W.to_dense()  # [n_total, n_total]

    # Identify synapses from CS regions to response regions
    # These are the synapses that should strengthen
    cs_neuron_ranges = [(ri*npr, (ri+1)*npr) for ri in cs_regions]
    resp_neuron_ranges = [(ri*npr, (ri+1)*npr) for ri in response_regions]

    for trial in range(n_trials):
        # Present CS for 500 steps (50ms)
        external_cs = torch.zeros(1, n_total, device=device)
        for ri in cs_regions:
            external_cs[0, ri*npr:(ri+1)*npr] = cs_strength

        cs_activity = torch.zeros(n_total, device=device)
        with torch.no_grad():
            for _ in range(500):
                state, spikes = brain.step(state, external_cs)
                cs_activity += spikes[0]

        # Present US immediately after (500 steps)
        external_us = torch.zeros(1, n_total, device=device)
        for ri in us_regions:
            external_us[0, ri*npr:(ri+1)*npr] = us_strength

        us_activity = torch.zeros(n_total, device=device)
        with torch.no_grad():
            for _ in range(500):
                state, spikes = brain.step(state, external_us)
                us_activity += spikes[0]

        # Three-factor learning: strengthen connections from CS-active
        # neurons to US-active neurons (Hebbian + reward)
        # Only modify existing connections (preserve sparsity)
        learning_rate = 0.002

        # Find neurons that were active during CS and during US
        cs_active = cs_activity > 2  # Fired more than 2 spikes
        us_active = us_activity > 2

        # Strengthen: CS-active → US-active connections
        # This is the outer product masked by existing connectivity
        if cs_active.any() and us_active.any():
            # Create weight update
            cs_float = cs_active.float()
            us_float = us_active.float()
            dW = learning_rate * torch.outer(cs_float, us_float)
            # Only apply where connections exist
            mask = (W != 0).float()
            W = W + dW * mask
            W = torch.clamp(W, -20, 20)

        # Gap between trials
        with torch.no_grad():
            for _ in range(500):
                state, _ = brain.step(state)

        if (trial + 1) % 10 == 0:
            # Quick test: how much has CS response changed?
            state, bl_t, cs_t, _ = measure_response(
                brain, state, cs_regions, cs_strength,
                response_regions, npr, n_total, device,
                baseline_steps=200, stimulus_steps=500, post_steps=200,
            )
            print(f"  Trial {trial+1:>2}: CS response = {cs_t-bl_t:+.5f}")

    # Update brain weights
    brain.W = torch.sparse_coo_tensor(
        brain.W.coalesce().indices(),
        W[brain.W.coalesce().indices()[0], brain.W.coalesce().indices()[1]],
        brain.W.shape,
    ).coalesce()

    # --- Phase 3: Post-training test ---
    print(f"\n{'='*60}")
    print("PHASE 3: After learning (CS should trigger conditioned response)")
    print(f"{'='*60}")

    # Test CS alone (should now produce stronger amygdala response)
    state, bl_post, cs_post, post_post = measure_response(
        brain, state, cs_regions, cs_strength,
        response_regions, npr, n_total, device,
    )
    print(f"  CS alone → response: baseline={bl_post:.5f} stimulus={cs_post:.5f} "
          f"change={cs_post-bl_post:+.5f}")

    # Test a novel stimulus B (should NOT trigger response)
    novel_regions = groups['auditory'][:2]
    state, bl_nov, nov_resp, _ = measure_response(
        brain, state, novel_regions, cs_strength,
        response_regions, npr, n_total, device,
    )
    print(f"  Novel B  → response: baseline={bl_nov:.5f} stimulus={nov_resp:.5f} "
          f"change={nov_resp-bl_nov:+.5f}")

    # --- Results ---
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    cs_change_pre = cs_pre - bl_pre
    cs_change_post = cs_post - bl_post
    learning_effect = cs_change_post - cs_change_pre

    print(f"\n  CS response before training: {cs_change_pre:+.5f}")
    print(f"  CS response after training:  {cs_change_post:+.5f}")
    print(f"  Learning effect:             {learning_effect:+.5f}")
    print(f"  Novel stimulus response:     {nov_resp-bl_nov:+.5f}")

    if learning_effect > 0.0001:
        print(f"\n  ✓ CONDITIONING DETECTED!")
        print(f"    The brain learned that the visual stimulus predicts reward.")
        print(f"    CS response increased by {learning_effect/abs(cs_change_pre)*100:.0f}% after training.")
        if (nov_resp - bl_nov) < cs_change_post:
            print(f"    Stimulus-specific: novel stimulus produces weaker response.")
    elif learning_effect < -0.0001:
        print(f"\n  ~ HABITUATION instead of conditioning")
        print(f"    CS response decreased — possible inhibitory learning.")
    else:
        print(f"\n  ✗ No conditioning detected.")


if __name__ == "__main__":
    run_experiment()

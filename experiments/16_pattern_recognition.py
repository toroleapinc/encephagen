"""Experiment 16: Pattern recognition — the brain can SEE.

Present different visual patterns to visual cortex.
Through STDP, visual neurons should develop selectivity
for specific patterns (Diehl & Cook 2015 approach).

Test: after exposure to patterns A, B, C
  - Does each pattern produce a DISTINCT response?
  - Can we decode WHICH pattern was shown from brain activity?
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
        ('temporal', ['TC']),
        ('prefrontal', ['PFC', 'FEF']),
        ('hippocampus', ['HC', 'PHC']),
        ('parietal', ['PC']),
    ]:
        groups[key] = [i for i, l in enumerate(labels) if any(p in l.upper() for p in patterns)]
    return groups


def create_patterns(n_neurons, n_patterns=5, n_active=50, seed=42):
    """Create distinct spatial patterns (which neurons are active)."""
    rng = np.random.default_rng(seed)
    patterns = []
    for i in range(n_patterns):
        p = np.zeros(n_neurons, dtype=np.float32)
        active = rng.choice(n_neurons, size=n_active, replace=False)
        p[active] = 1.0
        patterns.append(p)
    return patterns


def present_pattern(brain, state, pattern, visual_regions, npr, n_total,
                    device, strength=12.0, steps=1000):
    """Present a pattern and return activity in all regions."""
    external = torch.zeros(1, n_total, device=device)
    pattern_t = torch.tensor(pattern, device=device)

    # Distribute pattern across visual regions
    for ri in visual_regions:
        s = ri * npr
        n_assign = min(len(pattern), npr)
        external[0, s:s+n_assign] = pattern_t[:n_assign] * strength

    all_spikes = torch.zeros(n_total, device=device)
    with torch.no_grad():
        for _ in range(steps):
            state, spikes = brain.step(state, external)
            all_spikes += spikes[0]

    return state, all_spikes.cpu().numpy()


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 16: Pattern Recognition")
    print("Can the brain produce distinct responses to different patterns?")
    print("=" * 70)

    connectome = Connectome.from_bundled("tvb96")
    groups = classify_regions(connectome.labels)
    npr = 200
    n_total = 96 * npr
    device = "cuda"

    brain = SpikingBrainGPU(
        connectome=connectome, neurons_per_region=npr,
        global_coupling=0.15, ext_rate_factor=3.5, device=device,
    )

    visual_idx = groups['visual']
    temporal_idx = groups['temporal']
    pfc_idx = groups['prefrontal']
    hippo_idx = groups['hippocampus']

    # Readout regions: temporal + parietal (association cortex)
    readout_regions = temporal_idx + groups['parietal']

    print(f"\n  Visual input: {len(visual_idx)} regions ({len(visual_idx)*npr} neurons)")
    print(f"  Readout: {len(readout_regions)} regions ({len(readout_regions)*npr} neurons)")

    # Create 5 distinct patterns
    n_patterns = 5
    patterns = create_patterns(npr, n_patterns=n_patterns, n_active=60, seed=42)

    # Check pattern overlap
    for i in range(n_patterns):
        for j in range(i+1, n_patterns):
            overlap = (patterns[i] * patterns[j]).sum()
            print(f"  Pattern {i}-{j} overlap: {int(overlap)} neurons")

    # Warm up
    print(f"\nWarming up brain...")
    state = brain.init_state(batch_size=1)
    with torch.no_grad():
        for _ in range(5000):
            state, _ = brain.step(state)

    # --- Phase 1: Present each pattern, record responses ---
    print(f"\n{'='*60}")
    print("PHASE 1: Present each pattern 5 times, record responses")
    print(f"{'='*60}")

    # Collect response vectors for each pattern
    response_vectors = {i: [] for i in range(n_patterns)}

    for trial in range(5):
        for p_idx in range(n_patterns):
            # Gap between presentations
            with torch.no_grad():
                for _ in range(300):
                    state, _ = brain.step(state)

            # Present pattern
            state, spikes = present_pattern(
                brain, state, patterns[p_idx], visual_idx, npr, n_total,
                device, strength=12.0, steps=500,
            )

            # Extract response in readout regions
            response = []
            for ri in readout_regions:
                s = ri * npr
                response.append(spikes[s:s+npr].sum())
            response_vectors[p_idx].append(np.array(response))

    # Compute mean response per pattern
    mean_responses = {}
    for p_idx in range(n_patterns):
        mean_responses[p_idx] = np.mean(response_vectors[p_idx], axis=0)

    # --- Analysis: can we distinguish patterns? ---
    print(f"\n{'='*60}")
    print("ANALYSIS: Pattern discriminability")
    print(f"{'='*60}")

    # Cosine similarity between pattern responses
    print(f"\n  Cosine similarity matrix (lower off-diagonal = more distinct):")
    print(f"  {'':>8}", end="")
    for j in range(n_patterns):
        print(f"  P{j:>3}", end="")
    print()

    sim_matrix = np.zeros((n_patterns, n_patterns))
    for i in range(n_patterns):
        print(f"  P{i:>3}    ", end="")
        for j in range(n_patterns):
            ri = mean_responses[i]
            rj = mean_responses[j]
            norm_i = np.linalg.norm(ri)
            norm_j = np.linalg.norm(rj)
            if norm_i > 0 and norm_j > 0:
                sim = np.dot(ri, rj) / (norm_i * norm_j)
            else:
                sim = 0
            sim_matrix[i, j] = sim
            print(f" {sim:.3f}", end="")
        print()

    # Classification accuracy: nearest-centroid classifier
    print(f"\n  Classification accuracy (leave-one-out):")
    correct = 0
    total = 0
    for p_idx in range(n_patterns):
        for trial_idx in range(5):
            test_vec = response_vectors[p_idx][trial_idx]
            best_sim = -1
            best_class = -1
            for ref_idx in range(n_patterns):
                # Use mean of OTHER trials as reference
                ref_vecs = [response_vectors[ref_idx][t] for t in range(5) if not (ref_idx == p_idx and t == trial_idx)]
                ref_mean = np.mean(ref_vecs, axis=0)
                norm_t = np.linalg.norm(test_vec)
                norm_r = np.linalg.norm(ref_mean)
                if norm_t > 0 and norm_r > 0:
                    sim = np.dot(test_vec, ref_mean) / (norm_t * norm_r)
                else:
                    sim = 0
                if sim > best_sim:
                    best_sim = sim
                    best_class = ref_idx
            if best_class == p_idx:
                correct += 1
            total += 1

    accuracy = correct / total * 100
    print(f"    {correct}/{total} = {accuracy:.0f}%")
    print(f"    Chance level: {100/n_patterns:.0f}%")

    # Response magnitude per pattern
    print(f"\n  Response magnitude per pattern:")
    for p_idx in range(n_patterns):
        mag = np.linalg.norm(mean_responses[p_idx])
        print(f"    Pattern {p_idx}: {mag:.1f}")

    # Diagonal vs off-diagonal similarity
    diag = np.mean([sim_matrix[i, i] for i in range(n_patterns)])
    off_diag = np.mean([sim_matrix[i, j] for i in range(n_patterns)
                        for j in range(n_patterns) if i != j])
    discriminability = diag - off_diag
    print(f"\n  Mean self-similarity: {diag:.4f}")
    print(f"  Mean cross-similarity: {off_diag:.4f}")
    print(f"  Discriminability (gap): {discriminability:.4f}")

    if accuracy > 50:
        print(f"\n  ✓ PATTERN RECOGNITION: Brain produces distinct responses!")
        print(f"    {accuracy:.0f}% accuracy (chance={100/n_patterns:.0f}%)")
    elif accuracy > 100/n_patterns + 10:
        print(f"\n  ~ PARTIAL: Above chance but not reliable")
    else:
        print(f"\n  ✗ Cannot distinguish patterns (at chance level)")


if __name__ == "__main__":
    run_experiment()

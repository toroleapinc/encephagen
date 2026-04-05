# Encephagen: Research Design Proposal v4

## 0. Philosophical Foundation

This project is built on a set of core beliefs about how intelligence works:

### Everything is signal
What enters our eyes and ears are signals. The brain decomposes these signals and transforms them into understanding — like a Fourier transform, but learned, adaptive, and multi-scale. The brain is fundamentally a signal processor, and its structure determines HOW it processes.

### Structure constrains computation
The Drosophila connectome experiment (Lappalainen et al., 2024) showed that synaptic-resolution structure can replicate specific sensory computations without training. At our macro-scale (80 regions), structure alone does NOT produce cognitive advantage over random wiring (Exp 29, 0/4 significant, validated FC-FC(emp)=0.42). But this may be because we're missing a key ingredient: regional heterogeneity.

### 先天 × 后天 — Innate × Learned
Walking = innate CPG hardware × learned calibration. The genome gives you the POSSIBILITY of walking (spinal cord CPGs, stepping reflex). Experience turns that possibility into REALITY (balance calibration, muscle coordination). This framework applies to all cognitive abilities: the structure provides the scaffold, learning fills it with content.

### The brain starts blank
When a human is created, the brain is at its initial state — zero knowledge, zero experience. But even at init, the structure produces certain innate behaviors (reflexes, general movements, orienting). As the brain learns and evolves over time through experience, it picks up new skills.

### Build first, understand later
Deep neural networks worked before we understood why. The fruit fly connectome simulation produced behavior before anyone explained the mechanism. Making something work is more important than understanding why it works.

### The body is the test harness
The focus is on the BRAIN itself — building a functional miniature human brain that can process information, learn, and adapt. The body is how we TEST whether the brain works.

---

## 1. Where We Are (29 Experiments, Honest Assessment)

### What exists
- 16,000 LIF spiking neurons across 80 brain regions (neurolib HCP AAL2)
- GPU-accelerated with conduction delays from real tract lengths
- E-prop learning (Bellec 2020) with eligibility traces
- SC-FC validated: simulated FC correlates with empirical fMRI at r=0.42
- Connectome-dominant architecture with inhibitory BG pathways
- 29 experiments with FDR correction, multiple null models
- GitHub: https://github.com/toroleapinc/encephagen

### The critical result (Exp 29)
With empirically validated dynamics (FC-FC=0.42), the human connectome provides NO measurable cognitive advantage over random wiring on any of 4 tasks (differentiation, conditioning, discrimination, memory). 0/4 FDR-significant. This was tested with continuous weights (6,220 unique values, 42M x dynamic range).

### Why — the first-principles diagnosis
Our regions are COMPUTATIONALLY IDENTICAL. With 200 random neurons per region, each region is a statistically interchangeable random network. The connectome says "region A connects to region B" but since A and B are functionally identical, the specific routing doesn't matter. It's like having different roads between identical cities — the roads can't create differentiation if the cities are the same.

The fly model works because **each neuron is unique** — its identity is defined by its specific input/output pattern. In our model, neurons within a region are **interchangeable**.

---

## 2. What the Frontier Tells Us (April 2026)

### Has anyone demonstrated cognitive advantage from macro-scale connectome topology?
**Almost no one.** The closest:
- **BT-SNN (Zhao et al. 2024, Frontiers in Neuroscience):** Used Allen mouse connectome topology (213 regions) in a spiking RL agent. OUTPERFORMED random topologies and LSTM on MuJoCo tasks. Key difference: each region has different computational properties.
- **conn2res (Suárez et al. 2024, Nature Communications):** Connectome-based reservoir computing has topology-dependent memory capacity. Rate-based, not spiking.
- No one has shown it in a full spiking simulation with identical parameters. **We are in genuinely uncharted territory.**

### The consensus from Deco/TVB literature
Regional parameter gradients are NECESSARY. T1w/T2w myelination ratio → tau_m per region. Without regional heterogeneity, structure alone fails to match empirical data (Burt et al. 2018, Gao et al. 2020, Chaudhuri et al. 2015).

### Latest state of the art
- **BAAIWorm (Dec 2024, Nature Comp Sci):** Multicompartment C. elegans, closed-loop brain-body-environment. Required parameter fitting even WITH complete connectome.
- **Eon Systems fly (Mar 2026):** Full FlyWire 140K neurons embodied in MuJoCo. Foraging, grooming — all hardwired, NO learning. Authors: "should not be interpreted as proof that structure alone is sufficient."
- **Allen Institute (Nov 2025):** Full mouse cortex simulation (10M neurons, 26B synapses) on Fugaku supercomputer. No cognitive tasks — purely dynamics.
- **Arbor-TVB (2025):** Multi-scale co-simulation bridging spiking (NEST) with mean-field (TVB).

---

## 3. First-Principles Analysis: What We Got Wrong

### Mistake 1: Identical parameters everywhere
We treated this as a feature ("emergence from topology alone"). It's actually the reason structure doesn't help. Real brains invest enormous evolutionary effort in making every region DIFFERENT — different cell types, different receptor densities, different time constants, different layer composition. By making everything identical, we guaranteed that topology would be irrelevant, because identical nodes are interchangeable regardless of how you wire them.

**The fix:** Implement the Murray/Chaudhuri timescale gradient. The T1w/T2w myelination ratio varies across cortex and directly predicts regional intrinsic timescale (tau_m). Sensory regions are fast (tau_m ≈ 10ms), prefrontal regions are slow (tau_m ≈ 30ms). This is well-established neuroscience (Gao et al. 2020, eLife). One parameter gradient, derived from MRI data, not hand-tuned.

### Mistake 2: Testing the wrong tasks
We tested conditioning, discrimination, and memory — all LEARNED tasks on a brain that hasn't learned yet. The fly model's success was in INNATE computation (motion detection, hardwired sensorimotor loops). We're testing whether a blank computer architecture performs differently from a random one — of course it doesn't, because the architecture only matters when it's running SOFTWARE (learned representations).

**The fix:** First test INNATE dynamics (resting state patterns, stimulus response propagation, oscillation frequencies). Then test whether the connectome provides a LEARNING advantage — not whether it performs better at tasks it hasn't learned, but whether it LEARNS FASTER.

### Mistake 3: Wrong level of comparison
We compared connectome vs random on cognitive PERFORMANCE. But the fly model showed the connectome provides SPECIFIC innate computations (motion detection circuits). The human connectome might do the same — providing specific computational motifs (visual hierarchy, cortico-thalamic loops, default mode network) that emerge from topology but aren't captured by our generic cognitive tasks.

**The fix:** Test for SPECIFIC known human brain phenomena:
- Sensory-to-frontal propagation delay hierarchy
- Default mode network anti-correlation with task-positive network
- Alpha rhythm (8-12 Hz) at rest
- Stimulus-locked gamma oscillations (30-80 Hz)

### Mistake 4: Treating regions as populations of random neurons
200 random neurons per region is not "a cortical column." It's a random noise generator. Real cortical columns have specific laminar structure (layers 2/3/4/5/6), specific input-output patterns (layer 4 receives thalamic input, layer 5 sends output), and specific local circuit motifs.

**The fix (ambitious but realistic):** Give each region a minimal cortical microcircuit: input layer, processing layer, output layer. Not 6 layers — just 3 functional subdivisions with specific connectivity. Thalamic input → input layer → processing → output → between-region connections. This transforms regions from "random noise generators" to "structured computational units."

---

## 4. The Path Forward (Realistic with RTX 5070)

### Phase A: Regional Heterogeneity — DONE
T1w/T2w gradient implemented. Hierarchy emerges (r=-0.45, p<0.0001) but is identical for connectome and random. The gradient drives hierarchy, not topology.

### Phase B: Innate Dynamics — DONE (hit wall)
Stimulus doesn't propagate beyond visual cortex. Root cause: all dMRI long-range connections are excitatory. Added feedforward inhibition (FC-FC improved to 0.30+) but stimulus cascade still blocked. **This is the fundamental dMRI wall.**

### Phase C: Learning Scaffold — DONE (inconclusive)
Neither connectome nor random learns 3-way stimulus-action task (both at ~36%, chance=33%). E-prop doesn't produce meaningful learning at this scale. Can't compare learning speed when neither learns.

### Phase D: Interactive Demo — DONE
`python demo.py` — 16K neurons, 80 regions, T1w/T2w gradient, delays, CPG walking, stimulus response. The brain is alive but structure doesn't differentiate from random.

### VRAM Budget
All of the above fits in 12GB:
- 80 regions × 200 neurons = 16,000 neurons
- ~700K synapses = ~3MB sparse matrix
- State tensors = ~1MB
- E-prop eligibility = ~3MB
- **Total: ~10MB** — massive headroom

---

## 5. What's Outside the Box

### Idea 1: Multi-scale co-simulation
Don't spike everywhere. Use Wilson-Cowan at the regional level (which produces realistic SC-FC) and spiking neurons only in regions of interest (PFC for working memory, visual cortex for perception). This is what Arbor-TVB does. The macro-scale dynamics provide the CONTEXT, spiking provides the COMPUTATION.

### Idea 2: The connectome as a learning prior
Stop testing whether the connectome produces better innate performance. Test whether it produces a better LEARNING SUBSTRATE. Train two identical agents — one with connectome architecture, one with random — on the same sequence of experiences. The connectome's role isn't to know things; it's to be better at LEARNING things. This is the 先天 × 后天 hypothesis in its purest form.

### Idea 3: Developmental approach
Real brains aren't born with adult connectivity. They start with a rough scaffold and REFINE it through experience (synaptic pruning, myelination). Start with a sparse, noisy version of the connectome and let STDP/e-prop refine it. Compare the final connectivity pattern to the real connectome. If learning converges toward the real pattern, that proves the connectome is an ATTRACTOR of the learning dynamics — the most powerful possible evidence for 先天 × 后天.

---

## 6. Resource Requirements

| Phase | GPU Time | VRAM | Effort |
|-------|----------|------|--------|
| A: Timescale gradient | ~2 hours | <1GB | 1 day |
| B: Innate dynamics | ~4 hours | <1GB | 1 week |
| C: Learning advantage | ~8 hours | <2GB | 2 weeks |
| D: Microcircuits | ~4 hours | <2GB | 2-4 weeks |

All within RTX 5070 (12GB VRAM, 16GB system RAM).

---

## 7. Success Criteria

| Milestone | Criterion | Status |
|-----------|-----------|--------|
| SC-FC validation | FC-FC(emp) > 0.3 | **DONE** (r=0.42) |
| Regional heterogeneity | Timescale hierarchy matches Murray 2014 | **DONE** (r=-0.45) |
| Innate dynamics | Stimulus propagation through connectome | **BLOCKED** (dMRI wall) |
| Learning advantage | Connectome learns faster than random | **INCONCLUSIVE** (neither learns) |
| Working memory | PFC persistence with tau_m gradient | Not tested with gradient |
| Interactive demo | Brain responds to stimuli, drives body | **DONE** (demo.py) |

### The dMRI wall
dMRI tractography provides excitatory-only, undirected macro-scale routing. Without inhibitory long-range connections (needs neurotransmitter identity), stimulus propagation and differentiated computation are blocked. This is a DATA limitation, not a SOFTWARE limitation.

### Paths through the wall
1. **Estimated inhibitory long-range** — from neuroanatomy literature (~30% inhibitory)
2. **Multi-scale** — Wilson-Cowan regionally + spiking locally (Arbor-TVB)
3. **BT-SNN approach** — connectome as RL architecture (Zhao et al. 2024 showed this works)
4. **Developmental** — start noisy, let learning refine toward real connectome

---

*v4 — April 5, 2026. Major revision incorporating 29 experiments, expert panel feedback, first-principles analysis, and latest literature review.*

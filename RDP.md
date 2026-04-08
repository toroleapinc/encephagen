# Encephagen: Research Design Proposal v6

## The Complete Miniature Human Brain

**17,530 spiking neurons across 10 brain structures, biologically wired.**

We built every major organ of the human brain as spiking neural circuits: cortex (16,000 neurons, canonical microcircuit with 7 cell types), thalamus (200, sensory gateway + transthalamic relay), basal ganglia (200, action selection via direct/indirect pathways), cerebellum (500, motor coordination), superior colliculus (100, visual orienting), hippocampus (200, memory), amygdala (100, fear/emotion), hypothalamus (50, survival drives), neuromodulators (100, DA/5HT/NE/ACh state control), and spinal CPG (80, identified interneuron stepping circuit). All are connected through biologically correct pathways.

42 experiments proved that macro-scale dMRI connectome topology does NOT confer computational advantage (0/41 tests). What DOES matter: the "genetic recipe" — specific cell types, layer structure, identified interneuron circuits, and feedforward/feedback asymmetry. This is why the fly project works at synaptic resolution and our cortical connectome doesn't: behavior comes from specific circuits, not general routing.

## 0. Philosophical Foundation

### Everything is signal
What enters our eyes and ears are signals. The brain decomposes and transforms them. Its structure determines HOW it processes.

### Structure constrains but does not determine computation
At synaptic resolution (Drosophila, Lappalainen 2024), structure can replicate specific sensory computations. At macro-scale dMRI resolution (80 regions), structure does NOT provide cognitive advantage over random wiring (0/33 experiments). The computational advantages of human brain organization may reside in local microcircuit properties, cell-type-specific connectivity, and neuromodulatory systems — not in the macro-connectome topology (Maier-Hein et al. 2017: dMRI tractography produces systematically biased false positives in long-range connections).

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

### Experiment 32: Newborn Closed-Loop — DONE (body too stable)
Brain→CPG→Body→Brain loop with inverted pendulum. Both connectome and random survive full 10s with identical metrics. The pendulum self-stabilizes.

### Experiment 33: MuJoCo Walker2d — DONE (brain works, no structural advantage)
Brain controls properly unstable Walker2d biped. **210 steps vs 119 baseline = 78% improvement.** But connectome vs random: 0/5 significant.

### Newborn Demo (biologically correct) — DONE
Subcortical architecture: brainstem reflexes + spinal CPG + BG gating. Cortex observes, doesn't control. **Mean survival: 246 steps = 2.1x baseline.**

### Spiking CPG with identified interneurons — DONE
80 LIF neurons (Shox2, V0d, V0v, V1, V2a, V2b, V3, MNs) from Rybak/Danner/Kiehn. CMA-ES calibrated weights. **Sustained oscillation (fitness 5.5 → 2.4 verified).** Integrated into newborn demo.

### Lateralized brain assessment — DONE
Pure brain = noise (25 steps, worse than zero). Lateralized brain righting = 97% of PD controller (234 vs 240). The cortex provides real corrective routing but needs weight calibration.

Run: `python newborn.py --spiking-cpg --video`

### VRAM Budget
All of the above fits in 12GB:
- 80 regions × 200 neurons = 16,000 neurons
- ~700K synapses = ~3MB sparse matrix
- State tensors = ~1MB
- E-prop eligibility = ~3MB
- **Total: ~10MB** — massive headroom

---

## 5. 先天 Phase: COMPLETE

### What pure innate structure achieves (no learning):
| Achievement | Status |
|-------------|--------|
| SC-FC validated cortex (r=0.42 vs real fMRI) | ✅ |
| Timescale hierarchy (r=-0.45, matches Murray 2014) | ✅ |
| 10/15 neonatal reflexes on Humanoid body | ✅ |
| 80-neuron spiking CPG with identified interneuron classes | ✅ |
| Lateralized brain produces 97% of PD controller from structure alone | ✅ |
| Breathing rhythm, general movements, Moro, startle, grasp, ATNR | ✅ |
| Cortex observes through real human connectome (not controlling) | ✅ |

### What pure innate structure does NOT achieve:
| Finding | Status |
|---------|--------|
| Connectome advantage over random (0/33 experiments) | ❌ |
| Stimulus propagation through cortex | ❌ (dMRI all-excitatory wall) |
| Pure brain body control (brain = noise without calibration) | ❌ |
| Learning from experience | ✅ STARTED — 7 learning rules, 4 months simulated |

### Key scientific findings (from expert panel review):

**Finding 1: "Cortex as observer" (Exp 34)** — The most scientifically important result. A 16,000-neuron spiking cortex connected to motor output produces noise, not control. Behavioral signal lives in brainstem and subcortical circuits. This is consistent with the motor control literature (Grillner, Llinas, Pruszynski) and developmental neuroscience. This finding needs parameter sensitivity testing to confirm it's not an artifact of the specific LIF parameter regime.

**Finding 2: 0/33 null result** — A carefully designed 33-experiment test of whether human connectome topology confers computational advantage returned a comprehensive null. This constrains theories that attribute computational power to macro-connectome topology and suggests advantages reside in local microcircuit properties, cell-type connectivity, or neuromodulatory systems.

**Finding 3: CMA-ES CPG** — The spiking CPG required evolutionary search to oscillate. This raises the question: does the connectome-constrained architecture have a LARGER basin of oscillation in parameter space than a random-wired CPG? A larger basin = genuine innate architectural advantage. This experiment has not been done and is now the scientific heart of the 先天 question.

### The 先天 ceiling:
The newborn simulation matches real newborn capability: reflexes, breathing, fidgeting, stepping. Everything beyond requires 后天 (learning).

### 后天 Phase: STARTED

7 distributed learning rules active simultaneously:
- Cerebellum: supervised LTD (error → weaken Purkinje synapses)
- Basal ganglia: dopamine RL (reward prediction error → D1/D2)
- Hippocampus: Hebbian LTP (co-active CA3 → strengthen)
- Amygdala: Pavlovian (threat → strengthen LA→CeA)
- Brainstem: habituation (repeated stimulus → depression)
- CPG: sensory adaptation (tilt → adjust drive)
- Cortex: homeostatic (very slow rate adaptation)

**4 months of development results:**
- Motor control: flat (251 → 251 steps)
- Amygdala: **+51% weight change** (learned fear of falling)
- Thalamus: **-12%** (habituated to constant input)
- Hippocampus: **+2.4%** (memory traces formed)
- Dopamine dropped, arousal spiked, serotonin depleted

The brain learns internally but motor improvement requires cortical takeover — the cortex (0% change) hasn't started controlling the body yet.

**Innate baseline captured:** `snapshots/innate_baseline.json` — every synapse tracked.

### Outstanding experiments from expert review:
1. **Peak SC-FC real vs null across G sweep** — Does the real connectome achieve higher MAXIMUM SC-FC than random at any G?
2. **Tractography thresholding sensitivity** — Does r=0.42 survive thresholding at different streamline densities? (Maier-Hein et al. 2017)
3. **CPG parameter basin analysis** — Is the oscillation basin larger for connectome-constrained vs random CPG? (THE key 先天 test)
4. **Spike-based FC validation** — Compare spiking output statistics to Allen Brain Institute Neuropixels recordings (firing rates 2-20Hz, ISI CV>0.5, pairwise correlations 0.01-0.1)
5. **"Cortex as observer" parameter sensitivity** — Does changing E/I balance or synaptic timescales in the cortex change the conclusion?

---

## 6. 后天 Phase: NOT STARTED (next chapter)

The cortex (16,000 neurons) is ready to learn.

### Experiment design (pre-registered, from expert panel):
**Task:** Task FAMILY, not single task. Train on environments A1, A2, A3. Test on A4 (novel, same family).
**Comparison:** Connectome-structured vs random-wired, identical architecture otherwise.
**Seeds:** Minimum 20 random seeds per condition.
**Metric:** Learning curve slope AND final generalization performance on held-out environment.
**Statistical test:** Wilcoxon rank-sum on generalization performance across seeds (not t-test — weight distributions are non-normal after learning).
**Outcome:** If connectome does not show significant advantage in generalization after proper multi-seed comparison → definitive null for 先天 learning advantage. This null result, with 37-experiment trajectory, is a legitimate scientific contribution.

### Three approaches:

### Approach 1: Cortical takeover
Over simulated developmental time, cortex learns to modulate brainstem reflexes via e-prop. Primitive reflexes weaken as cortical control strengthens. Mirrors corticospinal myelination at 2-4 months.

### Approach 2: Connectome as learning prior (generalization test)
Train two newborns — one with real connectome, one random — through identical experiences. Test on NOVEL environments. If connectome generalizes better → structure provides learning scaffold.

### Approach 3: Developmental refinement
Start with noisy connectome, let STDP/e-prop refine it through experience. Compare learned connectivity to real connectome. If learning converges toward real pattern → connectome is an attractor of learning dynamics.

---

## 7. Full Success Criteria

| Milestone | Status |
|-----------|--------|
| SC-FC validation (r > 0.3) | **DONE** (r=0.42) |
| Timescale hierarchy | **DONE** (r=-0.45) |
| Spiking CPG with identified interneurons | **DONE** (80 neurons, CMA-ES) |
| 10+ innate behaviors | **DONE** (10/15) |
| Full newborn on Humanoid body | **DONE** (1.2x baseline) |
| Lateralized brain corrective output | **DONE** (97% of PD controller) |
| Structural advantage over random | **NOT FOUND** (0/33) |
| Complete brain (all 10 organs) | **DONE** (17,530 neurons, all wired) |
| Integrated body demo | **DONE** (2.8x baseline with all structures) |
| Innate baseline captured | **DONE** (873K synapses, all states saved) |
| Distributed learning (7 rules) | **DONE** (amygdala +51%, thalamus -12%) |
| Motor improvement from learning | NOT YET (cortex hasn't taken over) |
| Cortical developmental takeover | NOT YET (needs corticospinal myelination sim) |

### The dMRI wall
dMRI tractography provides excitatory-only, undirected macro-scale routing. Without inhibitory long-range connections (needs neurotransmitter identity), stimulus propagation and differentiated computation are blocked. This is a DATA limitation, not a SOFTWARE limitation.

### Paths through the wall
1. **MuJoCo Walker2d + BT-SNN approach** — Wire heterogeneous brain to properly unstable body. Test if connectome architecture learns motor control faster than random. Has positive evidence from mouse (Zhao et al. 2024). Most likely to succeed.
2. **Estimated inhibitory long-range** — from neuroanatomy literature (~30% inhibitory). Fixes the all-excitatory problem but needs subcortical parcellation.
3. **Multi-scale** — Wilson-Cowan regionally + spiking locally (Arbor-TVB). The regional model WORKS for propagation.
4. **Developmental** — start noisy, let learning refine toward real connectome. Most elegant 先天 × 后天 test.

---

*v4 — April 5, 2026. Major revision incorporating 29 experiments, expert panel feedback, first-principles analysis, and latest literature review.*

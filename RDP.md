# Research Design Proposal: Encephagen

## 1. Title & Abstract

**Formal Title:** Encephagen: Emergent Brain-Like Dynamics from Human Macro-Connectome Topology Simulated as a Living Dynamical System

### Abstract

The human brain's remarkable capabilities emerge not from any single neuron but from the specific topology of connections between regions. While connectome-scale simulations exist for small organisms (C. elegans, Drosophila), no project has taken the human brain's macro-connectome --- the wiring diagram between cortical and subcortical regions --- instantiated it as a biophysical dynamical system, and simply observed what emerges. Encephagen does exactly this. Using structural connectivity data from the Human Connectome Project (HCP), we construct a network of 76 brain regions (expandable to 360 via the Glasser parcellation), each governed by Wilson-Cowan excitatory-inhibitory dynamics with region-type-specific parameters (cortical oscillators, thalamic burst/tonic gates, hippocampal theta generators). The simulation runs in continuous time with no backpropagation, no loss function, and no training. We hypothesize that the topology alone --- the specific pattern of long-range and short-range connections unique to the human brain --- will produce spontaneous oscillations, stimulus-selective responses, and functional differentiation that qualitatively resemble real brain dynamics. Subsequent phases introduce Hebbian and spike-timing-dependent plasticity (STDP) to test whether the system can learn from experience, and ultimately embodiment via physics simulation to test sensorimotor learning. This project aims to demonstrate that meaningful brain-like behavior can emerge from structure alone, without machine learning.

---

## 2. Research Questions

### Primary Question

Can a miniature simulation using real human brain macro-connectome topology produce emergent brain-like dynamics and behavior without any training?

Specifically:
- Do spontaneous oscillations in biologically plausible frequency bands (delta, theta, alpha, beta, gamma) emerge from the network topology and Wilson-Cowan dynamics alone?
- Does sensory stimulation applied to appropriate input regions produce activity patterns that preferentially activate the correct downstream regions (e.g., visual input activating occipital cortex)?
- Does the real human connectome topology produce qualitatively different dynamics than a random network with matched degree distribution?

### Secondary Questions

1. **Structural variation:** Do connectome variations across age (neonatal vs. adult) and sex produce measurably different emergent behaviors? Can we observe signatures of developmental maturation in simulation?
2. **Learning from structure:** Can biologically plausible learning rules (Hebbian, STDP, reward-modulated plasticity) layered onto the connectome topology enable the system to learn from repeated experience --- distinguishing patterns, forming associations, and adapting behavior?
3. **Embodiment:** Can a learned connectome-based controller drive meaningful sensorimotor behavior in a virtual body, bootstrapping from structure rather than reinforcement learning from scratch?

---

## 3. Background & Motivation

### 3.1 The Connectome Revolution

The past decade has seen transformative advances in mapping neural connectivity at multiple scales:

- **C. elegans** (302 neurons): The complete wiring diagram has been known since the 1980s (White et al., 1986). The OpenWorm project simulated the full connectome as a dynamical system and demonstrated emergent locomotion behavior, proving that structure can drive function without training.
- **Drosophila melanogaster** (~140,000 neurons): Lappalainen et al. (2024, Nature) published the first complete whole-brain connectome of an adult fruit fly, enabling unprecedented analysis of circuit motifs and information flow architecture.
- **Human macro-connectome** (76--360 regions): The Human Connectome Project (HCP; Van Essen et al., 2013) provides diffusion MRI-derived structural connectivity matrices for hundreds of subjects, parcellated using the Glasser et al. (2016) multi-modal atlas into 360 cortical regions. The developing Human Connectome Project (dHCP; Bastiani et al., 2019) extends this to neonatal brains.

### 3.2 Large-Scale Brain Simulation

- **The Blue Brain Project / EBRAINS** (Markram et al., 2015): Simulated a cortical column of ~31,000 neurons with biophysically detailed compartmental models. Demonstrated that detailed biophysics can produce emergent oscillatory dynamics, but required supercomputer-scale resources and focused on a single cortical column rather than whole-brain topology.
- **The Virtual Brain (TVB)** (Sanz-Leon et al., 2013): Simulates whole-brain dynamics using neural mass models coupled by connectome data. Closest to our approach, but designed for fitting clinical neuroimaging data (EEG/fMRI), not for observing emergent behavior from a "blank" brain or testing learning.

### 3.3 Theoretical Foundations

- **Buzsaki's "Inside Out"** (2019): Argues that the brain is not a passive stimulus-response machine but an active, self-organizing system whose internal dynamics are primary. Brain rhythms and sequences exist before sensory experience. This directly supports our hypothesis that connectome topology alone should produce rich spontaneous dynamics.
- **Friston's Free Energy Principle** (2010): Proposes that biological systems minimize surprise (free energy) through prediction and action. While we do not implement the full Free Energy framework, our learning phases (Hebbian/STDP) implement a simplified version: the network's synaptic weights will self-organize to predict and respond to regularities in its inputs.
- **Sporns' "Networks of the Brain"** (2010): Established that brain network topology --- small-world architecture, rich-club organization, modular structure --- is not arbitrary but optimized for efficient information integration. This motivates our core hypothesis that topology matters.

### 3.4 The Gap

Despite these advances, **no project has**:

1. Taken the human macro-connectome as a starting point
2. Instantiated each region with biophysically motivated dynamics (not just linear coupling)
3. Treated the system as a "newborn brain" with no prior training
4. Systematically observed what emergent dynamics and behaviors arise from structure alone
5. Then layered biologically plausible learning to test whether the system can learn like a developing brain

### 3.5 Why Human (Not Fly or Worm)?

While C. elegans and Drosophila connectomes offer completeness at the neuron level, the human brain has unique structural features that may produce qualitatively different dynamics:

- **Enlarged prefrontal cortex**: Massive expansion of PFC relative to other primates, with dense long-range connections to virtually all other cortical and subcortical regions
- **Long-range cortico-cortical connections**: The human brain has uniquely developed association fiber bundles (arcuate fasciculus, superior longitudinal fasciculus) connecting distant cortical regions
- **Cortical layer proportions**: Human cortex has expanded supragranular layers (II/III), associated with cortico-cortical communication and abstract processing
- **Rich-club topology**: Human brain networks show pronounced rich-club organization --- a densely interconnected core of hub regions that may serve as an integration backbone

We hypothesize that these features will produce richer spontaneous dynamics, more complex stimulus responses, and greater learning capacity than would be seen with simpler organism topologies.

---

## 4. Methodology

### 4.1 Connectome Data Sources

| Source | Resolution | Subjects | Use Case |
|--------|-----------|----------|----------|
| HCP (Human Connectome Project) | 76 regions (Desikan-Killiany) | 1,000+ adults | Initial prototype, fast iteration |
| HCP + Glasser Atlas | 360 cortical regions + subcortical | 1,000+ adults | Full-resolution simulation |
| dHCP (developing HCP) | 76--90 regions | Neonatal (37--44 weeks) | Neonatal brain simulation |
| Demographic subsets | Matched resolution | Age/sex stratified | Variation experiments |

**Connectivity matrix format:** Weighted, undirected structural connectivity matrices derived from diffusion MRI tractography. Edge weights represent streamline counts normalized by region volume (connection density). Matrices will be thresholded to remove spurious connections (retaining top 20--30% of connections by weight) and log-transformed to compress the heavy-tailed weight distribution.

### 4.2 Region Dynamics: Wilson-Cowan Model

Each brain region is modeled as a Wilson-Cowan excitatory-inhibitory (E-I) coupled oscillator:

```
tau_E * dE/dt = -E + S(w_EE*E - w_EI*I + P + sum_j(C_ij * E_j) + I_ext)
tau_I * dI/dt = -I + S(w_IE*E - w_II*I + Q)
```

Where:
- `E`, `I`: Excitatory and inhibitory population activities for a given region
- `w_EE, w_EI, w_IE, w_II`: Local coupling weights (region-type specific)
- `P, Q`: Baseline drives
- `C_ij`: Structural connectivity weight from region j to region i (from connectome)
- `I_ext`: External stimulus input
- `S(x)`: Sigmoidal activation function: `S(x) = 1 / (1 + exp(-a*(x - theta))) - 1 / (1 + exp(a*theta))`
- `tau_E, tau_I`: Time constants (excitatory ~10ms, inhibitory ~20ms)

### 4.3 Specialized Region Types

Not all brain regions behave identically. We define three region archetypes with distinct parameter sets:

**Cortical regions** (default):
- Standard Wilson-Cowan E-I oscillator
- Parameters tuned to produce alpha-band (~10 Hz) oscillations at rest
- Includes all Glasser parcellation cortical regions

**Thalamic regions** (thalamus, LGN, MGN, pulvinar):
- Modified dynamics with tonic and burst firing modes
- Low-threshold calcium current approximation enabling state-dependent gating
- Acts as a relay/gate for sensory input and cortico-cortical communication
- Parameters: lower excitatory time constant, higher burst threshold

**Hippocampal regions** (hippocampus, entorhinal cortex, subiculum):
- Theta-frequency oscillator (~4--8 Hz)
- Activity-dependent trace variable for memory formation (used in learning phases)
- Slower inhibitory time constant to support theta rhythm
- Parameters: tau_I ~50ms, stronger recurrent excitation

### 4.4 Simulation Architecture

- **Integration method:** 4th-order Runge-Kutta (RK4) with adaptive timestep (dt = 0.1--1.0 ms)
- **Backend:** NumPy for prototyping; CuPy/JAX for GPU acceleration on RTX 5070
- **No backpropagation:** All dynamics are forward-simulated. No loss functions, no gradient computation.
- **State representation:** Each region stores (E, I, trace) = 3 floats. For 360 regions: 1,080 state variables total --- trivially fits in GPU memory.
- **Connectivity:** Sparse matrix multiplication for inter-region coupling. For 360 regions with ~30% density: ~39,000 non-zero connections.
- **Output recording:** E and I activity for all regions sampled at 1 kHz (1,000 Hz), stored as time-series for offline analysis.

### 4.5 Stimulus Protocol

Stimuli are delivered as external current `I_ext` to designated input regions:

| Modality | Target Regions | Stimulus Types |
|----------|---------------|----------------|
| Visual | V1, V2, V4 (occipital) | Flashing (step function), edges (graded activation across retinotopic regions), patterns |
| Auditory | A1, A2 (temporal) | Pure tones (sinusoidal), tone sequences, frequency sweeps |
| Somatosensory | S1, S2 (parietal) | Touch pulses, sustained pressure, moving stimuli |

Stimulus intensities are parameterized as multiples of the spontaneous activity baseline (1x, 2x, 5x threshold).

### 4.6 Learning Rules (Phase 7)

**Hebbian learning:**
```
dW_ij/dt = eta * E_i * E_j - lambda * W_ij
```
Connection weights between co-active regions strengthen; a decay term prevents runaway excitation.

**Spike-Timing-Dependent Plasticity (STDP):**
```
dW_ij = A+ * exp(-dt/tau+)  if pre before post (LTP)
dW_ij = A- * exp(dt/tau-)   if post before pre (LTD)
```
Adapted for rate-coded populations using activity traces as spike-time proxies.

**Reward modulation:**
```
dW_ij/dt = eta * E_i * E_j * R(t)
```
Where R(t) is a global reward signal, enabling reinforcement-like learning without backpropagation.

### 4.7 Embodiment (Phase 8, Future)

Integration with MuJoCo or NVIDIA Isaac Gym to provide a virtual body:
- Motor output regions (M1, premotor, SMA) drive joint torques
- Sensory regions receive proprioceptive and exteroceptive feedback
- Closed-loop sensorimotor interaction enables grounded learning

---

## 5. Experiments

### Experiment 1: Spontaneous Dynamics (No Input)

**Question:** Does the connectome topology, coupled with Wilson-Cowan dynamics, produce spontaneous oscillatory activity in biologically plausible frequency bands?

**Protocol:**
1. Initialize all regions at low random activity (E ~ 0.1, I ~ 0.05)
2. Run simulation for 60 seconds (simulated time) with no external input
3. Record all region activities at 1 kHz

**Analysis:**
- Power spectral density (PSD) of each region's E activity
- Cross-correlation and coherence between region pairs
- Phase-amplitude coupling analysis
- Comparison with resting-state fMRI/EEG power spectra from literature

**Success criteria:** Emergence of peaks in the alpha (8--12 Hz) and/or theta (4--8 Hz) bands that are not present in the single-region dynamics alone (i.e., topology-dependent).

**Falsification:** If the network produces only flat-spectrum noise or uniform fixed-point activity regardless of parameter tuning, the macro-connectome resolution may be insufficient.

### Experiment 2: Stimulus Response

**Question:** Does sensory stimulation of appropriate input regions produce activity that preferentially propagates to the correct downstream regions?

**Protocol:**
1. Apply visual stimulus (step function, 500ms duration) to V1/V2 regions
2. Record propagation of activity across all regions
3. Repeat with auditory stimulus to A1/A2
4. Repeat with somatosensory stimulus to S1/S2

**Analysis:**
- Latency maps: time-to-peak activation for each region after stimulus onset
- Activation magnitude maps: which regions show strongest response?
- Modality selectivity: does visual input preferentially activate the ventral/dorsal visual streams? Does auditory input preferentially activate temporal regions?

**Success criteria:** Stimulus-evoked activity patterns that are modality-specific and consistent with known functional anatomy (e.g., visual input activates occipital > temporal > frontal, not random spread).

**Falsification:** If all three stimulus modalities produce identical activation patterns, the topology does not meaningfully constrain information flow.

### Experiment 3: Real vs. Random Wiring

**Question:** Does the specific topology of the human connectome matter, or would any network with similar statistical properties produce the same dynamics?

**Protocol:**
1. Generate null-model networks: (a) Erdos-Renyi random graph with matched density, (b) degree-preserving randomization, (c) weight-preserving randomization
2. Run Experiments 1 and 2 on each null model
3. Compare dynamics between real connectome and null models

**Analysis:**
- Compare oscillatory power spectra
- Compare stimulus response selectivity
- Compare network-level measures (synchrony, metastability, functional connectivity structure)
- Statistical comparison using permutation testing (N=100 null networks)

**Success criteria:** Real connectome produces significantly different (and more brain-like) dynamics than all null models.

**Falsification:** If null models produce equivalent dynamics, macro-connectome topology does not carry meaningful information at this scale.

### Experiment 4: Structural Variations

**Question:** Do connectome differences across developmental stage and sex produce different emergent dynamics?

**Protocol:**
1. **Neonatal vs. adult:** Compare dHCP neonatal connectome vs. HCP adult connectome
2. **Male vs. female:** Compare sex-stratified HCP connectomes (matched for age)
3. Run Experiments 1--3 on each variant

**Analysis:**
- Compare spontaneous oscillatory frequencies (hypothesis: neonatal shows slower dominant frequencies)
- Compare stimulus response latencies (hypothesis: neonatal shows slower, more diffuse responses due to less-developed myelination reflected in weaker long-range connections)
- Compare functional connectivity patterns

**Success criteria:** Observable, statistically significant differences in emergent dynamics that parallel known developmental and sex differences in human neuroimaging literature.

### Experiment 5: Hebbian Learning

**Question:** Can Hebbian plasticity layered onto the connectome enable the system to learn from repeated stimulus exposure?

**Protocol:**
1. Enable Hebbian learning rule on inter-region connections
2. Present stimulus A repeatedly (100 trials), then stimulus B (100 trials)
3. Test: does the network respond differently to A vs. B after training?
4. Test: does the network show faster/stronger response to A (familiar) vs. C (novel)?

**Analysis:**
- Response amplitude and latency before vs. after training
- Weight change maps: which connections strengthened/weakened?
- Habituation and novelty detection metrics

**Success criteria:** Network shows measurable learning effects: faster response to familiar stimuli, distinct representations for different stimuli, without any gradient-based training.

### Experiment 6: Simple Number Learning and MNIST

**Question:** Can the connectome-based system learn a concrete cognitive task --- recognizing handwritten digits?

**Protocol:**
1. **Phase A:** Encode digits 0--9 as spatial patterns across visual input regions (simplified: 7-segment display mapped to V1 subregions)
2. Train with Hebbian/STDP learning: present each digit 1,000 times with label signal to association cortex
3. Test: present digit without label, read out from association regions
4. **Phase B:** Encode MNIST images as graded activation patterns across V1/V2 (28x28 pixels mapped to ~50 visual regions via spatial averaging)
5. Train and test classification accuracy

**Analysis:**
- Classification accuracy (chance = 10%)
- Confusion matrix: which digits are confused?
- Comparison with: (a) random network baseline, (b) simple feedforward network with same number of parameters

**Success criteria:** Above-chance classification (>30% would be remarkable; >50% would be extraordinary). This is NOT expected to compete with deep learning --- the goal is to demonstrate that structure + simple learning rules can produce non-trivial cognitive behavior.

### Experiment 7: Embodiment

**Question:** Can the connectome-based controller learn to control a virtual body through sensorimotor interaction?

**Protocol:**
1. Connect motor output regions to a simple virtual agent (MuJoCo ant or humanoid)
2. Feed proprioceptive/sensory signals back to somatosensory input regions
3. Enable reward-modulated learning: reward for forward movement
4. Run for 10,000+ simulation steps

**Analysis:**
- Distance traveled over time (learning curve)
- Motor pattern analysis: does coordinated locomotion emerge?
- Comparison with random controller baseline

**Success criteria:** Any measurable improvement over random motor output after learning. Coordinated locomotion would be a major result.

---

## 6. Expected Outcomes

### What We Expect to See

1. **Spontaneous oscillations** (Exp 1): Wilson-Cowan dynamics coupled by realistic topology should produce oscillatory activity. We expect alpha-band oscillations in cortical regions and theta-band in hippocampal regions, with the specific frequency and spatial distribution shaped by the connectome topology.

2. **Topology-dependent stimulus routing** (Exp 2): The connectome's hierarchical organization (primary sensory -> association -> prefrontal) should naturally route stimulus-evoked activity along known functional pathways. Activity should not spread uniformly.

3. **Real > Random** (Exp 3): The real connectome should produce more structured dynamics (higher metastability, richer spectral content, more modality-selective responses) than degree-matched random networks. This is our strongest prediction, supported by graph-theoretic analyses showing that brain topology is non-random.

4. **Developmental signatures** (Exp 4): Neonatal connectomes should show slower, more diffuse dynamics reflecting less-developed long-range connectivity. This parallels known EEG developmental trajectories.

5. **Hebbian learning works** (Exp 5): Simple learning should produce measurable changes --- but we expect it to be noisy and require careful parameter tuning. The topology should provide a useful inductive bias (prior structure) that makes learning easier than in a random network.

6. **Modest MNIST performance** (Exp 6): We expect above-chance but not impressive accuracy. The real value is demonstrating that a biologically structured system with no backpropagation can learn a standardized task at all.

7. **Rudimentary motor control** (Exp 7): Highly speculative. We expect noisy, uncoordinated movement that slowly improves. Any directed locomotion would be a major finding.

### What Would Falsify the Hypothesis

- **No oscillations from topology** (Exp 1): If the network produces only noise or fixed-point activity across all reasonable parameter regimes, the macro-connectome resolution is insufficient for meaningful dynamics. Mitigation: try finer parcellation (360 regions), add conduction delays.
- **No modality selectivity** (Exp 2): If visual and auditory stimuli produce identical activation patterns, the topology does not constrain information flow at this scale. This would be a significant negative result.
- **Real = Random** (Exp 3): If null models produce equivalent dynamics, the specific topology carries no dynamically relevant information at the macro scale. This would challenge a core assumption of computational connectomics.

---

## 7. Timeline

### Month 1: Foundation
- **Week 1--2:** Data pipeline --- download and preprocess HCP connectome data, build connectivity matrix loader, visualize network topology
- **Week 3--4:** Core simulation engine --- implement Wilson-Cowan dynamics, RK4 integrator, GPU acceleration with CuPy/JAX

### Month 2: Spontaneous Dynamics
- **Week 5--6:** Run Experiment 1 (spontaneous dynamics), parameter sweep for Wilson-Cowan parameters, identify oscillatory regimes
- **Week 7--8:** Run Experiment 3 (real vs. random), implement null network generators, statistical comparison framework

### Month 3: Stimulus Response
- **Week 9--10:** Implement stimulus delivery system, run Experiment 2 (stimulus response), analyze modality selectivity
- **Week 11--12:** Run Experiment 4 (structural variations), download and preprocess dHCP data, comparative analysis

### Month 4: Visualization & Analysis
- **Week 13--14:** Build real-time visualization dashboard (brain activity heatmap, oscillation traces, connectivity graph)
- **Week 15--16:** Write up results from Experiments 1--4, prepare first preprint/blog post

### Month 5: Learning
- **Week 17--18:** Implement Hebbian and STDP learning rules, run Experiment 5 (Hebbian learning)
- **Week 19--20:** Run Experiment 6 (MNIST), analyze and compare performance

### Month 6: Integration & Publication
- **Week 21--22:** Run Experiment 7 (embodiment, if feasible) or expand learning experiments
- **Week 23--24:** Final analysis, write full paper, prepare for submission, release open-source code and interactive demo

---

## 8. Hardware Requirements

### Primary Development Machine
- **GPU:** NVIDIA RTX 5070 (12GB VRAM)
  - Sufficient for: 360-region simulation with full connectivity (state vector ~4KB, connectivity matrix ~500KB) --- trivially fits in VRAM
  - CuPy/JAX kernel launch overhead is the bottleneck, not memory
  - Estimated performance: 1 second of simulated brain time in ~1--5 seconds wall-clock time (at dt=0.1ms)
- **CPU:** Modern multi-core (for data preprocessing, analysis, visualization)
- **RAM:** 16GB system memory
  - Sufficient for: HCP data loading (~2GB per subject), analysis pipelines, visualization
  - Multiple subjects loaded simultaneously may require memory management

### Scaling Considerations
- **76-region simulation:** Runs comfortably on CPU alone. GPU provides ~10x speedup.
- **360-region simulation:** GPU recommended. ~20x more connections than 76-region.
- **Batch runs (null models, parameter sweeps):** GPU essential. 100 null models x 60s simulation each = ~100 GPU-hours at worst case.
- **Embodiment (Phase 8):** MuJoCo runs on CPU; Isaac Gym requires GPU. May need to time-share GPU between brain simulation and physics.

### Software Stack
- Python 3.11+
- NumPy, SciPy (core numerics)
- CuPy or JAX (GPU acceleration)
- Matplotlib, Plotly (visualization)
- NetworkX, graph-tool (network analysis)
- MuJoCo / Isaac Gym (embodiment, Phase 8)
- Jupyter (interactive exploration)

---

## 9. Risks & Mitigations

### Risk 1: Macro-Connectome Resolution Insufficient
**Severity:** High | **Probability:** Medium

**Risk:** The 76--360 region parcellation may be too coarse to produce meaningful dynamics. Real brain function depends on micro-circuit motifs within regions, not just macro-connectivity.

**Mitigation:**
- Start with 76 regions for fast iteration; scale to 360 if promising
- Add within-region structure (e.g., laminar sub-populations per region) if macro-level results are flat
- Use conduction delays proportional to fiber tract length (available from HCP) to add temporal structure
- Compare with TVB (The Virtual Brain) literature which shows that macro-scale models CAN produce meaningful dynamics

### Risk 2: Wilson-Cowan Dynamics Produce Only Noise
**Severity:** High | **Probability:** Medium

**Risk:** Without careful parameter tuning, coupled Wilson-Cowan oscillators can easily fall into fixed-point attractors (boring) or chaotic noise (uninterpretable).

**Mitigation:**
- Extensive parameter sweep in Month 2 to map the bifurcation landscape
- Use known Wilson-Cowan parameter regimes from literature (Deco et al., 2009; Breakspear et al., 2010)
- Operate near the critical point (edge of chaos) where complex dynamics are richest
- If Wilson-Cowan fails, fall back to simpler Kuramoto oscillators (phase-only) or more complex Jansen-Rit neural mass models

### Risk 3: Hebbian Learning Instability
**Severity:** Medium | **Probability:** High

**Risk:** Hebbian learning is notoriously unstable --- positive feedback loops cause runaway excitation or winner-take-all dynamics that kill diversity.

**Mitigation:**
- Weight decay term to prevent unbounded growth
- Synaptic scaling (homeostatic normalization of total input weights)
- BCM (Bienenstock-Cooper-Munro) rule as alternative: includes sliding threshold that stabilizes learning
- Careful monitoring of weight distributions during learning

### Risk 4: MNIST Performance Too Low to Be Meaningful
**Severity:** Low | **Probability:** High

**Risk:** The system may achieve only marginally above-chance classification, making results hard to interpret or publish.

**Mitigation:**
- Frame the experiment correctly: the goal is not SOTA accuracy but demonstration of learning in a biologically structured system
- Compare against meaningful baselines (random network with same learning rules, not deep learning)
- If accuracy is very low, analyze what the system DOES learn (e.g., maybe it learns to distinguish "round" vs. "angular" digits even if it cannot classify all 10)

### Risk 5: Computational Bottleneck at Scale
**Severity:** Medium | **Probability:** Low

**Risk:** GPU kernel launch overhead or memory bandwidth may limit simulation speed for 360-region models with high temporal resolution.

**Mitigation:**
- Profile early and optimize hot loops
- Use JAX's JIT compilation to fuse operations
- Reduce temporal resolution (dt=0.5ms instead of 0.1ms) if dynamics are smooth enough
- The state space is tiny (~1,080 floats for 360 regions) --- the bottleneck will be connectivity matrix operations, which are well-optimized in CuPy/JAX

### Risk 6: Scope Creep
**Severity:** Medium | **Probability:** High

**Risk:** The project's ambition (from oscillations to embodiment) invites scope creep. Trying to do everything in 6 months risks doing nothing well.

**Mitigation:**
- Hard phase gates: Experiments 1--4 (structure-only) must be complete and publishable before starting learning experiments
- Embodiment (Phase 8) is explicitly marked as future/aspirational --- not a 6-month deliverable
- First publication target covers only Experiments 1--4

---

## 10. Publication Target

### Primary Target: Computational Neuroscience Journals

| Journal | Impact Factor | Fit | Timeline |
|---------|--------------|-----|----------|
| **PLoS Computational Biology** | ~4.5 | Excellent: computational neuroscience, open-access, reproducibility emphasis | Month 6 submission |
| **NeuroImage** | ~5.7 | Good: connectome-based modeling, neuroimaging-relevant predictions | Month 6--8 |
| **Nature Computational Science** | ~12 | Stretch: if results are striking (topology-dependent dynamics clearly demonstrated) | Month 6--8 |
| **eLife** | ~7.7 | Good: open science, computational biology, welcomes novel approaches | Month 6--8 |

### High-Impact Stretch Targets (If Results Are Extraordinary)

| Journal | Rationale |
|---------|-----------|
| **Nature Neuroscience** | If we demonstrate that human connectome topology alone predicts known functional organization |
| **PNAS** | Broad interest: emergence, complexity, brain simulation |
| **Science** | If embodiment produces qualitatively brain-like sensorimotor behavior |

### Conference Presentations

- **OHBM (Organization for Human Brain Mapping):** Connectome analysis and simulation results
- **CNS (Computational Neuroscience):** Wilson-Cowan dynamics and emergent oscillations
- **NeurIPS (Workshop on Neuro-AI):** Bridge between neuroscience-inspired models and AI

### Preprint Strategy

- First preprint (bioRxiv/arXiv) at Month 4: Experiments 1--4 results
- Second preprint at Month 6: Learning experiments
- All code, data pipelines, and trained models released as open-source Python package (`encephagen` on PyPI)

### Media & Viral Potential

This project has unusually high media potential:

- **"Digital newborn brain" narrative:** The idea of simulating a baby's brain and watching it "wake up" is inherently compelling to general audiences
- **Visual output:** Real-time brain activity visualizations (heatmaps on 3D brain surface) produce striking imagery for social media, blog posts, and talks
- **Interactive demo:** A web-based demo where users can "stimulate" the virtual brain and watch activity propagate would generate significant engagement
- **Philosophical resonance:** The project touches on consciousness, emergence, nature vs. nurture, and what makes the human brain unique --- topics with broad public interest
- **Open science angle:** All data, code, and results open-source, inviting community contribution and extension

Target platforms for media outreach:
- Twitter/X science community
- Hacker News / Reddit r/neuroscience, r/compsci
- YouTube: animated visualization of brain dynamics
- Blog post series documenting the journey (building in public)

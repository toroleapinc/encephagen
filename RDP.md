# Encephagen: Research Design Proposal v3

## 0. Philosophical Foundation

This project is built on a set of core beliefs about how intelligence works:

### Everything is signal
What enters our eyes and ears are signals. The brain decomposes these signals and transforms them into understanding — like a Fourier transform, but learned, adaptive, and multi-scale. The brain is fundamentally a signal processor, and its structure determines HOW it processes.

### Structure constrains computation (not "IS intelligence")
The Drosophila connectome experiment (Lappalainen et al., 2024) showed that synaptic-resolution structure (54M individually resolved synapses) can replicate specific sensory computations (motion detection) without training. However, this was at cellular resolution — 6 orders of magnitude finer than our dMRI tractography. At our macro-scale (96 regions), structure creates ORGANIZATION (regional differentiation, p=0.0002) and CHANNELING (concentrated signal flow through specific pathways), but does not provide universal cognitive advantages. The original "structure IS intelligence" framing was an overstatement of what our evidence supports.

### 先天 × 后天 — Innate × Learned
Walking = innate CPG hardware × learned calibration. The genome gives you the POSSIBILITY of walking (spinal cord CPGs, stepping reflex). Experience turns that possibility into REALITY (balance calibration, muscle coordination). Like a phone that ships with an OS but still needs activation and updates. This framework applies to all cognitive abilities: the structure provides the scaffold, learning fills it with content.

### The brain starts blank
When a human is created, the brain is at its initial state — zero knowledge, zero experience. But even at init, the structure produces certain innate behaviors (reflexes, general movements, orienting). As the brain learns and evolves over time through experience, it picks up new skills. We should be able to monitor how the brain and body change over time by teaching it different skills.

### Build first, understand later
Deep neural networks worked before we understood why. The fruit fly connectome simulation produced behavior before anyone explained the mechanism. Making something work is more important than understanding why it works. The engineering mindset: get it running, then analyze.

### The body is the test harness
The focus is on the BRAIN itself — building a functional miniature human brain that can process information, learn, and adapt. The body is how we TEST whether the brain works. It's not an RL agent optimizing a reward function; it's a brain that happens to have a body for testing purposes.

### Key findings so far
Experiments 21-24 tested the core thesis across 4 levels of biophysical realism:
- **Exp 21 (Hebbian, no delays):** Structure creates organization (p=0.0002, survives FDR) but no cognitive advantage.
- **Exp 22 (E-prop, no delays):** Structure helps conditioning (p=0.011, survives BH-FDR). The channeling vs distributing trade-off.
- **Exp 24 (E-prop + delays + ALIF + neuron types):** ALIF adaptation reverses the advantage — concentrated thalamocortical volleys drive adaptation too high in the structured brain.

The overall picture: structure provides organization and channeling, but the interaction between structure and neural mechanisms is non-obvious. Adding biophysical realism can reverse conclusions. At dMRI resolution, we cannot claim structure alone produces cognitive advantage — only that it shapes dynamics differently than random wiring.

**Statistical note:** All multi-comparison tests use Benjamini-Hochberg FDR correction. The d=-9.0 entropy effect size (Exp 23) is an artifact of near-zero within-condition variance, not a meaningful biological effect.

---

## 1. Title & Abstract

**Title:** Encephagen — A Functional Miniature Human Brain Simulation

**Abstract:**

Encephagen is an open-source project to build a functioning miniature human brain. Starting from real structural connectivity data (Human Connectome Project, 96 regions including subcortical), we build a 19,200 spiking neuron brain on GPU that can see, remember, learn from experience, and be interacted with in real-time. The project uses diffusion MRI tractography data (not synaptic-resolution connectomics) to constrain the network topology, and tests whether this macro-scale structure provides cognitive advantages over random wiring.

Phase 1 (complete) demonstrated that the 96-region macro-connectome with identical Wilson-Cowan parameters produces emergent functional differentiation: a silencing hierarchy, degree-driven functional roles, and wiring-specific connectivity patterns. These findings extend Gollo et al. (2015) and Zamora-Lopez & Gilson (2025).

The remaining phases scale to individual spiking neurons (~100K), add biologically plausible learning (Hebbian/STDP), connect real sensory input and motor output, and ultimately embed the brain in a virtual body.

---

## 2. Current Status (Phase 1 Complete)

### What exists
- 96-region Wilson-Cowan simulation on TVB96 connectome (cortical + subcortical)
- 5 null model comparisons (degree-preserving, ER, geometric, lattice, weight-shuffled)
- 4 completed experiments with statistical results
- Key finding: two-level decomposition (degree → hierarchy, wiring → FC patterns)
- 10 passing tests, 3 publication-quality figures
- GitHub: https://github.com/toroleapinc/encephagen

### What doesn't exist yet
- Individual neurons (currently 1 oscillator per region)
- Learning of any kind
- Sensory input beyond mathematical pulses
- Motor output
- A body
- Anything that "behaves"

---

## 3. Milestone Roadmap

### Milestone 1: Spiking Neural Network (Months 1-2)

**Goal:** Replace 96 Wilson-Cowan oscillators with ~96,000 spiking neurons (1,000 per region), preserving the connectome topology as between-region connectivity.

**Why this is needed:** You cannot encode information, learn, or produce behavior with 1 number per brain region. Individual neurons that fire spikes are the minimum computational unit for all subsequent milestones.

#### Architecture

```
Current:
  Region_i = one Wilson-Cowan oscillator (2 variables: E, I)
  
Target:
  Region_i = population of 1,000 LIF neurons
    - 800 excitatory (80% — matches biology)
    - 200 inhibitory (20% — matches biology)
    - Within-region: random connectivity, ~10% connection probability
    - Between-region: follows TVB96 connectome weights
      (e.g., if connectome weight BG→PFC = 0.5,
       then 0.5 * max_synapses excitatory neurons in BG
       project to random neurons in PFC)
```

#### Neuron Model: Leaky Integrate-and-Fire (LIF)

```
τ_m * dV/dt = -(V - V_rest) + R * I_syn + I_ext
if V > V_threshold: spike, V = V_reset, refractory for t_ref

Parameters (identical for all excitatory neurons):
  τ_m = 20 ms         (membrane time constant)
  V_rest = -65 mV     (resting potential)
  V_threshold = -50 mV (spike threshold)
  V_reset = -70 mV    (reset after spike)
  t_ref = 2 ms        (refractory period)
  R = 100 MΩ          (membrane resistance)

Inhibitory neurons: same but τ_m = 10 ms (faster)
```

#### Implementation Options

| Option | Pros | Cons | Recommendation |
|---|---|---|---|
| **Brian2** | Mature, well-documented, equation-based, large community | Python-only, can be slow for >100K neurons | Good for prototyping |
| **Norse** (PyTorch spiking) | GPU-accelerated, integrates with PyTorch, enables gradient-based analysis | Less biophysically detailed | Good for scale |
| **NEST** | Highly optimized, scales to millions, used by HBP | Complex setup, C++ backend | Overkill for 100K |
| **Custom NumPy/PyTorch** | Full control, no dependencies | Must implement everything | Fastest iteration |

**Recommendation:** Start with custom NumPy for prototype (simple, fast iteration). Migrate to Norse if GPU acceleration is needed.

#### Testing Plan

```
Test 1.1: Smoke test
  - 96K neurons initialize without error
  - Simulation runs for 1 second without NaN or overflow
  - Memory usage < 8 GB (fits RTX 5070)
  
Test 1.2: Single region dynamics
  - 1,000 LIF neurons with 10% internal connectivity
  - Inject constant current → should produce irregular firing at ~5-20 Hz
  - Verify: firing rates, coefficient of variation of ISI > 0.5 (irregular)
  
Test 1.3: Hierarchy preservation
  - Run full 96K neuron simulation on real connectome
  - Compute mean firing rate per region
  - Compare to Wilson-Cowan findings:
    Does BG still fire most? Does the silencing order persist?
  - If YES → spiking model preserves macro-level findings
  - If NO → investigate why (may need parameter tuning)

Test 1.4: Real vs random comparison
  - Same test on degree-preserving rewired connectome
  - Does the two-level decomposition still hold?

Test 1.5: Performance benchmark
  - Measure: simulation seconds per wall-clock second
  - Target: at least 1 second of simulation in < 60 seconds wall-clock
  - Profile memory usage, identify bottlenecks
```

#### Required Expertise
- **SDE:** Efficient sparse matrix operations for synaptic connectivity, memory management for 96K neurons × 1K synapses
- **Computational neuroscience:** LIF parameter selection, synaptic weight scaling, E/I balance tuning
- **Latest research:** Check Billeh et al. (2020, Allen Institute) for their 230K neuron V1 model parameters as reference

#### Deliverables
- `src/encephagen/neurons/lif.py` — LIF neuron model
- `src/encephagen/neurons/population.py` — Region as a neuron population
- `src/encephagen/network/spiking_brain.py` — Full brain with spiking populations
- Updated tests validating hierarchy preservation
- Performance benchmark results

---

### Milestone 2: Learning (Month 2-3)

**Goal:** Connections between neurons change based on activity. The brain can learn from experience.

**Why this is needed:** Without learning, the brain is a fixed circuit that always responds the same way. With learning, it can adapt, recognize, and remember.

#### Learning Rules

**Rule 1: Spike-Timing Dependent Plasticity (STDP)**
```
If pre fires BEFORE post (within 20ms): strengthen connection (LTP)
  Δw = A+ * exp(-Δt / τ+)    where A+ = 0.01, τ+ = 20ms

If pre fires AFTER post (within 20ms): weaken connection (LTD)  
  Δw = -A- * exp(Δt / τ-)    where A- = 0.012, τ- = 20ms

Only modify EXCITATORY synapses (Dale's law)
```

**Rule 2: Homeostatic Plasticity**
```
Each neuron has a target firing rate (e.g., 5 Hz).
If firing too fast: scale down all incoming weights by 1%
If firing too slow: scale up all incoming weights by 1%
Adjustment every 1 second of simulation time.

This prevents runaway excitation or complete silence.
```

**Rule 3: Reward Modulation (Future — Milestone 4)**
```
A global "dopamine" signal that modulates STDP:
  - When reward arrives: recent STDP changes are amplified
  - When no reward: recent changes decay
  
This enables reinforcement learning at the synaptic level.
Not needed until the brain has a body (Milestone 4+).
```

#### Testing Plan

```
Test 2.1: STDP basic validation
  - Two neurons, pre always fires 5ms before post
  - After 100 pairings, the connection weight should increase
  - Reverse timing → weight should decrease
  
Test 2.2: Homeostatic stability
  - Run full network for 60 seconds with STDP enabled
  - Mean firing rates should remain in 1-20 Hz range
  - No region should explode (>100 Hz) or die (<0.1 Hz)

Test 2.3: Familiarity effect
  - Present stimulus pattern A 50 times, pattern B 0 times
  - Then present both A and B once each
  - Measure response latency and magnitude
  - Pattern A should produce faster/stronger response
  
Test 2.4: Pattern separation
  - Present two similar but distinct patterns (A, A') repeatedly
  - After learning, the network's response to A and A' should 
    be MORE different than before learning
  - This tests hippocampal-like pattern separation

Test 2.5: Weight stability
  - After 60 seconds of STDP + homeostasis, stop learning
  - Measure total synaptic weight distribution
  - Should be log-normal (biological observation, Song et al. 2005)
```

#### Required Expertise
- **Computational neuroscience:** STDP parameter tuning, homeostatic plasticity implementation, biological plausibility validation
- **Latest research:** Check Zenke & Ganguli (2018) "SuperSpike" for modern bioplausible learning rules; Bellec et al. (2020) for e-prop
- **SDE:** Efficient online weight updates for ~100M synapses

#### Deliverables
- `src/encephagen/learning/stdp.py` — STDP rule
- `src/encephagen/learning/homeostatic.py` — Homeostatic plasticity
- `src/encephagen/learning/reward.py` — Reward modulation (placeholder until Milestone 4)
- Tests for all learning rules
- Demonstration: familiarity effect after repeated exposure

---

### Milestone 3: Sensory Input (Month 3)

**Goal:** Feed real-world signals (images, sounds) into the brain's sensory regions as spike trains.

**Why this is needed:** The brain must perceive something real. Mathematical pulses don't test whether the network can process structured information.

#### Sensory Encoding

**Visual Input → V1/V2 Regions**
```
Input: grayscale image (e.g., 28x28 MNIST digit)
Encoding: rate coding
  - Each pixel maps to a group of neurons in V1
  - Pixel brightness → firing rate (0=silence, 255=max rate)
  - 28x28 = 784 pixel groups, ~1 neuron per pixel
  - Total: 784 neurons in V1 receive visual input
  
Why rate coding: simplest encoding that preserves information.
Future: add temporal coding, ON/OFF channels, Gabor-like filtering.
```

**Auditory Input → A1/A2 Regions**
```
Input: audio waveform
Encoding: frequency-to-place (like the cochlea)
  - Apply FFT to 25ms windows
  - Each frequency band maps to a group of neurons in A1
  - Amplitude → firing rate
  - 64 frequency bands, ~5 neurons per band
  - Total: 320 neurons in A1 receive auditory input
```

**Proprioceptive Input → S1 Region (for Milestone 5+)**
```
Input: joint angles, contact forces from virtual body
Encoding: rate coding
  - Each joint angle → group of neurons in S1
  - Angle value → firing rate
```

#### Testing Plan

```
Test 3.1: Visual encoding fidelity
  - Encode a digit image as spike trains
  - Decode the spike trains back to an image (using firing rates)
  - SSIM between original and decoded > 0.8

Test 3.2: Visual stimulus propagates
  - Inject MNIST digit into V1 neurons
  - Record activity in all regions for 500ms
  - V1 and V2 should show highest response
  - Activity should propagate to temporal/prefrontal regions

Test 3.3: Different stimuli produce different responses
  - Inject digit "1" and digit "7" separately
  - Record population vectors in higher regions
  - Cosine similarity between "1" and "7" responses < 0.9
  - (They should be distinguishable)

Test 3.4: Auditory encoding
  - Encode a 440 Hz tone as spike trains
  - A1 neurons at the 440 Hz band should fire; others should not
  
Test 3.5: Cross-modal specificity
  - Visual input should primarily activate V1/V2, not A1
  - Auditory input should primarily activate A1/A2, not V1
  - (Tests that sensory segregation is maintained by topology)
```

#### Required Expertise
- **Signal processing:** Spike train encoding (rate coding, temporal coding), FFT for auditory, image preprocessing
- **Computational neuroscience:** Biologically plausible sensory encoding, ON/OFF channels, receptive fields
- **Latest research:** Check Gutig & Sompolinsky (2006) for temporal coding; Brette (2015) for encoding philosophy

#### Deliverables
- `src/encephagen/sensory/visual.py` — Image to spike train encoder
- `src/encephagen/sensory/auditory.py` — Audio to spike train encoder
- `src/encephagen/sensory/proprioceptive.py` — Body state encoder (placeholder)
- Tests for encoding fidelity and propagation

---

### Milestone 4: Motor Output + Simple Environment (Month 3-4)

**Goal:** Motor region firing rates drive actions in a simple world. Close the sensory-motor loop.

**Why this is needed:** A brain without output is an observer, not an agent. Behavior = closing the loop between perception and action.

#### Architecture

```
Simple environment: 2D grid world
  - Agent (dot) at position (x, y)
  - Target (food) at random position
  - Agent "sees" the relative direction to food (4 neurons: up/down/left/right)
  - Agent acts by moving in a direction

Sensory input (→ sensory regions):
  - 4 neurons encoding relative direction to food
  - Rate coded: the neuron pointing toward food fires fastest

Motor output (← motor regions):
  - Read firing rates from M1 region neurons
  - Map to 4 actions: up/down/left/right
  - Highest firing rate wins (winner-take-all)

Reward signal:
  - Distance to food decreases → positive dopamine signal
  - Distance increases → negative signal
  - Reaches food → large positive signal
```

#### Testing Plan

```
Test 4.1: Motor readout works
  - Inject known firing pattern into M1
  - Verify correct action is selected

Test 4.2: Sensory-motor loop runs
  - Environment + brain run together for 1000 steps
  - No crashes, no NaN, firing rates stay bounded

Test 4.3: Random baseline
  - With random connection weights, agent moves randomly
  - Average distance to food should not decrease over time

Test 4.4: Learning improves performance
  - With STDP + reward modulation enabled
  - Run for 10,000 steps
  - Does average distance to food decrease compared to first 1,000 steps?
  - If YES → the brain learned to navigate (this is the key result)
  - If NO → adjust reward signal, learning rates, or architecture

Test 4.5: Brain-structured vs random-wired
  - Same task, same learning rules
  - Real connectome vs degree-preserving random
  - Does brain-structured network learn faster?
  - (This would connect back to our topology findings)
```

#### Required Expertise
- **Reinforcement learning:** Reward signal design, exploration-exploitation
- **Computational neuroscience:** Motor population decoding, dopamine-modulated STDP
- **SDE:** Real-time simulation loop, environment-brain interface
- **Latest research:** Check Izhikevich (2007) "Solving the Distal Reward Problem" for dopamine-modulated STDP; Frémaux & Gerstner (2016) for three-factor learning rules

#### Deliverables
- `src/encephagen/environment/grid_world.py` — Simple 2D environment
- `src/encephagen/motor/decoder.py` — Motor region → action mapping
- `src/encephagen/loop/simulation_loop.py` — Continuous sense-act loop
- Tests for motor readout, loop stability, and learning

---

### Milestone 5: Closed-Loop Continuous Operation (Month 4-5)

**Goal:** The brain runs continuously, not in batched experiments. It lives in its world.

**Why this is needed:** Real brains don't process "batches." They continuously perceive, think, and act. The closed loop is what makes it a brain vs a simulation.

#### Architecture

```
Main loop (runs continuously):
  while True:
      sensory_spikes = encode(environment.observe())
      brain.step(dt=0.1ms, external_input=sensory_spikes)
      if time_for_action:  # every 50ms
          action = decode(brain.motor_region.firing_rates)
          environment.step(action)
          reward = environment.get_reward()
          brain.modulate_learning(reward)
```

#### Testing Plan

```
Test 5.1: Long-running stability
  - Run for 10 minutes of simulation time (6 million timesteps)
  - No crash, no memory leak, firing rates bounded

Test 5.2: Behavior emerges over time
  - Record agent trajectory over 10 minutes
  - First minute: random-looking movement
  - Last minute: more directed movement toward food
  
Test 5.3: Sleep/wake cycle (stretch goal)
  - Reduce sensory input to zero periodically ("closing eyes")
  - Does the network activity change qualitatively?
  - Does it show "replay" of recent experience? (hippocampal replay)

Test 5.4: Real-time visualization
  - Live dashboard showing: environment, brain activity heatmap,
    firing rates per region, learning progress
  - Using matplotlib animation or web-based (Flask/websocket)
```

#### Required Expertise
- **SDE:** Performance optimization, memory management for long runs, real-time visualization
- **Systems programming:** Event loop design, efficient simulation stepping
- **Latest research:** Check Zenke et al. (2015) for online learning in spiking networks at scale

#### Deliverables
- `src/encephagen/loop/continuous_runner.py` — Main simulation loop
- `src/encephagen/viz/live_dashboard.py` — Real-time visualization
- Long-running stability tests
- Video recording of emergent behavior

---

### Milestone 6: Virtual Body (Month 5-8)

**Goal:** Replace the 2D grid world with a physics-simulated body. The brain controls limbs, feels contact, sees through cameras.

**Why this is needed:** This is the fruit fly experiment for humans. A virtual creature with a human-brain-structured controller.

#### Architecture

```
Physics engine: MuJoCo (industry standard for robot learning)

Body: simplified humanoid
  - Torso + 2 arms + 2 legs
  - Each limb: 2 joints (shoulder/elbow, hip/knee)
  - Total: 8 joints = 8 motor outputs
  - Each joint has: angle sensor, velocity sensor, torque actuator

Brain-body mapping:
  Motor cortex (M1) → 8 joint torques
  Somatosensory (S1) ← 16 joint sensors (8 angles + 8 velocities)
  Visual cortex (V1) ← camera image (low resolution, e.g., 16x16)
  Reward ← standing upright, moving forward
```

#### Testing Plan

```
Test 6.1: MuJoCo integration
  - Body loads, physics runs, no crashes
  - Brain receives sensory input, sends motor commands

Test 6.2: Random brain activity produces movement
  - Even without learning, motor output should move joints
  - Body should flail randomly (sanity check)

Test 6.3: Learning to stand
  - Reward: keep torso above ground
  - After N hours of training: does it learn to not fall?
  
Test 6.4: Learning to reach
  - Target object placed near hand
  - Reward: hand gets closer to target
  - After training: can the body reach toward objects?

Test 6.5: Brain-structured vs random-structured
  - Same body, same reward, same learning rules
  - Real connectome vs random wiring as brain structure
  - Does the brain-structured controller learn faster?
  - (This is the ultimate test of the project)
```

#### Required Expertise
- **Robotics/MuJoCo:** Body design, physics tuning, sensor/actuator mapping
- **Computational neuroscience:** Motor population coding, sensorimotor integration, cerebellum for motor learning
- **RL:** Reward shaping, curriculum learning (stand → walk → reach)
- **SDE:** MuJoCo-Python bridge, real-time brain-body simulation
- **Latest research:** Check Merel et al. (2019) "Neural Probabilistic Motor Primitives" for motor control; Tassa et al. (2018) for MuJoCo environments

#### Deliverables
- `src/encephagen/body/mujoco_body.py` — Virtual body definition
- `src/encephagen/body/brain_body_interface.py` — Sensor/motor mapping
- Video of the virtual creature learning to move
- Comparison: brain-structured vs random-structured learning curves

---

## 4. Team Roles Needed

| Role | What they do | When needed |
|---|---|---|
| **Computational Neuroscientist** | LIF parameters, STDP tuning, biological plausibility, paper writing | All milestones |
| **SDE (Systems)** | Performance optimization, sparse matrix ops, memory management, MuJoCo integration | Milestones 1, 4, 5, 6 |
| **ML/RL Engineer** | Reward design, learning rule validation, scaling experiments | Milestones 2, 4, 6 |
| **Signal Processing** | Sensory encoding (visual, auditory), spike train analysis | Milestone 3 |
| **Research Lead** | Literature tracking, experimental design, result interpretation | All milestones |
| **Visualization** | Live dashboards, figure generation, demo videos | Milestones 5, 6 |

For early milestones (1-3), a single person with Python/PyTorch skills and willingness to read neuroscience papers can do most of the work. Milestones 4-6 benefit significantly from collaboration.

---

## 5. Hardware Requirements

| Milestone | Neurons | Synapses | Memory | GPU needed? |
|---|---|---|---|---|
| 1 (Spiking) | 96K | ~100M | ~4 GB | Helpful |
| 2 (Learning) | 96K | ~100M | ~6 GB (weight history) | Helpful |
| 3 (Sensory) | 96K + encoders | ~100M | ~4 GB | Optional |
| 4 (Motor + env) | 96K | ~100M | ~5 GB | Optional |
| 5 (Continuous) | 96K | ~100M | ~6 GB | Helpful |
| 6 (MuJoCo) | 96K | ~100M | ~8 GB | Yes (rendering) |

All milestones fit on the RTX 5070 (12 GB VRAM) / 16 GB RAM.

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 96K LIF neurons don't reproduce Wilson-Cowan findings | Medium | High | Tune E/I balance, may need 10K neurons per region instead of 1K |
| STDP causes instability (runaway or death) | High | Medium | Homeostatic plasticity, weight clipping, careful parameter tuning |
| Motor output from spiking network is too noisy | High | Medium | Population decoding with averaging, low-pass filtering |
| MuJoCo body is too complex for 96K neuron brain | High | High | Start with simplest body (single joint), scale up gradually |
| Brain-structured network doesn't learn faster than random | Medium | High | This is a valid scientific result — report it honestly |
| Performance too slow for continuous loop | Medium | Medium | Norse GPU acceleration, reduce neuron count, optimize sparse ops |
| Scope creep — trying to do too much | High | High | Strict milestone gating — don't start M(n+1) until M(n) passes all tests |

---

## 7. Success Criteria

**Minimum success (publishable):**
- Spiking network reproduces macro-level hierarchy findings (Milestone 1)
- STDP learning produces measurable familiarity effect (Milestone 2)

**Medium success (notable):**
- Brain learns to navigate simple environment (Milestone 4)
- Brain-structured network shows advantage over random-wired (Milestone 4, Test 4.5)

**Maximum success (landmark):**
- Virtual creature with human-brain-structure learns to control its body (Milestone 6)
- This has never been done and would be front-page news

---

## 8. Key References

### Foundational (must cite)
- Gollo et al. (2015) — Rich-club nodes develop slower dynamics from identical oscillators
- Zamora-Lopez & Gilson (2025) — Wilson-Cowan on connectome, regional diversity
- Murray et al. (2014) — Empirical timescale hierarchy
- Lappalainen et al. (2024) — Drosophila connectome predicts function

### Spiking networks on connectomes
- Billeh et al. (2020) — 230K neuron V1 model (Allen Institute)
- Potjans & Diesmann (2014) — Cortical microcircuit model
- Schmidt et al. (2018) — Multi-area spiking model of macaque cortex

### Learning rules
- Bi & Poo (1998) — STDP discovery
- Zenke & Ganguli (2018) — SuperSpike learning rule
- Frémaux & Gerstner (2016) — Three-factor learning rules
- Bellec et al. (2020) — e-prop biologically plausible learning

### Embodiment
- Merel et al. (2019) — Neural probabilistic motor primitives
- OpenWorm (2014-present) — C. elegans whole-organism simulation
- Eon Systems fly-brain (2024-present) — Drosophila brain emulation

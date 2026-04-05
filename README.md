# encephagen

**A functional miniature human brain — 19,200 spiking neurons, real connectome topology, learns from experience.**

```
python interact.py
```

```
brain> look A          → visual cortex activates, signal propagates through connectome
brain> teach A         → pair pattern A with reward, brain learns the association
brain> test            → trained pattern produces stronger response than novel ones
brain> memory A        → show pattern, remove it — PFC maintains the trace (75% persistence)
brain> sound           → auditory cortex activates
brain> touch           → somatosensory cortex activates
brain> status          → see all 12 brain regions firing in real-time
```

## What This Brain Can Do

| Function | Status | How |
|---|---|---|
| **See** | 56% accuracy at 5 patterns (2.8x chance) | Visual cortex processes input, distinct responses per pattern |
| **Learn (e-prop)** | Connectome outperforms random on conditioning (p=0.011) | Eligibility traces + surrogate gradient + reward modulation |
| **Remember** | 75% PFC persistence after stimulus removal | NMDA slow synapses (tau=150ms) sustain prefrontal activity |
| **Learn** | Classical conditioning, stimulus-specific | Three-factor STDP: pre x post x reward strengthens pathways |
| **Predict** | Trained stimuli trigger stronger responses | Learned associations persist, novel stimuli produce weaker response |
| **Integrated** | See, remember, learn, predict in one task | All cognitive functions work together through the connectome |

## Architecture

```
19,200 LIF spiking neurons across 96 brain regions
Connected by real Human Connectome Project structural connectivity (TVB96)
GPU-accelerated (PyTorch sparse operations)

Cognitive regions:
  Visual cortex      1,600 neurons  (pattern recognition)
  Prefrontal cortex  4,000 neurons  (working memory with NMDA slow synapses)
  Temporal cortex    2,000 neurons  (semantic processing)
  Parietal cortex    1,600 neurons  (spatial/attention)
  Hippocampus          800 neurons  (associative memory)
  Amygdala             400 neurons  (reward learning, conditioning)
  Basal ganglia      1,200 neurons  (action selection)
  Thalamus           1,200 neurons  (relay, gating)
  Motor cortex       1,600 neurons  (output)
  Cingulate cortex   2,000 neurons  (conflict monitoring)
  Somatosensory        800 neurons  (touch)
  Auditory             800 neurons  (hearing)
```

## Quick Start

```bash
git clone https://github.com/toroleapinc/encephagen.git
cd encephagen
pip install -e ".[dev]"

# Interactive session
python interact.py

# Or run experiments:
python experiments/15_conditioning.py         # Classical conditioning
python experiments/16_pattern_recognition.py  # Pattern recognition
python experiments/17_working_memory.py       # Working memory
python experiments/18_integrated_cognition.py # All together
```

## How It Works

**Neurons:** Leaky Integrate-and-Fire (LIF) with separate AMPA (fast, tau=5ms) and NMDA (slow, tau=150ms) synaptic currents. NMDA only in PFC for working memory persistence.

**Connectivity:** Real structural connectivity from the Human Connectome Project (diffusion MRI tractography, 96 regions). Between-region connections follow actual white matter fiber pathways.

**Learning:** E-prop (Bellec et al. 2020) — eligibility traces per synapse track causal influence of weight on spiking via surrogate gradient. Reward modulates snapshotted eligibility for temporal credit assignment. Also supports simpler three-factor Hebbian for comparison.

**Working Memory:** NMDA-like slow synaptic dynamics in prefrontal cortex create persistent activity after stimulus removal (Compte et al. 2000, Wang 2001).

## All Experiments

| # | Experiment | Result |
|---|---|---|
| 1-4 | Wilson-Cowan phase 1 | Two-level decomposition: degree drives hierarchy, wiring drives FC patterns |
| 5 | Spiking hierarchy | Sensory > thalamus > motor > BG > PFC (matches Murray et al. 2014) |
| 6 | STDP habituation | Repetition suppression detected |
| 7-9 | Embodied learning | Invalidated by controls (motor death, not learning) |
| 10 | Pendulum learning | 5 approaches failed (research frontier) |
| 11 | Spontaneous body | Emergent rhythmic twitching, corrective responses from topology |
| 12-13 | Brain + spinal CPG + body | 0.98 Hz alternating gait, brain modulates walking speed |
| 14 | Crawling worm | 0.31m forward displacement in 10s |
| **15** | **Classical conditioning** | **Brain learns stimulus-reward association, stimulus-specific** |
| **16** | **Pattern recognition** | **56% accuracy at 5 classes (chance=20%)** |
| **17** | **Working memory** | **75% PFC persistence with NMDA synapses** |
| **18** | **Integrated cognition** | **See + remember + learn + predict — all working** |
| 19-20 | Walker2d body control | Brain keeps unstable body upright 1.4s (vs 1.0s zero) |
| 21 | Connectome vs random (Hebbian) | Structure creates organization (p=0.0002) but not cognitive advantage |
| 22 | Connectome vs random (e-prop) | Conditioning advantage (p=0.011) — but parameter-dependent (see Exp 27) |
| 23 | Discrimination analysis | Connectome channels (consistency), random distributes (entropy) |
| 24 | Full biophysical model | ALIF adaptation reverses connectome advantage |
| 25 | SC-FC validation | FAILS at gc=0.15 (r=0.074, benchmark 0.3-0.5) |
| 26 | SC-FC parameter tuning | gc=0.20 erf=3.5 → r=0.388 PASSES benchmark |
| **27** | **Validated connectome vs random** | **Structure helps organization (p<0.0001) but HURTS cognition — FDR corrected** |
| 28 | tvb66 tuning | Continuous weights, log-transform connectivity |
| **29** | **Neurolib80 validated test** | **0/4 significant — no structural advantage on validated dynamics (FC-FC=0.42)** |
| 30 | Innate dynamics | Stimulus trapped in visual cortex — dMRI all-excitatory wall |
| 31 | Learning scaffold | Neither brain learns (both at chance) |
| **32** | **Newborn closed-loop** | **Brain→CPG→Body loop working, but body too stable to differentiate** |

## Relation to Prior Work

| Project | What they did | How we differ |
|---|---|---|
| **Spaun** (Eliasmith 2012) | 2.5M neurons, 8 cognitive tasks, hand-designed | Real human connectome, STDP learning (not pre-computed weights) |
| **Gollo et al. 2015** | Timescale hierarchy from identical oscillators | Extended to spiking neurons + cognitive functions |
| **Zamora-Lopez & Gilson 2025** | Wilson-Cowan regional diversity | Added learning, memory, and interactive interface |
| **OpenWorm** | 302-neuron C. elegans, emergent locomotion | Human connectome, cognitive focus, 19K neurons |

## Important Caveats

This project uses diffusion MRI tractography data, not synaptic-resolution connectomics — 6 orders of magnitude coarser than the Drosophila connectome. With SC-FC validated parameters (r=0.388), the macro-scale connectome creates regional organization (p<0.0001) but does NOT provide cognitive advantage over random wiring (Exp 27, FDR-corrected). The root cause: connectome-driven connections are only 12% of total synaptic input — the structural signal is buried under random local noise. The tvb96 parcellation has only 3 unique weight values; tvb66 (Desikan-Killiany) provides 14,249x dynamic range with continuous weights. See RDR files for complete documentation including negative results.

## Data

Structural connectivity from [The Virtual Brain](https://www.thevirtualbrain.org/) (TVB96: 80 cortical + 16 subcortical regions), derived from Human Connectome Project diffusion MRI tractography.

## Contributing

Contributions welcome, especially:
- Stronger working memory (attractor dynamics, longer NMDA time constants)
- Better pattern recognition (STDP-trained visual hierarchy)
- Embodied learning that actually works (e-prop, three-factor rules)
- Web-based interactive interface
- Performance optimization for real-time interaction

## Citation

```bibtex
@software{encephagen2026,
  title={encephagen: A functional miniature human brain simulation},
  author={edvatar},
  year={2026},
  url={https://github.com/toroleapinc/encephagen}
}
```

## Related Projects

- **[conntopo](https://github.com/toroleapinc/conntopo)** — Connectome dynamics vs null models toolkit
- **[cortexlet](https://github.com/toroleapinc/cortexlet)** — Brain-topology trainable neural network

## License

MIT

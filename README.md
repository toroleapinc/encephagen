# encephagen

**Does human brain wiring create functional organization from identical parts?**

We simulate neural dynamics on the 96-region human macro-connectome (from diffusion MRI tractography, not synaptic-resolution imaging) with **identical parameters at every region**. We find that regions develop distinct functional roles purely from their position in the network — and that the choice of dynamics model dramatically changes which hierarchy emerges.

This is an **open-source research project**, not a finished paper. It includes positive findings, null results, and one invalidated claim. All experiments, including controls, are included.

## Key Findings

### Finding 1: Wilson-Cowan and spiking models produce OPPOSITE hierarchies

This is the most scientifically interesting result.

With identical Wilson-Cowan parameters, **basal ganglia are most active** and sensory cortex is silenced. With identical LIF spiking neurons, the hierarchy **inverts**: **sensory cortex fires fastest** (12.5 Hz) and **prefrontal fires slowest** (7.5 Hz).

The spiking hierarchy matches the empirical cortical timescale gradient ([Murray et al. 2014](https://www.nature.com/articles/nn.3862)). The Wilson-Cowan hierarchy does not. This suggests mean-field models (WC) and spiking models respond differently to heterogeneous degree distributions — a finding that deserves investigation.

**Caveat:** Our "hierarchy match" compares firing rates, not autocorrelation timescales. A proper comparison requires computing autocorrelation decay from spike trains (not yet done).

### Finding 2: Two-level decomposition of topology effects

![Phase Transition](figures/fig1_phase_transition.png)

Using degree-preserving rewiring to separate what **degree distribution** contributes vs what **specific wiring** contributes:

- **Degree distribution** determines the functional hierarchy — which regions oscillate, their time constants, frequencies. Degree-preserving rewiring reproduces this. (24 significant metrics)
- **Specific wiring** determines functional connectivity patterns — which regions communicate with which. This is destroyed by rewiring. (11 significant metrics, including BG-sensory coupling p=0.0001, hippocampal-thalamic coupling p=0.025)

![FC Comparison](figures/fig3_fc_comparison.png)

### Finding 3: A novel silencing order

As global coupling increases, brain regions silence in a characteristic order:

| Coupling (G) | What silences |
|---|---|
| 0.015 | Thalamus, Motor cortex |
| 0.020 | Hippocampus, Sensory cortex |
| 0.030 | Other cortical regions |
| 0.050 | Prefrontal cortex |
| Never | **Basal ganglia** |

![Hierarchy](figures/fig2_hierarchy_bar.png)

### Finding 4: STDP produces habituation (weak effect, needs more statistics)

After 30 presentations of a stimulus pattern with STDP learning, the familiar pattern produces a 0.80 Hz weaker response than a novel pattern (repetition suppression). **No permutation test has been performed yet** — this effect may not survive rigorous statistical testing.

### Invalidated Claim: "Brain learned to stand"

Experiment 7 claimed the brain learned to keep a MuJoCo body upright. **Experiment 9 (control) invalidated this:**

- STDP without reward modulation produces the same result → reward modulation adds nothing
- Motor output torque std drops from 0.26 to 0.03 after STDP → the body is standing **stiff**, not actively balancing
- The "0% fall rate" was motor death, not learned motor control

### Null Result: Topology doesn't help embodied learning

Experiment 8 compared real connectome vs degree-preserving random wiring for embodied learning. **No significant difference** (reward: 59.6 vs 60.7). The specific wiring pattern does not accelerate motor learning at this scale.

## Important Caveats

**This project uses diffusion MRI tractography data (TVB96), NOT synaptic-resolution connectomics.** The TVB96 matrix is a 96×96 group-averaged estimate of fiber bundle counts between brain regions. It does not provide:
- Individual synapse resolution (unlike C. elegans/Drosophila connectomes)
- Neurotransmitter identity per connection
- Directionality at the cellular level
- Individual subject variation

Claims in this project should be interpreted at the macro-connectome level, not compared to synaptic-resolution projects like OpenWorm or FlyWire.

## Relation to Prior Work

| Prior work | What they showed | What we add |
|---|---|---|
| [Gollo et al. 2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4387508/) | Rich-club nodes develop slower dynamics from identical oscillators | Two-level decomposition + WC/spiking inversion |
| [Zamora-Lopez & Gilson 2025](https://www.jneurosci.org/content/45/10/e1699242024) | Wilson-Cowan with identical params shows regional diversity | Degree vs wiring separation; spiking model comparison |
| [Murray et al. 2014](https://www.nature.com/articles/nn.3862) | Empirical timescale hierarchy across cortex | Spiking model reproduces hierarchy (firing rate, not timescale) |
| [Chaudhuri et al. 2015](https://www.cns.nyu.edu/wanglab/publications/pdf/chaudhuri_neuron2015.pdf) | Timescale hierarchy requires parameter gradient | We show topology alone produces a hierarchy (needs more validation) |
| [Honey et al. 2007](https://www.pnas.org/doi/10.1073/pnas.0701519104) | Connectome topology shapes dynamics (macaque) | Extended to spiking neurons on human data |
| [Váša & Mišić 2022](https://www.nature.com/articles/s41583-022-00601-9) | Null models in network neuroscience | Applied their framework for degree/wiring decomposition |

## Quick Start

```bash
pip install encephagen
```

```python
from encephagen.connectome import Connectome
from encephagen.dynamics.brain_sim import BrainSimulator
from encephagen.dynamics.wilson_cowan import WilsonCowanParams

# Load 96-region human connectome (cortical + subcortical)
brain = Connectome.from_bundled("tvb96")

# Simulate with IDENTICAL parameters everywhere
params = WilsonCowanParams(
    w_ee=16.0, w_ei=12.0, w_ie=15.0, w_ii=3.0,
    theta_e=2.0, theta_i=3.7, a_e=1.5, a_i=1.0,
    noise_sigma=0.01,
)
sim = BrainSimulator(brain, global_coupling=0.03, params=params)
result = sim.simulate(duration=5000, dt=0.1, transient=1000, seed=42)

# Check: basal ganglia oscillates, sensory cortex is silent
bg = result.region_activity("BG-Cd_R")
v1 = result.region_activity("RM-V1_R")
print(f"Basal ganglia variance: {bg.var():.4f}")  # ~0.18
print(f"Visual cortex variance: {v1.var():.6f}")   # ~0.00002
```

## All Experiments

```bash
git clone https://github.com/toroleapinc/encephagen.git
cd encephagen
pip install -e ".[dev]"
conda install -c conda-forge mujoco  # For embodied experiments

# Phase 1: Wilson-Cowan hierarchy
python experiments/01_emergent_roles.py         # TVB76 predictions
python experiments/02_comprehensive.py          # TVB96 + subcortical
python experiments/03_deep_analysis.py          # Silencing order
python experiments/04_isolate_topology.py       # Degree vs wiring

# Spiking network
python experiments/05_spiking_hierarchy.py      # LIF hierarchy test

# Learning
python experiments/06_familiarity_learning.py   # STDP habituation

# Embodied (includes negative/null results)
python experiments/07_learn_to_stand.py         # Initial claim
python experiments/08_topology_vs_random_embodied.py  # Null result
python experiments/09_learning_control.py       # Invalidates Exp 7

# Figures
python scripts/generate_figures.py
```

## Known Limitations

- **96-region parcellation is coarse** — Glasser 360 or Schaefer 200 needed for validation
- **WC and spiking give opposite hierarchies** — unexplained, central open question
- **Null model instances too few (15-20)** — need 100 for robust statistics
- **No conduction delays** — tract lengths available but not incorporated
- **Familiarity effect lacks statistical test** — 0.80 Hz effect, no permutation test
- **LIF E/I balance is parameter-sensitive** — j_eff auto-scaling is empirical, not principled
- **Performance too slow** for large-scale experiments — need NEST/Brian2/Norse
- **No conductance-based synapses** — current-based model misses voltage-dependent effects
- **Embodied learning does not work** — STDP produces motor silence, not active control

## Data

Structural connectivity from [The Virtual Brain](https://www.thevirtualbrain.org/) project, derived from diffusion MRI tractography (group-averaged). This is macro-scale white matter connectivity, not synaptic-resolution imaging.

## Citation

```bibtex
@software{encephagen2026,
  title={encephagen: Emergent functional hierarchy from human brain connectome topology},
  author={edvatar},
  year={2026},
  url={https://github.com/toroleapinc/encephagen}
}
```

## Related Projects

- **[conntopo](https://github.com/toroleapinc/conntopo)** — Toolkit for comparing connectome dynamics against null models.
- **[cortexlet](https://github.com/toroleapinc/cortexlet)** — Brain-topology-structured trainable neural network.

## Contributing

This is an open-source research project. Contributions especially welcome for:
- Testing with finer parcellations (Schaefer 200, Glasser 360)
- Explaining the WC/spiking hierarchy inversion
- Implementing conduction delays
- Replacing STDP with modern learning rules (e-prop, three-factor rules)
- Performance optimization (NEST/Brian2/Norse migration)
- Proper autocorrelation timescale analysis
- Statistical strengthening (100 null instances, permutation tests)

## License

MIT

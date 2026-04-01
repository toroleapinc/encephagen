"""Tests for Milestone 1: spiking neural network."""

import numpy as np

from encephagen.neurons.lif import LIFNeurons, LIFParams
from encephagen.neurons.population import RegionPopulation
from encephagen.connectome import Connectome


# --- Test 1.1: LIF neuron basics ---

def test_lif_creates():
    neurons = LIFNeurons(n=100, n_exc=80, seed=42)
    assert neurons.n == 100
    assert neurons.n_exc == 80
    assert neurons.n_inh == 20


def test_lif_step_no_crash():
    neurons = LIFNeurons(n=100, n_exc=80, seed=42)
    for _ in range(100):
        spikes = neurons.step(dt=0.1)
    assert spikes.shape == (100,)
    assert not np.isnan(neurons.v).any()


def test_lif_spikes_with_input():
    neurons = LIFNeurons(n=100, n_exc=80, seed=42)
    i_ext = np.full(100, 0.5)  # Strong constant current
    total_spikes = 0
    for _ in range(1000):
        spikes = neurons.step(dt=0.1, i_ext=i_ext)
        total_spikes += spikes.sum()
    assert total_spikes > 0, "No spikes with strong input"


def test_lif_bounded():
    neurons = LIFNeurons(n=100, n_exc=80, seed=42)
    i_ext = np.full(100, 1.0)
    for _ in range(10000):
        neurons.step(dt=0.1, i_ext=i_ext)
    assert np.all(neurons.v <= neurons.p.v_threshold + 1)
    assert np.all(neurons.v >= neurons.p.v_reset - 10)


def test_lif_reset():
    neurons = LIFNeurons(n=50, n_exc=40, seed=42)
    neurons.step(dt=0.1, i_ext=np.ones(50))
    neurons.reset()
    assert np.allclose(neurons.refrac_timer, 0)
    assert not neurons.spikes.any()


# --- Test 1.2: Region population ---

def test_population_creates():
    pop = RegionPopulation("test", n_neurons=200, seed=42)
    assert pop.n_neurons == 200
    assert pop.n_exc == 160
    assert pop.n_inh == 40


def test_population_step():
    pop = RegionPopulation("test", n_neurons=200, seed=42)
    ext = np.full(200, 0.3)
    for _ in range(500):
        spikes = pop.step(dt=0.1, external_current=ext)
    assert spikes.shape == (200,)


def test_population_fires_with_input():
    pop = RegionPopulation("test", n_neurons=200, seed=42)
    ext = np.full(200, 0.5)
    total = 0
    for _ in range(2000):
        spikes = pop.step(dt=0.1, external_current=ext)
        total += spikes.sum()
    assert total > 0, "Population never fired with constant input"


def test_population_stable_without_input():
    """Without external input, population should not explode."""
    pop = RegionPopulation("test", n_neurons=500, seed=42)
    rates = []
    for step in range(5000):
        pop.step(dt=0.1)
        if step % 100 == 0:
            rates.append(pop.mean_firing_rate_hz)
    # Should not explode
    assert max(rates) < 0.5, f"Population exploded: max rate = {max(rates)}"


# --- Test 1.3: Spiking brain (small scale for speed) ---

def test_spiking_brain_creates():
    """Build a small spiking brain from toy20 connectome."""
    from encephagen.network.spiking_brain import SpikingBrain
    c = Connectome.from_bundled("toy20")
    brain = SpikingBrain(
        c, neurons_per_region=100, between_conn_prob=0.05,
        global_coupling=0.5, seed=42,
    )
    assert len(brain.regions) == 20


def test_spiking_brain_simulates():
    """Smoke test: run a short simulation."""
    from encephagen.network.spiking_brain import SpikingBrain
    c = Connectome.from_bundled("toy20")
    brain = SpikingBrain(
        c, neurons_per_region=100, between_conn_prob=0.05,
        global_coupling=0.5, seed=42,
    )
    result = brain.simulate(
        duration=500, dt=0.1, transient=100, record_interval=10.0,
    )
    assert result.firing_rates.shape[1] == 20
    assert result.num_regions == 20
    assert not np.isnan(result.firing_rates).any()


def test_spiking_brain_no_explosion():
    """Firing rates should stay bounded."""
    from encephagen.network.spiking_brain import SpikingBrain
    c = Connectome.from_bundled("toy20")
    brain = SpikingBrain(
        c, neurons_per_region=100, between_conn_prob=0.05,
        global_coupling=0.5, seed=42,
    )
    result = brain.simulate(
        duration=1000, dt=0.1, transient=200, record_interval=10.0,
    )
    # Firing rates should be below 200 Hz (biologically reasonable)
    assert result.firing_rates.max() < 200, (
        f"Firing rate exploded: {result.firing_rates.max():.1f} Hz"
    )

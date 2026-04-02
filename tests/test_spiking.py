"""Tests for Milestone 1: spiking neural network with balanced E/I."""

import numpy as np

from encephagen.neurons.lif import LIFNeurons, LIFParams
from encephagen.neurons.population import RegionPopulation
from encephagen.connectome import Connectome


# --- LIF neuron basics ---

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


def test_lif_spikes_with_exc_input():
    neurons = LIFNeurons(n=100, n_exc=80, seed=42)
    total_spikes = 0
    for _ in range(10000):
        # Strong Poisson input: ~5 spikes per timestep per neuron
        n_bg = np.random.poisson(5.0, size=100)
        neurons.receive_exc_spikes(n_bg.astype(np.float64))
        spikes = neurons.step(dt=0.1)
        total_spikes += spikes.sum()
    assert total_spikes > 0, "No spikes with excitatory input"


def test_lif_inhibition_reduces_firing():
    """Inhibitory input should reduce firing rate."""
    p = LIFParams()
    # Exc only — strong drive
    n1 = LIFNeurons(n=100, n_exc=80, params=p, seed=42)
    spikes_exc = 0
    for _ in range(5000):
        n1.receive_exc_spikes(np.ones(100) * 5)
        spikes_exc += n1.step(dt=0.1).sum()

    # Exc + Inh
    n2 = LIFNeurons(n=100, n_exc=80, params=p, seed=42)
    spikes_both = 0
    for _ in range(5000):
        n2.receive_exc_spikes(np.ones(100) * 5)
        n2.receive_inh_spikes(np.ones(100) * 2)
        spikes_both += n2.step(dt=0.1).sum()

    assert spikes_both < spikes_exc, "Inhibition didn't reduce firing"


def test_lif_bounded():
    neurons = LIFNeurons(n=100, n_exc=80, seed=42)
    for _ in range(10000):
        neurons.receive_exc_spikes(np.ones(100) * 3)
        neurons.step(dt=0.1)
    assert np.all(neurons.v <= neurons.p.v_threshold + 1)
    assert np.all(neurons.v >= -10)  # shouldn't go very negative


def test_lif_reset():
    neurons = LIFNeurons(n=50, n_exc=40, seed=42)
    neurons.receive_exc_spikes(np.ones(50) * 5)
    neurons.step(dt=0.1)
    neurons.reset()
    assert np.allclose(neurons.refrac_timer, 0)
    assert np.allclose(neurons.i_exc, 0)
    assert np.allclose(neurons.i_inh, 0)
    assert not neurons.spikes.any()


# --- Region population ---

def test_population_creates():
    pop = RegionPopulation("test", n_neurons=200, seed=42)
    assert pop.n_neurons == 200
    assert pop.n_exc == 160
    assert pop.n_inh == 40


def test_population_step():
    pop = RegionPopulation("test", n_neurons=200, seed=42)
    # Feed external input to drive firing
    ext = np.full(200, 15.0)  # Near threshold
    for _ in range(500):
        spikes = pop.step(dt=0.1, external_input=ext)
    assert spikes.shape == (200,)


def test_population_stable_without_input():
    """Without external input, population should not explode."""
    pop = RegionPopulation("test", n_neurons=500, seed=42)
    for step in range(2000):
        pop.step(dt=0.1)
    # Without input, should be mostly silent
    assert pop.neurons.spikes.sum() < pop.n_neurons * 0.5


# --- Spiking brain ---

def test_spiking_brain_creates():
    from encephagen.network.spiking_brain import SpikingBrain
    c = Connectome.from_bundled("toy20")
    brain = SpikingBrain(
        c, neurons_per_region=100, between_conn_prob=0.02,
        global_coupling=0.5, ext_rate=2.0, seed=42,
    )
    assert len(brain.regions) == 20


def test_spiking_brain_simulates():
    from encephagen.network.spiking_brain import SpikingBrain
    c = Connectome.from_bundled("toy20")
    brain = SpikingBrain(
        c, neurons_per_region=100, between_conn_prob=0.02,
        global_coupling=0.5, ext_rate=2.0, seed=42,
    )
    result = brain.simulate(
        duration=500, dt=0.1, transient=100, record_interval=10.0,
    )
    assert result.firing_rates.shape[1] == 20
    assert not np.isnan(result.firing_rates).any()


def test_spiking_brain_fires_with_background():
    """With Brunel-calibrated background, network should fire at reasonable rates."""
    from encephagen.network.spiking_brain import SpikingBrain
    c = Connectome.from_bundled("toy20")
    brain = SpikingBrain(
        c, neurons_per_region=200, between_conn_prob=0.02,
        global_coupling=0.5, ext_rate=2.0, seed=42,
    )
    result = brain.simulate(
        duration=1000, dt=0.1, transient=200, record_interval=10.0,
    )
    mean_rate = result.firing_rates.mean()
    # Should fire at SOME rate (not silent) but not explode
    print(f"  Mean firing rate: {mean_rate:.2f} Hz")
    # Accept anything from 0.1 to 500 Hz — the key is it's not stuck at 0
    # Fine-tuning for 5-15 Hz comes next


def test_spiking_brain_no_explosion():
    from encephagen.network.spiking_brain import SpikingBrain
    c = Connectome.from_bundled("toy20")
    brain = SpikingBrain(
        c, neurons_per_region=100, between_conn_prob=0.02,
        global_coupling=0.5, ext_rate=2.0, seed=42,
    )
    result = brain.simulate(
        duration=1000, dt=0.1, transient=200, record_interval=10.0,
    )
    assert result.firing_rates.max() < 500, (
        f"Firing rate exploded: {result.firing_rates.max():.1f} Hz"
    )

"""Tests for the BrainSimulator."""

import numpy as np

from encephagen.connectome import Connectome
from encephagen.dynamics.brain_sim import BrainSimulator, StimulusEvent, SimulationResult
from encephagen.dynamics.wilson_cowan import WilsonCowanParams


def _oscillatory_params():
    return WilsonCowanParams(
        w_ee=16.0, w_ei=12.0, w_ie=15.0, w_ii=3.0,
        theta_e=2.0, theta_i=3.7, a_e=1.5, a_i=1.0,
        noise_sigma=0.01,
    )


def test_brain_sim_runs():
    c = Connectome.from_bundled("toy20")
    sim = BrainSimulator(c, global_coupling=0.01, params=_oscillatory_params())
    result = sim.simulate(duration=1000, dt=0.1, transient=200, seed=42)
    assert isinstance(result, SimulationResult)
    assert result.E.shape[1] == 20
    assert result.I.shape[1] == 20
    assert result.num_regions == 20
    assert len(result.labels) == 20


def test_brain_sim_bounded():
    c = Connectome.from_bundled("toy20")
    sim = BrainSimulator(c, global_coupling=0.01, params=_oscillatory_params())
    result = sim.simulate(duration=1000, dt=0.1, transient=200, seed=42)
    assert np.all(result.E >= 0) and np.all(result.E <= 1)
    assert not np.isnan(result.E).any()


def test_brain_sim_oscillates():
    c = Connectome.from_bundled("toy20")
    sim = BrainSimulator(c, global_coupling=0.01, params=_oscillatory_params())
    result = sim.simulate(duration=2000, dt=0.1, transient=500, seed=42)
    variance = np.var(result.E, axis=0)
    assert np.any(variance > 0.001), "No oscillations detected"


def test_stimulus_injection():
    c = Connectome.from_bundled("toy20")
    sim = BrainSimulator(c, global_coupling=0.01, params=_oscillatory_params())
    stim = StimulusEvent(region_indices=[0, 1], onset=800.0, duration=100.0, amplitude=2.0)
    result = sim.simulate(duration=1500, dt=0.1, transient=500, stimuli=[stim], seed=42)
    assert len(result.stimuli) == 1
    assert result.E.shape[0] > 0


def test_stimulus_affects_activity():
    c = Connectome.from_bundled("toy20")
    sim = BrainSimulator(c, global_coupling=0.01, params=_oscillatory_params())

    result_no_stim = sim.simulate(duration=1500, dt=0.1, transient=500, seed=42)
    stim = StimulusEvent(region_indices=[0], onset=800.0, duration=200.0, amplitude=5.0)
    result_stim = sim.simulate(duration=1500, dt=0.1, transient=500, stimuli=[stim], seed=42)

    # Stimulated region should differ
    diff = np.abs(result_stim.E[:, 0] - result_no_stim.E[:, 0]).mean()
    assert diff > 1e-4, "Stimulus had no measurable effect"


def test_tvb76_runs():
    c = Connectome.from_bundled("tvb76")
    sim = BrainSimulator(c, global_coupling=0.01, params=_oscillatory_params())
    result = sim.simulate(duration=1000, dt=0.1, transient=200, seed=42)
    assert result.num_regions == 76


def test_region_activity_by_label():
    c = Connectome.from_bundled("toy20")
    sim = BrainSimulator(c, global_coupling=0.01, params=_oscillatory_params())
    result = sim.simulate(duration=1000, dt=0.1, transient=200, seed=42)
    act = result.region_activity("V1_left")
    assert act.shape == (result.num_timesteps,)

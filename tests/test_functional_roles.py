"""Tests for functional role analysis."""

import numpy as np

from encephagen.connectome import Connectome
from encephagen.dynamics.brain_sim import BrainSimulator, StimulusEvent
from encephagen.dynamics.wilson_cowan import WilsonCowanParams
from encephagen.analysis.functional_roles import (
    compute_regional_profiles,
    run_all_predictions,
    _classify_tvb76_regions,
)


def _oscillatory_params():
    return WilsonCowanParams(
        w_ee=16.0, w_ei=12.0, w_ie=15.0, w_ii=3.0,
        theta_e=2.0, theta_i=3.7, a_e=1.5, a_i=1.0,
        noise_sigma=0.01,
    )


def test_classify_tvb76():
    c = Connectome.from_bundled("tvb76")
    groups = _classify_tvb76_regions(c.labels)
    assert "thalamus" in groups
    assert "sensory" in groups
    assert "prefrontal" in groups
    total = sum(len(v) for v in groups.values())
    assert total == 76


def test_regional_profiles():
    c = Connectome.from_bundled("tvb76")
    sim = BrainSimulator(c, global_coupling=0.01, params=_oscillatory_params())
    result = sim.simulate(duration=2000, dt=0.1, transient=500, seed=42)
    profiles = compute_regional_profiles(result)
    assert len(profiles) == 76
    for label, p in profiles.items():
        assert "mean_activity" in p
        assert "variance" in p
        assert "autocorrelation_tau" in p
        assert "peak_frequency" in p


def test_run_all_predictions():
    c = Connectome.from_bundled("tvb76")
    sim = BrainSimulator(c, global_coupling=0.01, params=_oscillatory_params())
    result = sim.simulate(duration=2000, dt=0.1, transient=500, seed=42)
    preds = run_all_predictions(result)
    assert "P1_thalamic_gating" in preds
    assert "P2_prefrontal_sustained" in preds
    assert "P4_frequency_differentiation" in preds
    assert "P5_regional_differentiation" in preds
